#!/usr/bin/env python3
"""WxFP8 Bit Partitioned Group MoE —— e4m3 激活 + 混合 bit 权重的 MoE。

**存储格式与 WxA16/WxA8 完全相同**（同一份 packed indices / codebook / norms，
checkpoint 通用），直接继承 `WxA16BitPartitionedGroupMoE`，覆盖三处：

  1. `_get_bit_context`  —— 追加每 bit 的 e4m3 码本 LUT 与 cb_scale
  2. `_build_hoisted_rotations` —— 预旋转之后顺带量化成 e4m3
  3. `_gate_up_matmul` / `_down_matmul` —— 换成 wxfp8 kernel

与 WxA8（quantization/wxa8/bit_partitioned_moe.py）逐点对应，差异只有：
  - build_int8_codebook → build_fp8_codebook（cb_step → cb_scale）
  - rotate_quantize_fused → rotate_quantize_fused_fp8
  - wxa8_* kernel → wxfp8_* kernel（无 IDENTITY_CB：e4m3 网格对 idx 非线性）

定位（roadmaps/wxfp8-plan-260924.md）：MoE 路径 fp8 速度中性
（实测 0.89~1.00x），精度上 2-bit 码本 e4m3 化近无损、激活 e4m3 2.57%
是主误差源——本路线是 WxFP4 的基础设施预备步骤，不为在 8-bit 上赢过 wxa8。
"""

import numpy as np
import torch

from quantization.wxa16.bit_partitioned_moe import WxA16BitPartitionedGroupMoE
from turboquant_utils.rotation import generate_batch_rotation_matrices
from turboquant_utils.triton_kernels_fp8 import (
    build_fp8_codebook,
    rotate_quantize_fused_fp8,
    wxfp8_matmul_grouped_slice_in_features_gf,
    wxfp8_matmul_grouped_slice_rows_gf,
)


class WxFP8BitPartitionedGroupMoE(WxA16BitPartitionedGroupMoE):
    """WxFP8 版 MoE。构造/加载与 WxA16 一致，只有 forward 里的两个 matmul 不同。"""

    @classmethod
    def from_wxa16(cls, moe: WxA16BitPartitionedGroupMoE) -> "WxFP8BitPartitionedGroupMoE":
        """把已加载的 WxA16 模块原地转成 WxFP8。

        存储格式完全相同，只换 __class__，零张量拷贝、零显存增长。
        接受任何已是 WxA16 子类的实例（含已转成 WxA8 的——回到同一存储格式）。
        """
        if not isinstance(moe, WxA16BitPartitionedGroupMoE):
            raise TypeError(f"期望 WxA16BitPartitionedGroupMoE，得到 {type(moe)}")
        moe.__class__ = cls
        # bit context 需要重建（要追加 e4m3 码本）
        moe._bit_ctx_cache = None
        moe._fp8_ctx_ready = None
        return moe

    # ---------------------------------------------------------------- context

    def _get_bit_context(self, device):
        """在 WxA16 的 bit context 上追加 e4m3 码本 LUT 与 cb_scale。

        cb_scale（而非 cb_step）折进激活 scale —— 与 WxA8 的机制完全同构，
        权重侧直接复用 WxA16 的 fp16 `norms_gf`。
        """
        ctx = super()._get_bit_context(device)
        if getattr(self, "_fp8_ctx_ready", None) is ctx:
            return ctx

        for c in ctx.values():
            gu_cb_f8, gu_scale = build_fp8_codebook(c["gate_up_codebook"])
            dn_cb_f8, dn_scale = build_fp8_codebook(c["down_codebook"])
            c["gate_up_cb_f8"] = gu_cb_f8
            c["gate_up_cb_scale"] = gu_scale
            c["down_cb_f8"] = dn_cb_f8
            c["down_cb_scale"] = dn_scale
            # 激活 dtype：kernel 固定产出 fp16，down 输出 cast 回原激活 dtype
            # （真实模型 bf16），与 WxA8 的同款约定
            c["act_dtype"] = self.gate.weight.dtype

        self._fp8_ctx_ready = ctx
        return ctx

    # ------------------------------------------------------- hoisted rotation

    def _build_hoisted_rotations(self, x, bit_ctx, tokens_per_expert):
        """预旋转 + 量化成 e4m3（覆盖 WxA16 的纯旋转版本，判据同 WxA8）。

        Returns:
            {bit_str: (x_f8, x_scale)}，只包含判定为划算的 bit。
        """
        x_src = x if x.dtype == torch.float16 else x.half()

        T = x.shape[0]
        cum = np.asarray(tokens_per_expert)
        counts = np.diff(np.concatenate(([0], cum)))

        out = {}
        for bit_str, ctx in bit_ctx.items():
            if not ctx["gf_gate_up"]:
                continue
            gs = ctx["gate_up_group_size"]
            in_f = ctx["gate_up_in_features"]
            if in_f % gs != 0 or in_f // gs < 2:
                continue

            rows = int(counts[self._expert_mask_by_bit[bit_str]].sum())
            if rows <= T * self.rotation_hoist_threshold:
                continue  # 预旋转不划算（长尾 bit）

            rot = generate_batch_rotation_matrices(
                gs, ctx["gate_up_seed"], in_f // gs, stride=gs,
                device=x_src.device, dtype=torch.float16)
            out[bit_str] = rotate_quantize_fused_fp8(
                x_src, rot, gs, in_f // gs,
                extra_scale=ctx["gate_up_cb_scale"])

        return out

    # ------------------------------------------------------------- matmul 覆盖

    def _gate_up_matmul(self, ctx, bit, bit_str, x, x_rot_by_bit,
                        exp_token_idx, expert_tokens, start, end, prof):
        """gate_up 方向的 wxfp8 matmul（无条件走 fp8，取舍同 WxA8）。"""
        gu = ctx["gate_up"]
        gs = ctx["gate_up_group_size"]

        hoisted = x_rot_by_bit.get(bit_str)
        if hoisted is not None:
            x_f8_all, x_scale_all = hoisted
            with prof.stage("gather_rotated"):
                inp_f8 = x_f8_all[exp_token_idx]
                inp_scale = x_scale_all[exp_token_idx]
        else:
            # 该 bit 没做预旋转（长尾 bit）：逐 expert 旋转 + 量化（融合 kernel）
            if expert_tokens is None:
                with prof.stage("gather_x"):
                    expert_tokens = x[exp_token_idx]
            with prof.stage("act_quant"):
                src = expert_tokens
                if src.dtype != torch.float16:
                    src = src.half()
                in_f = ctx["gate_up_in_features"]
                rot = generate_batch_rotation_matrices(
                    gs, ctx["gate_up_seed"], in_f // gs, stride=gs,
                    device=src.device, dtype=torch.float16)
                inp_f8, inp_scale = rotate_quantize_fused_fp8(
                    src, rot, gs, in_f // gs,
                    extra_scale=ctx["gate_up_cb_scale"])

        with prof.stage("gate_up_kernel"):
            gate_up_out = wxfp8_matmul_grouped_slice_rows_gf(
                inp_f8, inp_scale,
                gu["indices_packed_gf"],
                ctx["gate_up_cb_f8"],
                gu["norms_gf"],
                gs,
                ctx["gate_up_in_features"],
                2 * start, 2 * end,  # row slice
                bit,
                norms_prescaled=ctx["gate_up_norms_prescaled"],
            )
        del inp_f8, inp_scale
        return gate_up_out, expert_tokens

    def _down_matmul(self, ctx, bit, act_out, start, end, prof):
        """down 方向的 wxfp8 matmul（旋转 seed 含 expert 偏移，只能逐 expert 量化）。"""
        dn = ctx["down"]
        gs = ctx["down_group_size"]
        num_groups = (end - start) // gs

        with prof.stage("act_quant_down"):
            src = act_out if act_out.dtype == torch.float16 else act_out.half()
            rot = generate_batch_rotation_matrices(
                gs, ctx["down_seed"] + start, num_groups, stride=gs,
                device=src.device, dtype=torch.float16)
            act_f8, act_scale = rotate_quantize_fused_fp8(
                src, rot, gs, num_groups,
                extra_scale=ctx["down_cb_scale"])

        with prof.stage("down_kernel"):
            down_out = wxfp8_matmul_grouped_slice_in_features_gf(
                act_f8, act_scale,
                dn["indices_packed_gf"],
                ctx["down_cb_f8"],
                dn["norms_gf"],
                gs,
                start, end,  # original_start, original_end
                bit,
                norms_prescaled=ctx["down_norms_prescaled"],
            )
        del act_f8, act_scale
        # 对齐 WxA16 wrapper 的语义：输出 cast 回原激活 dtype（真实模型 bf16）
        if down_out.dtype != ctx["act_dtype"]:
            down_out = down_out.to(ctx["act_dtype"])
        return down_out
