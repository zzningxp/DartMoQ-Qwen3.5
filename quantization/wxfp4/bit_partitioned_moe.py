#!/usr/bin/env python3
"""WxFP4 Bit Partitioned Group MoE —— 保守版混合部署（WF4-4 / 3-3）。

**保守版定义**（roadmaps/wxfp4-plan-260924.md WF4-4，2026-09-25 本人拍板）：
  - bit ∈ {1, 4} 的 expert → **e2m1 激活 + FP4 MX MMA**（1-bit 连续 scale 下
    码本精确落格 relerr 0；4-bit 等尺寸零膨胀）
  - bit == 2 的主力 expert → **维持 WxA8**（int8 路径，继承 WxA8 类原样复用）
  - attention / shared expert 8-bit linear → WxA8（由 convert_model_to_wxfp4
    按 wxa8 同款规则转换）

存储格式与 WxA16/WxA8/WxFP8 完全相同（checkpoint 通用）。fp4 特有产物全部
加载期构建：nibble LUT（连续 scale + residual）、常量 e8m0 块缩放、
norms×residual 的折算副本（**不改共享 buffer，只替换 ctx 字典项**——
bit 2 的 int8 路径读同一 buffer，不能污染）。

⚠ 运行环境：fp4 kernel 只在 triton ≥ 3.8 下可编译（dart312-t38）；
本模块 import 在 dart312 下安全（JIT 延迟到首次 forward），但 forward 会崩。
"""

import numpy as np
import torch

from quantization.wxa8.bit_partitioned_moe import WxA8BitPartitionedGroupMoE
from quantization.wxa16.bit_partitioned_moe import WxA16BitPartitionedGroupMoE
from turboquant_utils.rotation import generate_batch_rotation_matrices
from turboquant_utils.triton_kernels_a8 import rotate_quantize_fused
from turboquant_utils.triton_kernels_fp4 import (
    build_e2m1_codebook,
    rotate_quantize_fused_fp4,
    wxfp4_matmul_grouped_slice_in_features_gf_swapped,
    wxfp4_matmul_grouped_slice_rows_gf_swapped,
)


class WxFP4BitPartitionedGroupMoE(WxA8BitPartitionedGroupMoE):
    """保守版 wxfp4 MoE：bit 1/4 走 fp4，bit 2 走 wxa8（int8）。

    分派发生在 _gate_up_matmul / _down_matmul（bit 是 int）；
    hoisted 旋转量化按 bit_str 分派量化器。
    """

    # 保守版 fp4 集合；3-4 激进版把 2 加进来即可（入口条件见 roadmap）
    FP4_BITS = {1, 4}

    @classmethod
    def from_wxa16(cls, moe) -> "WxFP4BitPartitionedGroupMoE":
        if not isinstance(moe, WxA16BitPartitionedGroupMoE):
            raise TypeError(f"期望 WxA16BitPartitionedGroupMoE，得到 {type(moe)}")
        moe.__class__ = cls
        moe._bit_ctx_cache = None
        moe._a8_ctx_ready = None
        moe._fp4_ctx_ready = None
        return moe

    # ---------------------------------------------------------------- context

    def _get_bit_context(self, device):
        """在 WxA8 的 context 上追加 fp4 产物（仅 FP4_BITS，按 bit_str 键判定）。

        fp4 分解链：w ≈ nib[idx] × 2^E0 × (residual × norms_gf)
          - nib/2^E0/residual 来自 build_e2m1_codebook（连续 scale 搜索）
          - 2^E0 进 e8m0 通道：w_s 常量张量（gate_up / down 全矩阵）
          - residual 折进 norms：替换 ctx 字典项为折算副本（不动共享 buffer）
        """
        ctx = super()._get_bit_context(device)
        if getattr(self, "_fp4_ctx_ready", None) is ctx:
            return ctx

        for bit_str, c in ctx.items():
            try:
                bit = int(bit_str)
            except (TypeError, ValueError):
                continue
            if bit not in self.FP4_BITS:
                continue

            gs_gu = c["gate_up_group_size"]
            gu = c["gate_up"]
            nib, e0, resid, _ = build_e2m1_codebook(c["gate_up_codebook"])
            c["gate_up_nib"] = nib.to(device)
            n_gu = gu["indices_packed_gf"].shape[1]
            k_gu = c["gate_up_in_features"]
            c["gate_up_ws"] = torch.full(
                (n_gu, k_gu // 32), 127 + e0, dtype=torch.uint8, device=device)
            gu["norms_gf"] = (gu["norms_gf"].float() * resid).half().contiguous()

            gs_dn = c["down_group_size"]
            dn = c["down"]
            nib_d, e0_d, resid_d, _ = build_e2m1_codebook(c["down_codebook"])
            c["down_nib"] = nib_d.to(device)
            n_dn = dn["indices_packed_gf"].shape[1]
            k_dn = dn["indices_packed_gf"].shape[0] * gs_dn
            c["down_ws"] = torch.full(
                (n_dn, k_dn // 32), 127 + e0_d, dtype=torch.uint8, device=device)
            dn["norms_gf"] = (dn["norms_gf"].float() * resid_d).half().contiguous()

        self._fp4_ctx_ready = ctx
        return ctx

    # ------------------------------------------------------- hoisted rotation

    def _build_hoisted_rotations(self, x, bit_ctx, tokens_per_expert):
        """按 bit 分派量化器：FP4_BITS → e2m1 融合量化；其余 → int8（WxA8 同款）。

        判据/结构与 WxA8 版一致（bit_str 键、阈值跳过长尾 bit、
        旋转矩阵只依赖 (group_size, seed) 与 expert 无关）。
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
                continue

            rot = generate_batch_rotation_matrices(
                gs, ctx["gate_up_seed"], in_f // gs, stride=gs,
                device=x_src.device, dtype=torch.float16)
            try:
                is_fp4 = int(bit_str) in self.FP4_BITS
            except (TypeError, ValueError):
                is_fp4 = False
            if is_fp4:
                # fp4：纯 MX 量化（residual 已折进 norms，激活通道无 extra_scale）
                out[bit_str] = rotate_quantize_fused_fp4(x_src, rot, gs, in_f // gs)
            else:
                out[bit_str] = rotate_quantize_fused(
                    x_src, rot, gs, in_f // gs,
                    extra_scale=ctx["gate_up_cb_step"])
        return out

    # ------------------------------------------------------------- matmul 分派

    def _gate_up_matmul(self, ctx, bit, bit_str, x, x_rot_by_bit,
                        exp_token_idx, expert_tokens, start, end, prof):
        if bit not in self.FP4_BITS:
            return super()._gate_up_matmul(
                ctx, bit, bit_str, x, x_rot_by_bit,
                exp_token_idx, expert_tokens, start, end, prof)

        gu = ctx["gate_up"]
        gs = ctx["gate_up_group_size"]
        hoisted = x_rot_by_bit.get(bit_str)
        if hoisted is not None:
            x_f4_all, x_s_all = hoisted          # (in_f//2, T), (T, in_f//32)
            with prof.stage("gather_rotated"):
                inp_f4 = x_f4_all[:, exp_token_idx]
                inp_s = x_s_all[exp_token_idx]
        else:
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
                inp_f4, inp_s = rotate_quantize_fused_fp4(src, rot, gs, in_f // gs)

        with prof.stage("gate_up_kernel"):
            gate_up_out = wxfp4_matmul_grouped_slice_rows_gf_swapped(
                inp_f4, inp_s,
                gu["indices_packed_gf"],
                ctx["gate_up_nib"],
                ctx["gate_up_ws"],
                gu["norms_gf"],
                gs,
                ctx["gate_up_in_features"],
                2 * start, 2 * end,
                bit,
                norms_prescaled=ctx["gate_up_norms_prescaled"],
            )
        del inp_f4, inp_s
        return gate_up_out, expert_tokens

    def _down_matmul(self, ctx, bit, act_out, start, end, prof):
        if bit not in self.FP4_BITS:
            return super()._down_matmul(ctx, bit, act_out, start, end, prof)

        dn = ctx["down"]
        gs = ctx["down_group_size"]
        num_groups = (end - start) // gs

        with prof.stage("act_quant_down"):
            src = act_out if act_out.dtype == torch.float16 else act_out.half()
            rot = generate_batch_rotation_matrices(
                gs, ctx["down_seed"] + start, num_groups, stride=gs,
                device=src.device, dtype=torch.float16)
            act_f4, act_s = rotate_quantize_fused_fp4(src, rot, gs, num_groups)

        with prof.stage("down_kernel"):
            down_out = wxfp4_matmul_grouped_slice_in_features_gf_swapped(
                act_f4, act_s,
                dn["indices_packed_gf"],
                ctx["down_nib"],
                ctx["down_ws"],
                dn["norms_gf"],
                gs,
                start, end,
                bit,
                norms_prescaled=ctx["down_norms_prescaled"],
            )
        del act_f4, act_s
        if down_out.dtype != ctx["act_dtype"]:
            down_out = down_out.to(ctx["act_dtype"])
        return down_out
