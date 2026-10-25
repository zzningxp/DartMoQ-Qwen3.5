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

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from quantization.wxa8.bit_partitioned_moe import WxA8BitPartitionedGroupMoE
from quantization.wxa16.bit_partitioned_moe import WxA16BitPartitionedGroupMoE
from turboquant_utils.rotation import generate_batch_rotation_matrices
from turboquant_utils.triton_kernels_a8 import (
    get_wxa8_config,
    rotate_quantize_fused,
)
from turboquant_utils.triton_kernels_fp4 import (
    _merged_fp4_default_cfg,
    build_e2m1_codebook,
    fold_indices_to_e2m1,
    rotate_quantize_fused_fp4,
    rotate_quantize_fused_fp4_pooled,
    wxfp4_matmul_grouped_slice_in_features_gf_swapped,
    wxfp4_matmul_grouped_slice_rows_gf_swapped,
    wxfp4_matmul_merged_in_features_gf,
    wxfp4_matmul_merged_in_features_gf_static,
    wxfp4_matmul_merged_rows_gf,
    wxfp4_matmul_merged_rows_gf_static,
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
        # 类切换后已构建的静态派发状态（若有人提前触发过）与新 kernel 语义
        # 不再匹配，作废重建
        moe._static_decode_state = None
        return moe

    # WFF-5 修复（2026-10-20）：fold 预算。e2m1 折叠副本只在**小池**时值得
    # 持有（2bpw 的 bit1/4 长尾池全模型仅 +47MB）；全 4bit 池（4bpw
    # checkpoint）的副本 ≈ 全量 MoE 索引，40 层 ctx 缓存累计 ~+10GB 把
    # 31.4GB 卡吃满（1018.thrput.4/5 两次 OOM 实测）。大池跳过 fold，
    # kernel 走 in-kernel LUT 分支（W_FOLDED=False，同一 LUT 同一映射，
    # 与折叠路径 bit-exact）——内存受限的 bit 级自适应。
    _FOLD_BYTES_BUDGET = 64 * 1024 * 1024

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
            # WFF-1：索引 → e2m1 load 期折叠（kernel 直读 W_FOLDED 分支，bit-exact；
            # bit4 零膨胀 / bit1 ×4 小池，全模型 +47MB 上限，见 fold plan §二）。
            # 新增 key 不替换 indices_packed_gf——共享 buffer 原样保留
            # WFF-5：超预算的池不折叠（indices_e2m1_gf 缺席 → kernel 走
            # in-kernel LUT 分支），见类级 _FOLD_BYTES_BUDGET 注释
            gu_fold_bytes = gu["indices_packed_gf"].numel() * (4 // bit)
            if gu_fold_bytes <= self._FOLD_BYTES_BUDGET:
                gu["indices_e2m1_gf"] = fold_indices_to_e2m1(
                    gu["indices_packed_gf"], c["gate_up_nib"], bit, gs_gu)

            gs_dn = c["down_group_size"]
            dn = c["down"]
            nib_d, e0_d, resid_d, _ = build_e2m1_codebook(c["down_codebook"])
            c["down_nib"] = nib_d.to(device)
            n_dn = dn["indices_packed_gf"].shape[1]
            k_dn = dn["indices_packed_gf"].shape[0] * gs_dn
            c["down_ws"] = torch.full(
                (n_dn, k_dn // 32), 127 + e0_d, dtype=torch.uint8, device=device)
            dn["norms_gf"] = (dn["norms_gf"].float() * resid_d).half().contiguous()
            dn_fold_bytes = dn["indices_packed_gf"].numel() * (4 // bit)
            if dn_fold_bytes <= self._FOLD_BYTES_BUDGET:
                dn["indices_e2m1_gf"] = fold_indices_to_e2m1(
                    dn["indices_packed_gf"], c["down_nib"], bit, gs_dn)

        if not getattr(self, "_fold_skip_logged", False):
            self._fold_skip_logged = True
            skipped = [
                bit_str for bit_str, c in ctx.items()
                if (int(bit_str) in self.FP4_BITS
                    and "indices_e2m1_gf" not in c["gate_up"])
            ]
            if skipped:
                print(f"  [wxfp4] WFF-5 budget: fold skipped for bit pools "
                      f"{skipped} (in-kernel LUT path)", flush=True)

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
                w_e2m1_gf=gu.get("indices_e2m1_gf"),
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
                w_e2m1_gf=dn.get("indices_e2m1_gf"),
            )
        del act_f4, act_s
        if down_out.dtype != ctx["act_dtype"]:
            down_out = down_out.to(ctx["act_dtype"])
        return down_out

    # ------------------------------------------------- P5-2 重做：合并路径

    def _expert_compute(self, x, bit_ctx, x_rot_by_bit, sorted_tokens,
                        sorted_weights, tokens_per_expert,
                        final_hidden_states, prof):
        """P5-2：WxFP4 合并路径——同一 forward 内按池混合分派。

        bit ∈ FP4_BITS 的池走 e2m1 MX 合并 kernel（in-kernel LUT，有
        WFF-5 小池折叠副本时直读），bit 2 的池继承 WxA8 的 int8 合并
        kernel（_merged_pool_compute 默认实现）。仅 WxFP4 本类生效
        （精确类判定，防子类误继承）；DARTMOQ_MERGED_MOE=0 回退逐 expert。
        门控含 _merged_groups_ok：小形状（quant_io 骨架 down 1 group）
        回退逐 expert，与 wxa16 同判据。
        """
        if (self.__class__ is WxFP4BitPartitionedGroupMoE
                and os.environ.get("DARTMOQ_MERGED_MOE", "1") != "0"
                and self._merged_pools_ok()
                and self._merged_groups_ok(bit_ctx)
                and all(c.get("gf_gate_up") and c.get("gf_down")
                        for c in bit_ctx.values())):
            return self._expert_compute_merged(
                x, bit_ctx, x_rot_by_bit, sorted_tokens, sorted_weights,
                tokens_per_expert, final_hidden_states, prof)
        return super()._expert_compute(
            x, bit_ctx, x_rot_by_bit, sorted_tokens, sorted_weights,
            tokens_per_expert, final_hidden_states, prof)

    def _merged_pool_compute(self, bit_str, members, x, bit_ctx, x_rot_by_bit,
                             sorted_tokens, sorted_weights,
                             final_hidden_states, prof):
        """fp4 池的合并计算（覆盖 WxA8 的 int8 默认实现）。

        编排与 wxa8 版逐点对应（gather→合并 gate_up→silu×up→逐 expert
        旋转量化→合并 down→整池 scatter）；fp4 差异：激活量化产物是
        预转置 e2m1 (K//2, B)（hoisted 列 gather / 整池一次融合量化）、
        权重侧 nibble LUT / folded 直读、W 块缩放走 e8m0 常量通道。
        """
        if members[0][1] not in self.FP4_BITS:
            return super()._merged_pool_compute(
                bit_str, members, x, bit_ctx, x_rot_by_bit,
                sorted_tokens, sorted_weights, final_hidden_states, prof)

        t_pool = 0.0
        ctx = bit_ctx[bit_str]
        bit = members[0][1]
        inter = members[0][3] - members[0][2]
        gs_gu = ctx["gate_up_group_size"]
        in_f = ctx["gate_up_in_features"]
        gs_dn = ctx["down_group_size"]

        # 池内 token/权重（各成员行段按序拼接，一次 gather）
        pool_tokens = torch.cat(
            [sorted_tokens[ts:ts + n] for _e, _b, _s, _e2, ts, n in members])
        pool_weights = torch.cat(
            [sorted_weights[ts:ts + n] for _e, _b, _s, _e2, ts, n in members])

        # gate_up 输入：hoisted（e2m1 预转置）则列 gather；否则整池一次
        # 旋转量化（旋转/量化逐行独立，整池 ≡ 逐 expert 拼接，P6-1 同判据）
        hoisted = x_rot_by_bit.get(bit_str)
        if hoisted is not None:
            x_f4_all, x_s_all = hoisted            # (in_f//2, T), (T, in_f//32)
            x_pool_f4 = x_f4_all[:, pool_tokens]   # 列 gather（token 是列）
            x_pool_s = x_s_all[pool_tokens]
        else:
            src = x[pool_tokens]
            if src.dtype != torch.float16:
                src = src.half()
            rot = generate_batch_rotation_matrices(
                gs_gu, ctx["gate_up_seed"], in_f // gs_gu, stride=gs_gu,
                device=src.device, dtype=torch.float16)
            x_pool_f4, x_pool_s = rotate_quantize_fused_fp4(
                src, rot, gs_gu, in_f // gs_gu)

        # 合并 gate_up（列切片 2*start / 2*inter，池内统一）
        expert_info_gu = self._merged_expert_info_rows(members)

        with prof.stage("merged_gate_up"):
            t_k = time.time()
            gu = wxfp4_matmul_merged_rows_gf(
                x_pool_f4, x_pool_s,
                ctx["gate_up"]["indices_packed_gf"], ctx["gate_up_nib"],
                ctx["gate_up_ws"], ctx["gate_up"]["norms_gf"],
                expert_info_gu, gs_gu, in_f, bit,
                norms_prescaled=ctx["gate_up_norms_prescaled"],
                w_e2m1_gf=ctx["gate_up"].get("indices_e2m1_gf"))
            t_pool += time.time() - t_k
        del x_pool_f4, x_pool_s

        # silu×up 一次（整池扁平输出，gate 前 inter 列 up 后 inter 列）
        with prof.stage("silu_mul"):
            gate = gu[:, :inter]
            up = gu[:, inter:]
            del gu
            act = F.silu(gate) * up
            del gate, up

        # down：逐 expert 旋转量化（seed + start；v1 每 expert 一次 launch，
        # 产物预转置 (inter//2, n)，按列拼接）
        f4_parts, s_parts = [], []
        cursor = 0
        for _e, _b, start, _e2, _ts, n in members:
            act_e = act[cursor:cursor + n]
            rot = generate_batch_rotation_matrices(
                gs_dn, ctx["down_seed"] + start, inter // gs_dn,
                stride=gs_dn, device=act.device, dtype=torch.float16)
            a_f4, a_s = rotate_quantize_fused_fp4(
                act_e, rot, gs_dn, inter // gs_dn)
            f4_parts.append(a_f4)
            s_parts.append(a_s)
            cursor += n
        del act
        expert_info_dn = self._merged_expert_info_groups(members, gs_dn)

        with prof.stage("merged_down"):
            t_k = time.time()
            dn = wxfp4_matmul_merged_in_features_gf(
                torch.cat(f4_parts, dim=1), torch.cat(s_parts),
                ctx["down"]["indices_packed_gf"], ctx["down_nib"],
                ctx["down_ws"], ctx["down"]["norms_gf"],
                expert_info_dn, gs_dn, bit,
                norms_prescaled=ctx["down_norms_prescaled"],
                w_e2m1_gf=ctx["down"].get("indices_e2m1_gf"))
            t_pool += time.time() - t_k
        del f4_parts, s_parts

        # 对齐逐 expert 路径：输出 cast 回激活 dtype（真实模型 bf16）
        if dn.dtype != ctx["act_dtype"]:
            dn = dn.to(ctx["act_dtype"])

        # 整池 scatter 一次（同 token 跨池出现时由 index_add_ 累加）
        with prof.stage("scatter"):
            dn.mul_(pool_weights.unsqueeze(1))
            final_hidden_states.index_add_(0, pool_tokens, dn)
        del dn, pool_tokens, pool_weights

        return t_pool

    # ------------------------------------------------- 静态派发（graph decode）

    _STATIC_DISPATCH_READY = True

    @staticmethod
    def _static_cfg_getter(bit):
        """按池 bit 分派选档：FP4_BITS 池用 fp4 默认 tile，其余用 wxa8。"""
        if bit in WxFP4BitPartitionedGroupMoE.FP4_BITS:
            return lambda direction, B: _merged_fp4_default_cfg(B)
        return lambda direction, B: get_wxa8_config(bit, direction=direction, B=B)

    def _static_pool_compute(self, ps, x, tokens_c, weights_c, mask_rows,
                             final_hidden_states):
        """单个 bit 池的静态计算（按池 bit 分派，与 _merged_pool_compute 同数学）。

        FP4_BITS 池：e2m1 预转置激活（gate_up 全量一次融合旋转量化后**列**
        gather；down 池内逐成员 pooled kernel，预转置写出）；其余池走
        WxA8 的 int8 静态实现（原样复用）。fp4 无 extra_scale 通道（cb 类
        常量在权重侧 fold 进 nib/residual，激活侧恒 1）。
        """
        if ps.bit not in self.FP4_BITS:
            return super()._static_pool_compute(
                ps, x, tokens_c, weights_c, mask_rows, final_hidden_states)

        src = x if x.dtype == torch.float16 else x.half()
        x_f4_all, x_s_all = rotate_quantize_fused_fp4(
            src, ps.rot_gu, ps.gs_gu, ps.in_f // ps.gs_gu)
        x_pool = x_f4_all.index_select(1, tokens_c)
        xs_pool = x_s_all.index_select(0, tokens_c)
        del x_f4_all, x_s_all
        gu = wxfp4_matmul_merged_rows_gf_static(
            x_pool, xs_pool, ps.idx_gu_gf, ps.nib_gu, ps.ws_gu, ps.norms_gu_gf,
            ps.expert_info_gu, ps.block_info_gu, ps.n_slice, ps.n_limit,
            ps.gs_gu, ps.in_f, ps.bit,
            norms_prescaled=ps.prescaled_gu, w_e2m1_gf=ps.e2m1_gu,
            cfg=ps.cfg_gu)
        del x_pool, xs_pool
        act = F.silu(gu[:, :ps.inter]) * gu[:, ps.inter:]
        del gu
        act = act * mask_rows.unsqueeze(1).to(act.dtype)
        act16 = act if act.dtype == torch.float16 else act.half()
        a_f4, a_s = rotate_quantize_fused_fp4_pooled(
            act16, ps.rot_bank, ps.expert_info_dn, ps.block_info_b32,
            ps.gs_dn, ps.g_count)
        del act16
        dn = wxfp4_matmul_merged_in_features_gf_static(
            a_f4, a_s, ps.idx_dn_gf, ps.nib_dn, ps.ws_dn, ps.norms_dn_gf,
            ps.expert_info_dn, ps.block_info_dn, ps.gs_dn, ps.bit,
            norms_prescaled=ps.prescaled_dn, w_e2m1_gf=ps.e2m1_dn,
            cfg=ps.cfg_dn)
        del a_f4, a_s
        if dn.dtype != x.dtype:
            dn = dn.to(x.dtype)
        # padding 行显式清零再乘零权重（防 0×inf/NaN，见基类 _static_scatter_tail）
        self._static_scatter_tail(dn, tokens_c, weights_c, mask_rows,
                                  final_hidden_states)
        del dn, act
