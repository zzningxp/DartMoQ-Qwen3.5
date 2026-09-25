# -*- coding: utf-8 -*-
"""WxFP4 路线的 Triton kernel 与工具（roadmaps/wxfp4-plan-260924.md WF4-2）。

DP-2 定稿后的主线：全 e2m1×e2m1（mixed e2m1×e4m3 实测走 bf16 模拟，已否）。

**运行环境约束（WF4-P1 实测，详见 roadmap/memory）**：
  - 本文件的全部 kernel 只能在 **triton ≥ 3.8.0** 下编译（3.5.0 正确布局下
    dot_scaled 在 AccelerateMatmul pass 崩溃），且 ptxas-blackwell 须为 CUDA 12.8/12.9
    （13.3 产的 sm_120a cubin 被 570 驱动拒载）→ 用 dart312-t38 env 跑；
    dart312（triton 3.5）下 import 本模块安全（JIT 延迟到首次调用），但调用会崩。
  - rhs 布局：value (K/2, N) 沿 K 打包、scale (N, K/32)——scale 与 value 转置。
  - scale 不走 TMA（box 内维 4~8 字节违反 16B 对齐），指针加载。

数学结构（与 wxfp8 的 LUT + cb_scale 分解同构）：
  W 侧：w ≈ nib[idx] × 2^E0 × norms_gf[g,n]
    - nib[]：码本索引 → e2m1 nibble（0..15，sign<<3|mag）的全局 LUT，
      离线用「2 的幂约束」的 scale 搜索求最优（build_e2m1_codebook）
    - 2^E0：常量（W 侧 e8m0 块缩放全 0x7F+E0，可被 MMA 的 scale 通道吸收）
    - norms_gf：fp16 per-(group,n)，epilogue 兜底（沿用 WxA16 同一份 buffer）
  A 侧：x ≈ e2m1_nibble × 2^(per-32 块 e8m0)（MX 原生格式，无 fp16 scale）
"""

import torch

# e2m1 网格（nibble 低 3 位 = 幅度索引）
E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_MAX = 6.0


def build_e2m1_codebook(codebook: torch.Tensor):
    """把 fp16/fp32 码本转成 e2m1 nibble LUT，返回 (nib_lut, pow2_scale)。

    满足 `codebook[i] ≈ e2m1_value(nib_lut[i]) * pow2_scale`，且 pow2_scale
    是 2 的整数幂——这样 W 侧的 per-32 e8m0 块缩放可以取全局常量
    （block_scale = 0x7F + log2(pow2_scale)），分解链与 wxfp8 的
    `cb ≈ lut × cb_scale`（折进激活 extra_scale）完全同构，norms 侧不动。

    scale 搜索空间限制为 2^e（e ∈ [-31, 31]）：e2m1 浮点网格尺度不变，
    但只有 7 个幅度，比 e4m3 的搜索（连续 scale）余量小；
    2-bit 码本 4 级的最优比值匹配误差实测见 test（预期 ~5-10%，
    被 2-bit 量化噪声本身 ~34% 淹没，端到端影响可忽略）。
    """
    cb = codebook.float()
    cb_max = cb.abs().max()
    if cb_max <= 0:
        raise ValueError("码本全零，无法转 e2m1")
    grid = torch.tensor(E2M1_GRID, device=cb.device)

    best_e, best_err = 0, float("inf")
    best_lut = None
    for e in range(-31, 32):
        s = 2.0 ** e
        if s > cb_max * 4:  # 网格最左非零 0.5×s 已超码本最大值太多，跳过粗扫头部
            continue
        y = (cb / s).clamp(-E2M1_MAX, E2M1_MAX)
        mag = torch.argmin((y.abs().unsqueeze(-1) - grid).abs(), dim=-1)
        nib = (mag & 0x7) | ((y < 0).to(torch.uint8) << 3)
        rec = grid[mag] * y.sign() * s
        err = ((rec - cb).norm() / cb.norm()).item()
        if err < best_err:
            best_e, best_err, best_lut = e, err, nib
    return best_lut.contiguous(), 2.0 ** best_e, best_e


def quantize_act_e2m1_per32(x_rot: torch.Tensor):
    """per-32 块 e8m0 缩放的 e2m1 激活量化（torch 参考实现）。

    MX 原生格式：scale = 2^ceil(log2(amax/6))（纯 2 的幂），余量由 satfinite
    语义的网格选择吸收。relerr 实测 0.1155（vs e4m3 per-128 的 0.0257，
    见 WF4-P1 ②）。

    Args:
        x_rot: (B, K) 已旋转的激活（K 为 32 的倍数）

    Returns:
        packed: (B, K/2) uint8（偶 k 在低 nibble）
        scale_b: (B, K/32) uint8（e8m0 编码 byte = exp + 127）
    """
    B, K = x_rot.shape
    xb = x_rot.float().reshape(B, K // 32, 32)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    e = torch.ceil(torch.log2(amax / E2M1_MAX))
    s = torch.pow(2.0, e).clamp(min=1e-30)
    scale_b = (e + 127.0).clamp(0, 254).to(torch.uint8).reshape(B, K // 32)
    y = (xb / s).clamp(-E2M1_MAX, E2M1_MAX)
    grid = torch.tensor(E2M1_GRID, device=x_rot.device)
    mag = torch.argmin((y.abs().unsqueeze(-1) - grid).abs(), dim=-1)
    nib = ((mag & 0x7).to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)).reshape(B, K)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return packed.contiguous(), scale_b.contiguous()


# ===========================================================================
# WF4-2：融合 GEMM kernel（A=e2m1+per-32 e8m0，W=码本索引→e2m1 nibble）
# ===========================================================================
# ⚠ 只能在 triton ≥ 3.8 的 env 下编译运行（见模块 docstring）；bring-up 版为
# 全指针加载（WF4-P1 已证指针加载同样产出原生 mxf4nvf4），TMA 优化后置。

import triton
import triton.language as tl


@triton.jit
def _wxfp4_fused_matmul_kernel_grouped_gf(
    # A 侧（MX 原生格式）
    a_ptr,        # (B, K_total//2) uint8 —— e2m1 沿 K 打包（偶 k 在低 nibble）
    a_s_ptr,      # (B, K_total//32) uint8 —— per-32 e8m0（byte = exp+127）
    # W 侧（packed 索引 + nibble LUT，checkpoint 不动）
    indices_ptr,  # (NUM_GROUPS_TOTAL, N, PACKED_PER_GROUP) uint8 —— 同 fp8 kernel
    nib_ptr,      # (2^bit,) uint8 —— 码本索引 → e2m1 nibble（build_e2m1_codebook）
    w_s_ptr,      # (N, K_total//32) uint8 —— 全常量 0x7F+E0（W 块缩放，转置布局）
    norms_ptr,    # (NUM_GROUPS_TOTAL, N) fp16 —— 与 WxA16 同源（norms 在 epilogue）
    # 输出
    output_ptr,   # (B, N) fp16
    # 形状
    B, N, K_total,
    INDICES_G0_STRIDE, NORMS_G0_STRIDE,
    # constexpr
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BIT_WIDTH: tl.constexpr,
    BLOCK_B: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """WxFP4 融合反量化 + MXFP4 matmul kernel（group-first 布局，bring-up 版）。

    与 _wxfp8_fused_matmul_kernel_grouped_gf 的结构对照：
      - A 侧从 e4m3+fp16 scale 变为 e2m1 打包 + per-32 e8m0（MX 原生）
      - W 侧从 e4m3 LUT 查表变为 nibble LUT 查表 + kernel 内拼 byte
        （byte = nib[k] | nib[k+1]<<4，tl.split 实现寄存器内配对）
      - dot 从 tl.dot(fp8,fp8) 变为 tl.dot_scaled(e2m1,e2m1)——per-32 缩放
        在 MMA 内部完成，epilogue 只剩 per-(g,n) 的 norms
    数学链：out[b,n] = Σ_g norm_g[n] · Σ_k (a_nib·2^a_e8) · (w_nib·2^E0)
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    total_acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    PACKED_PER_GROUP = GROUP_SIZE // ELEMENTS_PER_BYTE
    row_base = rn * PACKED_PER_GROUP

    for g in range(NUM_GROUPS):
        g_start = g * GROUP_SIZE
        g_base = g * INDICES_G0_STRIDE
        norm_g = tl.load(norms_ptr + g * NORMS_G0_STRIDE + rn, mask=mask_n, other=0.0)

        acc_g = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

        # A/scale 在 packed 域取 arange（byte 域 / e8m0 域），不是元素域除法
        rk_pk = tl.arange(0, BLOCK_K // 2)     # packed byte 索引
        rk_sc = tl.arange(0, BLOCK_K // 32)    # e8m0 块索引

        for k_start in range(0, GROUP_SIZE, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < GROUP_SIZE
            mask_kp = (k_start // 2 + rk_pk) < (GROUP_SIZE // 2)

            # --- A 侧：packed e2m1 (BM, BK/2) + e8m0 (BM, BK/32)，指针加载 ---
            a_off = rb[:, None] * (K_total // 2) + (g_start + k_start) // 2 + rk_pk[None, :]
            a_tile = tl.load(a_ptr + a_off,
                             mask=mask_b[:, None] & mask_kp[None, :], other=0)
            as_off = rb[:, None] * (K_total // 32) + (g_start + k_start) // 32 + rk_sc[None, :]
            as_tile = tl.load(a_s_ptr + as_off,
                              mask=mask_b[:, None]
                              & ((k_start // 32 + rk_sc) < (GROUP_SIZE // 32))[None, :],
                              other=0)

            # --- W 侧：unpack 索引 → nibble LUT → 拼 byte → (BK/2, BN) ---
            w_mask = mask_n[:, None] & mask_k[None, :]
            if BIT_WIDTH == 8:
                byte_off = g_base + row_base[:, None] + rk[None, :]
                idx = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.int32)
            else:
                BIT_MASK = (1 << BIT_WIDTH) - 1
                byte_col = rk // ELEMENTS_PER_BYTE
                pos_in_byte = rk % ELEMENTS_PER_BYTE
                byte_off = g_base + row_base[:, None] + byte_col[None, :]
                packed_w = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
                shift = pos_in_byte * BIT_WIDTH
                idx = ((packed_w >> shift[None, :]) & BIT_MASK).to(tl.int32)
            nib = tl.load(nib_ptr + idx, mask=w_mask, other=0)      # (BN, BK) uint8

            # 寄存器内配对打包：nib (BN, BK) -> (BN, BK/2, 2) -> lo|hi<<4
            nib3 = tl.reshape(nib, (BLOCK_N, BLOCK_K // 2, 2))
            lo, hi = tl.split(nib3)
            w_byte = (lo | (hi << 4)).to(tl.uint8)                   # (BN, BK/2)

            # W 块缩放（常量内容，转置布局 (BN, BK/32)）
            ws_off = rn[:, None] * (K_total // 32) + (g_start + k_start) // 32 + rk_sc[None, :]
            ws_tile = tl.load(w_s_ptr + ws_off,
                              mask=mask_n[:, None]
                              & ((k_start // 32 + rk_sc) < (GROUP_SIZE // 32))[None, :],
                              other=0)

            acc_g = tl.dot_scaled(a_tile, as_tile, "e2m1",
                                  tl.trans(w_byte), ws_tile, "e2m1", acc_g)

        total_acc += acc_g * norm_g[None, :]

    tl.store(output_ptr + rb[:, None] * N + rn[None, :],
             total_acc.to(output_ptr.dtype.element_ty),
             mask=mask_b[:, None] & mask_n[None, :])


def wxfp4_matmul_grouped_gf(
    a_packed, a_s, indices_packed_gf, nib, w_s, norms_gf,
    group_size, num_groups, bit_width: int,
    BLOCK_B=64, BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=3,
):
    """bring-up 版 wrapper（全矩阵，非切片）。"""
    if bit_width not in {1, 2, 4}:
        raise ValueError(f"WxFP4 支持 bit 1/2/4，got {bit_width}")
    if BLOCK_K > group_size or group_size % BLOCK_K != 0:
        raise ValueError(
            f"BLOCK_K ({BLOCK_K}) 必须 ≤ group_size ({group_size}) 且整除"
            "（W 侧 nibble 配对以 group 内连续 k 为界）")
    B = a_packed.shape[0]
    N = indices_packed_gf.shape[1]
    K_total = a_packed.shape[1] * 2
    out = torch.empty(B, N, dtype=torch.float16, device=a_packed.device)
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N, BLOCK_N))
    _wxfp4_fused_matmul_kernel_grouped_gf[grid](
        a_packed, a_s, indices_packed_gf, nib, w_s, norms_gf, out,
        B, N, K_total,
        indices_packed_gf.stride(0), norms_gf.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups, BIT_WIDTH=bit_width,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ===========================================================================
# WF4-2 优化迭代 1：swapped-operand（W 作 lhs 免 trans，A 预转置作 rhs）
# ===========================================================================

def quantize_act_e2m1_per32_t(x_rot: torch.Tensor):
    """quantize_act_e2m1_per32 的预转置变体：返回 (K/2, B) 的 packed 布局。

    swapped kernel 里 A 作 rhs 需要 (BK/2, BM) tile——把转置做在量化输出
    （transient，一次写），换取 kernel 内 W 侧完全免 trans。
    scale 仍为 (B, K/32)（dot_scaled 约定 scale 尾维 = K，不随 value 转置）。
    """
    packed, scale_b = quantize_act_e2m1_per32(x_rot)
    return packed.t().contiguous(), scale_b


@triton.jit
def _wxfp4_fused_matmul_kernel_swapped_gf(
    a_ptr,        # (K_total//2, B) uint8 —— e2m1 打包、预转置（行=k 对，列=b）
    a_s_ptr,      # (B, K_total//32) uint8 —— per-32 e8m0（scale 尾维=K，不转置）
    indices_ptr,  # (NUM_GROUPS_TOTAL, N, PACKED_PER_GROUP) uint8
    nib_ptr,      # (2^bit,) uint8
    w_s_ptr,      # (N, K_total//32) uint8 —— W 块缩放（lhs 尾维=K ✓）
    norms_ptr,    # (NUM_GROUPS_TOTAL, N) fp16
    output_ptr,   # (B, N) fp16
    B, N, K_total,
    INDICES_G0_STRIDE, NORMS_G0_STRIDE,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BIT_WIDTH: tl.constexpr,
    BLOCK_B: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """优化迭代 1：与 bring-up 版数学等价，布局变化——

      - lhs = W nibble 路径的天然方向 (BN, BK/2)——**免 tl.trans**（bring-up 版
        的 smem 布局转换大户）
      - rhs = A 预转置存储后的 (BK/2, BM) tile，直接加载
      - acc 为 (BN, BM)，store 时一次转置写出（每 tile 一次，代价远小于逐 k trans）
      - 乘法归约拼包：byte = Σ_{j∈{0,1}} nib3[...,j] × (16j+1)，替代 reshape+split
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    total_acc = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    PACKED_PER_GROUP = GROUP_SIZE // ELEMENTS_PER_BYTE
    row_base = rn * PACKED_PER_GROUP
    # 拼包权重 [1, 16]
    pk_w = tl.arange(0, 2) * 15 + 1

    for g in range(NUM_GROUPS):
        g_start = g * GROUP_SIZE
        g_base = g * INDICES_G0_STRIDE
        norm_g = tl.load(norms_ptr + g * NORMS_G0_STRIDE + rn, mask=mask_n, other=0.0)

        acc_g = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)

        rk_pk = tl.arange(0, BLOCK_K // 2)
        rk_sc = tl.arange(0, BLOCK_K // 32)

        for k_start in range(0, GROUP_SIZE, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < GROUP_SIZE
            mask_kp = (k_start // 2 + rk_pk) < (GROUP_SIZE // 2)

            # --- rhs = A：预转置 (BK/2, BM)，行连续 ---
            # ⚠ 转置布局下 group 偏移是行偏移：全局 packed 行号 = (g_start+k_start)/2 + j，
            # 地址 = 行号 × B（列数）+ b。group0 偏移为 0 时侥幸不错，>=1 必须乘 B。
            a_off = ((g_start + k_start) // 2 + rk_pk[:, None]) * B + rb[None, :]
            a_tile = tl.load(a_ptr + a_off,
                             mask=mask_kp[:, None] & mask_b[None, :], other=0)
            # rhs scale (BM, BK/32)（尾维 = K）
            as_off = rb[:, None] * (K_total // 32) + (g_start + k_start) // 32 + rk_sc[None, :]
            as_tile = tl.load(a_s_ptr + as_off,
                              mask=mask_b[:, None]
                              & ((k_start // 32 + rk_sc) < (GROUP_SIZE // 32))[None, :],
                              other=0)

            # --- lhs = W：unpack → nibble LUT → 乘法归约拼包（免 trans 免 split）---
            w_mask = mask_n[:, None] & mask_k[None, :]
            if BIT_WIDTH == 8:
                byte_off = g_base + row_base[:, None] + rk[None, :]
                idx = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.int32)
            else:
                BIT_MASK = (1 << BIT_WIDTH) - 1
                byte_col = rk // ELEMENTS_PER_BYTE
                pos_in_byte = rk % ELEMENTS_PER_BYTE
                byte_off = g_base + row_base[:, None] + byte_col[None, :]
                packed_w = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
                shift = pos_in_byte * BIT_WIDTH
                idx = ((packed_w >> shift[None, :]) & BIT_MASK).to(tl.int32)
            nib = tl.load(nib_ptr + idx, mask=w_mask, other=0)      # (BN, BK) uint8

            nib3 = tl.reshape(nib, (BLOCK_N, BLOCK_K // 2, 2))
            w_byte = tl.sum(nib3.to(tl.int32) * pk_w[None, None, :], axis=2).to(tl.uint8)

            # lhs scale (BN, BK/32)
            ws_off = rn[:, None] * (K_total // 32) + (g_start + k_start) // 32 + rk_sc[None, :]
            ws_tile = tl.load(w_s_ptr + ws_off,
                              mask=mask_n[:, None]
                              & ((k_start // 32 + rk_sc) < (GROUP_SIZE // 32))[None, :],
                              other=0)

            acc_g = tl.dot_scaled(w_byte, ws_tile, "e2m1",
                                  a_tile, as_tile, "e2m1", acc_g)

        total_acc += acc_g * norm_g[:, None]

    # (BN, BM) -> output (B, N)：一次转置写
    tl.store(output_ptr + rb[None, :] * N + rn[:, None],
             total_acc.to(output_ptr.dtype.element_ty),
             mask=mask_b[None, :] & mask_n[:, None])


def wxfp4_matmul_grouped_gf_swapped(
    a_packed_t, a_s, indices_packed_gf, nib, w_s, norms_gf,
    group_size, num_groups, bit_width: int,
    BLOCK_B=128, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=2,
):
    """swapped-operand 版 wrapper（a_packed_t 为 quantize_act_e2m1_per32_t 产物）。"""
    if bit_width not in {1, 2, 4}:
        raise ValueError(f"WxFP4 支持 bit 1/2/4，got {bit_width}")
    if BLOCK_K > group_size or group_size % BLOCK_K != 0:
        raise ValueError(
            f"BLOCK_K ({BLOCK_K}) 必须 ≤ group_size ({group_size}) 且整除")
    B = a_packed_t.shape[1]
    N = indices_packed_gf.shape[1]
    K_total = a_packed_t.shape[0] * 2
    out = torch.empty(B, N, dtype=torch.float16, device=a_packed_t.device)
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N, BLOCK_N))
    _wxfp4_fused_matmul_kernel_swapped_gf[grid](
        a_packed_t, a_s, indices_packed_gf, nib, w_s, norms_gf, out,
        B, N, K_total,
        indices_packed_gf.stride(0), norms_gf.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups, BIT_WIDTH=bit_width,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ===========================================================================
# WF4-2 迭代 2：专家级通用纯 MX GEMM（普通 MoE 主路径）
# ===========================================================================
# 定位（2026-09-25 本人定调）：以专家为单位、面向普通 MoE（无 bit-partitioned
# 子专家结构）。W 在量化/加载期直量化为 e2m1 + per-32 e8m0（等尺寸，0.5B/elem，
# 不经 Lloyd-Max 码本），kernel 内零转换——与 546 TF 参照 kernel 同构。
# 本项目的 bit-partitioned checkpoint 由转换 kernel 适配（见后续）。

def quantize_weight_e2m1(w: torch.Tensor, global_scale_search: bool = True):
    """权重直量化为 MX 格式（e2m1 + per-32 e8m0 + 可选全局 fp32 scale）。

    与 NVFP4 的两级缩放思想同构：全局 scale（任意 fp32，epilogue 一次乘）
    吸收 per-32 e8m0 只能取 2 的幂造成的量化余量，可显著降低总误差
    （搜索在量化期一次完成，权重是静态的）。

    Args:
        w: (N, K) fp16/fp32 权重
    Returns:
        packed: (N, K/2) uint8
        scales: (N, K/32) uint8（e8m0）
        global_scale: float（global_scale_search=False 时为 1.0）
    """
    N, K = w.shape
    grid = torch.tensor(E2M1_GRID, device=w.device)

    def quant_with(g):
        wb = w.float().reshape(N, K // 32, 32) * g
        amax = wb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        e = torch.ceil(torch.log2(amax / E2M1_MAX))
        s = torch.pow(2.0, e).clamp(min=1e-30)
        y = (wb / s).clamp(-E2M1_MAX, E2M1_MAX)
        mag = torch.argmin((y.abs().unsqueeze(-1) - grid).abs(), dim=-1)
        nib = ((mag & 0x7) | (((y < 0).to(torch.uint8) << 3).long())).to(torch.uint8)
        deq = (grid[mag] * y.sign() * s / g).reshape(N, K)
        return nib.reshape(N, K), ((deq - w.float()).norm() / w.float().norm()).item()

    best_g, best_err, best_nib = 1.0, float("inf"), None
    if global_scale_search:
        # 对数粗扫 + 两轮细化（g 放大让块用满 6 的倍率更接近 amax）
        import math
        cands = [2.0 ** (i / 16.0) for i in range(-16, 17)]
        for g in cands:
            nib, err = quant_with(g)
            if err < best_err:
                best_g, best_err, best_nib = g, err, nib
        for _ in range(2):
            span = cands[1] / cands[0]
            lo, hi = best_g / span, best_g * span
            cands = [2.0 ** (math.log2(lo) + (math.log2(hi) - math.log2(lo)) * i / 64)
                     for i in range(65)]
            for g in cands:
                nib, err = quant_with(g)
                if err < best_err:
                    best_g, best_err, best_nib = g, err, nib
    else:
        best_nib, _ = quant_with(1.0)

    # 重算一次拿到 scale bytes（best 配方）
    wb = w.float().reshape(N, K // 32, 32) * best_g
    amax = wb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    e = torch.ceil(torch.log2(amax / E2M1_MAX))
    scale_b = (e + 127.0).clamp(0, 254).to(torch.uint8).reshape(N, K // 32)
    nib = best_nib.reshape(N, K)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    # 返回 epilogue 应乘的因子：packed/scales 代表 w·best_g 的量化，真值 = 值/best_g
    return packed.contiguous(), scale_b.contiguous(), 1.0 / best_g, best_err


@triton.jit
def _wxfp4_expert_matmul_kernel(
    a_ptr,     # (K_total//2, B) uint8 —— e2m1 预转置（行=k 对，列=b）= rhs
    a_s_ptr,   # (B, K_total//32) uint8 —— per-32 e8m0（rhs scale 尾维=K，不随 value 转）
    w_ptr,     # (N, K_total//2) uint8 —— e2m1 打包（行=n，沿 K 打包）= lhs
    w_s_ptr,   # (N, K_total//32) uint8 —— per-32 e8m0（lhs 尾维=K ✓）
    c_ptr,     # (B, N) fp16
    B, N, K_total,
    GLOBAL_SCALE,       # fp32 全局 scale（=1/best_g，epilogue 一次乘）
    BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """专家级通用纯 MX GEMM：out[b,n] = GLOBAL_SCALE · Σ_k (a·2^as)(w·2^ws)。

    纯预打包操作数（kernel 内零转换零查表）；无 group/norms（普通 MoE 语义）。
    操作数取向（2026-09-25 定稿）：W 作 lhs（自然方向免转置）、A 预转置作
    rhs。此前怀疑的"lhs 真实 scale bug"实为全局 scale 乘反（packed 代表
    w·g，epilogue 应乘 1/g 而非 g；两种取向的 scale 语义都正确）；
    W-lhs 取向实测更快（dense 373 vs 302 TF）。
    """
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_b
    num_tiles = num_pid_b * num_pid_n
    group_id = pid // num_in_group
    first_pid = group_id * GROUP_M
    group_sz = min(num_pid_n - first_pid, GROUP_M)
    pid_n = first_pid + ((pid % num_in_group) % group_sz)
    pid_b = (pid % num_in_group) // group_sz

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    acc = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)
    rk_pk = tl.arange(0, BLOCK_K // 2)
    rk_sc = tl.arange(0, BLOCK_K // 32)

    for k in range(0, K_total, BLOCK_K):
        mask_kp = (k // 2 + rk_pk) < tl.cdiv(K_total, 2)
        # rhs = A 预转置 (BK/2, BM)
        a_off = (k // 2 + rk_pk[:, None]) * B + rb[None, :]
        a_tile = tl.load(a_ptr + a_off, mask=mask_kp[:, None] & mask_b[None, :], other=0)
        as_off = rb[:, None] * (K_total // 32) + k // 32 + rk_sc[None, :]
        as_tile = tl.load(a_s_ptr + as_off, mask=mask_b[:, None], other=0)
        # lhs = W (BN, BK/2) 自然方向直读
        w_off = rn[:, None] * (K_total // 2) + k // 2 + rk_pk[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=mask_n[:, None] & mask_kp[None, :], other=0)
        ws_off = rn[:, None] * (K_total // 32) + k // 32 + rk_sc[None, :]
        ws_tile = tl.load(w_s_ptr + ws_off, mask=mask_n[:, None], other=0)
        acc = tl.dot_scaled(w_tile, ws_tile, "e2m1", a_tile, as_tile, "e2m1", acc)

    acc = acc * GLOBAL_SCALE
    tl.store(c_ptr + rb[None, :] * N + rn[:, None],
             acc.to(c_ptr.dtype.element_ty),
             mask=mask_b[None, :] & mask_n[:, None])


def wxfp4_expert_matmul(a_packed_t, a_s, w_packed, w_s, global_scale=1.0,
                        BLOCK_B=128, BLOCK_N=128, BLOCK_K=128,
                        num_warps=8, num_stages=3):
    """专家级通用 MX GEMM wrapper。

    Args:
        a_packed_t: (K/2, B) uint8（quantize_act_e2m1_per32_t 产物）
        a_s: (B, K/32) uint8
        w_packed: (N, K/2) uint8（quantize_weight_e2m1 产物，自然方向）
        w_s: (N, K/32) uint8
        global_scale: quantize_weight_e2m1 返回的 global_scale（=1/best_g，
            epilogue 乘法因子）
    """
    B = a_packed_t.shape[1]
    N = w_packed.shape[0]
    K_total = a_packed_t.shape[0] * 2
    c = torch.empty(B, N, dtype=torch.float16, device=a_packed_t.device)
    grid = (triton.cdiv(B, BLOCK_B) * triton.cdiv(N, BLOCK_N),)
    _wxfp4_expert_matmul_kernel[grid](
        a_packed_t, a_s, w_packed, w_s, c, B, N, K_total, global_scale,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


# ===========================================================================
# WF4-2 迭代 3-1：rotate + quantize 融合 kernel（fp4 / 纯 MX）
# ===========================================================================

@triton.jit
def _rotate_quantize_kernel_fp4(
    x_ptr,        # (B, K_total) fp16 未旋转激活
    rot_ptr,      # (NUM_GROUPS, G0_STRIDE) fp16 每 group 一个旋转矩阵
    x_f4_ptr,     # (K_total//2, B) uint8 —— e2m1 预转置输出（专家 GEMM rhs 布局）
    x_s_ptr,      # (B, K_total//32) uint8 —— e8m0（byte = exp + 127）
    B, K_total,
    ROT_G0_STRIDE,
    GROUP_SIZE: tl.constexpr,    # 旋转组（128）；MX 块 32 在组内细分
    NUM_GROUPS: tl.constexpr,
    BLOCK_B: tl.constexpr = 32,
):
    """分组旋转 + per-32 块 e8m1/e8m0 量化（纯 MX，与 _rotate_quantize_kernel_fp8
    对应的 fp4 变体）。

    与 fp8 版的三处差异：
      1. scale 粒度：旋转组 128 内按 32 细分（4 个 MX 块/组）
      2. 输出：e2m1 nibble 打包（偶 k 低半字节）+ e8m0 字节；无 extra_scale
         通道——cb_scale 类常量只能进 GEMM epilogue（e8m0 是纯 2 的幂，
         折不进任意常量，这是与 fp8 版 fp16 scale 的本质差异）
      3. 存储直接写预转置 (K/2, B)（专家 GEMM 的 rhs 布局），每 tile 一次
         tl.trans（带宽型 kernel，代价可忽略，实测见 test）

    数学：x_rot = x @ P_g（fp32 累加，同 fp8 版约定）；
    对每个 32 块：e = ceil(log2(amax/6))，nibble = nearest_grid(x/2^e)。
    """
    pid_g = tl.program_id(0)
    pid_b = tl.program_id(1)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = rb < B

    x_off = rb[:, None] * K_total + pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=mask_b[:, None], other=0.0)

    acc = tl.zeros((BLOCK_B, GROUP_SIZE), dtype=tl.float32)
    rj = tl.arange(0, GROUP_SIZE)
    rk = tl.arange(0, GROUP_SIZE)
    p_off = pid_g * ROT_G0_STRIDE + rj[:, None] * GROUP_SIZE + rk[None, :]
    p_tile = tl.load(rot_ptr + p_off)
    acc += tl.dot(x_tile, tl.trans(p_tile), out_dtype=tl.float32)

    # ---- 按 32 子块量化 ----
    NBLK: tl.constexpr = GROUP_SIZE // 32
    acc3 = tl.reshape(acc, (BLOCK_B, NBLK, 32))
    amax = tl.max(tl.abs(acc3), axis=2)                      # (BLOCK_B, NBLK)
    amax = tl.maximum(amax, 1e-30)
    e = tl.ceil(tl.log2(amax / 6.0))
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    # e8m0 存出（自然 (B, K/32) 布局）
    sb_off = rb[:, None] * (K_total // 32) + pid_g * NBLK + tl.arange(0, NBLK)[None, :]
    tl.store(x_s_ptr + sb_off, (e + 127.0).to(tl.int32).to(tl.uint8), mask=mask_b[:, None])

    y = acc3 / tl.exp2(e)[:, :, None]
    y = tl.minimum(tl.maximum(y, -6.0), 6.0)
    # e2m1 网格最近邻（断点 0.25/0.75/1.25/1.75/2.5/3.5/5）
    ay = tl.abs(y)
    mag = tl.where(ay < 0.25, 0,
          tl.where(ay < 0.75, 1,
          tl.where(ay < 1.25, 2,
          tl.where(ay < 1.75, 3,
          tl.where(ay < 2.5, 4,
          tl.where(ay < 3.5, 5,
          tl.where(ay < 5.0, 6, 7))))))).to(tl.uint8)
    sign = (y < 0).to(tl.uint8)
    nib = (mag & 0x7) | (sign << 3)                           # (BLOCK_B, NBLK, 32)

    nib2 = tl.reshape(nib, (BLOCK_B, GROUP_SIZE // 2, 2))
    lo, hi = tl.split(nib2)
    byte = (lo | (hi << 4)).to(tl.uint8)                      # (BLOCK_B, GS/2)

    # 预转置写出：x_f4[(g*GS/2 + j), b]
    rj2 = tl.arange(0, GROUP_SIZE // 2)
    f4_off = (pid_g * (GROUP_SIZE // 2) + rj2)[:, None] * B + rb[None, :]
    tl.store(x_f4_ptr + f4_off, tl.trans(byte), mask=mask_b[None, :])


def rotate_quantize_fused_fp4(x: torch.Tensor, rot: torch.Tensor,
                              group_size: int, num_groups: int):
    """分组旋转 + e2m1/e8m0 量化（fp4 融合 kernel 的 wrapper）。

    Returns:
        x_f4: (K_total//2, B) uint8 —— 预转置（专家 GEMM rhs 布局）
        x_s:  (B, K_total//32) uint8
    """
    B, K_total = x.shape
    if K_total != num_groups * group_size:
        raise ValueError(f"K_total ({K_total}) != num_groups*group_size "
                         f"({num_groups * group_size})")
    if rot.shape != (num_groups, group_size, group_size):
        raise ValueError(f"rot 形状应为 {(num_groups, group_size, group_size)}, "
                         f"得到 {tuple(rot.shape)}")
    x_f4 = torch.empty((K_total // 2, B), dtype=torch.uint8, device=x.device)
    x_s = torch.empty((B, K_total // 32), dtype=torch.uint8, device=x.device)
    BLOCK_B = 32
    grid = (num_groups, triton.cdiv(B, BLOCK_B))
    _rotate_quantize_kernel_fp4[grid](
        x, rot, x_f4, x_s, B, K_total, rot.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups, BLOCK_B=BLOCK_B,
        num_warps=4, num_stages=3,
    )
    return x_f4, x_s


# ===========================================================================
# WF4-2 迭代 3-2：专家 GEMM 的 TMA + persistent 变体（大 B 优化项）
# ===========================================================================

@triton.jit
def _wxfp4_expert_matmul_tma_kernel(
    a_desc,     # (K_total//2, B) uint8 rhs —— box (BK/2, BM)
    a_s_ptr,    # (B, K_total//32) uint8 指针（TMA 16B 约束，不走 TMA）
    w_desc,     # (N, K_total//2) uint8 lhs —— box (BN, BK/2)
    w_s_ptr,    # (N, K_total//32) uint8 指针
    c_desc,     # (B, N) fp16 —— box (BM, BN)
    B, N, K_total,
    GLOBAL_SCALE,
    BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
):
    """TMA + persistent 版专家 MX GEMM（与 _wxfp4_expert_matmul_kernel 数学等价）。

    数据走 TMA（a/w/c）、scale 走指针（box 内维 4~8 字节违反 TMA 16B 对齐，
    WF4-P1 实测约束）；persistent 调度 + group-M swizzle。
    """
    start_pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_b
    num_tiles = num_pid_b * num_pid_n

    rk_sc = tl.arange(0, BLOCK_K // 32)

    for tid in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        group_id = tid // num_in_group
        first_pid = group_id * GROUP_M
        group_sz = min(num_pid_n - first_pid, GROUP_M)
        pid_n = first_pid + ((tid % num_in_group) % group_sz)
        pid_b = (tid % num_in_group) // group_sz

        rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_b = rb < B
        mask_n = rn < N

        acc = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)
        for k in range(0, K_total, BLOCK_K):
            a_tile = a_desc.load([k // 2, pid_b * BLOCK_B])      # (BK/2, BM)
            w_tile = w_desc.load([pid_n * BLOCK_N, k // 2])      # (BN, BK/2)
            as_off = rb[:, None] * (K_total // 32) + k // 32 + rk_sc[None, :]
            as_tile = tl.load(a_s_ptr + as_off, mask=mask_b[:, None], other=0)
            ws_off = rn[:, None] * (K_total // 32) + k // 32 + rk_sc[None, :]
            ws_tile = tl.load(w_s_ptr + ws_off, mask=mask_n[:, None], other=0)
            acc = tl.dot_scaled(w_tile, ws_tile, "e2m1", a_tile, as_tile, "e2m1", acc)

        acc = acc * GLOBAL_SCALE
        c_desc.store([pid_b * BLOCK_B, pid_n * BLOCK_N],
                     tl.trans(acc).to(c_desc.dtype))


def wxfp4_expert_matmul_tma(a_packed_t, a_s, w_packed, w_s, global_scale=1.0,
                            BLOCK_B=128, BLOCK_N=128, BLOCK_K=128,
                            num_warps=8, num_stages=3):
    """TMA+persistent 版 wrapper（参数语义同 wxfp4_expert_matmul）。"""
    from triton.tools.tensor_descriptor import TensorDescriptor
    B = a_packed_t.shape[1]
    N = w_packed.shape[0]
    K_total = a_packed_t.shape[0] * 2
    c = torch.empty(B, N, dtype=torch.float16, device=a_packed_t.device)
    NUM_SMS = torch.cuda.get_device_properties(a_packed_t.device).multi_processor_count
    a_desc = TensorDescriptor.from_tensor(a_packed_t, [BLOCK_K // 2, BLOCK_B])
    w_desc = TensorDescriptor.from_tensor(w_packed, [BLOCK_N, BLOCK_K // 2])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_B, BLOCK_N])
    _wxfp4_expert_matmul_tma_kernel[(NUM_SMS,)](
        a_desc, a_s, w_desc, w_s, c_desc, B, N, K_total, global_scale,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=8, NUM_SMS=NUM_SMS,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c
