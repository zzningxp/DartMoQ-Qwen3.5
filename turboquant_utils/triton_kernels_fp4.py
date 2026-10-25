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

import math

import torch

# e2m1 网格（nibble 低 3 位 = 幅度索引）
E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_MAX = 6.0


def build_e2m1_codebook(codebook: torch.Tensor):
    """把 fp16/fp32 码本转成 e2m1 nibble LUT（连续 scale 搜索 + residual 分解）。

    分解链：`codebook[i] ≈ e2m1_value(nib[i]) × 2^E0 × residual`
      - 2^E0：进 W 侧 e8m0 通道（常量块缩放，byte = 127 + E0）
      - residual：任意常数，折进 norms_gf（fp16，epilogue 兜底）——
        e8m0 只能取 2 的幂，非 2 幂残差必须走这条路（与 wxfp8 的
        cb_scale 折激活 scale 同构，只是 fp4 的激活通道装不下任意常数）

    scale 用连续搜索（对数粗扫 + 两轮细化）：1-bit 的 ±a 可精确落格
    （residual 吸收非 2 幂因子），各 bit 误差实测见 test_wxfp4_gemm_align。
    """
    import math
    cb = codebook.float()
    cb_max = cb.abs().max()
    if cb_max <= 0:
        raise ValueError("码本全零，无法转 e2m1")
    grid = torch.tensor(E2M1_GRID, device=cb.device)

    def fit(s):
        y = (cb / s).clamp(-E2M1_MAX, E2M1_MAX)
        mag = torch.argmin((y.abs().unsqueeze(-1) - grid).abs(), dim=-1)
        nib = ((mag & 0x7) | (((y < 0).to(torch.uint8) << 3).long())).to(torch.uint8)
        rec = grid[mag] * y.sign() * s
        return nib, ((rec - cb).norm() / cb.norm()).item()

    s0 = cb_max / E2M1_MAX
    best_s, best_err = s0, float("inf")
    best_lut = None
    coarse = [s0 * (2.0 ** (i / 16.0)) for i in range(-16, 17)]
    for s in coarse:
        nib, err = fit(s)
        if err < best_err:
            best_s, best_err, best_lut = s, err, nib
    for _ in range(2):
        span = 2.0 ** (1 / 16)
        lo, hi = best_s / span, best_s * span
        fine = [lo * (hi / lo) ** (i / 64.0) for i in range(65)]
        for s in fine:
            nib, err = fit(s)
            if err < best_err:
                best_s, best_err, best_lut = s, err, nib

    e0 = math.floor(math.log2(best_s))
    residual = best_s / (2.0 ** e0)
    return best_lut.contiguous(), e0, residual, best_err


def fold_indices_to_e2m1(indices_packed_gf: torch.Tensor, nib: torch.Tensor,
                         bit_width: int, group_size: int):
    """W 侧 load 期折叠：packed 码本索引 → packed e2m1（WFF-1，
    roadmaps/wxfp4-fold-plan-260927.md）。

    把 kernel 内逐 tile 的「unpack → nib LUT → 拼 byte」挪到 ctx 构建期一次完成，
    GEMM kernel 走 W_FOLDED 分支直读。同一 LUT 同一映射，输出与 legacy 构造
    bit-exact（byte = nib[k] | nib[k+1]<<4，偶 k 低 nibble）。

    显存：bit4 零膨胀（4→4 bit）；bit1 ×4 / bit2 ×2——fp4 线内仅 bit1/4 小池，
    全模型 +47MB 上限（fold plan §二）。

    Args:
        indices_packed_gf: (G, N, group_size // (8//bit_width)) uint8
            （group-first 布局，与 kernel legacy 路径同一 buffer）
        nib: (2^bit_width,) uint8 码本索引 → e2m1 nibble（build_e2m1_codebook 产物）
        bit_width: 1/2/4（bit2 当前不在 fp4 线，参数保留给 3-4 激进版）
        group_size: 量化组大小
    Returns:
        (G, N, group_size // 2) uint8 —— e2m1 沿组内 K 打包（偶 k 低 nibble）
    """
    from turboquant_utils.quantize import unpack_nbit
    if bit_width not in {1, 2, 4}:
        raise ValueError(f"fold 支持 bit 1/2/4，got {bit_width}")
    G, N, _ = indices_packed_gf.shape
    out = torch.empty((G, N, group_size // 2), dtype=torch.uint8,
                      device=indices_packed_gf.device)
    nib_d = nib.to(indices_packed_gf.device)
    # 分块 fold：整矩阵一次物化时 idx.long() 是 8B/元素，全 4bit 池（4bpw
    # checkpoint 全部 256 expert 都是 bit4）下单次分配可达 4GB 直接 OOM
    # （1018.thrput.1.w4 wxfp4 实测）；按 G 分块把瞬时峰值压到 ~数百 MB，
    # 逐块独立计算，输出与整矩阵路径 bit-exact
    per_g = N * (group_size // 2)
    chunk = max(1, (1 << 25) // max(per_g, 1))  # 每块 ~32M 输出元素
    for g0 in range(0, G, chunk):
        sub = indices_packed_gf[g0:g0 + chunk].reshape(-1)
        n_elem = sub.numel() * (8 // bit_width)
        # unpack_nbit 的 byte 内 low-bit-first 与 kernel 的 >> (pos*bit)) & mask 同约定
        idx = unpack_nbit(sub, bit_width, n_elem)
        nb = nib_d[idx.long()].reshape(-1, 2)  # (n/2, 2) uint8
        out[g0:g0 + chunk] = (
            nb[:, 0] | (nb[:, 1] << 4)).reshape(-1, N, group_size // 2)
    return out


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
    WS_STRIDE,          # w_s 行 stride（全局 K//32；切片时按全局布局免拷贝）
    WS_KOFF,            # w_s 的 k-block 全局偏移（down 切片 = original_start//32）
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BIT_WIDTH: tl.constexpr,
    W_FOLDED: tl.constexpr = False,   # WFF-1：indices_ptr 直读已折叠 e2m1
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
      - WFF-1 W_FOLDED：W 侧 ctx 期已折叠为 e2m1（fold_indices_to_e2m1），
        本 kernel 直读（byte 域寻址同 A 侧），跳过 unpack/LUT/拼包整块——
        indices_ptr 传 folded buffer、INDICES_G0_STRIDE 传其 stride(0)，
        nib_ptr 不读（wrapper 传占位）
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

            # --- lhs = W：W_FOLDED 直读（WFF-1）或 unpack → nibble LUT → 乘法归约拼包 ---
            if W_FOLDED:
                # ctx 期已折叠的 e2m1 (G, N, GS//2)：byte = nib[2j] | nib[2j+1]<<4
                # 已拼好，byte 域寻址（mask_kp），写法同 A 侧 packed 加载
                row_f = rn * (GROUP_SIZE // 2)
                wf_off = g_base + row_f[:, None] + (k_start // 2 + rk_pk)[None, :]
                w_byte = tl.load(indices_ptr + wf_off,
                                 mask=mask_n[:, None] & mask_kp[None, :], other=0)
            else:
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

            # lhs scale (BN, BK/32)——按全局 w_s 布局寻址（WS_STRIDE/WS_KOFF）
            ws_off = rn[:, None] * WS_STRIDE + WS_KOFF \
                + (g_start + k_start) // 32 + rk_sc[None, :]
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
        w_s.stride(0), 0,
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups, BIT_WIDTH=bit_width,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def _launch_swapped(x_f4_t, x_s, indices_slice, nib, w_s, norms_slice,
                    group_size, num_groups, bit_width, ws_stride, ws_koff, cfg=None,
                    w_folded=False):
    """swapped kernel 的通用启动入口（供切片 wrapper 使用）。

    w_folded=True 时 indices_slice 应传 fold_indices_to_e2m1 产物（切片后同布局），
    nib 传任意 uint8 占位张量（W_FOLDED constexpr 分支不读）。
    """
    # A 侧 kernel 按形状推导行距（(k/2+j)*B、rb*(K_total//32)），非连续
    # view（如整池预转置产物的列切片）会读错行——兜底拷贝成连续（真实
    # 路径 gather 产物本就连续，不触发拷贝）
    if not x_f4_t.is_contiguous():
        x_f4_t = x_f4_t.contiguous()
    if not x_s.is_contiguous():
        x_s = x_s.contiguous()
    B = x_f4_t.shape[1]
    N = indices_slice.shape[1]
    K_total = x_f4_t.shape[0] * 2
    out = torch.empty(B, N, dtype=torch.float16, device=x_f4_t.device)
    if cfg is None:
        BLOCK_B = 32 if B <= 256 else 128
        BLOCK_N, BLOCK_K, nw, ns = 128, 128, (4 if B <= 256 else 8), 2
    else:
        BLOCK_B, BLOCK_N, BLOCK_K, nw, ns = cfg
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N, BLOCK_N))
    _wxfp4_fused_matmul_kernel_swapped_gf[grid](
        x_f4_t, x_s, indices_slice, nib, w_s, norms_slice, out,
        B, N, K_total,
        indices_slice.stride(0), norms_slice.stride(0), ws_stride, ws_koff,
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups, BIT_WIDTH=bit_width,
        W_FOLDED=w_folded,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=nw, num_stages=ns,
    )
    return out


def wxfp4_matmul_grouped_slice_rows_gf_swapped(
    x_f4_t, x_s, indices_packed_gf, nib, w_s, norms_gf,
    group_size, in_features, row_start, row_end, bit_width,
    norms_prescaled=False, w_e2m1_gf=None,
):
    """gate_up 路径：行切片（输出维切片，对应 wxa8 的 slice_rows_gf）。

    Args:
        x_f4_t: (in_features//2, B) uint8 预转置（rotate_quantize_fused_fp4 产物）
        x_s: (B, in_features//32) uint8
        w_s: (N_total, in_features//32) uint8（全矩阵，行切片 view 免拷贝）
        norms_gf: (num_groups, N_total) fp16（**已含 residual 折算**）
        w_e2m1_gf: (num_groups, N_total, group_size//2) uint8 可选——
            fold_indices_to_e2m1 产物；非 None 时走 W_FOLDED 直读分支
            （nib 被忽略，可传任意 uint8 占位）
    """
    if bit_width not in {1, 2, 4}:
        raise ValueError(f"WxFP4 支持 bit 1/2/4，got {bit_width}")
    num_groups = in_features // group_size
    norms_slice = norms_gf[:, row_start:row_end]
    if not norms_prescaled:
        norms_slice = norms_slice.float() / math.sqrt(group_size)
    if w_e2m1_gf is not None:
        w_src = w_e2m1_gf[:, row_start:row_end, :]
        nib_arg = w_src          # 占位（W_FOLDED 分支不读）
    else:
        w_src = indices_packed_gf[:, row_start:row_end, :]
        nib_arg = nib
    return _launch_swapped(
        x_f4_t, x_s, w_src, nib_arg, w_s[row_start:row_end], norms_slice,
        group_size, num_groups, bit_width,
        w_s.stride(0), 0, w_folded=w_e2m1_gf is not None)


def wxfp4_matmul_grouped_slice_in_features_gf_swapped(
    x_f4_t, x_s, indices_packed_gf, nib, w_s, norms_gf,
    group_size, original_start, original_end, bit_width,
    norms_prescaled=False, w_e2m1_gf=None,
):
    """down 路径：in_features 切片（group 维切片，对应 wxa8 的 slice_in_features_gf）。

    Args:
        x_f4_t: (slice//2, B) uint8——per-expert 量化产物（局部 k）
        x_s: (B, slice//32) uint8
        w_s: (N, K_total//32) uint8 全矩阵（WS_STRIDE/WS_KOFF 全局寻址免拷贝）
        norms_gf: (K_total//group_size, N) fp16（已含 residual 折算）
        w_e2m1_gf: (K_total//group_size, N, group_size//2) uint8 可选——
            fold_indices_to_e2m1 产物；非 None 时走 W_FOLDED 直读分支
    """
    if bit_width not in {1, 2, 4}:
        raise ValueError(f"WxFP4 支持 bit 1/2/4，got {bit_width}")
    if original_start % group_size != 0:
        raise ValueError(f"original_start ({original_start}) 必须对齐 group_size")
    g_start = original_start // group_size
    g_end = original_end // group_size
    norms_slice = norms_gf[g_start:g_end]
    if not norms_prescaled:
        norms_slice = norms_slice.float() / math.sqrt(group_size)
    if w_e2m1_gf is not None:
        w_src = w_e2m1_gf[g_start:g_end]
        nib_arg = w_src          # 占位（W_FOLDED 分支不读）
    else:
        w_src = indices_packed_gf[g_start:g_end]
        nib_arg = nib
    return _launch_swapped(
        x_f4_t, x_s, w_src, nib_arg,
        w_s, norms_slice,
        group_size, g_end - g_start, bit_width,
        w_s.stride(0), original_start // 32, w_folded=w_e2m1_gf is not None)


# ===========================================================================
# P5-2 重做：按 bit 池合并多 expert 的 grouped kernel（wxfp4 版）
# ===========================================================================
# 与 _wxa8_merged_kernel_gf 同骨架（block_info 1D grid + expert_info 双模式），
# fp4 语义三处差异：A 侧 e2m1 预转置 + per-32 e8m0 双操作数 dot_scaled、
# W 侧 nibble LUT / WFF-1 folded 直读（W_FOLDED，大池免副本）、acc fp32。
# 天然 in-kernel LUT：4bpw 大池不需要 fold 副本（WFF-5 的 +10GB 问题消失），
# 有 folded 副本的小池直读（与逐 expert 折叠路径 bit-exact）。

from turboquant_utils.triton_kernels_a8 import (  # noqa: E402
    _build_merged_block_info,
)


@triton.jit
def _wxfp4_merged_kernel_gf(
    # A 侧（MX 原生格式，池内按 expert 顺序拼接）
    a_ptr,        # (K_a//2, B_total) uint8 —— e2m1 打包、预转置（行=k 对，列=b）
    a_s_ptr,      # (B_total, K_a//32) uint8 —— per-32 e8m0（scale 尾维=K）
    # W 侧（group-first 布局，本 bit 池共享，全局寻址免切片）
    indices_ptr,  # (G_total, N_total, PACKED_PER_GROUP | GS//2) uint8
                  # W_FOLDED 时传 fold_indices_to_e2m1 产物（stride(0) 随之）
    nib_ptr,      # (2^bit,) uint8（W_FOLDED 分支不读，传占位）
    w_s_ptr,      # (N_total, K_w_total//32) uint8 —— W 块缩放（全局布局）
    norms_ptr,    # (G_total, N_total) fp16
    # Expert / block 布局（与 _wxa8_merged_kernel_gf 一致）
    expert_info_ptr,  # (E, 4) int32: [token_start, token_count, slice_start, slice_count]
    block_info_ptr,   # (num_blocks, 3) int32: [expert_idx, block_b, block_n]
    # Output
    output_ptr,   # (B_total, N_out) fp16 扁平连续，行 stride = OUT_ROW_STRIDE
    # Shape / strides
    B_total, N_LIMIT,
    INDICES_G0_STRIDE, NORMS_G0_STRIDE, AS_ROW_STRIDE,
    WS_STRIDE, OUT_ROW_STRIDE,
    # Constexpr config
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,     # 本 expert 的有效 group 数
    BIT_WIDTH: tl.constexpr,
    W_FOLDED: tl.constexpr = False,
    SLICE_IS_GROUPS: tl.constexpr = False,
    BLOCK_B: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """按 bit 池合并多 expert 的 wxfp4 grouped GEMM（双模式，swapped 操作数）。

    SLICE_IS_GROUPS=False（gate_up / 行切片）：slice_start 是共享权重上的
    **列**切片（rn 全局，输出列回 [0, slice_count)），A 的 K 是完整
    in_features（k 局部 = 全局）。SLICE_IS_GROUPS=True（down / group 切片）：
    slice_start 是 **group** 切片（indices/norms 从全局 group 起），A 的 K
    是切片后 inter（k 局部），W 的 e8m0 k 偏移由 slice_start 在 kernel 内
    折算（ws_koff = slice_start*GROUP_SIZE//32）。

    与 _wxfp4_fused_matmul_kernel_swapped_gf 的 tile 数学完全一致（同一
    tile 分解、同一 dot_scaled 累加顺序；两档 tile 表 BLOCK_K 同为 128，
    fp32 acc 跨档仍位级一致）→ 与逐 expert 路径 bit-exact。
    """
    pid = tl.program_id(0)
    bi = block_info_ptr + pid * 3
    eid = tl.load(bi + 0).to(tl.int32)
    blk_b = tl.load(bi + 1).to(tl.int32)
    blk_n = tl.load(bi + 2).to(tl.int32)

    ei = expert_info_ptr + eid * 4
    token_start = tl.load(ei + 0).to(tl.int32)
    token_count = tl.load(ei + 1).to(tl.int32)
    slice_start = tl.load(ei + 2).to(tl.int32)

    rb = token_start + blk_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = rb < token_start + token_count

    if SLICE_IS_GROUPS:
        # down：N 维完整（从 0 起），group 维按 slice_start 偏移
        rn = blk_n * BLOCK_N + tl.arange(0, BLOCK_N)
    else:
        # gate_up：N 维是共享权重的列切片，从 slice_start 起
        rn = slice_start + blk_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < N_LIMIT

    # W 的 e8m0 k 偏移：rows 模式 0；groups 模式 = 切片起始的全局 k 块
    if SLICE_IS_GROUPS:
        ws_koff = (slice_start * GROUP_SIZE) // 32
    else:
        ws_koff = 0

    total_acc = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    PACKED_PER_GROUP = GROUP_SIZE // ELEMENTS_PER_BYTE
    row_base = rn * PACKED_PER_GROUP
    # 拼包权重 [1, 16]
    pk_w = tl.arange(0, 2) * 15 + 1

    for g in range(NUM_GROUPS):
        g_global = (slice_start + g) if SLICE_IS_GROUPS else g
        k_local = g * GROUP_SIZE      # A 侧始终局部 k（rows=全局 / groups=切片内）
        g_base = g_global * INDICES_G0_STRIDE

        norm_g = tl.load(norms_ptr + g_global * NORMS_G0_STRIDE + rn,
                         mask=mask_n, other=0.0)

        acc_g = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.float32)

        rk_pk = tl.arange(0, BLOCK_K // 2)
        rk_sc = tl.arange(0, BLOCK_K // 32)

        for k_start in range(0, GROUP_SIZE, BLOCK_K):
            # --- rhs = A：预转置 (BK/2, BM)，行连续；group 偏移是行偏移×B_total
            a_off = ((k_local + k_start) // 2 + rk_pk[:, None]) * B_total + rb[None, :]
            a_tile = tl.load(a_ptr + a_off,
                             mask=((k_start // 2 + rk_pk) < (GROUP_SIZE // 2))[:, None]
                             & mask_b[None, :], other=0)
            # rhs scale (BM, BK/32)（尾维 = K）
            as_off = rb[:, None] * AS_ROW_STRIDE + (k_local + k_start) // 32 + rk_sc[None, :]
            as_tile = tl.load(a_s_ptr + as_off,
                              mask=mask_b[:, None]
                              & ((k_start // 32 + rk_sc) < (GROUP_SIZE // 32))[None, :],
                              other=0)

            # --- lhs = W：W_FOLDED 直读（WFF-1）或 unpack → nibble LUT → 拼包
            if W_FOLDED:
                row_f = rn * (GROUP_SIZE // 2)
                wf_off = g_base + row_f[:, None] + (k_start // 2 + rk_pk)[None, :]
                w_byte = tl.load(indices_ptr + wf_off,
                                 mask=mask_n[:, None]
                                 & ((k_start // 2 + rk_pk) < (GROUP_SIZE // 2))[None, :],
                                 other=0)
            else:
                rk = k_start + tl.arange(0, BLOCK_K)
                w_mask = mask_n[:, None] & (rk < GROUP_SIZE)[None, :]
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

            # lhs scale (BN, BK/32)——按全局 w_s 布局寻址（WS_STRIDE/ws_koff）
            ws_off = rn[:, None] * WS_STRIDE + ws_koff \
                + (k_local + k_start) // 32 + rk_sc[None, :]
            ws_tile = tl.load(w_s_ptr + ws_off,
                              mask=mask_n[:, None]
                              & ((k_start // 32 + rk_sc) < (GROUP_SIZE // 32))[None, :],
                              other=0)

            acc_g = tl.dot_scaled(w_byte, ws_tile, "e2m1",
                                  a_tile, as_tile, "e2m1", acc_g)

        total_acc += acc_g * norm_g[:, None]

    # (BN, BM) -> output (B_total, N_out)：一次转置写（列回 [0, slice_count)）
    out_col = (rn - slice_start) if not SLICE_IS_GROUPS else rn
    tl.store(output_ptr + rb[None, :] * OUT_ROW_STRIDE + out_col[:, None],
             total_acc.to(output_ptr.dtype.element_ty),
             mask=mask_b[None, :] & mask_n[:, None])


def _merged_fp4_default_cfg(B_total):
    """合并 fp4 的默认 tile（与 _launch_swapped 两档一致：BLOCK_K 恒 128）。"""
    BLOCK_B = 32 if B_total <= 256 else 128
    return BLOCK_B, 128, 128, (4 if B_total <= 256 else 8), 2


def _launch_wxfp4_merged_gf(
    a, a_s, w_src, nib_arg, w_s, norms_used,
    expert_info_dev, block_info, out, n_extent,
    group_size, num_groups, bit_width,
    w_folded, slice_is_groups, cfg,
):
    """wxfp4 merged kernel 统一发射入口（host 编排路径与静态派发路径共用）。

    参数序与 _wxfp4_merged_kernel_gf 形参一一对应；w_src/nib_arg 由调用方
    按 W_FOLDED 分支预先解析（folded 时 nib 占位传 w_src 本身）；expert_info_dev
    必须是 (E,4) int32 设备张量。
    """
    BLOCK_B, BLOCK_N, BLOCK_K, nw, ns = cfg
    _wxfp4_merged_kernel_gf[(block_info.shape[0],)](
        a, a_s, w_src, nib_arg, w_s, norms_used,
        expert_info_dev, block_info, out,
        out.shape[0], n_extent,
        w_src.stride(0), norms_used.stride(0), a_s.stride(0),
        w_s.stride(0), out.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        BIT_WIDTH=bit_width, W_FOLDED=w_folded,
        SLICE_IS_GROUPS=slice_is_groups,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=nw, num_stages=ns,
    )


def wxfp4_matmul_merged_rows_gf(
    x_f4_t, x_s, indices_packed_gf, nib, w_s, norms_gf,
    expert_info, group_size, in_features, bit_width: int,
    norms_prescaled: bool = False, w_e2m1_gf=None, cfg=None,
):
    """gate_up 合并版（wxfp4）：同一 bit 池所有 expert 一次 launch（行切片语义）。

    与逐 expert 的 wxfp4_matmul_grouped_slice_rows_gf_swapped 逐段等价。
    x_f4_t 是 (in_features//2, B_total) 预转置 e2m1（hoisted 列 gather 或
    整池一次 rotate_quantize_fused_fp4），池内按 expert 顺序列拼接；
    expert_info 里 slice_start/slice_count 是该 expert 在共享 indices/norms
    上的列切片（主流程为 e*2*inter / 2*inter）。W 侧全局寻址（rn 全局），
    免逐 expert 切片。

    W_FOLDED：w_e2m1_gf 非 None（WFF-5 小池折叠副本）时直读 folded buffer
    （nib 占位不读），与逐 expert 折叠路径 bit-exact；None 走 in-kernel
    LUT（大池免副本，wxfp4 合并版的自然设计）。

    Returns:
        output: (B_total, 2*inter) fp16 扁平连续（按 expert 顺序行拼接）
    """
    if bit_width not in {1, 2, 4}:
        raise ValueError(f"WxFP4 支持 bit 1/2/4，got {bit_width}")
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) 必须对齐 group_size ({group_size})")

    expert_info = [tuple(int(v) for v in row) for row in expert_info]
    B_total = x_f4_t.shape[1]
    num_groups = in_features // group_size
    # A 侧 kernel 按形状推导行距（(k/2+j)*B_total），兜底非连续 view
    # （真实路径 hoisted 列 gather / 整池融合量化产物均连续，不触发拷贝）
    if not x_f4_t.is_contiguous():
        x_f4_t = x_f4_t.contiguous()
    n_slice = expert_info[0][3] if expert_info else 0
    if any(row[3] != n_slice for row in expert_info):
        raise ValueError(
            f"merged rows kernel 要求池内列切片一致（inter_size 相同），"
            f"得到 slice_count 集合 {sorted({row[3] for row in expert_info})}")

    norms_used = norms_gf if norms_prescaled else (
        norms_gf.float() / math.sqrt(group_size))

    if cfg is not None:
        BLOCK_B, BLOCK_N, BLOCK_K, nw, ns = cfg
    else:
        BLOCK_B, BLOCK_N, BLOCK_K, nw, ns = _merged_fp4_default_cfg(B_total)

    block_info = _build_merged_block_info(expert_info, BLOCK_B, BLOCK_N,
                                          n_slice, x_f4_t.device)
    out = torch.empty(B_total, n_slice, dtype=torch.float16, device=x_f4_t.device)
    n_limit = max(row[2] for row in expert_info) + n_slice

    w_src = w_e2m1_gf if w_e2m1_gf is not None else indices_packed_gf
    nib_arg = w_src if w_e2m1_gf is not None else nib

    _launch_wxfp4_merged_gf(
        x_f4_t, x_s, w_src, nib_arg, w_s, norms_used,
        torch.tensor(expert_info, dtype=torch.int32, device=x_f4_t.device),
        block_info, out, n_limit,
        group_size, num_groups, bit_width,
        w_e2m1_gf is not None, slice_is_groups=False,
        cfg=(BLOCK_B, BLOCK_N, BLOCK_K, nw, ns))
    return out


def wxfp4_matmul_merged_in_features_gf(
    x_f4_t, x_s, indices_packed_gf, nib, w_s, norms_gf,
    expert_info, group_size, bit_width: int,
    norms_prescaled: bool = False, w_e2m1_gf=None, cfg=None,
):
    """down 合并版（wxfp4）：同一 bit 池所有 expert 一次 launch（in_features 切片）。

    x_f4_t 是 (inter//2, B_total) 预转置 e2m1——每 expert 段已用
    seed + original_start 旋转并量化（v1 由调用方逐 expert 调
    rotate_quantize_fused_fp4），列拼接；expert_info 里 slice_start/slice_count
    是 group 维切片（主流程为 e*inter/gs / inter/gs），N 维完整 hidden。
    W 的 e8m0 k 偏移由 slice_start 在 kernel 内折算。

    Returns:
        output: (B_total, N) fp16 扁平连续（按 expert 顺序行拼接）
    """
    if bit_width not in {1, 2, 4}:
        raise ValueError(f"WxFP4 支持 bit 1/2/4，got {bit_width}")

    expert_info = [tuple(int(v) for v in row) for row in expert_info]
    B_total = x_f4_t.shape[1]
    # A 侧 kernel 按形状推导行距（(k/2+j)*B_total），兜底非连续 view
    if not x_f4_t.is_contiguous():
        x_f4_t = x_f4_t.contiguous()
    g_count = expert_info[0][3] if expert_info else 0
    if any(row[3] != g_count for row in expert_info):
        raise ValueError(
            f"merged in_features kernel 要求池内 group 切片一致（inter_size "
            f"相同），得到集合 {sorted({row[3] for row in expert_info})}")
    if x_f4_t.shape[0] * 2 != g_count * group_size:
        raise ValueError(
            f"x_f4_t K ({x_f4_t.shape[0] * 2}) != g_count*group_size "
            f"({g_count * group_size})")

    norms_used = norms_gf if norms_prescaled else (
        norms_gf.float() / math.sqrt(group_size))
    N = norms_used.shape[1]

    if cfg is not None:
        BLOCK_B, BLOCK_N, BLOCK_K, nw, ns = cfg
    else:
        BLOCK_B, BLOCK_N, BLOCK_K, nw, ns = _merged_fp4_default_cfg(B_total)

    block_info = _build_merged_block_info(expert_info, BLOCK_B, BLOCK_N,
                                          N, x_f4_t.device)
    out = torch.empty(B_total, N, dtype=torch.float16, device=x_f4_t.device)

    w_src = w_e2m1_gf if w_e2m1_gf is not None else indices_packed_gf
    nib_arg = w_src if w_e2m1_gf is not None else nib

    _launch_wxfp4_merged_gf(
        x_f4_t, x_s, w_src, nib_arg, w_s, norms_used,
        torch.tensor(expert_info, dtype=torch.int32, device=x_f4_t.device),
        block_info, out, N,
        group_size, g_count, bit_width,
        w_e2m1_gf is not None, slice_is_groups=True,
        cfg=(BLOCK_B, BLOCK_N, BLOCK_K, nw, ns))
    return out


def wxfp4_matmul_merged_rows_gf_static(
    x_f4_t, x_s, indices_packed_gf, nib, w_s, norms_gf,
    expert_info_dev, block_info, n_slice, n_limit,
    group_size, in_features, bit_width: int,
    norms_prescaled: bool = False, w_e2m1_gf=None, cfg=None,
):
    """gate_up 合并版·静态派发（CUDA graph decode 用）。

    与 wxfp4_matmul_merged_rows_gf 同一 kernel、同一 tile 数学；差别仅在
    编排来源：expert_info（(E,4) int32，行段列每步设备端重写、列切片列
    为加载期常量）与 block_info（最坏情况静态调度表）均为设备常驻张量。
    B_total 取 x_f4_t.shape[1]（静态容量）。函数内零 host 读、零 H2D。
    """
    norms_used = norms_gf if norms_prescaled else (
        norms_gf.float() / math.sqrt(group_size))
    out = torch.empty(x_f4_t.shape[1], n_slice, dtype=torch.float16,
                      device=x_f4_t.device)
    if cfg is None:
        cfg = _merged_fp4_default_cfg(x_f4_t.shape[1])
    w_src = w_e2m1_gf if w_e2m1_gf is not None else indices_packed_gf
    nib_arg = w_src if w_e2m1_gf is not None else nib
    _launch_wxfp4_merged_gf(
        x_f4_t, x_s, w_src, nib_arg, w_s, norms_used,
        expert_info_dev, block_info, out, n_limit,
        group_size, in_features // group_size, bit_width,
        w_e2m1_gf is not None, slice_is_groups=False, cfg=cfg)
    return out


def wxfp4_matmul_merged_in_features_gf_static(
    x_f4_t, x_s, indices_packed_gf, nib, w_s, norms_gf,
    expert_info_dev, block_info, group_size, bit_width: int,
    norms_prescaled: bool = False, w_e2m1_gf=None, cfg=None,
):
    """down 合并版·静态派发（CUDA graph decode 用），契约同 rows 静态版。

    N（完整输出列数）取 norms_gf.shape[1]（静态形状）；NUM_GROUPS 即池内
    group 切片计数，由 x_f4_t 的 K 维隐含（kernel 不收 g_count 参数列）。
    """
    norms_used = norms_gf if norms_prescaled else (
        norms_gf.float() / math.sqrt(group_size))
    N = norms_used.shape[1]
    out = torch.empty(x_f4_t.shape[1], N, dtype=torch.float16,
                      device=x_f4_t.device)
    if cfg is None:
        cfg = _merged_fp4_default_cfg(x_f4_t.shape[1])
    w_src = w_e2m1_gf if w_e2m1_gf is not None else indices_packed_gf
    nib_arg = w_src if w_e2m1_gf is not None else nib
    _launch_wxfp4_merged_gf(
        x_f4_t, x_s, w_src, nib_arg, w_s, norms_used,
        expert_info_dev, block_info, out, N,
        group_size, x_f4_t.shape[0] * 2 // group_size, bit_width,
        w_e2m1_gf is not None, slice_is_groups=True, cfg=cfg)
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


@triton.jit
def _rotate_quantize_kernel_fp4_pooled(
    x_ptr,        # (B_cap, K_total) fp16 —— 池缓冲（行序 = 成员序）
    rot_ptr,      # 旋转矩阵库（TOTAL, G0_STRIDE），第 k 个矩阵基址 = k*stride
    expert_info_ptr,  # (E, 4) int32 [token_start, token_count, rot_base, g_count]
    block_info_ptr,   # (num_blocks, 2) int32 [member_idx, block_b]
    x_f4_ptr,     # (K_total//2, B_cap) uint8 —— e2m1 预转置输出
    x_s_ptr,      # (B_cap, K_total//32) uint8 —— e8m0
    B_cap, K_total,
    ROT_G0_STRIDE,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_B: tl.constexpr = 32,
):
    """池内逐成员旋转 + e2m1/e8m0 量化（静态派发 down 方向）。

    与 `_rotate_quantize_kernel_fp4` 同一数学、同一量化尾（含预转置写出）；
    差别只在旋转矩阵来源：down 的 seed 含 expert 偏移，整批不能共用一个
    矩阵。按"成员 × 行块"的最坏情况静态调度表分发程序，行段与矩阵基序号
    由 kernel 运行时从 expert_info 读（列 2 = start // group_size，与 GEMM
    的 group 切片语义一致，共用同一份缓冲）。token_count=0 的成员整块早退。
    """
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)
    bi = block_info_ptr + pid * 2
    midx = tl.load(bi + 0).to(tl.int32)
    blk_b = tl.load(bi + 1).to(tl.int32)

    ei = expert_info_ptr + midx * 4
    token_start = tl.load(ei + 0).to(tl.int32)
    token_count = tl.load(ei + 1).to(tl.int32)
    rot_base = tl.load(ei + 2).to(tl.int32)
    if token_count <= 0:
        return

    rb = token_start + blk_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = rb < token_start + token_count

    x_off = rb[:, None] * K_total + pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=mask_b[:, None], other=0.0)

    acc = tl.zeros((BLOCK_B, GROUP_SIZE), dtype=tl.float32)
    rj = tl.arange(0, GROUP_SIZE)
    rk = tl.arange(0, GROUP_SIZE)
    p_off = (rot_base + pid_g) * ROT_G0_STRIDE + rj[:, None] * GROUP_SIZE + rk[None, :]
    p_tile = tl.load(rot_ptr + p_off)
    acc += tl.dot(x_tile, tl.trans(p_tile), out_dtype=tl.float32)

    # ---- 按 32 子块量化（与 _rotate_quantize_kernel_fp4 逐行对应）----
    NBLK: tl.constexpr = GROUP_SIZE // 32
    acc3 = tl.reshape(acc, (BLOCK_B, NBLK, 32))
    amax = tl.max(tl.abs(acc3), axis=2)
    amax = tl.maximum(amax, 1e-30)
    e = tl.ceil(tl.log2(amax / 6.0))
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    sb_off = rb[:, None] * (K_total // 32) + pid_g * NBLK + tl.arange(0, NBLK)[None, :]
    tl.store(x_s_ptr + sb_off, (e + 127.0).to(tl.int32).to(tl.uint8), mask=mask_b[:, None])

    y = acc3 / tl.exp2(e)[:, :, None]
    y = tl.minimum(tl.maximum(y, -6.0), 6.0)
    ay = tl.abs(y)
    mag = tl.where(ay < 0.25, 0,
          tl.where(ay < 0.75, 1,
          tl.where(ay < 1.25, 2,
          tl.where(ay < 1.75, 3,
          tl.where(ay < 2.5, 4,
          tl.where(ay < 3.5, 5,
          tl.where(ay < 5.0, 6, 7))))))).to(tl.uint8)
    sign = (y < 0).to(tl.uint8)
    nib = (mag & 0x7) | (sign << 3)

    nib2 = tl.reshape(nib, (BLOCK_B, GROUP_SIZE // 2, 2))
    lo, hi = tl.split(nib2)
    byte = (lo | (hi << 4)).to(tl.uint8)

    rj2 = tl.arange(0, GROUP_SIZE // 2)
    f4_off = (pid_g * (GROUP_SIZE // 2) + rj2)[:, None] * B_cap + rb[None, :]
    tl.store(x_f4_ptr + f4_off, tl.trans(byte), mask=mask_b[None, :])


def rotate_quantize_fused_fp4_pooled(x: torch.Tensor, rot_bank: torch.Tensor,
                                     expert_info: torch.Tensor,
                                     block_info: torch.Tensor,
                                     group_size: int, num_groups: int):
    """池内逐成员旋转 + e2m1/e8m0 量化的 python wrapper（静态派发用）。

    参数语义与 triton_kernels_a8.rotate_quantize_fused_pooled 一一对应；
    fp4 无 extra_scale 通道（cb 类常量只能进 GEMM epilogue）。

    Returns:
        x_f4: (K_total//2, B_cap) uint8 预转置；x_s: (B_cap, K_total//32) uint8
    """
    B_cap, K_total = x.shape
    if K_total != num_groups * group_size:
        raise ValueError(f"K_total ({K_total}) != num_groups*group_size "
                         f"({num_groups * group_size})")
    x_f4 = torch.empty((K_total // 2, B_cap), dtype=torch.uint8, device=x.device)
    x_s = torch.empty((B_cap, K_total // 32), dtype=torch.uint8, device=x.device)
    grid = (block_info.shape[0], num_groups)
    _rotate_quantize_kernel_fp4_pooled[grid](
        x, rot_bank, expert_info, block_info, x_f4, x_s,
        B_cap, K_total, rot_bank.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups, BLOCK_B=32,
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
