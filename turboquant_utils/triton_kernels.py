"""Triton fused dequant + matmul kernels for on-the-fly inference.

These kernels avoid materializing the full dequantized weight by fusing
nbit unpack → codebook lookup → matmul → norm rescale in one kernel launch.

Supports 1/2/4/8 bit quantization with efficient bitwise unpacking.

Optimizations applied:
  1. Autotune — @triton.autotune searches (BLOCK_B, BLOCK_N, BLOCK_K, num_warps,
     num_stages) per problem shape; cached after first invocation.
  2. Shared-memory codebook — small codebook stays in L1/registers
     after first load in each K-tile; repeated gather hits cache.
  3. TF32 tensor cores — allow_tf32=True in tl.dot for ~2× throughput on
     fp32 Ampere+/Ada/Hopper.
  4. Pre-scaled norms — norms / sqrt(K) computed once on host, eliminating
     per-element division in the kernel epilogue.
  5. Software pipelining — num_stages in autotune configs controls prefetch depth.
  6. Transpose elimination — accumulates in natural (B, N) layout; no extra
     transpose required.
  7. Efficient bitwise unpack — parallel unpack for 1/2/4/8 bit.
  8. Pre-compute rotations — compute all rotated inputs outside kernel loop.

Main kernel: _turboquant_fused_matmul_kernel_nbit
  - Input: x_rot (pre-rotated activations), packed indices, codebook, norms_scaled
  - Output: x_rot @ codebook[indices].T * norms_scaled

Supports group-wise calls: pass a packed index slice with K=g_dim.
"""

from __future__ import annotations

import time
import math

import torch
import triton
import triton.language as tl

from .rotation import generate_rotation_matrix, batch_rotate_input


@triton.jit
def _turboquant_fused_matmul_kernel_nbit(
    # Input
    input_ptr,        # (B, K) pre-rotated activations
    # Quantized weight
    indices_ptr,      # (N, PACKED_K) packed uint8
    codebook_ptr,     # (n_levels,) float16 — Step 2: codebook 改为 fp16
    norms_ptr,        # (N,) float16 — pre-scaled by 1/scale on host, Step 2: fp16
    # Output
    output_ptr,       # (B, N)
    # Dims
    B, N, K,
    PACKED_K,         # packed dimension (stride) for indices (FULL tensor width)
    # 注意：COL_START 必须是运行时参数，不能声明为 tl.constexpr。
    # constexpr 会参与编译缓存 key：真实 MoE 里 col_start 随 expert×bit 变化有
    # 成百上千个取值，每个取值触发一次完整重编译（实测 ~205ms/个），
    # 导致首个 mini_batch 240s 的编译风暴（见 test/test_colstart_recompile.py）。
    COL_START,        # byte-column offset of this slice within the full tensor
    N_LEVELS: tl.constexpr,
    BIT_WIDTH: tl.constexpr,
    BLOCK_B: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    """Fused dequant-matmul: output[b,n] = norms_scaled[n] * Σ_k x[b,k] * codebook[idx[n,k]]

    Supports 1/2/4/8 bit with efficient bitwise unpacking:
    - 1-bit: 8 elements per byte
    - 2-bit: 4 elements per byte
    - 4-bit: 2 elements per byte
    - 8-bit: 1 element per byte
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    # Step 3 (FP16 链路改造): accumulator 改为 fp16，完整 FP16 链路
    # - tl.dot 两边都是 fp16 → FP16 Tensor Core
    # - 累加在 fp16 下进行（K=128, 权重归一化，累加值≈128，远小于 fp16 max=65504）
    # - 去掉 allow_tf32（fp16 输入下无效）
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float16)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K

        inp_off = rb[:, None] * K + rk[None, :]
        inp_mask = mask_b[:, None] & mask_k[None, :]
        inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0)

        if BIT_WIDTH == 8:
            # 8-bit fast path: no unpacking needed
            byte_off = rn[:, None] * PACKED_K + (COL_START + rk[None, :])
            w_mask = mask_n[:, None] & mask_k[None, :]
            idx = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.int32)
            w_quant = tl.load(codebook_ptr + idx, mask=w_mask, other=0.0)
        else:
            # 1/2/4-bit: need bit unpacking
            BIT_MASK = (1 << BIT_WIDTH) - 1
            byte_col = rk // ELEMENTS_PER_BYTE
            pos_in_byte = rk % ELEMENTS_PER_BYTE

            byte_off = rn[:, None] * PACKED_K + (COL_START + byte_col[None, :])
            w_mask = mask_n[:, None] & mask_k[None, :]
            packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)

            shift = pos_in_byte * BIT_WIDTH
            shift_broadcast = shift[None, :]
            idx = (packed >> shift_broadcast) & BIT_MASK
            idx = idx.to(tl.int32)

            w_quant = tl.load(codebook_ptr + idx, mask=w_mask, other=0.0)

        acc += tl.dot(
            inp_tile,
            tl.trans(w_quant),
            out_dtype=tl.float16,
        )

    norm_vals = tl.load(norms_ptr + rn, mask=mask_n, other=1.0)
    acc = acc * norm_vals[None, :]

    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def triton_fused_matmul(
    x_rot: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    K: int,
    bit_width: int = 4,
    scale: float | None = None,
    col_start: int = 0,
) -> torch.Tensor:
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    B = x_rot.shape[0]
    N = indices_packed.shape[0]
    PACKED_K = indices_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    # 统一在 fp16 下计算（FP16 Tensor Core 吞吐最高）
    # 输入为 bf16/fp32 时自动转 fp16，输出再转回原 dtype
    orig_dtype = x_rot.dtype
    if orig_dtype != torch.float16:
        x_rot = x_rot.half()
        codebook = codebook.half()
        norms = norms.half()

    # Step 2: norms 已是 fp16，除以 scale 后仍为 fp16
    norms_scaled = norms / scale

    # 输出为 fp16（kernel 内部链路全 fp16）
    output = torch.empty(B, N, dtype=torch.float16, device=x_rot.device)

    grid = (
        triton.cdiv(B, 16),
        triton.cdiv(N, 64),
    )

    _turboquant_fused_matmul_kernel_nbit[grid](
        x_rot, indices_packed, codebook, norms_scaled, output,
        B, N, K, PACKED_K, col_start,
        N_LEVELS=codebook.shape[0],
        BIT_WIDTH=bit_width,
    )

    # 输出转回原 dtype（如 bf16）
    if orig_dtype != torch.float16:
        output = output.to(orig_dtype)

    return output


@triton.jit
def _turboquant_fused_dual_matmul_kernel_nbit(
    input1_ptr,
    indices1_ptr,
    codebook1_ptr,  # (n_levels,) float16 — Step 2: codebook 改为 fp16
    norms1_ptr,     # (N,) float16 — Step 2: norms 改为 fp16
    input2_ptr,
    indices2_ptr,
    codebook2_ptr,  # (n_levels,) float16 — Step 2: codebook 改为 fp16
    norms2_ptr,     # (N,) float16 — Step 2: norms 改为 fp16
    output_ptr,
    B, N, K,
    PACKED_K,
    N_LEVELS: tl.constexpr,
    BIT_WIDTH: tl.constexpr,
    SAME_INPUT: tl.constexpr,
    BLOCK_B: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    # Step 3: accumulator 改为 fp16，完整 FP16 链路
    acc1 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float16)
    acc2 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float16)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        inp_mask = mask_b[:, None] & mask_k[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]

        inp1_off = rb[:, None] * K + rk[None, :]
        inp1 = tl.load(input1_ptr + inp1_off, mask=inp_mask, other=0.0)

        if BIT_WIDTH == 8:
            # 8-bit fast path: no unpacking needed
            byte_off = rn[:, None] * PACKED_K + rk[None, :]
            idx1 = tl.load(indices1_ptr + byte_off, mask=w_mask, other=0).to(tl.int32)
            w1 = tl.load(codebook1_ptr + idx1, mask=w_mask, other=0.0)

            acc1 += tl.dot(inp1, tl.trans(w1))

            if SAME_INPUT:
                inp2 = inp1
            else:
                inp2_off = rb[:, None] * K + rk[None, :]
                inp2 = tl.load(input2_ptr + inp2_off, mask=inp_mask, other=0.0)

            idx2 = tl.load(indices2_ptr + byte_off, mask=w_mask, other=0).to(tl.int32)
            w2 = tl.load(codebook2_ptr + idx2, mask=w_mask, other=0.0)

            acc2 += tl.dot(inp2, tl.trans(w2))
        else:
            # 1/2/4-bit: need bit unpacking
            BIT_MASK = (1 << BIT_WIDTH) - 1
            byte_col = rk // ELEMENTS_PER_BYTE
            pos_in_byte = rk % ELEMENTS_PER_BYTE
            byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
            shift = pos_in_byte * BIT_WIDTH
            shift_broadcast = shift[None, :]

            packed1 = tl.load(indices1_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
            idx1 = (packed1 >> shift_broadcast) & BIT_MASK
            idx1 = idx1.to(tl.int32)
            w1 = tl.load(codebook1_ptr + idx1, mask=w_mask, other=0.0)

            acc1 += tl.dot(inp1, tl.trans(w1))

            if SAME_INPUT:
                inp2 = inp1
            else:
                inp2_off = rb[:, None] * K + rk[None, :]
                inp2 = tl.load(input2_ptr + inp2_off, mask=inp_mask, other=0.0)

            packed2 = tl.load(indices2_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
            idx2 = (packed2 >> shift_broadcast) & BIT_MASK
            idx2 = idx2.to(tl.int32)
            w2 = tl.load(codebook2_ptr + idx2, mask=w_mask, other=0.0)

            acc2 += tl.dot(inp2, tl.trans(w2))

    n1 = tl.load(norms1_ptr + rn, mask=mask_n, other=1.0)
    n2 = tl.load(norms2_ptr + rn, mask=mask_n, other=1.0)
    result = acc1 * n1[None, :] + acc2 * n2[None, :]

    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, result.to(output_ptr.dtype.element_ty), mask=out_mask)


def triton_fused_dual_matmul(
    x_rot1: torch.Tensor,
    indices1_packed: torch.Tensor,
    codebook1: torch.Tensor,
    norms1: torch.Tensor,
    x_rot2: torch.Tensor,
    indices2_packed: torch.Tensor,
    codebook2: torch.Tensor,
    norms2: torch.Tensor,
    K: int,
    bit_width: int = 4,
    scale: float | None = None,
) -> torch.Tensor:
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    B = x_rot1.shape[0]
    N = indices1_packed.shape[0]
    PACKED_K = indices1_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    # 统一在 fp16 下计算
    orig_dtype = x_rot1.dtype
    if orig_dtype != torch.float16:
        x_rot1 = x_rot1.half()
        x_rot2 = x_rot2.half()
        codebook1 = codebook1.half()
        codebook2 = codebook2.half()
        norms1 = norms1.half()
        norms2 = norms2.half()

    norms1_scaled = norms1 / scale
    norms2_scaled = norms2 / scale

    same_input = x_rot1.data_ptr() == x_rot2.data_ptr()

    output = torch.empty(B, N, dtype=torch.float16, device=x_rot1.device)

    grid = (
        triton.cdiv(B, 16),
        triton.cdiv(N, 64),
    )

    _turboquant_fused_dual_matmul_kernel_nbit[grid](
        x_rot1, indices1_packed, codebook1, norms1_scaled,
        x_rot2, indices2_packed, codebook2, norms2_scaled,
        output,
        B, N, K, PACKED_K,
        N_LEVELS=codebook1.shape[0],
        BIT_WIDTH=bit_width,
        SAME_INPUT=1 if same_input else 0,
    )

    # 输出转回原 dtype
    if orig_dtype != torch.float16:
        output = output.to(orig_dtype)

    return output


# ---------------------------------------------------------------------------
# P4-4: Multi-group fused kernel
# ---------------------------------------------------------------------------
# 单个 kernel launch 内处理多个 group 的 dequant + matmul + 累加，
# 省去多次 launch + 中间张量读写 + Python 循环开销。
#
# 约束：所有 group 大小相同（= group_size），即 in_features % group_size == 0
# ---------------------------------------------------------------------------

@triton.jit
def _turboquant_fused_matmul_kernel_grouped(
    # Input
    input_ptr,        # (B, K_total) 拼接后的旋转输入（连续存储）
    # Quantized weight
    indices_ptr,      # (N, PACKED_K_stride) packed uint8（行 stride = PACKED_K_stride）
    codebook_ptr,     # (n_levels,) float16
    norms_ptr,        # (N, NORMS_COL_STRIDE) float16 — per-group norms (pre-scaled)
    # Output
    output_ptr,       # (B, N)
    # Shape
    B, N,
    K_total,              # 总 K = num_groups * group_size
    PACKED_K_stride,      # packed indices 的行 stride（= 原矩阵总列数）
    PACKED_COL_START,     # packed indices 的列起始偏移（group 0 对应的列）
    NORMS_COL_STRIDE,     # norms 的列 stride（= 原矩阵总 group 数）
    NORMS_GROUP_START,    # norms 的起始 group 索引
    # Constexpr config
    GROUP_SIZE: tl.constexpr,     # 每个 group 的 K 大小
    NUM_GROUPS: tl.constexpr,     # group 数量（本次处理的）
    BIT_WIDTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    BLOCK_B: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    """Multi-group fused dequant + matmul kernel.

    外层循环 NUM_GROUPS 个 group，每个 group 独立 dequant + matmul，
    乘对应 norms 后累加到输出。

    支持 packed indices 和 norms 的偏移/stride，可用于 slice_in_features 场景：
    - packed indices 传原始大矩阵，用 PACKED_COL_START 指定切片起始列
    - norms 传原始 (N, total_groups) 矩阵，用 NORMS_GROUP_START 指定起始 group
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    total_acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float16)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    PACKED_PER_GROUP = GROUP_SIZE // ELEMENTS_PER_BYTE

    for g in range(NUM_GROUPS):
        g_start = g * GROUP_SIZE
        p_start = PACKED_COL_START + g * PACKED_PER_GROUP

        # 加载当前 group 的 norms: (BLOCK_N,)
        # norms 存储: (N, total_groups) row-major → 偏移 = rn * NORMS_COL_STRIDE + (NORMS_GROUP_START + g)
        norm_off = rn * NORMS_COL_STRIDE + (NORMS_GROUP_START + g)
        norm_g = tl.load(norms_ptr + norm_off, mask=mask_n, other=1.0)

        acc_g = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float16)

        for k_start in range(0, GROUP_SIZE, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < GROUP_SIZE

            # Load input tile (with group offset in K dimension)
            inp_k = g_start + rk
            inp_off = rb[:, None] * K_total + inp_k[None, :]
            inp_mask = mask_b[:, None] & mask_k[None, :]
            inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0)

            if BIT_WIDTH == 8:
                byte_col = rk
                byte_off = rn[:, None] * PACKED_K_stride + (p_start + byte_col[None, :])
                w_mask = mask_n[:, None] & mask_k[None, :]
                idx = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.int32)
                w_quant = tl.load(codebook_ptr + idx, mask=w_mask, other=0.0)
            else:
                BIT_MASK = (1 << BIT_WIDTH) - 1
                byte_col = rk // ELEMENTS_PER_BYTE
                pos_in_byte = rk % ELEMENTS_PER_BYTE
                pbc = p_start + byte_col
                byte_off = rn[:, None] * PACKED_K_stride + pbc[None, :]
                w_mask = mask_n[:, None] & mask_k[None, :]
                packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
                shift = pos_in_byte * BIT_WIDTH
                idx = (packed >> shift[None, :]) & BIT_MASK
                idx = idx.to(tl.int32)
                w_quant = tl.load(codebook_ptr + idx, mask=w_mask, other=0.0)

            acc_g += tl.dot(inp_tile, tl.trans(w_quant), out_dtype=tl.float16)

        # 乘 norms 后加到总累加器
        total_acc += acc_g * norm_g[None, :]

    tl.store(
        output_ptr + rb[:, None] * N + rn[None, :],
        total_acc.to(output_ptr.dtype.element_ty),
        mask=mask_b[:, None] & mask_n[None, :],
    )


# P4-2: 各 bit-width 的最优 kernel 配置（离线调优，针对 RTX 5090）
# 调优脚本: test_p4_tune.py
# 测试形状: group_size=128, B≈32, 典型 MoE 场景
# 格式: (BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages)
_FUSED_GROUPED_CONFIG = {
    1: (16, 32, 128, 2, 3),   # 1-bit: 计算轻, warps 少好
    2: (32, 32, 128, 2, 2),   # 2-bit: 计算中等
    4: (32, 32, 128, 8, 3),   # 4-bit: 访存重, warps 多藏延迟
    8: (16, 32, 128, 4, 3),   # 8-bit: 默认配置（attention 用，暂未精细调）
}


def _get_fused_grouped_config(bit_width):
    """获取指定 bit-width 的最优配置。不在表中返回默认值。"""
    return _FUSED_GROUPED_CONFIG.get(bit_width, (16, 64, 64, 4, 3))


def _triton_fused_matmul_grouped_fused(
    x_rot_concat, indices_packed, codebook, norms_scaled,
    group_size, num_groups, bit_width,
    packed_col_start=0, norms_group_start=0,
    packed_k_stride=None, norms_col_stride=None,
):
    """内部函数：调用 fused grouped kernel。

    Args:
        x_rot_concat: (B, K_total) 拼接后的旋转输入，连续存储
        indices_packed: (N, PACKED_K_stride) 权重 packed indices
        codebook: (n_levels,)
        norms_scaled: (N, NORMS_COL_STRIDE) pre-scaled norms
        group_size: 每个 group 的 K 大小
        num_groups: 本次处理的 group 数量
        bit_width: 1/2/4/8
        packed_col_start: packed indices 的列起始偏移（group 0 对应的列）
        norms_group_start: norms 的起始 group 索引
        packed_k_stride: packed indices 的行 stride，默认 = indices_packed.shape[1]
        norms_col_stride: norms 的列 stride，默认 = norms_scaled.shape[1]

    Returns:
        output: (B, N)
    """
    B = x_rot_concat.shape[0]
    N = indices_packed.shape[0]
    K_total = x_rot_concat.shape[1]
    if packed_k_stride is None:
        packed_k_stride = indices_packed.shape[1]
    if norms_col_stride is None:
        norms_col_stride = norms_scaled.shape[1]

    output = torch.empty(B, N, dtype=x_rot_concat.dtype, device=x_rot_concat.device)

    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages = _get_fused_grouped_config(bit_width)

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N, BLOCK_N))

    _turboquant_fused_matmul_kernel_grouped[grid](
        x_rot_concat, indices_packed, codebook, norms_scaled, output,
        B, N, K_total,
        packed_k_stride, packed_col_start,
        norms_col_stride, norms_group_start,
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        BIT_WIDTH=bit_width, N_LEVELS=codebook.shape[0],
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    return output


def triton_fused_matmul_grouped(
    x, indices_packed, codebook, norms, seed, group_size, in_features, bit_width: int = 4
):
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    if isinstance(seed, torch.Tensor):
        seed = int(seed.item())

    batch_size = x.shape[0]
    out_features = indices_packed.shape[0]
    ELEMENTS_PER_BYTE = 8 // bit_width

    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    # P4-7: Batch rotation via bmm（fast path，in_features 对齐 group_size）
    # 把 num_groups 次小 matmul 合并为一次 bmm，消除 Python 循环 + 多次 launch 开销。
    if in_features % group_size == 0:
        num_groups = in_features // group_size
        # P4-4: Multi-group fused fast path 条件：num_groups >= 2
        if num_groups >= 2:
            orig_dtype = x.dtype
            if orig_dtype != torch.float16:
                codebook = codebook.half()
                norms = norms.half()
                x_rot = batch_rotate_input(x.half(), group_size, seed)
            else:
                x_rot = batch_rotate_input(x, group_size, seed)

            scale = math.sqrt(group_size)
            norms_scaled = norms / scale

            output_fp16 = _triton_fused_matmul_grouped_fused(
                x_rot, indices_packed, codebook, norms_scaled,
                group_size, num_groups, bit_width,
            )

            if orig_dtype != torch.float16:
                return output_fp16.to(orig_dtype)
            return output_fp16

        # num_groups == 1 的 fast path：直接单次旋转 + triton_fused_matmul
        else:
            Pi = generate_rotation_matrix(group_size, seed, device=x.device).to(x.dtype)
            x_rot = x @ Pi.T
            if norms.dim() == 2:
                norms = norms.squeeze(1)
            return triton_fused_matmul(x_rot, indices_packed, codebook, norms, in_features, bit_width)

    # Fallback: 逐 group 调用（处理非对齐 in_features 等情况）
    num_groups = (in_features + group_size - 1) // group_size
    x_rot_list = []
    for group_idx in range(num_groups):
        g_start = group_idx * group_size
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start
        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=x.device).to(x.dtype)
        x_rot_g = x[:, g_start:g_end] @ Pi.T
        x_rot_list.append(x_rot_g)

    output = torch.zeros(batch_size, out_features, dtype=x.dtype, device=x.device)

    group_idx = 0
    for g_start in range(0, in_features, group_size):
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start

        x_rot_g = x_rot_list[group_idx]

        packed_start = g_start // ELEMENTS_PER_BYTE
        g_end = g_start + g_dim
        packed_end = g_end // ELEMENTS_PER_BYTE
        if g_end % ELEMENTS_PER_BYTE != 0:
            packed_end += 1

        norms_g = norms[:, group_idx]

        out_g = triton_fused_matmul(x_rot_g, indices_packed, codebook, norms_g, g_dim, bit_width, col_start=packed_start)

        output += out_g

        group_idx += 1

    return output



def triton_fused_matmul_grouped_slice_rows(
    x, indices_packed, codebook, norms, seed, group_size,
    in_features, row_start, row_end, bit_width: int = 4
):
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    if isinstance(seed, torch.Tensor):
        seed = int(seed.item())

    batch_size = x.shape[0]
    slice_out_features = row_end - row_start
    ELEMENTS_PER_BYTE = 8 // bit_width

    indices_packed_slice = indices_packed[row_start:row_end]
    norms_slice = norms[row_start:row_end] if norms.dim() == 1 else norms[row_start:row_end, :]

    if norms_slice.dim() == 1:
        norms_slice = norms_slice.unsqueeze(1)

    # P4-7: Batch rotation via bmm（fast path，in_features 对齐 group_size）
    if in_features % group_size == 0:
        num_groups = in_features // group_size
        if num_groups >= 2:
            orig_dtype = x.dtype
            if orig_dtype != torch.float16:
                codebook = codebook.half()
                norms_slice = norms_slice.half()
                x_rot = batch_rotate_input(x.half(), group_size, seed)
            else:
                x_rot = batch_rotate_input(x, group_size, seed)

            scale = math.sqrt(group_size)
            norms_scaled = norms_slice / scale

            output_fp16 = _triton_fused_matmul_grouped_fused(
                x_rot, indices_packed_slice, codebook, norms_scaled,
                group_size, num_groups, bit_width,
            )

            if orig_dtype != torch.float16:
                return output_fp16.to(orig_dtype)
            return output_fp16

        # num_groups == 1
        else:
            Pi = generate_rotation_matrix(group_size, seed, device=x.device).to(x.dtype)
            x_rot = x @ Pi.T
            if norms_slice.dim() == 2:
                norms_slice = norms_slice.squeeze(1)
            return triton_fused_matmul(x_rot, indices_packed_slice, codebook, norms_slice, in_features, bit_width)

    # Fallback: 逐 group 调用（非对齐情况）
    num_groups = (in_features + group_size - 1) // group_size
    x_rot_list = []
    for group_idx in range(num_groups):
        g_start = group_idx * group_size
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start
        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=x.device).to(x.dtype)
        x_rot_g = x[:, g_start:g_end] @ Pi.T
        x_rot_list.append(x_rot_g)

    output = torch.zeros(batch_size, slice_out_features, dtype=x.dtype, device=x.device)

    group_idx = 0
    for g_start in range(0, in_features, group_size):
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start

        x_rot_g = x_rot_list[group_idx]

        packed_start = g_start // ELEMENTS_PER_BYTE

        norms_g = norms_slice[:, group_idx]

        out_g = triton_fused_matmul(x_rot_g, indices_packed_slice, codebook, norms_g, g_dim, bit_width, col_start=packed_start)

        output += out_g

        group_idx += 1

    return output


def triton_fused_matmul_grouped_slice_in_features(
    x, indices_packed, codebook, norms, seed, group_size,
    original_start, original_end, full_in_features, bit_width: int = 4
):
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    if original_start % group_size != 0:
        raise ValueError(f"original_start ({original_start}) must be aligned to group_size ({group_size})")

    if isinstance(seed, torch.Tensor):
        seed = int(seed.item())

    batch_size = x.shape[0]
    out_features = indices_packed.shape[0]
    slice_in_features = original_end - original_start
    ELEMENTS_PER_BYTE = 8 // bit_width

    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    # P4-7: Batch rotation via bmm（fast path，slice_in_features 对齐 group_size）
    if slice_in_features % group_size == 0:
        num_groups_in_slice = slice_in_features // group_size
        # seed_base 用 original_start 对齐的 seed
        seed_base = seed + original_start

        if num_groups_in_slice >= 2:
            orig_dtype = x.dtype
            if orig_dtype != torch.float16:
                codebook = codebook.half()
                norms = norms.half()
                x_rot = batch_rotate_input(x.half(), group_size, seed_base)
            else:
                x_rot = batch_rotate_input(x, group_size, seed_base)

            scale = math.sqrt(group_size)
            norms_scaled = norms / scale

            # packed 起始列 = original_start 的 packed 位置
            packed_col_start = original_start // ELEMENTS_PER_BYTE
            # norms 起始 group = original_start // group_size
            norms_group_start = original_start // group_size

            output_fp16 = _triton_fused_matmul_grouped_fused(
                x_rot, indices_packed, codebook, norms_scaled,
                group_size, num_groups_in_slice, bit_width,
                packed_col_start=packed_col_start,
                norms_group_start=norms_group_start,
                packed_k_stride=indices_packed.shape[1],
                norms_col_stride=norms_scaled.shape[1],
            )

            if orig_dtype != torch.float16:
                return output_fp16.to(orig_dtype)
            return output_fp16

        # num_groups_in_slice == 1
        else:
            Pi = generate_rotation_matrix(group_size, seed_base, device=x.device).to(x.dtype)
            x_rot = x @ Pi.T
            group_idx_original = original_start // group_size
            norms_g = norms[:, group_idx_original] if norms.dim() == 2 else norms
            packed_start = original_start // ELEMENTS_PER_BYTE
            return triton_fused_matmul(x_rot, indices_packed, codebook, norms_g, group_size, bit_width, col_start=packed_start)

    # Fallback: 逐 group 调用（非对齐情况）
    num_groups_in_slice = (slice_in_features + group_size - 1) // group_size
    x_rot_list = []
    for group_idx_in_slice in range(num_groups_in_slice):
        g_start_in_slice = group_idx_in_slice * group_size
        g_end_in_slice = min(g_start_in_slice + group_size, slice_in_features)
        g_dim = g_end_in_slice - g_start_in_slice
        g_start_original = original_start + g_start_in_slice
        Pi = generate_rotation_matrix(g_dim, seed + g_start_original, device=x.device).to(x.dtype)
        x_rot_g = x[:, g_start_in_slice:g_end_in_slice] @ Pi.T
        x_rot_list.append(x_rot_g)

    output = torch.zeros(batch_size, out_features, dtype=x.dtype, device=x.device)

    group_idx_in_slice = 0
    for g_start_in_slice in range(0, slice_in_features, group_size):
        g_end_in_slice = min(g_start_in_slice + group_size, slice_in_features)
        g_dim = g_end_in_slice - g_start_in_slice

        g_start_original = original_start + g_start_in_slice

        x_rot_g = x_rot_list[group_idx_in_slice]

        packed_start = g_start_original // ELEMENTS_PER_BYTE

        group_idx_original = g_start_original // group_size
        norms_g = norms[:, group_idx_original]

        out_g = triton_fused_matmul(x_rot_g, indices_packed, codebook, norms_g, g_dim, bit_width, col_start=packed_start)

        output += out_g

        group_idx_in_slice += 1

    return output
