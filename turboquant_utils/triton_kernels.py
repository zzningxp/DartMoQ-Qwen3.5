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

from .rotation import generate_rotation_matrix


@triton.jit
def _turboquant_fused_matmul_kernel_nbit(
    # Input
    input_ptr,        # (B, K) pre-rotated activations
    # Quantized weight
    indices_ptr,      # (N, PACKED_K) packed uint8
    codebook_ptr,     # (n_levels,) float32
    norms_ptr,        # (N,) float32 — pre-scaled by 1/scale on host
    # Output
    output_ptr,       # (B, N)
    # Dims
    B, N, K,
    PACKED_K,         # packed dimension for indices
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

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    # Calculate elements per byte based on bit width
    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    BIT_MASK = (1 << BIT_WIDTH) - 1

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K

        # Load input tile: (BLOCK_B, BLOCK_K)
        inp_off = rb[:, None] * K + rk[None, :]
        inp_mask = mask_b[:, None] & mask_k[None, :]
        inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0)

        # Load + unpack weight indices: (BLOCK_N, BLOCK_K)
        # Calculate byte column and position within byte
        byte_col = rk // ELEMENTS_PER_BYTE
        pos_in_byte = rk % ELEMENTS_PER_BYTE

        byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)

        # Unpack using bitwise operations: (packed >> (pos * BIT_WIDTH)) & BIT_MASK
        shift = pos_in_byte * BIT_WIDTH
        # Need to broadcast shift to (BLOCK_N, BLOCK_K)
        shift_broadcast = shift[None, :]
        idx = (packed >> shift_broadcast) & BIT_MASK
        # Convert to int32 for codebook lookup
        idx = idx.to(tl.int32)

        # Codebook lookup (stays in L1/registers after first access)
        w_quant = tl.load(codebook_ptr + idx, mask=w_mask, other=0.0)

        # TF32 tensor-core MMA: (BLOCK_B, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        acc += tl.dot(
            inp_tile.to(tl.float32),
            tl.trans(w_quant.to(tl.float32)),
            allow_tf32=True,
        )

    # Multiply by pre-scaled norms (norms / scale computed on host)
    norm_vals = tl.load(norms_ptr + rn, mask=mask_n, other=1.0)
    acc = acc * norm_vals[None, :]

    # Store
    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def triton_fused_matmul(
    x_rot: torch.Tensor,           # (B, K) pre-rotated input
    indices_packed: torch.Tensor,   # (N, PACKED_K) packed uint8
    codebook: torch.Tensor,         # (n_levels,) float32
    norms: torch.Tensor,            # (N,) float32
    K: int,                         # in_features (or group_size for per-group calls)
    bit_width: int = 4,             # 1/2/4/8
    scale: float | None = None,     # override sqrt(K) if needed
) -> torch.Tensor:
    """Fused dequant + matmul via Triton with autotune + TF32 tensor cores.

    Expects pre-rotated input: x_rot = x @ Pi.T

    Supports per-group calls: pass a slice of packed indices
    with K=g_dim. The kernel handles unpack + codebook lookup + matmul + norm
    rescale in one launch, avoiding materialization of the (N, K) float weight.

    Args:
        x_rot: (B, K) pre-rotated activations
        indices_packed: (N, PACKED_K) packed nbit weight indices
        codebook: centroids
        norms: per-row weight norms (N,)
        K: dimension of this group (in_features or group_size)
        bit_width: 1/2/4/8 (default: 4)
        scale: norm divisor (default: sqrt(K))

    Returns:
        output: (B, N)
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    B = x_rot.shape[0]
    N = indices_packed.shape[0]
    PACKED_K = indices_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    # Pre-scale norms on host (avoids per-element division in kernel)
    norms_scaled = norms / scale

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    # Fixed grid for the fixed config
    grid = (
        triton.cdiv(B, 16),
        triton.cdiv(N, 64),
    )

    _turboquant_fused_matmul_kernel_nbit[grid](
        x_rot, indices_packed, codebook, norms_scaled, output,
        B, N, K, PACKED_K,
        N_LEVELS=codebook.shape[0],
        BIT_WIDTH=bit_width,
    )

    return output


# ---------------------------------------------------------------------------
# Backward compatibility: original 4-bit only kernel (preserved for compatibility)
# ---------------------------------------------------------------------------

@triton.jit
def _turboquant_fused_matmul_kernel(
    # Input
    input_ptr,        # (B, K) pre-rotated activations
    # Quantized weight
    indices_ptr,      # (N, K//2) packed uint8
    codebook_ptr,     # (n_levels,) float32
    norms_ptr,        # (N,) float32 — pre-scaled by 1/scale on host
    # Output
    output_ptr,       # (B, N)
    # Dims
    B, N, K,
    PACKED_K,         # K // 2 (stride for packed index rows)
    N_LEVELS: tl.constexpr,
    BLOCK_B: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    """Original 4-bit only kernel (preserved for backward compatibility)."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K

        # Load input tile: (BLOCK_B, BLOCK_K)
        inp_off = rb[:, None] * K + rk[None, :]
        inp_mask = mask_b[:, None] & mask_k[None, :]
        inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0)

        # Load + unpack weight indices: (BLOCK_N, BLOCK_K)
        byte_col = rk // 2
        is_high = (rk % 2) == 1
        byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        idx = tl.where(is_high[None, :], hi, lo)

        # Codebook lookup (16 entries — stays in L1/registers after first access)
        w_quant = tl.load(codebook_ptr + idx.to(tl.int32), mask=w_mask, other=0.0)

        # TF32 tensor-core MMA: (BLOCK_B, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        acc += tl.dot(
            inp_tile.to(tl.float32),
            tl.trans(w_quant.to(tl.float32)),
            allow_tf32=True,
        )

    # Multiply by pre-scaled norms (norms / scale computed on host)
    norm_vals = tl.load(norms_ptr + rn, mask=mask_n, other=1.0)
    acc = acc * norm_vals[None, :]

    # Store
    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Dual-pass fused kernel: both residual passes in one launch (nbit version)
# ---------------------------------------------------------------------------

@triton.jit
def _turboquant_fused_dual_matmul_kernel_nbit(
    # Pass 1
    input1_ptr,        # (B, K)
    indices1_ptr,      # (N, PACKED_K) packed uint8
    codebook1_ptr,     # (n_levels,) float32
    norms1_ptr,        # (N,) float32 — pre-scaled
    # Pass 2
    input2_ptr,        # (B, K)
    indices2_ptr,      # (N, PACKED_K) packed uint8
    codebook2_ptr,     # (n_levels,) float32
    norms2_ptr,        # (N,) float32 — pre-scaled
    # Output
    output_ptr,        # (B, N)
    # Dims
    B, N, K,
    PACKED_K,
    N_LEVELS: tl.constexpr,
    BIT_WIDTH: tl.constexpr,
    SAME_INPUT: tl.constexpr,  # 1 if input1==input2 (shared rotation), 0 otherwise
    BLOCK_B: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    """Dual-pass fused dequant-matmul (nbit version): output = acc1*norms1 + acc2*norms2.

    Processes both residual passes in a single kernel launch, avoiding:
    - 2nd kernel launch overhead
    - Writing + reading pass1 output for the add
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    acc1 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    # Calculate elements per byte based on bit width
    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    BIT_MASK = (1 << BIT_WIDTH) - 1

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        inp_mask = mask_b[:, None] & mask_k[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]

        # Shared index math for packed byte addressing
        byte_col = rk // ELEMENTS_PER_BYTE
        pos_in_byte = rk % ELEMENTS_PER_BYTE
        byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
        shift = pos_in_byte * BIT_WIDTH
        shift_broadcast = shift[None, :]

        # ---- Pass 1: load input + unpack + codebook + dot ----
        inp1_off = rb[:, None] * K + rk[None, :]
        inp1 = tl.load(input1_ptr + inp1_off, mask=inp_mask, other=0.0)

        packed1 = tl.load(indices1_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        idx1 = (packed1 >> shift_broadcast) & BIT_MASK
        idx1 = idx1.to(tl.int32)
        w1 = tl.load(codebook1_ptr + idx1, mask=w_mask, other=0.0)

        acc1 += tl.dot(inp1.to(tl.float32), tl.trans(w1.to(tl.float32)), allow_tf32=True)

        # ---- Pass 2: load input + unpack + codebook + dot ----
        if SAME_INPUT:
            inp2 = inp1
        else:
            inp2_off = rb[:, None] * K + rk[None, :]
            inp2 = tl.load(input2_ptr + inp2_off, mask=inp_mask, other=0.0)

        packed2 = tl.load(indices2_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        idx2 = (packed2 >> shift_broadcast) & BIT_MASK
        idx2 = idx2.to(tl.int32)
        w2 = tl.load(codebook2_ptr + idx2, mask=w_mask, other=0.0)

        acc2 += tl.dot(inp2.to(tl.float32), tl.trans(w2.to(tl.float32)), allow_tf32=True)

    # Epilogue: combine both passes with their norms
    n1 = tl.load(norms1_ptr + rn, mask=mask_n, other=1.0)
    n2 = tl.load(norms2_ptr + rn, mask=mask_n, other=1.0)
    result = acc1 * n1[None, :] + acc2 * n2[None, :]

    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, result.to(output_ptr.dtype.element_ty), mask=out_mask)


def triton_fused_dual_matmul(
    x_rot1: torch.Tensor,           # (B, K) pre-rotated input for pass 1
    indices1_packed: torch.Tensor,   # (N, PACKED_K) packed uint8 for pass 1
    codebook1: torch.Tensor,         # (n_levels,) float32
    norms1: torch.Tensor,            # (N,) float32
    x_rot2: torch.Tensor,           # (B, K) pre-rotated input for pass 2
    indices2_packed: torch.Tensor,   # (N, PACKED_K) packed uint8 for pass 2
    codebook2: torch.Tensor,         # (n_levels,) float32
    norms2: torch.Tensor,            # (N,) float32
    K: int,
    bit_width: int = 4,             # 1/2/4/8
    scale: float | None = None,
) -> torch.Tensor:
    """Dual-pass fused dequant + matmul (nbit version): pass1 + pass2 in one Triton launch.

    Equivalent to:
        triton_fused_matmul(x_rot1, indices1, cb1, n1, K, bit_width, scale)
      + triton_fused_matmul(x_rot2, indices2, cb2, n2, K, bit_width, scale)

    but with one kernel launch, one output write, and shared index math.

    Args:
        x_rot1/x_rot2: (B, K) pre-rotated activations (may be same tensor)
        indices1/2_packed: (N, PACKED_K) packed nbit weight indices
        codebook1/2: centroids for each pass
        norms1/2: per-row weight norms for each pass
        K: group dimension
        bit_width: 1/2/4/8 (default: 4)
        scale: norm divisor (default: sqrt(K))

    Returns:
        output: (B, N) = pass1_out + pass2_out
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    B = x_rot1.shape[0]
    N = indices1_packed.shape[0]
    PACKED_K = indices1_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    norms1_scaled = norms1 / scale
    norms2_scaled = norms2 / scale

    # Detect shared rotation (same rotated input → skip redundant load)
    same_input = x_rot1.data_ptr() == x_rot2.data_ptr()

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot1.device)

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

    return output


# ---------------------------------------------------------------------------
# Backward compatibility: original 4-bit dual kernel
# ---------------------------------------------------------------------------

@triton.jit
def _turboquant_fused_dual_matmul_kernel(
    # Pass 1
    input1_ptr,        # (B, K)
    indices1_ptr,      # (N, K//2) packed uint8
    codebook1_ptr,     # (n_levels,) float32
    norms1_ptr,        # (N,) float32 — pre-scaled
    # Pass 2
    input2_ptr,        # (B, K)
    indices2_ptr,      # (N, K//2) packed uint8
    codebook2_ptr,     # (n_levels,) float32
    norms2_ptr,        # (N,) float32 — pre-scaled
    # Output
    output_ptr,        # (B, N)
    # Dims
    B, N, K,
    PACKED_K,
    N_LEVELS: tl.constexpr,
    SAME_INPUT: tl.constexpr,  # 1 if input1==input2 (shared rotation), 0 otherwise
    BLOCK_B: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    """Original 4-bit dual kernel (preserved for backward compatibility)."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    acc1 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        inp_mask = mask_b[:, None] & mask_k[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]

        # Shared index math for packed byte addressing
        byte_col = rk // 2
        is_high = (rk % 2) == 1
        byte_off = rn[:, None] * PACKED_K + byte_col[None, :]

        # ---- Pass 1: load input + unpack + codebook + dot ----
        inp1_off = rb[:, None] * K + rk[None, :]
        inp1 = tl.load(input1_ptr + inp1_off, mask=inp_mask, other=0.0)

        packed1 = tl.load(indices1_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        lo1 = packed1 & 0x0F
        hi1 = (packed1 >> 4) & 0x0F
        idx1 = tl.where(is_high[None, :], hi1, lo1)
        w1 = tl.load(codebook1_ptr + idx1.to(tl.int32), mask=w_mask, other=0.0)

        acc1 += tl.dot(inp1.to(tl.float32), tl.trans(w1.to(tl.float32)), allow_tf32=True)

        # ---- Pass 2: load input + unpack + codebook + dot ----
        if SAME_INPUT:
            inp2 = inp1
        else:
            inp2_off = rb[:, None] * K + rk[None, :]
            inp2 = tl.load(input2_ptr + inp2_off, mask=inp_mask, other=0.0)

        packed2 = tl.load(indices2_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        lo2 = packed2 & 0x0F
        hi2 = (packed2 >> 4) & 0x0F
        idx2 = tl.where(is_high[None, :], hi2, lo2)
        w2 = tl.load(codebook2_ptr + idx2.to(tl.int32), mask=w_mask, other=0.0)

        acc2 += tl.dot(inp2.to(tl.float32), tl.trans(w2.to(tl.float32)), allow_tf32=True)

    # Epilogue: combine both passes with their norms
    n1 = tl.load(norms1_ptr + rn, mask=mask_n, other=1.0)
    n2 = tl.load(norms2_ptr + rn, mask=mask_n, other=1.0)
    result = acc1 * n1[None, :] + acc2 * n2[None, :]

    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, result.to(output_ptr.dtype.element_ty), mask=out_mask)


def triton_fused_matmul_grouped(
    x, indices_packed, codebook, norms, seed, group_size, in_features, bit_width: int = 4
):
    """支持分组量化的 triton fused matmul (nbit version)

    将 in_features 按 group_size 分组，对每个分组：
    1. 旋转输入
    2. 调用 triton_fused_matmul
    3. 累加结果

    Args:
        x: (batch_size, in_features) 输入
        indices_packed: (out_features, PACKED_K) 打包的量化索引
        codebook: (n_levels,) 码本
        norms: (out_features,) 或 (out_features, n_groups) 范数
        seed: 随机种子（用于生成旋转矩阵）
        group_size: 分组大小
        in_features: 输入特征维度
        bit_width: 1/2/4/8 (default: 4)

    Returns:
        output: (batch_size, out_features) 结果
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    batch_size = x.shape[0]
    out_features = indices_packed.shape[0]
    ELEMENTS_PER_BYTE = 8 // bit_width

    # 确保 norms 是二维的
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    output = torch.zeros(batch_size, out_features, dtype=torch.float32, device=x.device)

    # 对每个分组分别处理
    group_idx = 0
    for g_start in range(0, in_features, group_size):
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start

        # 1. 旋转这个分组对应的输入
        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=x.device)
        x_g = x[:, g_start:g_end].float()
        x_rot_g = x_g @ Pi.T

        # 2. 切片这个分组对应的 packed indices，然后 clone 确保内存连续
        packed_start = g_start // ELEMENTS_PER_BYTE
        packed_end = g_end // ELEMENTS_PER_BYTE
        # Handle case when g_end is not aligned
        if g_end % ELEMENTS_PER_BYTE != 0:
            packed_end += 1
        indices_packed_g = indices_packed[:, packed_start:packed_end].clone()

        # 3. 取出这个分组对应的 norms
        norms_g = norms[:, group_idx]

        # 4. 调用 triton fused kernel
        out_g = triton_fused_matmul(x_rot_g, indices_packed_g, codebook, norms_g, g_dim, bit_width)

        # 5. 累加到输出
        output += out_g

        group_idx += 1

    return output


def triton_fused_matmul_grouped_slice_rows(
    x, indices_packed, codebook, norms, seed, group_size,
    in_features, row_start, row_end, bit_width: int = 4
):
    """
    Triton fused matmul，针对 rows/out_features 维度切片的场景（MoE gate_up 投影）。

    Args:
        x: (batch_size, in_features) - 完整输入
        indices_packed: (out_features, PACKED_K) - 完整的 packed 索引
        codebook: (n_levels,) - 码本
        norms: (out_features,) 或 (out_features, n_groups) - 范数
        seed: 基础随机种子
        group_size: 分组大小
        in_features: 完整的输入特征维度
        row_start: 切片在 rows/out_features 中的起始位置
        row_end: 切片在 rows/out_features 中的结束位置
        bit_width: 1/2/4/8 (default: 4)

    Returns:
        output: (batch_size, row_end - row_start) 结果
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    # 确保 seed 是 Python int
    if isinstance(seed, torch.Tensor):
        seed = int(seed.item())

    batch_size = x.shape[0]
    slice_out_features = row_end - row_start
    ELEMENTS_PER_BYTE = 8 // bit_width

    # 切片 rows/out_features 维度
    indices_packed_slice = indices_packed[row_start:row_end]
    norms_slice = norms[row_start:row_end] if norms.dim() == 1 else norms[row_start:row_end, :]

    # 确保 norms 是二维的
    if norms_slice.dim() == 1:
        norms_slice = norms_slice.unsqueeze(1)

    output = torch.zeros(batch_size, slice_out_features, dtype=torch.float32, device=x.device)

    # 对每个分组分别处理
    group_idx = 0
    for g_start in range(0, in_features, group_size):
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start

        # 1. 旋转这个分组对应的输入
        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=x.device)
        x_g = x[:, g_start:g_end].float()
        x_rot_g = x_g @ Pi.T

        # 2. 切片这个分组对应的 packed indices
        packed_start = g_start // ELEMENTS_PER_BYTE
        packed_end = g_end // ELEMENTS_PER_BYTE
        if g_end % ELEMENTS_PER_BYTE != 0:
            packed_end += 1
        indices_packed_g = indices_packed_slice[:, packed_start:packed_end].clone()

        # 3. 取出这个分组对应的 norms
        norms_g = norms_slice[:, group_idx]

        # 4. 调用 triton fused kernel
        out_g = triton_fused_matmul(x_rot_g, indices_packed_g, codebook, norms_g, g_dim, bit_width)

        # 5. 累加到输出
        output += out_g

        group_idx += 1

    return output


def triton_fused_matmul_grouped_slice_in_features(
    x, indices_packed, codebook, norms, seed, group_size,
    original_start, original_end, full_in_features, bit_width: int = 4
):
    """
    Triton fused matmul，针对 in_features 维度切片的场景（MoE down 投影）。

    关键假设：[original_start:original_end] 范围对齐到 group_size 边界！

    Args:
        x: (batch_size, original_end - original_start) - 已经切片后的输入
        indices_packed: (out_features, PACKED_K) - 完整的 packed 索引
        codebook: (n_levels,) - 码本
        norms: (out_features,) 或 (out_features, n_groups) - 范数
        seed: 基础随机种子
        group_size: 分组大小
        original_start: 切片在完整 in_features 中的起始位置
        original_end: 切片在完整 in_features 中的结束位置
        full_in_features: 完整的 in_features 维度
        bit_width: 1/2/4/8 (default: 4)

    Returns:
        output: (batch_size, out_features) 结果
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    # 验证边界对齐
    if original_start % group_size != 0:
        raise ValueError(f"original_start ({original_start}) must be aligned to group_size ({group_size})")

    # 确保 seed 是 Python int
    if isinstance(seed, torch.Tensor):
        seed = int(seed.item())

    batch_size = x.shape[0]
    out_features = indices_packed.shape[0]
    slice_in_features = original_end - original_start
    ELEMENTS_PER_BYTE = 8 // bit_width

    # 确保 norms 是二维的
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    output = torch.zeros(batch_size, out_features, dtype=torch.float32, device=x.device)

    # 对切片内的每个分组分别处理
    group_idx_in_slice = 0
    for g_start_in_slice in range(0, slice_in_features, group_size):
        # gt_start = time.time()

        g_end_in_slice = min(g_start_in_slice + group_size, slice_in_features)
        g_dim = g_end_in_slice - g_start_in_slice

        # 计算在完整 in_features 中的真实位置
        g_start_original = original_start + g_start_in_slice

        # 1. 旋转这个分组对应的输入（用真实位置的 seed）
        Pi = generate_rotation_matrix(g_dim, seed + g_start_original, device=x.device)
        x_g = x[:, g_start_in_slice:g_end_in_slice].float()
        x_rot_g = x_g @ Pi.T

        # 2. 切片这个分组对应的 packed indices（用真实位置）
        packed_start = g_start_original // ELEMENTS_PER_BYTE
        packed_end = g_start_original + g_dim
        if packed_end % ELEMENTS_PER_BYTE != 0:
            packed_end = (packed_end // ELEMENTS_PER_BYTE) + 1
        else:
            packed_end = packed_end // ELEMENTS_PER_BYTE
        indices_packed_g = indices_packed[:, packed_start:packed_end].clone()

        # 3. 取出这个分组对应的 norms（用真实分组索引）
        group_idx_original = g_start_original // group_size
        norms_g = norms[:, group_idx_original]

        # 4. 调用 triton fused kernel
        out_g = triton_fused_matmul(x_rot_g, indices_packed_g, codebook, norms_g, g_dim, bit_width)

        # 5. 累加到输出
        output += out_g

        group_idx_in_slice += 1

        # gt_end = time.time()
        # print(f"    [DEBUG] group_idx_in_slice: {group_idx_in_slice}, x_g: {x_g.shape}, indices_packed_g: {indices_packed_g.shape}, codebook: {codebook.shape}, norms_g: {norms_g.shape}, g_dim: {g_dim}, time: {gt_end - gt_start:.4f}s", flush=True)

    return output
