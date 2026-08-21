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

from .rotation import generate_rotation_matrix


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

    # Pre-compute all rotations first
    # Step 2 (FP16 链路改造): 旋转计算与输入同 dtype（fp16/bf16）
    # - 旋转矩阵 Pi 从 fp32 转到 x.dtype（正交矩阵元素在 [-1,1]，低精度足够）
    # - 输入 x 保持原 dtype，不再 .float() 升 fp32
    # - 目的: tl.dot 两边 dtype 一致，走对应精度的 Tensor Core
    num_groups = (in_features + group_size - 1) // group_size
    x_rot_list = []
    for group_idx in range(num_groups):
        g_start = group_idx * group_size
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start

        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=x.device).to(x.dtype)
        x_rot_g = x[:, g_start:g_end] @ Pi.T
        x_rot_list.append(x_rot_g)

    output = torch.zeros(batch_size, out_features, dtype=x.dtype, device=x.device)  # Step 1: output 与输入同 dtype

    group_idx = 0
    for g_start in range(0, in_features, group_size):
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start

        x_rot_g = x_rot_list[group_idx]

        packed_start = g_start // ELEMENTS_PER_BYTE
        g_end = g_start + g_dim  # in_features 结束位置
        packed_end = g_end // ELEMENTS_PER_BYTE
        if g_end % ELEMENTS_PER_BYTE != 0:
            packed_end += 1

        norms_g = norms[:, group_idx]

        # 去除列切片 clone：直接传整张 indices_packed，由内核按 col_start 偏移寻址
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

    # Pre-compute all rotations first
    # Step 2: 旋转计算改为 fp16（同 triton_fused_matmul_grouped）
    num_groups = (in_features + group_size - 1) // group_size
    x_rot_list = []
    for group_idx in range(num_groups):
        g_start = group_idx * group_size
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start

        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=x.device).to(x.dtype)
        x_rot_g = x[:, g_start:g_end] @ Pi.T
        x_rot_list.append(x_rot_g)

    output = torch.zeros(batch_size, slice_out_features, dtype=x.dtype, device=x.device)  # 输出与输入同 dtype

    group_idx = 0
    for g_start in range(0, in_features, group_size):
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start

        x_rot_g = x_rot_list[group_idx]

        packed_start = g_start // ELEMENTS_PER_BYTE
        packed_end = g_end // ELEMENTS_PER_BYTE
        if g_end % ELEMENTS_PER_BYTE != 0:
            packed_end += 1

        norms_g = norms_slice[:, group_idx]

        # 去除列切片 clone：传行切片后的整张 indices_packed_slice，由内核按 col_start 偏移寻址
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

    # Pre-compute all rotations first
    # Step 2: 旋转计算改为 fp16（同 triton_fused_matmul_grouped）
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

    output = torch.zeros(batch_size, out_features, dtype=x.dtype, device=x.device)  # 输出与输入同 dtype

    group_idx_in_slice = 0
    for g_start_in_slice in range(0, slice_in_features, group_size):
        g_end_in_slice = min(g_start_in_slice + group_size, slice_in_features)
        g_dim = g_end_in_slice - g_start_in_slice

        g_start_original = original_start + g_start_in_slice

        x_rot_g = x_rot_list[group_idx_in_slice]

        packed_start = g_start_original // ELEMENTS_PER_BYTE
        g_end_original = g_start_original + g_dim  # in_features 结束位置
        packed_end = g_end_original // ELEMENTS_PER_BYTE
        if g_end_original % ELEMENTS_PER_BYTE != 0:
            packed_end += 1

        group_idx_original = g_start_original // group_size
        norms_g = norms[:, group_idx_original]

        # 去除列切片 clone：传整张 indices_packed，由内核按 col_start 偏移寻址
        out_g = triton_fused_matmul(x_rot_g, indices_packed, codebook, norms_g, g_dim, bit_width, col_start=packed_start)

        output += out_g

        group_idx_in_slice += 1

    return output
