#!/usr/bin/env python3
"""
混合精度 Triton 融合 kernel 测试程序。

测试三个算子：
1. triton_fused_matmul_grouped：单独的 linear 层
2. triton_fused_matmul_grouped_slice_rows：MoE up_gate (切 rows，混精)
3. triton_fused_matmul_grouped_slice_in_features：MoE down (切 in_features，混精)
"""

import argparse
import time
import torch
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant_utils.triton_kernels import (
    triton_fused_matmul_grouped,
    triton_fused_matmul_grouped_slice_rows,
    triton_fused_matmul_grouped_slice_in_features
)
from turboquant_utils.codebook import get_codebook
from turboquant_utils.rotation import generate_rotation_matrix
from turboquant_utils.quantize import pack_nbit, unpack_nbit


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 显式设置 TF32，确保 PyTorch 和 Triton 行为一致
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def quantize_weight_simple(W, bit_width=4, group_size=None, seed=42):
    """
    简单的量化函数，直接返回 triton_fused_matmul 需要的所有数据。
    支持 1/2/4/8 bit。
    """
    M, K = W.shape
    if group_size is None:
        group_size = K

    W = W.float()
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    # 收集每个分组的范数和索引
    all_norms = []
    all_indices = []

    for g_start in range(0, K, group_size):
        g_end = min(g_start + group_size, K)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        # 行归一化
        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        # 旋转
        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=W.device)
        Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        # 量化
        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_indices) > 1 else all_indices[0]

    # 打包 (使用通用 nbit 打包函数)
    packed = pack_nbit(full_indices, bit_width)

    return {
        "indices_packed": packed,
        "codebook": centroids,
        "norms": norms_out,
        "seed": seed,
        "group_size": group_size,
        "bit_width": bit_width,
    }


def dequantize_weight_simple(packed_data, W_shape):
    """简单的反量化函数"""
    indices_packed = packed_data["indices_packed"]
    codebook = packed_data["codebook"]
    norms = packed_data["norms"]
    seed = packed_data["seed"]
    group_size = packed_data["group_size"]
    bit_width = packed_data["bit_width"]

    M, K = W_shape
    device = indices_packed.device

    # 解包索引
    full_indices = unpack_nbit(indices_packed, bit_width, K)

    # 确保 norms 是二维的
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    # 反量化
    W_approx = torch.zeros((M, K), dtype=torch.float32, device=device)

    group_idx = 0
    for g_start in range(0, K, group_size):
        g_end = min(g_start + group_size, K)
        g_dim = g_end - g_start

        indices_g = full_indices[:, g_start:g_end]
        norms_g = norms[:, group_idx].unsqueeze(1)
        group_idx += 1

        Y_quant_scaled = codebook[indices_g]
        scale = math.sqrt(g_dim)
        Y_unscaled = Y_quant_scaled / scale

        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=device)
        W_g_approx = Y_unscaled @ Pi

        W_approx[:, g_start:g_end] = W_g_approx * norms_g

    return W_approx


def get_slice_boundaries(K, proportions=None):
    """
    计算切片的边界位置
    """
    if proportions is None:
        proportions = [2, 1, 1]

    total = sum(proportions)
    boundaries = []
    current_start = 0
    for i, p in enumerate(proportions):
        size = K * p // total
        # 最后一片取剩余所有
        if current_start + size > K or i == len(proportions) - 1:
            size = K - current_start
        boundaries.append((current_start, current_start + size))
        current_start += size

    return boundaries


def test_linear(args):
    """场景1：单独的 linear 层，测试 triton_fused_matmul_grouped"""
    print("\n" + "=" * 70)
    print("场景1: 单独 Linear - triton_fused_matmul_grouped")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.out_features
    K = args.in_features
    B = args.batch_size
    group_size = args.group_size

    # 确保 K 对齐到 group_size 边界
    K = ((K + group_size - 1) // group_size) * group_size

    print(f"\n测试配置:")
    print(f"  Linear shape: {H} x {K} (out_features x in_features)")
    print(f"  Input x shape: {B} x {K}")
    print(f"  group_size: {group_size}")

    # 创建随机权重和输入
    torch.manual_seed(args.seed)
    W = torch.randn(H, K, dtype=torch.float16, device=device)
    x = torch.randn(B, K, dtype=torch.float16, device=device)

    # 量化权重
    packed_data = quantize_weight_simple(W, bit_width=4, group_size=group_size, seed=args.seed)

    # ===== Warmup =====
    print("\nWarmup...")
    for _ in range(5):
        _ = x @ W.T
        w_dequant = dequantize_weight_simple(packed_data, W.shape)
        _ = x.float() @ w_dequant.T
        _ = triton_fused_matmul_grouped(
            x, packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], K, bit_width=4
        )

    # 1. FP16 Baseline
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_fp16 = x @ W.T
    torch.cuda.synchronize()
    t_fp16 = (time.time() - t0) / 10

    # 2. 反量化 + GEMM
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        w_dequant = dequantize_weight_simple(packed_data, W.shape)
        out_dequant = x.float() @ w_dequant.T
    torch.cuda.synchronize()
    t_dequant = (time.time() - t0) / 10

    # 3. Triton Fused
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_triton = triton_fused_matmul_grouped(
            x, packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], K, bit_width=4
        )
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / 10

    # 计算误差
    out_fp16_float = out_fp16.float()
    out_dequant_float = out_dequant.float()
    out_triton_float = out_triton.float()

    abs_diff_dequant = torch.abs(out_fp16_float - out_dequant_float)
    abs_diff_triton = torch.abs(out_fp16_float - out_triton_float)
    abs_diff_between = torch.abs(out_dequant_float - out_triton_float)

    return {
        "t_fp16": t_fp16,
        "t_dequant": t_dequant,
        "t_triton": t_triton,
        "err_dequant_max": abs_diff_dequant.max().item(),
        "err_dequant_mean": abs_diff_dequant.mean().item(),
        "err_triton_max": abs_diff_triton.max().item(),
        "err_triton_mean": abs_diff_triton.mean().item(),
        "err_between_max": abs_diff_between.max().item(),
        "err_between_mean": abs_diff_between.mean().item(),
    }


def test_moe_up(args):
    """场景2.1：MoE up_gate，测试 triton_fused_matmul_grouped_slice_rows"""
    print("\n" + "=" * 70)
    print("场景2.1: MoE up_gate - triton_fused_matmul_grouped_slice_rows")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    K = args.in_features   # up输出维度
    D = K * 2             # up输入维度
    B = args.batch_size
    group_size = args.group_size

    # 确保对齐到 group_size 边界
    K = ((K + group_size - 1) // group_size) * group_size
    D = ((D + group_size - 1) // group_size) * group_size

    print(f"\n测试配置:")
    print(f"  Up_gate shape: {K} x {D}")
    print(f"  Input x shape: {B} x {D}")
    print(f"  group_size: {group_size}")

    # 创建随机权重和输入
    torch.manual_seed(args.seed)
    W_up = torch.randn(K, D, dtype=torch.float16, device=device)
    x = torch.randn(B, D, dtype=torch.float16, device=device)

    # 切片配置（混精 4:2:1）
    slice_boundaries_up = get_slice_boundaries(D)
    bit_widths = [4, 2, 1]
    W_up_slices = [W_up[:, start:end] for start, end in slice_boundaries_up]
    x_slices_up = [x[:, start:end] for start, end in slice_boundaries_up]

    # 分别量化
    packed_data_list_up = []
    for i, (w_slice, bw, (start, end)) in enumerate(zip(W_up_slices, bit_widths, slice_boundaries_up)):
        packed_data = quantize_weight_simple(
            w_slice, bit_width=bw, group_size=group_size, seed=args.seed + start
        )
        packed_data_list_up.append(packed_data)

    # ===== Warmup =====
    print("\nWarmup...")
    # warmup fp16
    for _ in range(5):
        _ = x @ W_up.T
    # warmup dequant
    for _ in range(5):
        out_up_dequant = torch.zeros(B, K, dtype=torch.float32, device=device)
        for i, (packed_data, w_slice, x_slice) in enumerate(zip(packed_data_list_up, W_up_slices, x_slices_up)):
            w_dequant = dequantize_weight_simple(packed_data, w_slice.shape)
            out_slice = x_slice.float() @ w_dequant.T
            out_up_dequant += out_slice
    # warmup triton
    for _ in range(5):
        out_up_triton_rows = torch.zeros(B, K, dtype=torch.float32, device=device)
        rows_mid = K // 2
        for row_start, row_end in [(0, rows_mid), (rows_mid, K)]:
            out_up_triton_part = torch.zeros(B, row_end - row_start, dtype=torch.float32, device=device)
            for i, (packed_data, x_slice) in enumerate(zip(packed_data_list_up, x_slices_up)):
                slice_out = triton_fused_matmul_grouped_slice_rows(
                    x_slice, packed_data["indices_packed"], packed_data["codebook"],
                    packed_data["norms"], packed_data["seed"], packed_data["group_size"],
                    x_slice.shape[1], row_start, row_end, bit_width=packed_data["bit_width"]
                )
                out_up_triton_part += slice_out
            out_up_triton_rows[:, row_start:row_end] = out_up_triton_part

    # 1. FP16 Baseline
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_up_fp16 = x @ W_up.T
    torch.cuda.synchronize()
    t_fp16 = (time.time() - t0) / 10

    # 2. 反量化 + GEMM
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_up_dequant = torch.zeros(B, K, dtype=torch.float32, device=device)
        for i, (packed_data, w_slice, x_slice) in enumerate(zip(packed_data_list_up, W_up_slices, x_slices_up)):
            w_dequant = dequantize_weight_simple(packed_data, w_slice.shape)
            out_slice = x_slice.float() @ w_dequant.T
            out_up_dequant += out_slice
    torch.cuda.synchronize()
    t_dequant = (time.time() - t0) / 10

    # 3. Triton Fused - slice_rows
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_up_triton_rows = torch.zeros(B, K, dtype=torch.float32, device=device)
        rows_mid = K // 2
        for row_start, row_end in [(0, rows_mid), (rows_mid, K)]:
            out_up_triton_part = torch.zeros(B, row_end - row_start, dtype=torch.float32, device=device)
            for i, (packed_data, x_slice) in enumerate(zip(packed_data_list_up, x_slices_up)):
                slice_out = triton_fused_matmul_grouped_slice_rows(
                    x_slice, packed_data["indices_packed"], packed_data["codebook"],
                    packed_data["norms"], packed_data["seed"], packed_data["group_size"],
                    x_slice.shape[1], row_start, row_end, bit_width=packed_data["bit_width"]
                )
                out_up_triton_part += slice_out
            out_up_triton_rows[:, row_start:row_end] = out_up_triton_part
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / 10

    # 计算误差
    out_up_fp16_float = out_up_fp16.float()
    out_up_dequant_float = out_up_dequant.float()
    out_up_triton_rows_float = out_up_triton_rows.float()

    abs_diff_up_dequant = torch.abs(out_up_fp16_float - out_up_dequant_float)
    abs_diff_up_triton = torch.abs(out_up_fp16_float - out_up_triton_rows_float)
    abs_diff_up_between = torch.abs(out_up_dequant_float - out_up_triton_rows_float)

    return {
        "t_fp16": t_fp16,
        "t_dequant": t_dequant,
        "t_triton": t_triton,
        "err_dequant_max": abs_diff_up_dequant.max().item(),
        "err_dequant_mean": abs_diff_up_dequant.mean().item(),
        "err_triton_max": abs_diff_up_triton.max().item(),
        "err_triton_mean": abs_diff_up_triton.mean().item(),
        "err_between_max": abs_diff_up_between.max().item(),
        "err_between_mean": abs_diff_up_between.mean().item(),
    }


def test_moe_down(args):
    """场景2.2：MoE down，测试 triton_fused_matmul_grouped_slice_in_features"""
    print("\n" + "=" * 70)
    print("场景2.2: MoE down - triton_fused_matmul_grouped_slice_in_features")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.out_features  # down输出维度
    K = args.in_features   # down输入维度
    B = args.batch_size
    group_size = args.group_size

    # 确保对齐到 group_size 边界
    K = ((K + group_size - 1) // group_size) * group_size

    print(f"\n测试配置:")
    print(f"  Down shape: {H} x {K}")
    print(f"  Input x shape: {B} x {K}")
    print(f"  group_size: {group_size}")

    # 创建随机权重和输入
    torch.manual_seed(args.seed)
    W_down = torch.randn(H, K, dtype=torch.float16, device=device)
    x = torch.randn(B, K, dtype=torch.float16, device=device)

    # 切片配置（混精 4:2:1）
    slice_boundaries_down = get_slice_boundaries(K)
    bit_widths = [4, 2, 1]
    W_down_slices = [W_down[:, start:end] for start, end in slice_boundaries_down]
    x_slices_down = [x[:, start:end] for start, end in slice_boundaries_down]

    # 分别量化
    packed_data_list_down = []
    for i, (w_slice, bw, (start, end)) in enumerate(zip(W_down_slices, bit_widths, slice_boundaries_down)):
        packed_data = quantize_weight_simple(
            w_slice, bit_width=bw, group_size=group_size, seed=args.seed + start
        )
        packed_data_list_down.append(packed_data)

    # ===== Warmup =====
    print("\nWarmup...")
    # warmup fp16
    for _ in range(5):
        _ = x @ W_down.T
    # warmup dequant
    for _ in range(5):
        out_down_dequant = torch.zeros(B, H, dtype=torch.float32, device=device)
        for i, (packed_data, w_slice, x_slice) in enumerate(zip(packed_data_list_down, W_down_slices, x_slices_down)):
            w_dequant = dequantize_weight_simple(packed_data, w_slice.shape)
            out_slice = x_slice.float() @ w_dequant.T
            out_down_dequant += out_slice
    # warmup triton
    for _ in range(5):
        out_down_triton_in = torch.zeros(B, H, dtype=torch.float32, device=device)
        for i, (packed_data, x_slice) in enumerate(zip(packed_data_list_down, x_slices_down)):
            slice_out = triton_fused_matmul_grouped_slice_in_features(
                x_slice, packed_data["indices_packed"], packed_data["codebook"],
                packed_data["norms"], packed_data["seed"], packed_data["group_size"],
                0, x_slice.shape[1], x_slice.shape[1], bit_width=packed_data["bit_width"]
            )
            out_down_triton_in += slice_out

    # 1. FP16 Baseline
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_down_fp16 = x @ W_down.T
    torch.cuda.synchronize()
    t_fp16 = (time.time() - t0) / 10

    # 2. 反量化 + GEMM
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_down_dequant = torch.zeros(B, H, dtype=torch.float32, device=device)
        for i, (packed_data, w_slice, x_slice) in enumerate(zip(packed_data_list_down, W_down_slices, x_slices_down)):
            w_dequant = dequantize_weight_simple(packed_data, w_slice.shape)
            out_slice = x_slice.float() @ w_dequant.T
            out_down_dequant += out_slice
    torch.cuda.synchronize()
    t_dequant = (time.time() - t0) / 10

    # 3. Triton Fused - slice_in_features
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_down_triton_in = torch.zeros(B, H, dtype=torch.float32, device=device)
        for i, (packed_data, x_slice) in enumerate(zip(packed_data_list_down, x_slices_down)):
            slice_out = triton_fused_matmul_grouped_slice_in_features(
                x_slice, packed_data["indices_packed"], packed_data["codebook"],
                packed_data["norms"], packed_data["seed"], packed_data["group_size"],
                0, x_slice.shape[1], x_slice.shape[1], bit_width=packed_data["bit_width"]
            )
            out_down_triton_in += slice_out
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / 10

    # 计算误差
    out_down_fp16_float = out_down_fp16.float()
    out_down_dequant_float = out_down_dequant.float()
    out_down_triton_in_float = out_down_triton_in.float()

    abs_diff_down_dequant = torch.abs(out_down_fp16_float - out_down_dequant_float)
    abs_diff_down_triton = torch.abs(out_down_fp16_float - out_down_triton_in_float)
    abs_diff_down_between = torch.abs(out_down_dequant_float - out_down_triton_in_float)

    return {
        "t_fp16": t_fp16,
        "t_dequant": t_dequant,
        "t_triton": t_triton,
        "err_dequant_max": abs_diff_down_dequant.max().item(),
        "err_dequant_mean": abs_diff_down_dequant.mean().item(),
        "err_triton_max": abs_diff_down_triton.max().item(),
        "err_triton_mean": abs_diff_down_triton.mean().item(),
        "err_between_max": abs_diff_down_between.max().item(),
        "err_between_mean": abs_diff_down_between.mean().item(),
    }

def print_summary(linear_result, moe_up_result, moe_down_result):
    """打印最终汇总"""
    print("\n" + "=" * 70)
    print("最终汇总")
    print("=" * 70)

    print("\n1. 单独 Linear (triton_fused_matmul_grouped):")
    print(f"  FP16:             {linear_result['t_fp16'] * 1000:8.2f} ms")
    print(f"  反量化+GEMM:      {linear_result['t_dequant'] * 1000:8.2f} ms")
    print(f"  Triton:           {linear_result['t_triton'] * 1000:8.2f} ms")
    print(f"\n  误差对比:")
    print(f"    Triton vs FP16: max={linear_result['err_triton_max']:.6f}, mean={linear_result['err_triton_mean']:.6f}")
    print(f"    反量化+GEMM vs Triton: max={linear_result['err_between_max']:.6f}, mean={linear_result['err_between_mean']:.6f}")

    print("\n2.1 MoE up_gate (triton_fused_matmul_grouped_slice_rows):")
    print(f"  FP16:             {moe_up_result['t_fp16'] * 1000:8.2f} ms")
    print(f"  反量化+GEMM:      {moe_up_result['t_dequant'] * 1000:8.2f} ms")
    print(f"  Triton:           {moe_up_result['t_triton'] * 1000:8.2f} ms")
    print(f"\n  误差对比:")
    print(f"    Triton vs FP16: max={moe_up_result['err_triton_max']:.6f}, mean={moe_up_result['err_triton_mean']:.6f}")
    print(f"    反量化+GEMM vs Triton: max={moe_up_result['err_between_max']:.6f}, mean={moe_up_result['err_between_mean']:.6f}")

    print("\n2.2 MoE down (triton_fused_matmul_grouped_slice_in_features):")
    print(f"  FP16:             {moe_down_result['t_fp16'] * 1000:8.2f} ms")
    print(f"  反量化+GEMM:      {moe_down_result['t_dequant'] * 1000:8.2f} ms")
    print(f"  Triton:           {moe_down_result['t_triton'] * 1000:8.2f} ms")
    print(f"\n  误差对比:")
    print(f"    Triton vs FP16: max={moe_down_result['err_triton_max']:.6f}, mean={moe_down_result['err_triton_mean']:.6f}")
    print(f"    反量化+GEMM vs Triton: max={moe_down_result['err_between_max']:.6f}, mean={moe_down_result['err_between_mean']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="混合精度 MoE 测试")

    parser.add_argument("--out_features", type=int, default=1024, help="Down 输出特征维度 H")
    parser.add_argument("--in_features", type=int, default=2048, help="Down 输入/Up 输出特征维度 K")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size B")
    parser.add_argument("--group_size", type=int, default=128, help="分组大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    setup_seed(args.seed)

    linear_result = test_linear(args)
    moe_up_result = test_moe_up(args)
    moe_down_result = test_moe_down(args)

    print_summary(linear_result, moe_up_result, moe_down_result)

if __name__ == "__main__":
    main()
