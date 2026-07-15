#!/usr/bin/env python3
import itertools
"""
简单直接的 Triton 融合 kernel 测试程序。
"""

import argparse
import time
import torch
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant_utils.triton_kernels import triton_fused_matmul, triton_fused_matmul_grouped
from turboquant_utils.codebook import get_codebook
from turboquant_utils.rotation import generate_rotation_matrix
from turboquant_utils.quantize import pack_nbit, unpack_nbit


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def quantize_weight_simple(W, bit_width=4, group_size=None, seed=42):
    """
    简单的量化函数，直接返回 triton_fused_matmul 需要的所有数据。
    """
    M, N = W.shape
    if group_size is None:
        group_size = N

    W = W.float()
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    # 收集每个分组的范数和索引
    all_norms = []
    all_indices = []

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
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
    norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]

    # 打包
    packed = pack_nbit(full_indices, bit_width)

    return {
        "indices_packed": packed,
        "codebook": centroids,
        "norms": norms_out,
        "seed": seed,
        "group_size": group_size,
    }


def dequantize_weight_simple(packed_data, W_shape):
    """简单的反量化函数"""
    indices_packed = packed_data["indices_packed"]
    codebook = packed_data["codebook"]
    norms = packed_data["norms"]
    seed = packed_data["seed"]
    group_size = packed_data["group_size"]

    M, N = W_shape
    device = indices_packed.device

    # 解包索引
    full_indices = unpack_nbit(indices_packed, 4, N)

    # 确保 norms 是二维的
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    # 反量化
    W_approx = torch.zeros((M, N), dtype=torch.float32, device=device)

    group_idx = 0
    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
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


def test_w1(args):
    """正确性测试"""
    print("\n" + "=" * 60)
    print("正确性测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    M, K, N = args.M, args.K, args.N
    group_size = args.group_size

    print(f"\n测试配置:")
    print(f"  w1 shape: {M} x {K} (out_features x in_features)")
    print(f"  x shape: {N} x {K} (batch x in_features)")
    print(f"  group_size: {group_size}")
    print(f"  组数: {K // group_size}")

    # 创建随机权重和输入
    w1 = torch.randn(M, K, dtype=torch.float16, device=device)
    x = torch.randn(N, K, dtype=torch.float16, device=device)

    # Baseline: 原始矩阵乘法（不量化）
    for i in range(3):
        _ = x @ w1.T

    print("\nBaseline: 原始矩阵乘法（不量化）...")
    t0 = time.time()
    out_baseline = x @ w1.T
    t_baseline = time.time() - t0
    print(f"  时间: {t_baseline * 1000:.2f} ms")

    # 量化
    print("\n量化权重 w1 ...")
    packed_data = quantize_weight_simple(
        w1, bit_width=4, group_size=group_size, seed=args.seed
    )

    indices_packed = packed_data["indices_packed"]
    codebook = packed_data["codebook"]
    norms = packed_data["norms"]
    seed_quant = packed_data["seed"]

    print(f"  indices_packed shape: {indices_packed.shape}")
    print(f"  codebook shape: {codebook.shape}")
    print(f"  norms shape: {norms.shape}")

    # 方法1：先反量化再做 GEMM
    print("\n方法1: 先反量化再做 GEMM ...")
    t0 = time.time()
    w1_dequant = dequantize_weight_simple(packed_data, (M, K))
    t_dequant = time.time() - t0

    t0 = time.time()
    out1 = x.float() @ w1_dequant.T
    t_gemm = time.time() - t0

    print(f"  反量化时间: {t_dequant * 1000:.2f} ms")
    print(f"  GEMM 时间: {t_gemm * 1000:.2f} ms")
    print(f"  合计: {(t_dequant + t_gemm) * 1000:.2f} ms")

    # 方法2：triton fused (支持分组)
    print("\n方法2: 使用 triton_fused_matmul (分组) ...")

    print("  预热 kernel ...")
    for _ in range(3):
        _ = triton_fused_matmul_grouped(x, indices_packed, codebook, norms, seed_quant, group_size, K)

    torch.cuda.synchronize()
    t0 = time.time()
    out2 = triton_fused_matmul_grouped(x, indices_packed, codebook, norms, seed_quant, group_size, K)
    torch.cuda.synchronize()
    t_fused = time.time() - t0

    print(f"  Fused kernel (分组) 时间: {t_fused * 1000:.2f} ms")

    # 比较结果（对比 baseline 和两种量化方法）
    print("\n结果对比:")
    print("  1. 先反量化再GEMM vs Baseline:")
    out_baseline_float = out_baseline.float()
    out1_float = out1.float()
    abs_diff1 = torch.abs(out_baseline_float - out1_float)
    print(f"    最大绝对误差: {abs_diff1.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff1.mean().item():.6f}")

    print("\n  2. Triton Fused vs Baseline:")
    out2_float = out2.float()
    abs_diff2 = torch.abs(out_baseline_float - out2_float)
    print(f"    最大绝对误差: {abs_diff2.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff2.mean().item():.6f}")

    print("\n  3. 先反量化再GEMM vs Triton Fused:")
    abs_diff3 = torch.abs(out1_float - out2_float)
    print(f"    最大绝对误差: {abs_diff3.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff3.mean().item():.6f}")

    atol = 1e-2
    rtol = 1e-2
    is_close1 = torch.allclose(out1_float, out_baseline_float, atol=atol, rtol=rtol)
    is_close2 = torch.allclose(out2_float, out_baseline_float, atol=atol, rtol=rtol)
    is_close3 = torch.allclose(out1_float, out2_float, atol=atol, rtol=rtol)
    print(f"\n  近似相等:")
    print(f"    先反量化再GEMM ≈ Baseline: {is_close1}")
    print(f"    Triton Fused ≈ Baseline: {is_close2}")
    print(f"    先反量化再GEMM ≈ Triton Fused: {is_close3}")

    if not is_close3:
        print("\n  样本对比:")
        for i in itertools.chain(range(3), range(N-3, N)):
            for j in itertools.chain(range(3), range(M-3, M)):
                 print(f"    baseline[{i},{j}] = {out_baseline_float[i,j]:.6f}, "
                      f"dequant+gemm[{i},{j}] = {out1_float[i,j]:.6f}, "
                      f"fused[{i},{j}] = {out2_float[i,j]:.6f}")

    return out_baseline_float, out1_float, out2_float


def test_w1_w2(args):
    """两个连续矩阵乘法的测试：w1(1024x512) -> w2(512x1024)"""
    print("\n" + "=" * 60)
    print("两个连续矩阵乘法测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 按照需求设置尺寸
    # w1: 1024 * 512 (out_features x in_features)
    # w2: 512 * 1024 (out_features x in_features)
    # x: 16 * 1024
    M1, K1 = 1024, 512  # w1 shape
    M2, K2 = 512, 1024  # w2 shape
    batch_size = 16
    group_size = args.group_size

    print(f"\n测试配置:")
    print(f"  w1 shape: {M1} x {K1}")
    print(f"  w2 shape: {M2} x {K2}")
    print(f"  x shape: {batch_size} x {K2} (batch x in_features for w2)")
    print(f"  group_size: {group_size}")
    print(f"  组数 (w1): {K1 // group_size}")
    print(f"  组数 (w2): {K2 // group_size}")

    # 创建随机权重和输入
    w1 = torch.randn(M1, K1, dtype=torch.float16, device=device)
    w2 = torch.randn(M2, K2, dtype=torch.float16, device=device)
    x = torch.randn(batch_size, K2, dtype=torch.float16, device=device)

    # Baseline: 原始矩阵乘法（不量化）
    print("\nBaseline: 原始矩阵乘法（不量化）...")
    t0 = time.time()
    y1_baseline = x.float() @ w2.float().T  # 16 x 512
    y2_baseline = y1_baseline @ w1.float().T  # 16 x 1024
    t_baseline = time.time() - t0
    print(f"  时间: {t_baseline * 1000:.2f} ms")

    # 量化 w1 和 w2
    print("\n量化权重 w1 ...")
    packed_data_w1 = quantize_weight_simple(
        w1, bit_width=4, group_size=group_size, seed=args.seed
    )
    print("量化权重 w2 ...")
    packed_data_w2 = quantize_weight_simple(
        w2, bit_width=4, group_size=group_size, seed=args.seed + 1000
    )

    indices_packed_w1 = packed_data_w1["indices_packed"]
    codebook_w1 = packed_data_w1["codebook"]
    norms_w1 = packed_data_w1["norms"]
    seed_w1 = packed_data_w1["seed"]

    indices_packed_w2 = packed_data_w2["indices_packed"]
    codebook_w2 = packed_data_w2["codebook"]
    norms_w2 = packed_data_w2["norms"]
    seed_w2 = packed_data_w2["seed"]

    print(f"  w1 indices_packed shape: {indices_packed_w1.shape}")
    print(f"  w2 indices_packed shape: {indices_packed_w2.shape}")

    # 方法1：先反量化再做 GEMM
    print("\n方法1: 先反量化再做 GEMM ...")
    t0 = time.time()
    w1_dequant = dequantize_weight_simple(packed_data_w1, (M1, K1))
    w2_dequant = dequantize_weight_simple(packed_data_w2, (M2, K2))
    t_dequant = time.time() - t0

    t0 = time.time()
    y1_dequant = x.float() @ w2_dequant.T
    y2_dequant = y1_dequant @ w1_dequant.T
    t_gemm = time.time() - t0

    print(f"  反量化时间: {t_dequant * 1000:.2f} ms")
    print(f"  GEMM 时间: {t_gemm * 1000:.2f} ms")
    print(f"  合计: {(t_dequant + t_gemm) * 1000:.2f} ms")

    # 方法2：triton fused (支持分组)
    print("\n方法2: 使用 triton_fused_matmul (分组) ...")

    print("  预热 kernel ...")
    for _ in range(3):
        _ = triton_fused_matmul_grouped(x, indices_packed_w2, codebook_w2, norms_w2, seed_w2, group_size, K2)
        _ = triton_fused_matmul_grouped(y1_baseline.half(), indices_packed_w1, codebook_w1, norms_w1, seed_w1, group_size, K1)

    torch.cuda.synchronize()
    t0 = time.time()
    y1_fused = triton_fused_matmul_grouped(x, indices_packed_w2, codebook_w2, norms_w2, seed_w2, group_size, K2)
    y2_fused = triton_fused_matmul_grouped(y1_fused.half(), indices_packed_w1, codebook_w1, norms_w1, seed_w1, group_size, K1)
    torch.cuda.synchronize()
    t_fused = time.time() - t0

    print(f"  Fused kernel (分组) 时间: {t_fused * 1000:.2f} ms")

    # 比较结果
    print("\n结果对比 (y2 最终输出 16x1024):")
    print("  1. 先反量化再GEMM vs Baseline:")
    y2_baseline_float = y2_baseline.float()
    y2_dequant_float = y2_dequant.float()
    abs_diff1 = torch.abs(y2_baseline_float - y2_dequant_float)
    print(f"    最大绝对误差: {abs_diff1.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff1.mean().item():.6f}")

    print("\n  2. Triton Fused vs Baseline:")
    y2_fused_float = y2_fused.float()
    abs_diff2 = torch.abs(y2_baseline_float - y2_fused_float)
    print(f"    最大绝对误差: {abs_diff2.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff2.mean().item():.6f}")

    print("\n  3. 先反量化再GEMM vs Triton Fused:")
    abs_diff3 = torch.abs(y2_dequant_float - y2_fused_float)
    print(f"    最大绝对误差: {abs_diff3.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff3.mean().item():.6f}")

    atol = 1e-2
    rtol = 1e-2
    is_close1 = torch.allclose(y2_dequant_float, y2_baseline_float, atol=atol, rtol=rtol)
    is_close2 = torch.allclose(y2_fused_float, y2_baseline_float, atol=atol, rtol=rtol)
    is_close3 = torch.allclose(y2_dequant_float, y2_fused_float, atol=atol, rtol=rtol)
    print(f"\n  近似相等:")
    print(f"    先反量化再GEMM ≈ Baseline: {is_close1}")
    print(f"    Triton Fused ≈ Baseline: {is_close2}")
    print(f"    先反量化再GEMM ≈ Triton Fused: {is_close3}")

    if not is_close3:
        print("\n  样本对比 (y2):")
        for i in itertools.chain(range(3), range(batch_size-3, batch_size)):
            for j in itertools.chain(range(3), range(M1-3, M1)):
                 print(f"    baseline[{i},{j}] = {y2_baseline_float[i,j]:.6f}, "
                      f"dequant+gemm[{i,j}] = {y2_dequant_float[i,j]:.6f}, "
                      f"fused[{i,j}] = {y2_fused_float[i,j]:.6f}")

    print("\n中间结果 y1 (16x512) 对比:")
    print("  Triton Fused y1 vs Baseline y1:")
    y1_fused_float = y1_fused.float()
    y1_baseline_float = y1_baseline.float()
    abs_diff_y1 = torch.abs(y1_baseline_float - y1_fused_float)
    print(f"    最大绝对误差: {abs_diff_y1.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff_y1.mean().item():.6f}")

    return y2_baseline_float, y2_dequant_float, y2_fused_float


def main():
    parser = argparse.ArgumentParser(description="测试 Triton 融合反量化+GEMM kernel")

    parser.add_argument("--M", type=int, default=8192, help="w1 shape[0]")
    parser.add_argument("--K", type=int, default=1024, help="w1 shape[1]")
    parser.add_argument("--N", type=int, default=16, help="batch size")
    parser.add_argument("--group_size", type=int, default=128, help="分组大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    setup_seed(args.seed)

    print("\nTriton 融合反量化+GEMM 测试程序")

    test_w1(args)
    test_w1_w2(args)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
