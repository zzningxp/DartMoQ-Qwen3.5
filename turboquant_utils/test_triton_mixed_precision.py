#!/usr/bin/env python3
import itertools
"""
混合精度 Triton 融合 kernel 测试程序，针对 MoE down 投影的场景。

测试内容：
1. Down 权重 shape: (out_features=H, in_features=sub_set_neurons)
2. 在 in_features (K) 维度按 2:1:1 切分成三份：
   - W1: (H, K/2)
   - W2: (H, K/4)
   - W3: (H, K/4)
3. 分别量化每份切片，支持混合精度
4. 输出直接相加得到最终结果

测试场景：
- 场景1：所有切片都是 4bit（完整量化+新函数测试）
- 场景2：切片按 4bit:2bit:1bit 混合精度（分别量化）
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


def manual_fused_matmul_grouped(x, packed_data, in_features, bit_width):
    """
    手动版本：和 triton_fused_matmul_grouped 完全一样的数学步骤，
    但用 PyTorch 操作，方便 debug
    """
    indices_packed = packed_data["indices_packed"]
    codebook = packed_data["codebook"]
    norms = packed_data["norms"]
    seed = packed_data["seed"]
    group_size = packed_data["group_size"]

    batch_size = x.shape[0]
    out_features = indices_packed.shape[0]
    ELEMENTS_PER_BYTE = 8 // bit_width

    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    # 先 unpack indices
    full_indices = unpack_nbit(indices_packed, bit_width, in_features)

    output = torch.zeros(batch_size, out_features, dtype=torch.float32, device=x.device)

    num_groups = (in_features + group_size - 1) // group_size
    for group_idx in range(num_groups):
        g_start = group_idx * group_size
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start

        # 旋转 x
        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=x.device)
        x_g = x[:, g_start:g_end].float()
        x_rot = x_g @ Pi.T

        # 取出这个 group 的 indices，还原 weight
        indices_g = full_indices[:, g_start:g_end]
        w_quant_scaled = codebook[indices_g]  # 这是乘了 scale 的
        scale = math.sqrt(g_dim)
        w_quant = w_quant_scaled / scale

        # 逆旋转
        w_g = w_quant @ Pi

        # 乘 norms
        norms_g = norms[:, group_idx].unsqueeze(1)
        w_g_scaled = w_g * norms_g

        # 矩阵乘法
        out_g = x_g @ w_g_scaled.T

        output += out_g

    return output


def manual_fused_matmul_grouped_math_order(x, packed_data, in_features, bit_width):
    """
    手动版本：完全按照 Triton 的数学顺序计算：(x_rot @ w_quant.T) * (norms / scale)
    """
    indices_packed = packed_data["indices_packed"]
    codebook = packed_data["codebook"]
    norms = packed_data["norms"]
    seed = packed_data["seed"]
    group_size = packed_data["group_size"]

    batch_size = x.shape[0]
    out_features = indices_packed.shape[0]
    ELEMENTS_PER_BYTE = 8 // bit_width

    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    # 先 unpack indices
    full_indices = unpack_nbit(indices_packed, bit_width, in_features)

    output = torch.zeros(batch_size, out_features, dtype=torch.float32, device=x.device)

    num_groups = (in_features + group_size - 1) // group_size
    for group_idx in range(num_groups):
        g_start = group_idx * group_size
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start

        # 旋转 x (和 Triton 一样)
        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=x.device)
        x_g = x[:, g_start:g_end].float()
        x_rot = x_g @ Pi.T

        # 取出这个 group 的 indices，得到量化后的 weight (没有逆旋转，没有除以 scale)
        indices_g = full_indices[:, g_start:g_end]
        w_quant_scaled = codebook[indices_g]  # 这是 Y_scaled = Y * sqrt(g_dim)

        # 矩阵乘法：x_rot @ w_quant_scaled.T (和 Triton 一样)
        out_g = x_rot @ w_quant_scaled.T

        # 计算 scale 因子
        scale = math.sqrt(g_dim)
        norms_g = norms[:, group_idx]
        norms_scaled = norms_g / scale

        # 应用 norms_scaled
        out_g = out_g * norms_scaled[None, :]

        output += out_g

    return output


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 显式设置 TF32，确保 PyTorch 和 Triton 行为一致
    torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch也用TF32，匹配Triton
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

    Args:
        K: 完整的 in_features
        proportions: 切分比例，默认 [2, 1, 1]

    Returns:
        list of (start, end) tuples
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


def test_mixed_precision(args):
    """
    混合精度测试：Down 权重在 in_features 维度切分，分别量化，结果相加
    """
    print("\n" + "=" * 70)
    print("混合精度 MoE Down 测试")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    H = args.out_features
    K = args.in_features
    B = args.batch_size
    group_size = args.group_size

    # 确保 K 对齐到 group_size 边界
    K = ((K + group_size - 1) // group_size) * group_size

    print(f"\n测试配置:")
    print(f"  Down shape: {H} x {K} (out_features x in_features)")
    print(f"  Input x shape: {B} x {K}")
    print(f"  group_size: {group_size}")
    print(f"  切分比例: 2:1:1 (in_features维度)")

    # 创建随机权重和输入
    W_down = torch.randn(H, K, dtype=torch.float16, device=device)
    x = torch.randn(B, K, dtype=torch.float16, device=device)

    # 计算切片边界
    slice_boundaries = get_slice_boundaries(K)
    W_slices = [W_down[:, start:end] for start, end in slice_boundaries]
    x_slices = [x[:, start:end] for start, end in slice_boundaries]

    print(f"\n权重切片信息:")
    for i, (start, end) in enumerate(slice_boundaries):
        print(f"  W{i+1}: in_features [{start}:{end}], shape {W_slices[i].shape}")

    # Baseline: 原始矩阵乘法
    print("\n" + "-" * 70)
    print("Baseline: 原始矩阵乘法（不量化）")
    print("-" * 70)

    # # 预热
    # for _ in range(3):
    #     _ = x @ W_down.T

    torch.cuda.synchronize()
    t0 = time.time()
    out_baseline = x @ W_down.T
    torch.cuda.synchronize()
    t_baseline = time.time() - t0

    print(f"  时间: {t_baseline * 1000:.2f} ms")
    print(f"  Output shape: {out_baseline.shape}")

    # 测试场景1：所有切片都是 4bit
    print("\n" + "=" * 70)
    print("场景1: 所有切片都是 4bit")
    print("=" * 70)

    # 方案A：完整量化 + 分别切片调用（原方式）
    print("\n--- 方案A：完整量化后分别切片调用（原方式）---")

    packed_data_full_4bit = quantize_weight_simple(
        W_down, bit_width=4, group_size=group_size, seed=args.seed
    )

    # 方法1：先反量化再做 GEMM
    print("\n方法1: 先反量化再做 GEMM (4bit)...")

    torch.cuda.synchronize()
    t0 = time.time()

    w_dequant = dequantize_weight_simple(packed_data_full_4bit, W_down.shape)
    out_dequant_4bit = x.float() @ w_dequant.T

    torch.cuda.synchronize()
    t_dequant = time.time() - t0

    print(f"  时间: {t_dequant * 1000:.2f} ms")

    # 方法2：Triton fused 分别处理每个切片
    print("\n方法2: Triton fused kernel - 分别切片 (4bit)...")

    torch.cuda.synchronize()
    t0 = time.time()

    out_triton_4bit = torch.zeros(B, H, dtype=torch.float32, device=device)
    for i, (start, end) in enumerate(slice_boundaries):
        x_slice = x[:, start:end]
        slice_out = triton_fused_matmul_grouped(
            x_slice,
            packed_data_full_4bit["indices_packed"][:, start // 2: (end + 1) // 2].clone(),
            packed_data_full_4bit["codebook"],
            packed_data_full_4bit["norms"],
            packed_data_full_4bit["seed"],
            packed_data_full_4bit["group_size"],
            end - start,
            bit_width=4
        )
        out_triton_4bit += slice_out

    torch.cuda.synchronize()
    t_triton_separate = time.time() - t0

    print(f"  时间: {t_triton_separate * 1000:.2f} ms")

    # 方法4：优化版v1
    print("\n方法4: Triton fused kernel - 优化版v1 (4bit)...")

    # 预热
    # for _ in range(3):
    #     out_triton_test = torch.zeros(B, H, dtype=torch.float32, device=device)
    #     for i, (start, end) in enumerate(slice_boundaries):
    #         x_slice = x[:, start:end]
    #         slice_out = triton_fused_matmul_grouped_slice_in_features(
    #             x_slice,
    #             packed_data_full_4bit["indices_packed"],
    #             packed_data_full_4bit["codebook"],
    #             packed_data_full_4bit["norms"],
    #             packed_data_full_4bit["seed"],
    #             packed_data_full_4bit["group_size"],
    #             start, end, K,
    #             bit_width=4
    #         )
    #         out_triton_test += slice_out

    torch.cuda.synchronize()
    t0 = time.time()

    out_triton_optv1_4bit = torch.zeros(B, H, dtype=torch.float32, device=device)
    for i, (start, end) in enumerate(slice_boundaries):
        x_slice = x[:, start:end]
        slice_out = triton_fused_matmul_grouped_slice_in_features(
            x_slice,
            packed_data_full_4bit["indices_packed"],
            packed_data_full_4bit["codebook"],
            packed_data_full_4bit["norms"],
            packed_data_full_4bit["seed"],
            packed_data_full_4bit["group_size"],
            start, end, K,
            bit_width=4
        )
        out_triton_optv1_4bit += slice_out

    torch.cuda.synchronize()
    t_triton_optv1 = time.time() - t0

    print(f"  时间: {t_triton_optv1 * 1000:.2f} ms")

    # 验证结果
    print("\n结果对比 (场景1 - 全4bit):")
    out_baseline_float = out_baseline.float()
    out_dequant_float = out_dequant_4bit.float()
    out_triton_float = out_triton_4bit.float()
    out_triton_optv1_float = out_triton_optv1_4bit.float()

    abs_diff_dequant = torch.abs(out_baseline_float - out_dequant_float)
    abs_diff_triton = torch.abs(out_baseline_float - out_triton_float)
    abs_diff_triton_optv1 = torch.abs(out_baseline_float - out_triton_optv1_float)

    print(f"  反量化+GEMM vs Baseline:")
    print(f"    最大绝对误差: {abs_diff_dequant.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff_dequant.mean().item():.6f}")

    print(f"\n  Triton 分别切片 vs Baseline:")
    print(f"    最大绝对误差: {abs_diff_triton.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff_triton.mean().item():.6f}")

    print(f"\n  Triton 优化版v1 vs Baseline:")
    print(f"    最大绝对误差: {abs_diff_triton_optv1.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff_triton_optv1.mean().item():.6f}")

    # 测试场景2：混合精度 4bit:2bit:1bit
    print("\n" + "=" * 70)
    print("场景2: 混合精度 4bit:2bit:1bit")
    print("=" * 70)

    bit_widths = [4, 2, 1]

    # 分别量化每个切片，使用不同精度
    print("\n分别量化每个切片 (4bit, 2bit, 1bit)...")
    packed_data_list_mixed = []
    for i, (w_slice, bw, (start, end)) in enumerate(zip(W_slices, bit_widths, slice_boundaries)):
        print(f"  量化 W{i+1} (in_features [{start}:{end}], shape {w_slice.shape}) with {bw}bit")
        # 使用切片的 seed 偏移，但我们只对切片内的位置进行处理
        packed_data = quantize_weight_simple(
            w_slice, bit_width=bw, group_size=group_size, seed=args.seed + start
        )
        packed_data_list_mixed.append(packed_data)

    # 方法1：先反量化再做 GEMM
    print("\n方法1: 先反量化再做 GEMM (混合精度)...")

    torch.cuda.synchronize()
    t0 = time.time()

    out_dequant_mixed = torch.zeros(B, H, dtype=torch.float32, device=device)
    for i, (packed_data, w_slice, x_slice) in enumerate(zip(packed_data_list_mixed, W_slices, x_slices)):
        w_dequant = dequantize_weight_simple(packed_data, w_slice.shape)
        out_slice = x_slice.float() @ w_dequant.T
        out_dequant_mixed += out_slice

    torch.cuda.synchronize()
    t_dequant_mixed = time.time() - t0

    print(f"  时间: {t_dequant_mixed * 1000:.2f} ms")

    # 方法2：Triton fused
    print("\n方法2: Triton fused kernel (混合精度)...")

    torch.cuda.synchronize()

    # 方法3：Triton fused - 优化版v1（混合精度）
    print("\n方法3: Triton fused kernel - 优化版v1 (混合精度)...")

    torch.cuda.synchronize()
    t0 = time.time()

    out_triton_mixed_optv1 = torch.zeros(B, H, dtype=torch.float32, device=device)
    for i, (packed_data, x_slice) in enumerate(zip(packed_data_list_mixed, x_slices)):
        slice_out = triton_fused_matmul_grouped(
            x_slice,
            packed_data["indices_packed"],
            packed_data["codebook"],
            packed_data["norms"],
            packed_data["seed"],
            packed_data["group_size"],
            x_slice.shape[1],
            bit_width=packed_data["bit_width"]
        )
        out_triton_mixed_optv1 += slice_out

    torch.cuda.synchronize()
    t_triton_mixed_optv1 = time.time() - t0

    print(f"  时间: {t_triton_mixed_optv1 * 1000:.2f} ms")

    # 验证结果
    print("\n结果对比 (场景2 - 混合精度):")
    out_dequant_mixed_float = out_dequant_mixed.float()
    out_triton_mixed_optv1_float = out_triton_mixed_optv1.float()

    # 手动 debug 版本
    print("\n  运行手动 Debug 版本...")
    out_manual_dequant_order = torch.zeros(B, H, dtype=torch.float32, device=device)
    out_manual_triton_order = torch.zeros(B, H, dtype=torch.float32, device=device)
    for i, (packed_data, x_slice) in enumerate(zip(packed_data_list_mixed, x_slices)):
        # 手动版本1：和反量化完全一样的顺序
        slice_out1 = manual_fused_matmul_grouped(
            x_slice, packed_data, x_slice.shape[1], packed_data["bit_width"]
        )
        out_manual_dequant_order += slice_out1

        # 手动版本2：和 Triton 完全一样的数学顺序
        slice_out2 = manual_fused_matmul_grouped_math_order(
            x_slice, packed_data, x_slice.shape[1], packed_data["bit_width"]
        )
        out_manual_triton_order += slice_out2

    abs_diff_dequant_mixed = torch.abs(out_baseline_float - out_dequant_mixed_float)
    abs_diff_triton_mixed_optv1 = torch.abs(out_baseline_float - out_triton_mixed_optv1_float)
    abs_diff_between_mixed = torch.abs(out_dequant_mixed_float - out_triton_mixed_optv1_float)

    abs_diff_manual_dequant_vs_triton = torch.abs(out_manual_dequant_order - out_manual_triton_order)
    abs_diff_manual_triton_vs_real_triton = torch.abs(out_manual_triton_order - out_triton_mixed_optv1_float)
    abs_diff_manual_dequant_vs_real_dequant = torch.abs(out_manual_dequant_order - out_dequant_mixed_float)

    print(f"  反量化+GEMM vs Baseline:")
    print(f"    最大绝对误差: {abs_diff_dequant_mixed.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff_dequant_mixed.mean().item():.6f}")

    print(f"\n  Triton Fused 优化版v1 vs Baseline:")
    print(f"    最大绝对误差: {abs_diff_triton_mixed_optv1.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff_triton_mixed_optv1.mean().item():.6f}")

    print(f"\n  反量化+GEMM vs Triton Fused:")
    print(f"    最大绝对误差: {abs_diff_between_mixed.max().item():.6f}")
    print(f"    平均绝对误差: {abs_diff_between_mixed.mean():.6f}")

    print(f"\n  Debug 对比:")
    print(f"    手动(反量化顺序) vs 手动(Triton顺序): 最大误差 = {abs_diff_manual_dequant_vs_triton.max().item():.12f}")
    print(f"    手动(Triton顺序) vs 真实Triton:     最大误差 = {abs_diff_manual_triton_vs_real_triton.max().item():.12f}")
    print(f"    手动(反量化顺序) vs 真实反量化:     最大误差 = {abs_diff_manual_dequant_vs_real_dequant.max().item():.12f}")

    print("\n" + "=" * 70)
    print("性能汇总")
    print("=" * 70)
    print(f"  Baseline:              {t_baseline * 1000:8.2f} ms")
    print(f"  场景1 - 全4bit:")
    print(f"    反量化+GEMM:         {t_dequant * 1000:8.2f} ms")
    print(f"    Triton 分别切片:     {t_triton_separate * 1000:8.2f} ms")
    print(f"    Triton Fused:     {t_triton_optv1 * 1000:8.2f} ms")
    print(f"  场景2 - 混合4:2:1bit:")
    print(f"    反量化+GEMM:         {t_dequant_mixed * 1000:8.2f} ms")
    print(f"    Triton 优化版v1:     {t_triton_mixed_optv1 * 1000:8.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="混合精度 MoE Down 测试")

    parser.add_argument("--out_features", type=int, default=1024, help="Down 输出特征维度 H")
    parser.add_argument("--in_features", type=int, default=2048, help="Down 输入特征维度 K")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size B")
    parser.add_argument("--group_size", type=int, default=128, help="分组大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    setup_seed(args.seed)

    test_mixed_precision(args)

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
