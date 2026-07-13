#!/usr/bin/env python3
"""
WxA16 Linear Module - 存储 packed 量化权重，推理时反量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

from turboquant_utils.quantize import (
    turboquant_quantize_packed_full,
    turboquant_dequantize_packed,
    unpack_nbit,
)
from turboquant_utils.quantize import get_codebook
from turboquant_utils.rotation import generate_rotation_matrix, hadamard_rotate_inverse


class WxA16Linear(nn.Module):
    """
    WxA16 量化 Linear 层。

    存储 packed 权重而非 fp16，推理时先反量化再进行 GEMM。

    参数形状:
      - packed_indices: (out_features, packed_in_features)
      - norms: (out_features, n_groups) 或 (out_features,)
      - codebook: (2^bit_width,)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bit_width: int,
        group_size: int,
        packed_indices: torch.Tensor,
        codebook: torch.Tensor,
        norms: torch.Tensor,
        seed: int,
        rotation: str = "qr",
        bias = None,
        orig_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        self.group_size = group_size
        self.seed = seed
        self.rotation = rotation
        self.orig_dtype = orig_dtype

        # 存储量化参数 (注册为 buffers 以便 state_dict 保存)
        self.register_buffer("packed_indices", packed_indices)
        self.register_buffer("norms", norms)
        self.register_buffer("codebook", codebook)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        # 计算分组数
        self.n_groups = (in_features + group_size - 1) // group_size

        # 预计算掩码
        self._bit_mask = (1 << bit_width) - 1
        self._elements_per_byte = 8 // bit_width

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        bit_width: int,
        group_size = None,
        seed: int = 42,
        rotation: str = "qr",
        keep_on_gpu: bool = True,
    ) -> "WxA16Linear":
        """
        从 nn.Linear 量化转换为 WxA16Linear。

        Args:
            linear: 原始 nn.Linear
            bit_width: 1/2/4/8
            group_size: 分组大小 (默认 in_features)
            seed: 随机种子
            rotation: 旋转类型
            keep_on_gpu: 是否保持在 GPU

        Returns:
            WxA16Linear 实例
        """
        if group_size is None:
            group_size = linear.in_features

        # 执行 packed 量化
        packed_data = turboquant_quantize_packed_full(
            linear.weight.data,
            bit_width=bit_width,
            group_size=group_size,
            seed=seed,
            rotation=rotation,
            keep_on_gpu=keep_on_gpu,
        )

        # 处理 bias
        bias = linear.bias.data.clone() if linear.bias is not None else None

        # 创建实例
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bit_width=bit_width,
            group_size=group_size,
            packed_indices=packed_data["indices_packed"],
            codebook=packed_data["codebook"],
            norms=packed_data["norms"],
            seed=seed,
            rotation=rotation,
            bias=bias,
            orig_dtype=linear.weight.dtype,
        )

    @torch.no_grad()
    def dequantize(self) -> torch.Tensor:
        """
        反量化回完整权重矩阵。

        Returns:
            W_approx: (out_features, in_features)
        """
        device = self.packed_indices.device
        codebook = self.codebook.to(device)

        M, N = self.out_features, self.in_features

        # 解包索引
        full_indices = unpack_nbit(self.packed_indices, self.bit_width, N)

        # 反量化
        W_approx = torch.zeros((M, N), dtype=torch.float32, device=device)

        # 确保 norms 是二维的
        norms = self.norms
        if norms.dim() == 1:
            norms = norms.unsqueeze(1)

        group_idx = 0
        for g_start in range(0, N, self.group_size):
            g_end = min(g_start + self.group_size, N)
            g_dim = g_end - g_start

            indices_g = full_indices[:, g_start:g_end]
            norms_g = norms[:, group_idx].unsqueeze(1)
            group_idx += 1

            # 从码本还原
            Y_quant_scaled = codebook[indices_g]

            # 逆缩放
            scale = math.sqrt(g_dim)
            Y_unscaled = Y_quant_scaled / scale

            # 逆旋转
            if self.rotation == "none":
                W_g_approx = Y_unscaled
            elif self.rotation == "hadamard":
                W_g_approx = hadamard_rotate_inverse(Y_unscaled, seed=self.seed + g_start)
            else:  # qr
                Pi = generate_rotation_matrix(g_dim, seed=self.seed + g_start).to(device)
                W_g_approx = Y_unscaled @ Pi

            # 恢复原始尺度
            W_approx[:, g_start:g_end] = W_g_approx * norms_g

        return W_approx.to(self.orig_dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向推理：反量化 + GEMM。

        Args:
            x: (batch_size, seq_len, in_features) 或 (batch_size, in_features)

        Returns:
            out: (batch_size, seq_len, out_features) 或 (batch_size, out_features)
        """
        t_start = time.time()

        # 反量化权重
        t_dequant_start = time.time()
        W_approx = self.dequantize()
        t_dequant_end = time.time()

        # 普通线性层推理
        t_gemm_start = time.time()
        x_dtype = x.dtype
        out = F.linear(x.to(W_approx.dtype), W_approx, self.bias.to(W_approx.dtype) if self.bias is not None else None)
        t_gemm_end = time.time()

        t_end = time.time()

        # 打印详细时间（仅第一次，避免刷屏）
        if not hasattr(self, '_log_printed'):
            self._log_printed = True
            print(f"  [WxA16Linear {self.bit_width}bit] forward total: {t_end - t_start:.4f}s, dequant: {t_dequant_end - t_dequant_start:.4f}s, gemm: {t_gemm_end - t_gemm_start:.4f}s", flush=True)
            print(f"  [WxA16Linear {self.bit_width}bit] input shape: {x.shape}, output shape: {out.shape}, W_approx shape: {W_approx.shape}, norms shape: {self.norms.shape}", flush=True)

        return out.to(x_dtype)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bit_width={self.bit_width}, group_size={self.group_size}, "
                f"rotation={self.rotation}")

    def get_memory_usage(self) -> dict:
        """
        获取当前 WxA16Linear 的存储空间统计。

        Returns:
            dict with:
                - packed_bytes: packed 权重的字节数
                - metadata_bytes: 元数据的字节数 (norms, codebook, bias)
                - orig_fp16_bytes: 原始 fp16 权重的字节数 (用于对比)
                - compression_ratio: 压缩比
        """
        # packed_indices 的字节数
        packed_bytes = self.packed_indices.numel() * self.packed_indices.element_size()

        # norms 的字节数
        norms_bytes = self.norms.numel() * self.norms.element_size()

        # codebook 的字节数
        codebook_bytes = self.codebook.numel() * self.codebook.element_size()

        # bias 的字节数
        bias_bytes = 0
        if self.bias is not None:
            bias_bytes = self.bias.numel() * self.bias.element_size()

        metadata_bytes = norms_bytes + codebook_bytes + bias_bytes

        # 原始 fp16 的字节数 (权重 + bias)
        orig_weight_bytes = self.in_features * self.out_features * 2  # fp16 = 2 bytes
        orig_bias_bytes = self.out_features * 2 if self.bias is not None else 0
        orig_fp16_bytes = orig_weight_bytes + orig_bias_bytes

        # 压缩比
        total_bytes = packed_bytes + metadata_bytes
        compression_ratio = orig_fp16_bytes / total_bytes if total_bytes > 0 else float('inf')

        return {
            'packed_bytes': packed_bytes,
            'metadata_bytes': metadata_bytes,
            'total_bytes': total_bytes,
            'orig_fp16_bytes': orig_fp16_bytes,
            'compression_ratio': compression_ratio,
            'bit_width': self.bit_width,
        }


class W8A16Linear(WxA16Linear):
    """
    专用 W8A16 版本，对 8-bit 有潜在优化。
    """

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size = None,
        seed: int = 42,
        rotation: str = "qr",
        keep_on_gpu: bool = True,
    ) -> "W8A16Linear":
        return super().from_linear(
            linear,
            bit_width=8,
            group_size=group_size,
            seed=seed,
            rotation=rotation,
            keep_on_gpu=keep_on_gpu,
        )
