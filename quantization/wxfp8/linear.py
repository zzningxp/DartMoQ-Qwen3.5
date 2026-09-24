#!/usr/bin/env python3
"""WxFP8 Linear —— attention 路径的 e4m3 激活 + 8-bit 码本 LUT 推理。

与 WxA8Linear（quantization/wxa8/linear.py）的差异：
  1. 码本走 build_fp8_codebook 的 e4m3 LUT（kernel 内查表），无 IDENTITY_CB
     ——e4m3 网格对 idx 非线性，均匀码本也必须查表（这也是 attention W8
     fp8 比 int8 慢 ~0.58x 的原因，见 roadmap §七 WF-3）
  2. **对 Lloyd-Max 码本也开放**：int8 路线因均匀网格塌级把 Lloyd-Max 拒之
     门外，e4m3 LUT 对两种码本的转换误差同量级（实测 uniform 0.0231 /
     Lloyd-Max 0.0252，见 roadmap §七 WF-1）——旧 checkpoint 无需重量化即可
     试验 full-fp8（默认混合部署仍建议 attn 保 wxa8，见 convert_model_to_wxfp8）
  3. cb_scale 在 _ensure_gf 时构建缓存（scale 搜索 600+ 次迭代，不宜每
     forward 重算；WxA8Linear 的 uniform_codebook_step 是 O(1) 所以没这个问题）

存储格式与 checkpoint 不变：packed_indices / codebook / norms 与
WxA16Linear 完全同源，加载后 convert_model_to_wxfp8 原地切换。
"""

import math

import torch

from quantization.wxa16.linear import WxA16Linear
from turboquant_utils.rotation import generate_batch_rotation_matrices
from turboquant_utils.triton_kernels import convert_to_group_first
from turboquant_utils.triton_kernels_fp8 import (
    build_fp8_codebook,
    rotate_quantize_fused_fp8,
    wxfp8_matmul_grouped_gf,
)


class WxFP8Linear(WxA16Linear):
    """WxFP8 attention linear（8-bit 码本 e4m3 LUT + e4m3 激活）。"""

    @classmethod
    def from_wxa16(cls, linear: WxA16Linear) -> "WxFP8Linear":
        """把 WxA16Linear 原地切到 WxFP8（零拷贝）。只支持 8-bit。

        与 WxA8Linear.from_wxa16 不同：不校验码本均匀性——e4m3 LUT 对
        Lloyd-Max / uniform 的转换误差同量级（见模块 docstring）。
        """
        if not isinstance(linear, WxA16Linear):
            raise TypeError(f"期望 WxA16Linear，得到 {type(linear)}")
        if linear.bit_width != 8:
            raise ValueError(f"WxFP8Linear 只支持 8-bit，got {linear.bit_width}")
        linear.__class__ = cls
        linear._gf_built = False
        return linear

    def _ensure_gf(self, device):
        """lazy 转 group-first 布局 + 预乘 1/sqrt(group_size) + 构建 e4m3 LUT 缓存。"""
        if getattr(self, "_gf_built", False):
            return
        if self.in_features % self.group_size != 0:
            raise ValueError(
                f"WxFP8Linear 要求 in_features ({self.in_features}) 对齐 "
                f"group_size ({self.group_size})")
        indices_gf, norms_gf = convert_to_group_first(
            self.packed_indices, self.norms, self.group_size, self.bit_width)
        self._indices_gf = indices_gf.to(device).contiguous()
        self._norms_gf = (norms_gf.to(device) / math.sqrt(self.group_size)
                          ).half().contiguous()
        self._cb_f8, self._cb_scale = build_fp8_codebook(self.codebook.to(device))
        # 原始 buffer 移到 CPU 释放显存（与 WxA8Linear 一致）
        self.packed_indices = self.packed_indices.cpu()
        self.norms = self.norms.cpu()
        self._gf_built = True

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """e4m3 激活推理：旋转+量化（融合）→ FP8 matmul（LUT 反量化）→ bias → 原 dtype。"""
        orig_shape = x.shape
        batch_size = seq_len = None
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, self.in_features)

        self._ensure_gf(x.device)
        src = x if x.dtype == torch.float16 else x.half()

        gs = self.group_size
        G = self.in_features // gs
        rot = generate_batch_rotation_matrices(
            gs, self.seed, G, stride=gs, device=src.device, dtype=torch.float16)
        x_f8, x_scale = rotate_quantize_fused_fp8(
            src, rot, gs, G, extra_scale=self._cb_scale)

        out = wxfp8_matmul_grouped_gf(
            x_f8, x_scale, self._indices_gf, self._cb_f8, self._norms_gf,
            gs, G, self.bit_width)
        del x_f8, x_scale

        if self.bias is not None:
            out = out + self.bias.to(x.device, out.dtype)

        if batch_size is not None and seq_len is not None:
            out = out.reshape(batch_size, seq_len, self.out_features)
        return out.to(x.dtype)
