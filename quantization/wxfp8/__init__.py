"""WxFP8 量化推理实现（e4m3 激活 + 混合 bit 权重 + FP8 Tensor Core）。

存储格式与 WxA16/WxA8 完全相同，checkpoint 通用；只有 forward 的 matmul 不同。
定位：WxFP4 的基础设施预备步骤（roadmaps/wxfp8-plan-260924.md）。
"""

from .bit_partitioned_moe import WxFP8BitPartitionedGroupMoE
from .linear import WxFP8Linear

__all__ = ["WxFP8BitPartitionedGroupMoE", "WxFP8Linear"]
