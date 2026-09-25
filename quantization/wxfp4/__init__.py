"""WxFP4 量化推理实现（e2m1 激活 + FP4 MX MMA，保守版混合部署）。

存储格式与 WxA16/WxA8/WxFP8 完全相同，checkpoint 通用；只有 forward 的
matmul 不同（bit 1/4 expert 走 fp4，bit 2 维持 wxa8，attention 维持 wxa8）。
定位与保守版/激进版（3-4）定义见 roadmaps/wxfp4-plan-260924.md WF4-4。
⚠ 只能在 dart312-t38（triton 3.8）下 forward。
"""

from .bit_partitioned_moe import WxFP4BitPartitionedGroupMoE

__all__ = ["WxFP4BitPartitionedGroupMoE"]
