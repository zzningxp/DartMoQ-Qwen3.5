#!/usr/bin/env python3
"""Qwen3.5 按 bit 分组的 grouped_gemm MoE 实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
from collections import defaultdict


class Qwen35HybridExperts(nn.Module):
    """
    按 bit 分组的专家权重，保持 Qwen3.5 的 grouped_gemm 格式

    结构：
      - gate_up_proj_by_bit[bit]: (num_experts, 2*inter_size_by_bit, hidden_size)
      - down_proj_by_bit[bit]: (num_experts, hidden_size, inter_size_by_bit)
    """
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        bit_list: list,
        gate_up_proj_by_bit: dict = None,
        down_proj_by_bit: dict = None,
        inter_size_by_bit: dict = None
    ):
        super().__init__()

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.bit_list = bit_list
        self.inter_size_by_bit = inter_size_by_bit or {}

        # 使用 ParameterDict 保存按 bit 分组的权重
        self.gate_up_proj_by_bit = nn.ParameterDict()
        self.down_proj_by_bit = nn.ParameterDict()

        if gate_up_proj_by_bit is not None:
            for bit_str, weight in gate_up_proj_by_bit.items():
                self.gate_up_proj_by_bit[bit_str] = nn.Parameter(weight, requires_grad=False)

        if down_proj_by_bit is not None:
            for bit_str, weight in down_proj_by_bit.items():
                self.down_proj_by_bit[bit_str] = nn.Parameter(weight, requires_grad=False)


class Qwen35HybridMLP(nn.Module):
    """
    按 bit 分组的 Qwen3.5 MoE 层

    关键特性：
      - 复用原始 gate（保持路由一致性）
      - 按 bit 分组存储权重（grouped_gemm 格式）
      - 前向时对每个 bit 分别计算，然后累加
    """
    def __init__(
        self,
        gate: nn.Module,
        experts: Qwen35HybridExperts,
        shared_expert: nn.Module = None,
        shared_expert_gate: nn.Module = None,
        top_k: int = 6
    ):
        super().__init__()

        self.gate = gate
        self.experts = experts
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.top_k = top_k

        # 复制一些方便访问的属性
        self.num_experts = experts.num_experts
        self.hidden_size = experts.hidden_size
        self.bit_list = experts.bit_list

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 高效优化版

        数学等价性保证：
          out = sum_{e in topk} weight_e * expert_e(x)
              = sum_{e in topk} weight_e * [ sum_b expert_e^b(x) ]
              = sum_b [ sum_{e in topk} weight_e * expert_e^b(x) ]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)

        final_hidden_states = torch.zeros_like(x)

        # Shared expert 路径
        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_out = self.shared_expert(x)
            shared_gate_val = torch.sigmoid(self.shared_expert_gate(x))
            final_hidden_states.add_(shared_out * shared_gate_val)

        # Router
        gate_output = self.gate(x)
        if isinstance(gate_output, tuple):
            # Qwen3.5 风格: (logits, topk_weights, topk_indices)
            _, topk_weights, topk_indices = gate_output
        else:
            # 传统风格: logits
            router_logits = gate_output.softmax(dim=-1)
            topk_weights, topk_indices = router_logits.topk(self.top_k, dim=-1)

        # 对每个 bit 分别计算
        for bit in self.bit_list:
            bit_out = self._forward_bit_fast(x, bit, topk_indices, topk_weights)
            final_hidden_states.add_(bit_out)

        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

    def _forward_bit_fast(
        self,
        x: torch.Tensor,
        bit: int,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """单个 bit 宽度的前向计算 - 高性能优化版"""
        bit_str = str(bit)
        if bit_str not in self.experts.gate_up_proj_by_bit:
            return torch.zeros_like(x)

        num_tokens = x.shape[0]
        out = torch.zeros_like(x, device=x.device)

        # 获取该 bit 的所有权重
        gate_up_proj = self.experts.gate_up_proj_by_bit[bit_str]  # (num_experts, 2*inter_size, hidden_size)
        down_proj = self.experts.down_proj_by_bit[bit_str]        # (num_experts, hidden_size, inter_size)
        inter_size = self.experts.inter_size_by_bit[bit]

        # 先对所有 top-k 位置做索引
        # topk_indices: (num_tokens, top_k)
        # topk_weights: (num_tokens, top_k)

        # 一次性 gather 所有需要的专家权重
        # 对于每个 token 和每个 top-k 专家，我们需要:
        #   gate_up_w = gate_up_proj[expert_idx]
        #   down_w = down_proj[expert_idx]

        # 先展平
        flat_indices = topk_indices.reshape(-1)  # (num_tokens * top_k)
        flat_weights = topk_weights.reshape(-1, 1)  # (num_tokens * top_k, 1)

        # 一次性 gather 所有需要的权重
        # gate_up_weights: (num_tokens * top_k, 2*inter_size, hidden_size)
        # down_weights: (num_tokens * top_k, hidden_size, inter_size)
        gate_up_weights = gate_up_proj[flat_indices]  # (N*K, 2I, H)
        down_weights = down_proj[flat_indices]        # (N*K, H, I)

        # 扩展 x 到每个 top-k 位置
        x_expanded = x.repeat_interleave(self.top_k, dim=0)  # (N*K, H)

        # 一次性计算所有 gate_up
        # 使用 batch matrix multiply 或者 einsum
        # x_expanded: (N*K, H)
        # gate_up_weights: (N*K, 2I, H)
        # result: (N*K, 2I)
        gate_up_out = torch.bmm(x_expanded.unsqueeze(1), gate_up_weights.transpose(1, 2)).squeeze(1)

        # 分离 gate 和 up
        gate_out = gate_up_out[:, :inter_size]  # (N*K, I)
        up_out = gate_up_out[:, inter_size:]   # (N*K, I)

        # SILU activation
        act_out = F.silu(gate_out) * up_out  # (N*K, I)

        # 计算 down projection
        # act_out: (N*K, I)
        # down_weights: (N*K, H, I)
        # result: (N*K, H)
        down_out = torch.bmm(act_out.unsqueeze(1), down_weights.transpose(1, 2)).squeeze(1)

        # 乘以权重
        down_out = down_out * flat_weights  # (N*K, H)

        # 现在需要把属于同一个 token 的结果加起来
        # 用 view + sum
        down_out = down_out.view(num_tokens, self.top_k, self.hidden_size)  # (N, K, H)
        out = down_out.sum(dim=1)  # (N, H)

        return out


@torch.no_grad()
def restructure_to_grouped_gemm(
    moe: nn.Module,
    layer_metadata: dict,
    device: torch.device = None
) -> Qwen35HybridMLP:
    """
    从量化后的传统格式重组回 grouped_gemm 格式

    参数：
      moe: 量化后的 SimpleMoEBlock（包含 DartMoQHybridWrapper）
      layer_metadata: 量化过程中保存的元数据
      device: 目标设备

    返回：
      Qwen35HybridMLP: 按 bit 分组的 grouped_gemm 格式 MoE 层
    """
    if device is None:
        device = next(moe.parameters()).device

    num_experts = layer_metadata['num_experts']
    hidden_size = layer_metadata['hidden_size']
    bit_list = layer_metadata['bit_list']

    # 收集每个 bit 的权重
    gate_up_proj_by_bit = {}
    down_proj_by_bit = {}
    inter_size_by_bit = {}

    # 第一步：确定每个 bit 在所有专家中的最大神经元数
    max_neurons_by_bit = defaultdict(int)
    for expert_idx in range(num_experts):
        expert_bit_idx = layer_metadata['expert_bit_indices'][expert_idx]
        for bit in bit_list:
            indices = expert_bit_idx.get(bit, [])
            max_neurons_by_bit[bit] = max(max_neurons_by_bit[bit], len(indices))

    # 第二步：为每个 bit 初始化权重张量
    for bit in bit_list:
        max_n = max_neurons_by_bit[bit]
        if max_n == 0:
            continue

        gate_up = torch.zeros(
            num_experts, 2 * max_n, hidden_size,
            dtype=torch.bfloat16, device=device
        )
        down = torch.zeros(
            num_experts, hidden_size, max_n,
            dtype=torch.bfloat16, device=device
        )

        gate_up_proj_by_bit[str(bit)] = gate_up
        down_proj_by_bit[str(bit)] = down
        inter_size_by_bit[bit] = max_n

    # 第三步：从每个专家的子专家中提取权重
    for expert_idx in range(num_experts):
        wrapper = moe.experts[expert_idx]  # DartMoQHybridWrapper

        # 遍历这个专家的所有子专家
        for sub_expert in wrapper.sub_experts:
            bit = sub_expert._quant_bit
            bit_str = str(bit)

            if bit_str not in gate_up_proj_by_bit:
                continue

            n_neurons = sub_expert.gate_proj.weight.shape[0]

            # 复制权重
            gate_up_proj_by_bit[bit_str][expert_idx, :n_neurons, :] = sub_expert.gate_proj.weight.data
            gate_up_proj_by_bit[bit_str][expert_idx, n_neurons:2*n_neurons, :] = sub_expert.up_proj.weight.data
            down_proj_by_bit[bit_str][expert_idx, :, :n_neurons] = sub_expert.down_proj.weight.data

    # 构建 Qwen35HybridExperts
    experts = Qwen35HybridExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        bit_list=bit_list,
        gate_up_proj_by_bit=gate_up_proj_by_bit,
        down_proj_by_bit=down_proj_by_bit,
        inter_size_by_bit=inter_size_by_bit
    )

    # 构建 Qwen35HybridMLP
    hybrid_mlp = Qwen35HybridMLP(
        gate=moe.gate,
        experts=experts,
        shared_expert=getattr(moe, 'shared_expert', None),
        shared_expert_gate=getattr(moe, 'shared_expert_gate', None),
        top_k=getattr(moe, 'top_k', 6)
    )

    return hybrid_mlp
