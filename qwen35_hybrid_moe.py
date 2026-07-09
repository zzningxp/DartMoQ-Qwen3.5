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

    def get_gate(self, expert_idx: int, bit: int):
        """获取某个专家某个 bit 宽度的 gate_proj 权重"""
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj_by_bit:
            return None
        inter_size = self.inter_size_by_bit.get(bit, 0)
        if inter_size == 0:
            return None
        return self.gate_up_proj_by_bit[bit_str][expert_idx, :inter_size, :]

    def get_up(self, expert_idx: int, bit: int):
        """获取某个专家某个 bit 宽度的 up_proj 权重"""
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj_by_bit:
            return None
        inter_size = self.inter_size_by_bit.get(bit, 0)
        if inter_size == 0:
            return None
        return self.gate_up_proj_by_bit[bit_str][expert_idx, inter_size:, :]

    def get_down(self, expert_idx: int, bit: int):
        """获取某个专家某个 bit 宽度的 down_proj 权重"""
        bit_str = str(bit)
        if bit_str not in self.down_proj_by_bit:
            return None
        return self.down_proj_by_bit[bit_str][expert_idx, :, :]


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
        前向传播

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
            final_hidden_states = final_hidden_states + shared_out * shared_gate_val

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
            bit_out = self._forward_bit(x, bit, topk_indices, topk_weights)
            final_hidden_states = final_hidden_states + bit_out

        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

    def _forward_bit(
        self,
        x: torch.Tensor,
        bit: int,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """单个 bit 宽度的前向计算"""
        bit_str = str(bit)
        if bit_str not in self.experts.gate_up_proj_by_bit:
            return torch.zeros_like(x)

        batch_size = x.shape[0]
        out = torch.zeros_like(x)

        # 对每个 top-k 专家
        for i in range(self.top_k):
            expert_idx_flat = topk_indices[:, i]
            weight_flat = topk_weights[:, i].unsqueeze(-1)

            # 对每个 unique 的专家索引，批量计算
            unique_experts, inverse_indices = expert_idx_flat.unique(return_inverse=True)

            for e_idx in unique_experts:
                mask = expert_idx_flat == e_idx
                if not mask.any():
                    continue

                x_e = x[mask]

                # 获取该专家该 bit 的权重
                gate_w = self.experts.get_gate(e_idx, bit)
                up_w = self.experts.get_up(e_idx, bit)
                down_w = self.experts.get_down(e_idx, bit)

                if gate_w is None or up_w is None or down_w is None:
                    continue

                # 计算：silu(x @ gate_w.T) * (x @ up_w.T) @ down_w.T
                gate_out = F.linear(x_e, gate_w)
                up_out = F.linear(x_e, up_w)
                act_out = F.silu(gate_out) * up_out
                down_out = F.linear(act_out, down_w)

                out[mask] = out[mask] + weight_flat[mask] * down_out

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
