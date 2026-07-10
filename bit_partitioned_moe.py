#!/usr/bin/env python3
"""
BitPartitionedGroupMoE - 按 bit 分区的 grouped_gemm MoE 实现

设计思路：
- 每个 bit group 独立存储 (E, 2I_b, H) 和 (E, H, I_b)
- 前向计算优化：对每个专家，收集所有选了它的 token，批量计算
- 为后续真实量化存储预留接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class BitPartitionedGroupMoE(nn.Module):
    """
    按 bit 分区的 grouped_gemm MoE

    存储结构：
        - gate: 复用原始 router
        - gate_up_proj: ParameterDict, key=str(bit), value=(E, 2I_b, H)
        - down_proj: ParameterDict, key=str(bit), value=(E, H, I_b)
        - inter_size_by_bit: dict, key=bit, value=I_b
    """

    def __init__(
        self,
        gate,
        num_experts,
        hidden_size,
        top_k=6,
        shared_expert=None,
        shared_expert_gate=None,
    ):
        super().__init__()

        self.gate = gate
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate

        # 这些会在后面填充
        self.bit_list = []
        self.gate_up_proj = nn.ParameterDict()
        self.down_proj = nn.ParameterDict()
        self.inter_size_by_bit = {}

    @classmethod
    def from_simple_moe(cls, simple_moe, layer_metadata):
        """
        从 SimpleMoEBlock 重构为 BitPartitionedGroupMoE

        参数：
            simple_moe: SimpleMoEBlock 实例（包含 DartMoQHybridWrapper）
            layer_metadata: 量化过程中保存的元数据

        返回：
            BitPartitionedGroupMoE 实例
        """
        # 从 meta 中获取信息
        num_experts = layer_metadata['num_experts']
        hidden_size = layer_metadata['hidden_size']
        bit_list = layer_metadata['bit_list']
        expert_bit_indices = layer_metadata['expert_bit_indices']

        # 创建实例
        moe = cls(
            gate=simple_moe.gate,
            num_experts=num_experts,
            hidden_size=hidden_size,
            top_k=getattr(simple_moe, 'top_k', 6),
            shared_expert=getattr(simple_moe, 'shared_expert', None),
            shared_expert_gate=getattr(simple_moe, 'shared_expert_gate', None),
        )

        moe.bit_list = bit_list

        # 第一步：确定每个 bit 在所有专家中的最大神经元数
        max_neurons_by_bit = defaultdict(int)
        for expert_idx in range(num_experts):
            bit_indices = expert_bit_indices[expert_idx]
            for bit in bit_list:
                indices = bit_indices.get(bit, [])
                max_neurons_by_bit[bit] = max(max_neurons_by_bit[bit], len(indices))

        # 第二步：为每个 bit 初始化权重张量
        for bit in bit_list:
            max_n = max_neurons_by_bit[bit]
            if max_n == 0:
                continue

            bit_str = str(bit)
            dtype = next(simple_moe.parameters()).dtype
            device = next(simple_moe.parameters()).device

            # (E, 2I_b, H)
            gate_up = torch.zeros(
                num_experts, 2 * max_n, hidden_size,
                dtype=dtype, device=device
            )

            # (E, H, I_b)
            down = torch.zeros(
                num_experts, hidden_size, max_n,
                dtype=dtype, device=device
            )

            moe.gate_up_proj[bit_str] = nn.Parameter(gate_up, requires_grad=False)
            moe.down_proj[bit_str] = nn.Parameter(down, requires_grad=False)
            moe.inter_size_by_bit[bit] = max_n

        # 第三步：从每个 expert 的 sub_expert 中提取权重
        for expert_idx in range(num_experts):
            wrapper = simple_moe.experts[expert_idx]  # DartMoQHybridWrapper

            # 先为每个 bit 建立 sub_expert 的映射
            bit_to_subexpert = {}
            for sub_expert in wrapper.sub_experts:
                bit = sub_expert._quant_bit
                bit_to_subexpert[bit] = sub_expert

            # 按 bit_list 中的 bit 顺序处理
            for bit in bit_list:
                bit_str = str(bit)
                if bit_str not in moe.gate_up_proj:
                    continue

                if bit not in bit_to_subexpert:
                    continue

                sub_expert = bit_to_subexpert[bit]
                n_neurons = sub_expert.gate_proj.weight.shape[0]
                if n_neurons == 0:
                    continue

                # 填充 gate 和 up
                # gate: 前一半, up: 后一半
                moe.gate_up_proj[bit_str][expert_idx, :n_neurons, :] = sub_expert.gate_proj.weight.data
                moe.gate_up_proj[bit_str][expert_idx, moe.inter_size_by_bit[bit]:moe.inter_size_by_bit[bit]+n_neurons, :] = sub_expert.up_proj.weight.data

                # 填充 down
                moe.down_proj[bit_str][expert_idx, :, :n_neurons] = sub_expert.down_proj.weight.data

        return moe

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)

        final_hidden_states = torch.zeros_like(x)

        # Shared expert
        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_out = self.shared_expert(x)
            shared_gate_val = torch.sigmoid(self.shared_expert_gate(x))
            final_hidden_states.add_(shared_out * shared_gate_val)

        # Router
        gate_output = self.gate(x)
        if isinstance(gate_output, tuple):
            _, topk_weights, topk_indices = gate_output
        else:
            router_logits = gate_output.softmax(dim=-1)
            topk_weights, topk_indices = router_logits.topk(self.top_k, dim=-1)

        # 对每个 bit group 分别计算
        for bit in self.bit_list:
            bit_out = self._forward_bit_group_optimized(x, bit, topk_indices, topk_weights)
            final_hidden_states.add_(bit_out)

        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

    def _forward_bit_group_optimized(self, x, bit, topk_indices, topk_weights):
        """
        单个 bit group 的前向计算（高效版）

        思路：对每个专家，收集所有选了这个专家的 token，然后批量计算
        """
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj:
            return torch.zeros_like(x)

        num_tokens = x.shape[0]
        out = torch.zeros_like(x)

        gate_up = self.gate_up_proj[bit_str]
        down = self.down_proj[bit_str]
        inter_size = self.inter_size_by_bit[bit]

        # 对每个 k 位置单独处理（可以并行，但为了简单先这样）
        for k in range(self.top_k):
            expert_indices_k = topk_indices[:, k]  # (N,)
            weights_k = topk_weights[:, k:k+1]  # (N, 1)

            # 对每个专家，收集选了它的 token
            for expert_idx in range(self.num_experts):
                # 找到所有选了这个专家的位置
                mask = expert_indices_k == expert_idx
                if not mask.any():
                    continue

                # 取出这些 token
                token_x = x[mask]  # (M, H)
                token_w = weights_k[mask]  # (M, 1)

                # 获取这个专家的权重
                e_gate_up = gate_up[expert_idx]  # (2I_b, H)
                e_down = down[expert_idx]        # (H, I_b)

                # 批量计算
                # (M, 2I_b) = (M, H) @ (H, 2I_b)
                gate_up_out = token_x @ e_gate_up.t()

                gate_out = gate_up_out[:, :inter_size]
                up_out = gate_up_out[:, inter_size:]

                # SILU
                act_out = F.silu(gate_out) * up_out

                # (M, H) = (M, I_b) @ (I_b, H)
                down_out = act_out @ e_down.t()

                # 累加
                out[mask] += down_out * token_w

        return out
