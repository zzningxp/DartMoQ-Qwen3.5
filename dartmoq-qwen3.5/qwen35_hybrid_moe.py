#!/usr/bin/env python3
"""
Hybrid MoE utilities for Qwen3.5.
Phase 2: Hybrid MoE wrapper and utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TraditionalExpertMLP(nn.Module):
    """Traditional format expert MLP for quantization process."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DartMoQHybridWrapper(nn.Module):
    """
    Single expert wrapper for hybrid MoE.
    This class wraps multiple sub-experts (with different bit configs) into a single callable expert.
    When called, it forwards input through all sub-experts and sums the results.
    """
    def __init__(self, sub_experts):
        super().__init__()
        self.sub_experts = nn.ModuleList(sub_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if len(self.sub_experts) == 0:
            return torch.zeros_like(hidden_states)

        if len(self.sub_experts) == 1:
            return self.sub_experts[0](hidden_states)

        total_output = self.sub_experts[0](hidden_states)
        for sub_expert in self.sub_experts[1:]:
            total_output = total_output + sub_expert(hidden_states)

        return total_output

    def named_children(self):
        for i, sub_expert in enumerate(self.sub_experts):
            yield (f"sub_expert_{i}", sub_expert)

    def __getitem__(self, idx):
        return self.sub_experts[idx]

    def __len__(self):
        return len(self.sub_experts)


def restructure_hybrid_qscheme(qscheme_expert, slice_expert_num):
    restructured = []
    for expert_idx in range(len(qscheme_expert)):
        bit_counts = {}
        for bit in qscheme_expert[expert_idx]:
            bit_counts[bit] = bit_counts.get(bit, 0) + 1

        expert_bits = sorted(bit_counts.items(), reverse=True)
        restructured.append([bit for bit, count in expert_bits])

    return restructured


class Qwen35HybridExperts(nn.Module):
    """
    Qwen3.5-style hybrid MoE experts, maintaining merged weight format
    but grouping by bit width for different precisions.
    """
    def __init__(self, num_experts, hidden_size, intermediate_sizes_by_bit, bits_list):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.bits_list = bits_list

        self.gate_up_proj_by_bit = nn.ParameterDict()
        self.down_proj_by_bit = nn.ParameterDict()

        for bit, inter_size in zip(bits_list, intermediate_sizes_by_bit):
            self.gate_up_proj_by_bit[str(bit)] = nn.Parameter(
                torch.zeros(num_experts, 2 * inter_size, hidden_size)
            )
            self.down_proj_by_bit[str(bit)] = nn.Parameter(
                torch.zeros(num_experts, hidden_size, inter_size)
            )

    def forward(self, x, topk_indices=None):
        output = 0
        for bit in self.bits_list:
            bit_str = str(bit)
            if bit_str in self.gate_up_proj_by_bit:
                # This is just a placeholder - real implementation needs Qwen3.5's actual logic
                pass

        return output
