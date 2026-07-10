#!/usr/bin/env python3
"""
BitPartitionedGroupMoE - 按 bit 分区的 grouped_gemm MoE 实现

设计思路 v3（最终版）：
- 从 SimpleMoEBlock 重构时，先保存 expert_bit_indices（每个 bit 对应的原始神经元位置）
- 把所有 bit 的权重，按原始神经元位置，拼回完整的 (E, 2*I, H) 和 (E, H, I) 张量
- 这样我们就完全恢复了原始的 grouped_gemm 格式！
- 然后用最接近原始 Qwen3_5MoeSparseMoeBlock 的方式计算！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import time


class BitPartitionedGroupMoE(nn.Module):
    """
    按 bit 分区的 grouped_gemm MoE，但最终会把权重拼回原始格式

    存储结构（最终使用）：
        - gate: 复用原始 router
        - experts.gate_up_proj: (E, 2*I, H)  拼接后的完整权重
        - experts.down_proj: (E, H, I)      拼接后的完整权重
    """

    def __init__(
        self,
        gate,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k=6,
        shared_expert=None,
        shared_expert_gate=None,
    ):
        super().__init__()

        self.gate = gate
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate

        # 这些会在后面填充
        self.bit_list = []
        self._has_been_restored = False

        # 创建一个 experts 子模块，模仿原始结构
        class Experts(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = None
                self.down_proj = None

        self.experts = Experts()

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
        intermediate_size = layer_metadata['intermediate_size']
        bit_list = layer_metadata['bit_list']
        expert_bit_indices = layer_metadata['expert_bit_indices']

        # 创建实例
        moe = cls(
            gate=simple_moe.gate,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            top_k=getattr(simple_moe, 'top_k', 6),
            shared_expert=getattr(simple_moe, 'shared_expert', None),
            shared_expert_gate=getattr(simple_moe, 'shared_expert_gate', None),
        )

        moe.bit_list = bit_list

        # 获取 dtype 和 device
        dtype = next(simple_moe.parameters()).dtype
        device = next(simple_moe.parameters()).device

        # 初始化完整的权重张量
        full_gate_up = torch.zeros(
            num_experts, 2 * intermediate_size, hidden_size,
            dtype=dtype, device=device
        )
        full_down = torch.zeros(
            num_experts, hidden_size, intermediate_size,
            dtype=dtype, device=device
        )

        # 第四步：从每个 expert 的 sub_expert 中提取权重，并按原始位置填回去
        for expert_idx in range(num_experts):
            wrapper = simple_moe.experts[expert_idx]  # DartMoQHybridWrapper
            bit_indices = expert_bit_indices[expert_idx]

            # 先为每个 bit 建立 sub_expert 的映射
            bit_to_subexpert = {}
            for sub_expert in wrapper.sub_experts:
                bit = sub_expert._quant_bit
                bit_to_subexpert[bit] = sub_expert

            # 按 bit_list 中的 bit 顺序处理
            for bit in bit_list:
                if bit not in bit_to_subexpert:
                    continue

                sub_expert = bit_to_subexpert[bit]
                n_neurons = sub_expert.gate_proj.weight.shape[0]
                if n_neurons == 0:
                    continue

                # 获取这个 bit 对应的原始神经元索引
                indices = bit_indices[bit]
                if len(indices) != n_neurons:
                    print(f"  [Warning] bit={bit}, expert={expert_idx}: len(indices)={len(indices)} != n_neurons={n_neurons}")
                    indices = list(range(n_neurons))  # fallback

                # 填充 gate 和 up 到原始位置
                full_gate_up[expert_idx, indices, :] = sub_expert.gate_proj.weight.data
                full_gate_up[expert_idx, intermediate_size + torch.tensor(indices), :] = sub_expert.up_proj.weight.data

                # 填充 down 到原始位置
                full_down[expert_idx, :, indices] = sub_expert.down_proj.weight.data

        # 保存完整权重
        moe.experts.gate_up_proj = nn.Parameter(full_gate_up, requires_grad=False)
        moe.experts.down_proj = nn.Parameter(full_down, requires_grad=False)
        moe._has_been_restored = True

        return moe

    def forward(self, hidden_states):
        t0 = time.time()

        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)

        final_hidden_states = torch.zeros_like(x)
        t1 = time.time()

        # Shared expert
        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_out = self.shared_expert(x)
            shared_gate_val = torch.sigmoid(self.shared_expert_gate(x))
            final_hidden_states.add_(shared_out * shared_gate_val)
        t2 = time.time()

        # Router
        gate_output = self.gate(x)
        if isinstance(gate_output, tuple):
            _, topk_weights, topk_indices = gate_output
        else:
            router_logits = gate_output.softmax(dim=-1)
            topk_weights, topk_indices = router_logits.topk(self.top_k, dim=-1)
        t3 = time.time()

        # 现在用完全恢复的 grouped_gemm 格式计算！
        # 参考 SimpleMoEBlock，但用完整的权重，避免 bit 循环
        gate_up = self.experts.gate_up_proj  # (E, 2*I, H)
        down = self.experts.down_proj        # (E, H, I)
        I = self.intermediate_size

        # 详细统计 compute 内部各部分时间
        t_mask_total = 0.0
        t_gate_up_matmul_total = 0.0
        t_silu_total = 0.0
        t_down_matmul_total = 0.0
        t_accum_total = 0.0

        active_experts_total = 0

        for k in range(self.top_k):
            expert_indices_k = topk_indices[:, k]  # (N,)
            weights_k = topk_weights[:, k:k+1]  # (N, 1)

            # 对每个 expert，收集选了它的 token
            for expert_idx in range(self.num_experts):
                t_mask_start = time.time()
                mask = expert_indices_k == expert_idx
                if not mask.any():
                    continue
                active_experts_total += 1
                token_x = x[mask]  # (M, H)
                token_w = weights_k[mask]  # (M, 1)
                t_mask_end = time.time()
                t_mask_total += t_mask_end - t_mask_start

                # 获取这个 expert 的完整权重
                e_gate_up = gate_up[expert_idx]  # (2I, H)
                e_down = down[expert_idx]        # (H, I)

                # 一次计算完整的输出
                # (M, 2I) = (M, H) @ (H, 2I)
                t_gate_up_start = time.time()
                gate_up_out = token_x @ e_gate_up.t()
                t_gate_up_end = time.time()
                t_gate_up_matmul_total += t_gate_up_end - t_gate_up_start

                gate_out = gate_up_out[:, :I]
                up_out = gate_up_out[:, I:]

                # SILU
                t_silu_start = time.time()
                act_out = F.silu(gate_out) * up_out
                t_silu_end = time.time()
                t_silu_total += t_silu_end - t_silu_start

                # (M, H) = (M, I) @ (I, H)
                t_down_start = time.time()
                down_out = act_out @ e_down.t()
                t_down_end = time.time()
                t_down_matmul_total += t_down_end - t_down_start

                # 累加回去
                t_accum_start = time.time()
                final_hidden_states[mask] += down_out * token_w
                t_accum_end = time.time()
                t_accum_total += t_accum_end - t_accum_start

        t4 = time.time()

        result = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        t5 = time.time()

        # 打印详细 profiling
        print(f"  [BitPartitioned_v3_detail] total={t5-t0:.4f}s | init={t1-t0:.4f}s | "
              f"shared={t2-t1:.4f}s | router={t3-t2:.4f}s | reshape={t5-t4:.4f}s")
        print(f"    [Compute_detail] mask={t_mask_total:.4f}s | gate_up_matmul={t_gate_up_matmul_total:.4f}s | "
              f"silu={t_silu_total:.4f}s | down_matmul={t_down_matmul_total:.4f}s | accum={t_accum_total:.4f}s | "
              f"active_experts={active_experts_total}")

        return result
