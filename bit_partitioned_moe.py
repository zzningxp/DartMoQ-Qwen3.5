#!/usr/bin/env python3
"""
BitPartitionedGroupMoE - 按 bit 分区的 grouped_gemm MoE 实现

设计思路 ：
- 只保留按 bit 分开的权重，内存最优
- 优化前向计算：对每个 token，先收集所有需要的 expert-bit 组合，批量计算
- 目标：速度接近完整合并权重版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import time


class BitPartitionedGroupMoE(nn.Module):
    """
    按 bit 分区的 grouped_gemm MoE

    存储结构（无冗余）：
        - gate: 复用原始 router
        - bit_weights.gate_up: ParameterDict[str, Tensor] - 按 bit 分开的权重
        - bit_weights.down: ParameterDict[str, Tensor] - 按 bit 分开的权重
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

        # 创建 bit_weights 子模块，存放按 bit 分开的权重
        class BitWeights(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up = nn.ParameterDict()
                self.down = nn.ParameterDict()

        self.bit_weights = BitWeights()
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

        # 第一步：确定每个 bit 在所有专家中的最大神经元数
        max_neurons_by_bit = defaultdict(int)
        for expert_idx in range(num_experts):
            bit_indices = expert_bit_indices[expert_idx]
            for bit in bit_list:
                indices = bit_indices.get(bit, [])
                max_neurons_by_bit[bit] = max(max_neurons_by_bit[bit], len(indices))

        # 第二步：为每个 bit 初始化独立的权重张量
        for bit in bit_list:
            max_n = max_neurons_by_bit[bit]
            if max_n == 0:
                continue

            bit_str = str(bit)

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

            moe.bit_weights.gate_up[bit_str] = nn.Parameter(gate_up, requires_grad=False)
            moe.bit_weights.down[bit_str] = nn.Parameter(down, requires_grad=False)
            moe.inter_size_by_bit[bit] = max_n

        # 第三步：从每个 expert 的 sub_expert 中提取权重，填充到对应 bit 位置
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
                if bit_str not in moe.bit_weights.gate_up:
                    continue

                if bit not in bit_to_subexpert:
                    continue

                sub_expert = bit_to_subexpert[bit]
                n_neurons = sub_expert.gate_proj.weight.shape[0]
                if n_neurons == 0:
                    continue

                # 填充 gate 和 up
                # gate: 前一半, up: 后一半
                moe.bit_weights.gate_up[bit_str][expert_idx, :n_neurons, :] = sub_expert.gate_proj.weight.data
                moe.bit_weights.gate_up[bit_str][expert_idx, moe.inter_size_by_bit[bit]:moe.inter_size_by_bit[bit]+n_neurons, :] = sub_expert.up_proj.weight.data

                # 填充 down
                moe.bit_weights.down[bit_str][expert_idx, :, :n_neurons] = sub_expert.down_proj.weight.data

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

        # 详细统计 compute 内部各部分时间
        t_mask_total = 0.0
        t_gate_up_matmul_total = 0.0
        t_silu_total = 0.0
        t_down_matmul_total = 0.0
        t_accum_total = 0.0

        active_experts_total = 0

        # 创建 CUDA streams 用于并行发射不同 bit 的 kernel
        streams = []
        if x.is_cuda:
            streams = [torch.cuda.Stream(device=x.device) for _ in self.bit_list]

        # 优化后的结构：top_k -> expert -> bit (with CUDA streams)
        # 对每个 k 位置单独处理
        for k in range(self.top_k):
            expert_indices_k = topk_indices[:, k]  # (N,)
            weights_k = topk_weights[:, k:k+1]  # (N, 1)

            # 对每个专家，收集选了它的 token
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

                # 对同一个 (token, expert)，用 CUDA streams 并行处理所有 bit
                expert_out = torch.zeros_like(token_x)

                # 存储每个 bit 的中间结果，最后再累加
                bit_down_outs = []

                # 第一阶段：用不同 stream 并行发射所有 bit 的计算
                for bit_idx, bit in enumerate(self.bit_list):
                    bit_str = str(bit)
                    if bit_str not in self.bit_weights.gate_up:
                        bit_down_outs.append(None)
                        continue

                    gate_up = self.bit_weights.gate_up[bit_str]
                    down = self.bit_weights.down[bit_str]
                    inter_size = self.inter_size_by_bit[bit]

                    # 获取这个 expert 的权重
                    e_gate_up = gate_up[expert_idx]  # (2I_b, H)
                    e_down = down[expert_idx]        # (H, I_b)

                    if x.is_cuda and bit_idx < len(streams):
                        # 使用指定的 stream
                        stream = streams[bit_idx]
                        with torch.cuda.stream(stream):
                            t_gate_up_start = time.time()
                            gate_up_out = token_x @ e_gate_up.t()
                            t_gate_up_end = time.time()
                            t_gate_up_matmul_total += t_gate_up_end - t_gate_up_start

                            gate_out = gate_up_out[:, :inter_size]
                            up_out = gate_up_out[:, inter_size:]

                            # SILU
                            t_silu_start = time.time()
                            act_out = F.silu(gate_out) * up_out
                            t_silu_end = time.time()
                            t_silu_total += t_silu_end - t_silu_start

                            # (M, H) = (M, I_b) @ (I_b, H)
                            t_down_start = time.time()
                            down_out = act_out @ e_down.t()
                            t_down_end = time.time()
                            t_down_matmul_total += t_down_end - t_down_start

                            bit_down_outs.append(down_out)
                    else:
                        # CPU 或没有足够 streams，串行执行
                        t_gate_up_start = time.time()
                        gate_up_out = token_x @ e_gate_up.t()
                        t_gate_up_end = time.time()
                        t_gate_up_matmul_total += t_gate_up_end - t_gate_up_start

                        gate_out = gate_up_out[:, :inter_size]
                        up_out = gate_up_out[:, inter_size:]

                        # SILU
                        t_silu_start = time.time()
                        act_out = F.silu(gate_out) * up_out
                        t_silu_end = time.time()
                        t_silu_total += t_silu_end - t_silu_start

                        # (M, H) = (M, I_b) @ (I_b, H)
                        t_down_start = time.time()
                        down_out = act_out @ e_down.t()
                        t_down_end = time.time()
                        t_down_matmul_total += t_down_end - t_down_start

                        bit_down_outs.append(down_out)

                # 同步所有 streams，确保所有 bit 的计算都完成
                if x.is_cuda:
                    torch.cuda.synchronize(device=x.device)

                # 累加所有 bit 的结果
                for down_out in bit_down_outs:
                    if down_out is not None:
                        expert_out += down_out

                # 最后把所有 bit 的结果累加回 final_hidden_states
                t_accum_start = time.time()
                final_hidden_states[mask] += expert_out * token_w
                t_accum_end = time.time()
                t_accum_total += t_accum_end - t_accum_start

        t4 = time.time()

        result = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        t5 = time.time()

        # 打印详细 profiling
        print(f"  [BitPartitioned_stream] total={t5-t0:.4f}s | init={t1-t0:.4f}s | shared={t2-t1:.4f}s | router={t3-t2:.4f}s | compute={t4-t3:.4f}s | reshape={t5-t4:.4f}s")
        print(f"    [Compute_detail] mask={t_mask_total:.4f}s | gate_up_matmul={t_gate_up_matmul_total:.4f}s | silu={t_silu_total:.4f}s | down_matmul={t_down_matmul_total:.4f}s | accum={t_accum_total:.4f}s | active_experts={active_experts_total} | active_bits={self.bit_list}")

        return result

    def _forward_bit_group(self, x, bit, topk_indices, topk_weights):
        """
        单个 bit group 的前向计算

        思路：对每个专家，收集所有选了这个专家的 token，然后批量计算
        """
        stats = {
            'mask': 0.0,
            'gate_up_matmul': 0.0,
            'silu': 0.0,
            'down_matmul': 0.0,
            'accum': 0.0,
            'active_experts': 0
        }

        bit_str = str(bit)
        if bit_str not in self.bit_weights.gate_up:
            return torch.zeros_like(x), stats

        out = torch.zeros_like(x)

        gate_up = self.bit_weights.gate_up[bit_str]
        down = self.bit_weights.down[bit_str]
        inter_size = self.inter_size_by_bit[bit]

        # 对每个 k 位置单独处理
        for k in range(self.top_k):
            expert_indices_k = topk_indices[:, k]  # (N,)
            weights_k = topk_weights[:, k:k+1]  # (N, 1)

            # 对每个专家，收集选了它的 token
            for expert_idx in range(self.num_experts):
                t_mask_start = time.time()
                mask = expert_indices_k == expert_idx
                if not mask.any():
                    continue
                stats['active_experts'] += 1

                token_x = x[mask]  # (M, H)
                token_w = weights_k[mask]  # (M, 1)
                t_mask_end = time.time()
                stats['mask'] += t_mask_end - t_mask_start

                # 获取这个专家的权重
                e_gate_up = gate_up[expert_idx]  # (2I_b, H)
                e_down = down[expert_idx]        # (H, I_b)

                # 批量计算
                # (M, 2I_b) = (M, H) @ (H, 2I_b)
                t_gate_up_start = time.time()
                gate_up_out = token_x @ e_gate_up.t()
                t_gate_up_end = time.time()
                stats['gate_up_matmul'] += t_gate_up_end - t_gate_up_start

                gate_out = gate_up_out[:, :inter_size]
                up_out = gate_up_out[:, inter_size:]

                # SILU
                t_silu_start = time.time()
                act_out = F.silu(gate_out) * up_out
                t_silu_end = time.time()
                stats['silu'] += t_silu_end - t_silu_start

                # (M, H) = (M, I_b) @ (I_b, H)
                t_down_start = time.time()
                down_out = act_out @ e_down.t()
                t_down_end = time.time()
                stats['down_matmul'] += t_down_end - t_down_start

                # 累加
                t_accum_start = time.time()
                out[mask] += down_out * token_w
                t_accum_end = time.time()
                stats['accum'] += t_accum_end - t_accum_start

        return out, stats
