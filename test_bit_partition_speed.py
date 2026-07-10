#!/usr/bin/env python3
"""
BitPartitionedGroupMoE 性能验证脚本
对比三种方式在单层上的前向速度：
1. Origin grouped_gemm
2. SimpleMoEBlock
3. BitPartitionedGroupMoE (新设计)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

sys.path.insert(0, '..')

from qwen35_utils import load_model, DEV
from grouped_gemm_moe_adapter import convert_grouped_gemm_to_traditional, SimpleMoEBlock


class BitPartitionedGroupMoE(nn.Module):
    """
    按 bit 分区的 grouped_gemm MoE - 性能验证版

    每个 bit group 只保留需要的神经元，无 padding。
    """
    def __init__(self, original_mlp, num_bit_groups=3):
        super().__init__()

        # 复制原始 gate
        self.gate = original_mlp.gate

        # 复制 shared expert
        if hasattr(original_mlp, 'shared_expert'):
            self.shared_expert = original_mlp.shared_expert
        if hasattr(original_mlp, 'shared_expert_gate'):
            self.shared_expert_gate = original_mlp.shared_expert_gate

        # 获取原始权重形状
        gate_up_proj = original_mlp.experts.gate_up_proj
        down_proj = original_mlp.experts.down_proj

        self.num_experts = gate_up_proj.shape[0]
        self.inter_size = gate_up_proj.shape[1] // 2
        self.hidden_size = gate_up_proj.shape[2]
        self.top_k = getattr(original_mlp, 'top_k', 6)

        # 模拟按 bit 分区（均匀分割）
        self.num_bit_groups = num_bit_groups
        self.bit_list = list(range(num_bit_groups))
        inter_size_per_bit = self.inter_size // num_bit_groups

        # 初始化每个 bit 的权重
        self.gate_up_proj = nn.ParameterDict()
        self.down_proj = nn.ParameterDict()
        self.inter_size_by_bit = {}

        for bit in self.bit_list:
            bit_str = str(bit)
            inter_size_b = inter_size_per_bit

            # (E, 2I_b, H) - 只保留需要的神经元
            gate_up = torch.zeros(
                self.num_experts, 2 * inter_size_b, self.hidden_size,
                dtype=gate_up_proj.dtype, device=gate_up_proj.device
            )

            # (E, H, I_b)
            down = torch.zeros(
                self.num_experts, self.hidden_size, inter_size_b,
                dtype=down_proj.dtype, device=down_proj.device
            )

            # 从原始权重中截取
            start = bit * inter_size_per_bit
            end = start + inter_size_per_bit
            gate_up[:, :inter_size_b, :] = gate_up_proj[:, start:end, :]
            gate_up[:, inter_size_b:, :] = gate_up_proj[:, self.inter_size+start:self.inter_size+end, :]
            down[:, :, :inter_size_b] = down_proj[:, :, start:end]

            self.gate_up_proj[bit_str] = nn.Parameter(gate_up, requires_grad=False)
            self.down_proj[bit_str] = nn.Parameter(down, requires_grad=False)
            self.inter_size_by_bit[bit] = inter_size_b

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)

        final_hidden_states = torch.zeros_like(x)

        # Shared expert
        if hasattr(self, 'shared_expert') and hasattr(self, 'shared_expert_gate'):
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
            bit_out = self._forward_bit_group_v3(x, bit, topk_indices, topk_weights)
            final_hidden_states.add_(bit_out)

        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

    def _forward_bit_group_v1(self, x, bit, topk_indices, topk_weights):
        """版本 1：逐 token 逐专家（参考之前 Qwen35HybridMLP）"""
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj:
            return torch.zeros_like(x)

        num_tokens = x.shape[0]
        out = torch.zeros_like(x)

        gate_up = self.gate_up_proj[bit_str]
        down = self.down_proj[bit_str]
        inter_size = self.inter_size_by_bit[bit]

        for token_idx in range(num_tokens):
            token_x = x[token_idx:token_idx+1]
            token_indices = topk_indices[token_idx]
            token_weights = topk_weights[token_idx:token_idx+1]

            token_out = torch.zeros_like(token_x)
            for k_idx in range(self.top_k):
                expert_idx = token_indices[k_idx].item()
                weight = token_weights[0, k_idx]

                expert_gate_up = gate_up[expert_idx:expert_idx+1]
                expert_down = down[expert_idx:expert_idx+1]

                gate_up_out = torch.bmm(token_x.unsqueeze(1), expert_gate_up.transpose(1, 2)).squeeze(1)
                gate_out = gate_up_out[:, :inter_size]
                up_out = gate_up_out[:, inter_size:]

                act_out = F.silu(gate_out) * up_out
                down_out = torch.bmm(act_out.unsqueeze(1), expert_down.transpose(1, 2)).squeeze(1)
                token_out += down_out * weight

            out[token_idx:token_idx+1] = token_out

        return out

    def _forward_bit_group_v2(self, x, bit, topk_indices, topk_weights):
        """版本 2：按专家批量计算（之前设计）"""
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj:
            return torch.zeros_like(x)

        num_tokens = x.shape[0]
        out = torch.zeros_like(x)

        gate_up = self.gate_up_proj[bit_str]
        down = self.down_proj[bit_str]
        inter_size = self.inter_size_by_bit[bit]

        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx)
            if not mask.any():
                continue

            for k in range(self.top_k):
                token_mask = mask[:, k]
                if not token_mask.any():
                    continue

                token_x = x[token_mask]
                weight = topk_weights[token_mask, k:k+1]

                expert_gate_up = gate_up[expert_idx]
                expert_down = down[expert_idx]

                gate_up_out = token_x @ expert_gate_up.t()
                gate_out = gate_up_out[:, :inter_size]
                up_out = gate_up_out[:, inter_size:]

                act_out = F.silu(gate_out) * up_out
                down_out = act_out @ expert_down.t()
                out[token_mask] += down_out * weight

        return out

    def _forward_bit_group_v3(self, x, bit, topk_indices, topk_weights):
        """版本 3：更高效的 gather 方式（分块避免 OOM）"""
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj:
            return torch.zeros_like(x)

        num_tokens = x.shape[0]
        out = torch.zeros_like(x)

        gate_up = self.gate_up_proj[bit_str]
        down = self.down_proj[bit_str]
        inter_size = self.inter_size_by_bit[bit]

        # 分块处理避免 OOM
        chunk_size = 128
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            x_chunk = x[start:end]
            idx_chunk = topk_indices[start:end]
            w_chunk = topk_weights[start:end]

            # (C, K, 2I_b, H)
            selected_gate_up = gate_up[idx_chunk]
            # (C, K, H, I_b)
            selected_down = down[idx_chunk]

            # (C, K, H)
            x_expanded = x_chunk.unsqueeze(1).expand(-1, self.top_k, -1)

            # (C, K, 2I_b) = (C, K, H) @ (C, K, 2I_b, H).transpose
            gate_up_out = torch.matmul(x_expanded.unsqueeze(2), selected_gate_up.transpose(2, 3)).squeeze(2)

            gate_out = gate_up_out[:, :, :inter_size]
            up_out = gate_up_out[:, :, inter_size:]

            act_out = F.silu(gate_out) * up_out

            # (C, K, H) = (C, K, I_b) @ (C, K, H, I_b).transpose
            down_out = torch.matmul(act_out.unsqueeze(2), selected_down.transpose(2, 3)).squeeze(2)

            # (C, K, H) * (C, K, 1) -> (C, K, H)
            weighted_out = down_out * w_chunk.unsqueeze(-1)

            # (C, H)
            out[start:end] = weighted_out.sum(dim=1)

        return out

    def _forward_bit_group_v4(self, x, bit, topk_indices, topk_weights):
        """版本 4：循环 K 次，每次处理一个 expert 位置"""
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj:
            return torch.zeros_like(x)

        out = torch.zeros_like(x)

        gate_up = self.gate_up_proj[bit_str]
        down = self.down_proj[bit_str]
        inter_size = self.inter_size_by_bit[bit]

        for k in range(self.top_k):
            # 第 k 个位置的专家和权重
            expert_indices_k = topk_indices[:, k]  # (N,)
            weights_k = topk_weights[:, k:k+1]  # (N, 1)

            # Gather 权重 (N, 2I_b, H) and (N, H, I_b)
            gate_up_k = gate_up[expert_indices_k]
            down_k = down[expert_indices_k]

            # (N, 2I_b) = (N, H) @ (N, 2I_b, H).transpose
            gate_up_out = torch.bmm(x.unsqueeze(1), gate_up_k.transpose(1, 2)).squeeze(1)

            gate_out = gate_up_out[:, :inter_size]
            up_out = gate_up_out[:, inter_size:]

            act_out = F.silu(gate_out) * up_out

            # (N, H) = (N, I_b) @ (N, H, I_b).transpose
            down_out = torch.bmm(act_out.unsqueeze(1), down_k.transpose(1, 2)).squeeze(1)

            out += down_out * weights_k

        return out


def benchmark_forward(mlp, x, num_iters=10):
    """前向速度 benchmark"""
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = mlp(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = mlp(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    avg_time = (time.time() - start) / num_iters
    return avg_time


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model path')
    parser.add_argument('--layer-idx', type=int, default=0, help='Layer to test')
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, standby_cpu=False)

    layer = model.model.layers[args.layer_idx]
    print(f"\n=== Original Layer {args.layer_idx} ===")
    print(f"MLP type: {type(layer.mlp).__name__}")

    # 测试输入（用小一点的 seq_len 避免 OOM）
    batch_size = 1
    seq_len = 512
    hidden_size = model.config.hidden_size
    x = torch.randn(batch_size, seq_len, hidden_size,
                    dtype=next(layer.parameters()).dtype,
                    device=next(layer.parameters()).device)

    # 1. Benchmark Origin grouped_gemm
    print(f"\n=== Benchmark 1: Origin grouped_gemm ===")
    origin_time = benchmark_forward(layer.mlp, x, num_iters=5)
    print(f"Average forward time: {origin_time:.4f}s")

    # 2. Convert to SimpleMoEBlock and benchmark
    print(f"\n=== Benchmark 2: SimpleMoEBlock ===")
    original_mlp = layer.mlp
    from grouped_gemm_moe_adapter import convert_single_layer
    layer, _ = convert_single_layer(layer)
    simple_time = benchmark_forward(layer.mlp, x, num_iters=5)
    print(f"Average forward time: {simple_time:.4f}s")
    print(f"Slowdown vs origin: {simple_time/origin_time:.2f}x")

    # 3. Test BitPartitionedGroupMoE (four versions)
    print(f"\n=== Benchmark 3: BitPartitionedGroupMoE ===")
    layer.mlp = original_mlp
    bit_moe = BitPartitionedGroupMoE(original_mlp, num_bit_groups=3)

    # 测试 v1 - 太慢，跳过
    # print(f"\n--- Version 1: Per-token per-expert ---")
    # bit_moe._forward_bit_group = bit_moe._forward_bit_group_v1
    # v1_time = benchmark_forward(bit_moe, x, num_iters=2)
    # print(f"Average forward time: {v1_time:.4f}s")
    # print(f"Slowdown vs origin: {v1_time/origin_time:.2f}x")

    # 测试 v2
    print(f"\n--- Version 2: Per-expert batch ---")
    bit_moe._forward_bit_group = bit_moe._forward_bit_group_v2
    v2_time = benchmark_forward(bit_moe, x, num_iters=5)
    print(f"Average forward time: {v2_time:.4f}s")
    print(f"Slowdown vs origin: {v2_time/origin_time:.2f}x")

    # 测试 v3
    print(f"\n--- Version 3: Gather batch matmul (chunked) ---")
    bit_moe._forward_bit_group = bit_moe._forward_bit_group_v3
    v3_time = benchmark_forward(bit_moe, x, num_iters=5)
    print(f"Average forward time: {v3_time:.4f}s")
    print(f"Slowdown vs origin: {v3_time/origin_time:.2f}x")
    print(f"Speedup vs v2: {v2_time/v3_time:.2f}x")

    # 测试 v4
    print(f"\n--- Version 4: Loop K, gather per K ---")
    bit_moe._forward_bit_group = bit_moe._forward_bit_group_v4
    v4_time = benchmark_forward(bit_moe, x, num_iters=5)
    print(f"Average forward time: {v4_time:.4f}s")
    print(f"Slowdown vs origin: {v4_time/origin_time:.2f}x")
    print(f"Speedup vs v2: {v2_time/v4_time:.2f}x")
    print(f"Speedup vs v3: {v3_time/v4_time:.2f}x")

    # 4. Summary
    print(f"\n" + "="*80)
    print(f"=== Summary ===")
    print(f"{'Origin grouped_gemm':<40} {origin_time:>8.4f}s (1.00x)")
    print(f"{'SimpleMoEBlock':<40} {simple_time:>8.4f}s ({simple_time/origin_time:>5.2f}x slower)")
    print(f"{'BitPartitionedGroupMoE v2':<40} {v2_time:>8.4f}s ({v2_time/origin_time:>5.2f}x slower)")
    print(f"{'BitPartitionedGroupMoE v3':<40} {v3_time:>8.4f}s ({v3_time/origin_time:>5.2f}x slower)")
    print(f"{'BitPartitionedGroupMoE v4':<40} {v4_time:>8.4f}s ({v4_time/origin_time:>5.2f}x slower)")
    print(f"="*80)


if __name__ == '__main__':
    main()



if __name__ == '__main__':
    main()
