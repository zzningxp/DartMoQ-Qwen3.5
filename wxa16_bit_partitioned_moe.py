#!/usr/bin/env python3
"""
WxA16 Bit Partitioned Group MoE - 存储 packed 量化权重的 MoE 模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import time
import gc

from wxa16_linear import WxA16Linear
from turboquant_utils.quantize import turboquant_quantize_packed_full


class WxA16Weights(nn.Module):
    """
    单个 bit 宽度的权重存储。

    存储 gate_up 和 down 的 packed 量化表示。
    """

    def __init__(
        self,
        bit_width: int,
        hidden_size: int,
    ):
        super().__init__()
        self.bit_width = bit_width
        self.hidden_size = hidden_size

        # 实际的量化参数会在 later 填充
        self.gate_up_packed = None
        self.down_packed = None
        self.gate_up_metadata = None
        self.down_metadata = None


class WxA16BitPartitionedGroupMoE(nn.Module):
    """
    按 bit 分区的 WxA16 量化 MoE。

    存储格式:
      - 每个 bit 有自己的 packed gate_up/down 权重
      - 紧凑存储，无冗余
    """

    def __init__(
        self,
        gate,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int = 6,
        shared_expert = None,
        shared_expert_gate = None,
    ):
        super().__init__()

        self.gate = gate
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate

        # 按 bit 分组的权重
        self.bit_weights = nn.ModuleDict()  # "8" -> WxA16Weights, "4" -> ...

        # expert 位置信息
        self.inter_size_by_bit = {}
        self.expert_offsets = {}  # bit_str -> LongTensor

        self.bit_list = []

    @classmethod
    @torch.no_grad()
    def from_build_block(cls, build_block, layer_metadata, group_size: int = 128):
        """
        从 MoEBuildBlock 重构为 WxA16BitPartitionedGroupMoE。

        Args:
            build_block: MoEBuildBlock (包含 DartMoQHybridWrapper)
            layer_metadata: 量化过程中的元数据
            group_size: TurboQuant 分组大小

        Returns:
            WxA16BitPartitionedGroupMoE 实例
        """
        from bit_partitioned_moe import BitPartitionedGroupMoE

        # 先从 build_block 构建普通的 BitPartitionedGroupMoE（提取 fp16 权重）
        fp16_moe = BitPartitionedGroupMoE.from_build_block(build_block, layer_metadata)

        # 创建 WxA16 版本
        moe = cls(
            gate=fp16_moe.gate,
            num_experts=fp16_moe.num_experts,
            hidden_size=fp16_moe.hidden_size,
            intermediate_size=fp16_moe.intermediate_size,
            top_k=fp16_moe.top_k,
            shared_expert=fp16_moe.shared_expert,
            shared_expert_gate=fp16_moe.shared_expert_gate,
        )

        moe.bit_list = fp16_moe.bit_list

        # 获取 dtype 和 device
        dtype = next(fp16_moe.parameters()).dtype if hasattr(fp16_moe, 'parameters') else torch.float16
        device = next(build_block.parameters()).device

        # 对每个 bit 的权重进行 WxA16 量化
        for bit_str, gate_up_weight in fp16_moe.bit_weights.gate_up.items():
            bit = int(bit_str)

            print(f"  Quantizing MoE weights for {bit} bit...")

            # 量化 gate_up
            down_weight = fp16_moe.bit_weights.down[bit_str]

            # 由于 gate_up 是拼接的权重 (2x_neurons, hidden_size)，我们需要特殊处理
            # gate_proj 和 up_proj 是拼接的
            gate_up_packed = turboquant_quantize_packed_full(
                gate_up_weight.data,
                bit_width=bit,
                group_size=group_size,
                seed=42 + bit,
                keep_on_gpu=True,
            )

            down_packed = turboquant_quantize_packed_full(
                down_weight.data,
                bit_width=bit,
                group_size=group_size,
                seed=42 + bit + 1000,
                keep_on_gpu=True,
            )

            # 创建 WxA16Weights
            wxa16_weights = WxA16Weights(bit, moe.hidden_size)

            # 存储 packed 数据并注册为 buffer
            wxa16_weights.set_packed_data(gate_up_packed, down_packed)

            # 复制 offset 信息
            moe.expert_offsets[bit_str] = fp16_moe.expert_offsets[bit_str]
            moe.inter_size_by_bit[bit] = fp16_moe.inter_size_by_bit[bit]

            moe.bit_weights[bit_str] = wxa16_weights

            print(f"  Done quantizing {bit} bit")

        # 清理 fp16_moe
        del fp16_moe
        gc.collect()
        torch.cuda.empty_cache()

        return moe

    @torch.no_grad()
    def forward(self, hidden_states):
        """
        前向推理：按 expert 处理，对每个 bit 分别反量化 + GEMM。
        """
        t0 = time.time()

        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)

        final_hidden_states = torch.zeros_like(x)
        t1 = time.time()

        # Shared expert
        t_shared_start = time.time()
        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_out = self.shared_expert(x)
            shared_gate_val = torch.sigmoid(self.shared_expert_gate(x))
            final_hidden_states.add_(shared_out * shared_gate_val)
            del shared_out, shared_gate_val
        t_shared_end = time.time()
        t2 = t_shared_end

        # Router
        t_router_start = time.time()
        gate_output = self.gate(x)
        if isinstance(gate_output, tuple):
            _, topk_weights, topk_indices = gate_output
        else:
            router_logits = gate_output.softmax(dim=-1)
            topk_weights, topk_indices = router_logits.topk(self.top_k, dim=-1)
            del router_logits
        del gate_output
        t_router_end = time.time()
        t3 = t_router_end

        # 优化结构：expert -> bit
        # Flatten: (N, top_k) -> (N * top_k,)
        flat_expert_indices = topk_indices.flatten()
        flat_expert_weights = topk_weights.flatten()
        flat_token_indices = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.top_k)

        # 按 expert 排序
        idxs = flat_expert_indices.argsort()
        sorted_experts = flat_expert_indices[idxs]
        sorted_weights = flat_expert_weights[idxs]
        sorted_tokens = flat_token_indices[idxs]

        # 统计每个 expert 的 token 数
        tokens_per_expert = sorted_experts.bincount(minlength=self.num_experts).cpu().numpy().cumsum(0)

        # 对每个 expert 处理
        t_compute_start = time.time()

        # 统计各阶段时间
        time_dequant_total = 0.0
        time_gemm_total = 0.0
        active_experts_count = 0
        active_bits_count = 0

        for expert_idx in range(self.num_experts):
            end_idx = tokens_per_expert[expert_idx]
            start_idx = 0 if expert_idx == 0 else tokens_per_expert[expert_idx - 1]

            if start_idx == end_idx:
                continue

            active_experts_count += 1

            # 取当前 expert 的所有 token
            exp_token_idx = sorted_tokens[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_weights = sorted_weights[start_idx:end_idx].unsqueeze(1)

            # 对同一个 expert 的所有 token，处理所有 bit
            expert_out = torch.zeros_like(expert_tokens)

            for bit_str in self.bit_weights.keys():
                bit = int(bit_str)
                active_bits_count += 1

                wxa16_weights = self.bit_weights[bit_str]
                expert_offsets = self.expert_offsets[bit_str]

                start = expert_offsets[expert_idx]
                end = expert_offsets[expert_idx + 1]
                actual_inter_size = end - start

                if actual_inter_size == 0:
                    continue

                # ========== WxA16: 反量化 + 推理 ==========
                t_dequant_start = time.time()
                # 需要用 packed 数据重建出该 expert 切片的权重
                # 这里用简化方式：完整反量化然后切片
                from turboquant_utils.quantize import turboquant_dequantize_packed

                gate_up_full = turboquant_dequantize_packed(wxa16_weights.gate_up_packed, device=x.device)
                down_full = turboquant_dequantize_packed(wxa16_weights.down_packed, device=x.device)

                # 切片该 expert 的部分
                e_gate_up = gate_up_full[2*start : 2*end]  # (2*actual_I_b, H)
                e_down = down_full[:, start:end]           # (H, actual_I_b)

                del gate_up_full, down_full
                t_dequant_end = time.time()
                time_dequant_total += t_dequant_end - t_dequant_start

                # 普通推理
                t_gemm_start = time.time()
                gate_up_out = expert_tokens @ e_gate_up.t()
                gate_out = gate_up_out[:, :actual_inter_size]
                up_out = gate_up_out[:, actual_inter_size:]
                del gate_up_out

                act_out = F.silu(gate_out) * up_out
                del gate_out, up_out

                down_out = act_out @ e_down.t()
                del act_out, e_gate_up, e_down

                expert_out += down_out
                t_gemm_end = time.time()
                time_gemm_total += t_gemm_end - t_gemm_start
                # =========================================

            # 累加回最终结果
            expert_out.mul_(expert_weights)
            final_hidden_states.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce='sum'
            )

            del expert_out, expert_tokens, expert_weights, exp_token_idx

        # Cleanup
        del flat_expert_indices, flat_expert_weights, flat_token_indices
        del idxs, sorted_experts, sorted_weights, sorted_tokens, tokens_per_expert

        t_compute_end = time.time()
        t4 = t_compute_end

        result = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        t5 = time.time()

        # 打印详细时间（仅第一次）
        if not hasattr(self, '_log_printed'):
            self._log_printed = True
            print(f"  [WxA16BitPartitionedGroupMoE] forward total: {t5 - t0:.4f}s")
            print(f"    init: {t1 - t0:.4f}s, shared: {t_shared_end - t_shared_start:.4f}s, router: {t_router_end - t_router_start:.4f}s")
            print(f"    compute: {t_compute_end - t_compute_start:.4f}s (dequant: {time_dequant_total:.4f}s, gemm: {time_gemm_total:.4f}s)")
            print(f"    reshape: {t5 - t4:.4f}s, active_experts: {active_experts_count}, active_bits: {active_bits_count}")

        return result
