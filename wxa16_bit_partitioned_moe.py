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
from turboquant_utils.triton_kernels import (
    triton_fused_matmul_grouped_slice_rows,
    triton_fused_matmul_grouped_slice_in_features
)


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
        self._gate_up_packed = None
        self._down_packed = None

        # 保存元数据（非 tensor 值）
        self._gate_up_seed = None
        self._gate_up_group_size = None
        self._gate_up_shape = None
        self._gate_up_bit_width = None
        self._gate_up_rotation = None
        self._gate_up_orig_dtype = None

        self._down_seed = None
        self._down_group_size = None
        self._down_shape = None
        self._down_bit_width = None
        self._down_rotation = None
        self._down_orig_dtype = None

    @property
    def gate_up_packed(self):
        """获取 gate_up_packed 字典，同时恢复元数据"""
        if self._gate_up_packed is None:
            # 从所有 gate_up_* 开头的 buffer 重建字典
            self._gate_up_packed = {}
            for name, param in self.named_buffers():
                if name.startswith('gate_up_'):
                    key = name[len('gate_up_'):]
                    self._gate_up_packed[key] = param
            # 恢复元数据
            if self._gate_up_seed is not None:
                self._gate_up_packed['seed'] = self._gate_up_seed
            if self._gate_up_group_size is not None:
                self._gate_up_packed['group_size'] = self._gate_up_group_size
            if self._gate_up_shape is not None:
                self._gate_up_packed['shape'] = self._gate_up_shape
            if self._gate_up_bit_width is not None:
                self._gate_up_packed['bit_width'] = self._gate_up_bit_width
            if self._gate_up_rotation is not None:
                self._gate_up_packed['rotation'] = self._gate_up_rotation
            if self._gate_up_orig_dtype is not None:
                self._gate_up_packed['orig_dtype'] = self._gate_up_orig_dtype
        return self._gate_up_packed

    @gate_up_packed.setter
    def gate_up_packed(self, value):
        self._gate_up_packed = value

    @property
    def down_packed(self):
        """获取 down_packed 字典，同时恢复元数据"""
        if self._down_packed is None:
            # 从所有 down_* 开头的 buffer 重建字典
            self._down_packed = {}
            for name, param in self.named_buffers():
                if name.startswith('down_'):
                    key = name[len('down_'):]
                    self._down_packed[key] = param
            # 恢复元数据
            if self._down_seed is not None:
                self._down_packed['seed'] = self._down_seed
            if self._down_group_size is not None:
                self._down_packed['group_size'] = self._down_group_size
            if self._down_shape is not None:
                self._down_packed['shape'] = self._down_shape
            if self._down_bit_width is not None:
                self._down_packed['bit_width'] = self._down_bit_width
            if self._down_rotation is not None:
                self._down_packed['rotation'] = self._down_rotation
            if self._down_orig_dtype is not None:
                self._down_packed['orig_dtype'] = self._down_orig_dtype
        return self._down_packed

    @down_packed.setter
    def down_packed(self, value):
        self._down_packed = value

    def set_packed_data(self, gate_up_packed, down_packed):
        """
        设置 packed 数据并注册为 buffer，以便 state_dict 保存。
        """
        # Register gate_up packed data
        if gate_up_packed is not None:
            for key, value in gate_up_packed.items():
                if isinstance(value, torch.Tensor):
                    self.register_buffer(f"gate_up_{key}", value)
            self._gate_up_packed = gate_up_packed
            # 保存元数据
            self._gate_up_seed = gate_up_packed.get('seed')
            self._gate_up_group_size = gate_up_packed.get('group_size')
            self._gate_up_shape = gate_up_packed.get('shape')
            self._gate_up_bit_width = gate_up_packed.get('bit_width')
            self._gate_up_rotation = gate_up_packed.get('rotation')
            self._gate_up_orig_dtype = gate_up_packed.get('orig_dtype')

        # Register down packed data
        if down_packed is not None:
            for key, value in down_packed.items():
                if isinstance(value, torch.Tensor):
                    self.register_buffer(f"down_{key}", value)
            self._down_packed = down_packed
            # 保存元数据
            self._down_seed = down_packed.get('seed')
            self._down_group_size = down_packed.get('group_size')
            self._down_shape = down_packed.get('shape')
            self._down_bit_width = down_packed.get('bit_width')
            self._down_rotation = down_packed.get('rotation')
            self._down_orig_dtype = down_packed.get('orig_dtype')


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
            build_block: MoEBuildBlock (包含 DartMoQHybridWrapper，权重为原始 fp16)
            layer_metadata: 量化过程中的元数据
            group_size: TurboQuant 分组大小

        Returns:
            WxA16BitPartitionedGroupMoE 实例
        """
        tick_start = time.time()
        from bit_partitioned_moe import BitPartitionedGroupMoE

        # 先从 build_block 构建普通的 BitPartitionedGroupMoE（提取原始 fp16 权重）
        tick_fp16moe = time.time()
        fp16_moe = BitPartitionedGroupMoE.from_build_block(build_block, layer_metadata)
        print(f"  [DEBUG] BitPartitionedGroupMoE.from_build_block: {time.time() - tick_fp16moe:.4f}s")

        # 创建 WxA16 版本
        tick_create = time.time()
        moe = cls(
            gate=fp16_moe.gate,
            num_experts=fp16_moe.num_experts,
            hidden_size=fp16_moe.hidden_size,
            intermediate_size=fp16_moe.intermediate_size,
            top_k=fp16_moe.top_k,
            shared_expert=fp16_moe.shared_expert,
            shared_expert_gate=fp16_moe.shared_expert_gate,
        )
        print(f"  [DEBUG] Create WxA16BitPartitionedGroupMoE instance: {time.time() - tick_create:.4f}s")

        moe.bit_list = fp16_moe.bit_list

        # 获取 dtype 和 device
        dtype = next(fp16_moe.parameters()).dtype if hasattr(fp16_moe, 'parameters') else torch.float16
        device = next(build_block.parameters()).device

        # 对每个 bit 的权重进行 WxA16 量化
        tick_quantize = time.time()
        for bit_str, gate_up_weight in fp16_moe.bit_weights.gate_up.items():
            bit = int(bit_str)
            tick_bit = time.time()

            print(f"  Quantizing MoE weights for {bit} bit...")

            # 量化 gate_up
            down_weight = fp16_moe.bit_weights.down[bit_str]

            # 由于 gate_up 是拼接的权重 (2x_neurons, hidden_size)，我们需要特殊处理
            # gate_proj 和 up_proj 是拼接的
            tick_gateup = time.time()
            gate_up_packed = turboquant_quantize_packed_full(
                gate_up_weight.data,
                bit_width=bit,
                group_size=group_size,
                seed=42 + bit,
                keep_on_gpu=True,
            )
            print(f"    [DEBUG] turboquant_quantize_packed_full (gate_up): {time.time() - tick_gateup:.4f}s")

            tick_down = time.time()
            down_packed = turboquant_quantize_packed_full(
                down_weight.data,
                bit_width=bit,
                group_size=group_size,
                seed=42 + bit + 1000,
                keep_on_gpu=True,
            )
            print(f"    [DEBUG] turboquant_quantize_packed_full (down): {time.time() - tick_down:.4f}s")

            # 创建 WxA16Weights
            tick_weights = time.time()
            wxa16_weights = WxA16Weights(bit, moe.hidden_size)
            print(f"    [DEBUG] Create WxA16Weights: {time.time() - tick_weights:.4f}s")

            # 存储 packed 数据并注册为 buffer
            tick_setpacked = time.time()
            wxa16_weights.set_packed_data(gate_up_packed, down_packed)
            print(f"    [DEBUG] set_packed_data: {time.time() - tick_setpacked:.4f}s")

            # 复制 offset 信息
            moe.expert_offsets[bit_str] = fp16_moe.expert_offsets[bit_str]
            moe.inter_size_by_bit[bit] = fp16_moe.inter_size_by_bit[bit]

            moe.bit_weights[bit_str] = wxa16_weights

            print(f"  Done quantizing {bit} bit: {time.time() - tick_bit:.4f}s")

        print(f"  [DEBUG] Quantize all bits: {time.time() - tick_quantize:.4f}s")

        # 清理 fp16_moe
        tick_cleanup = time.time()
        del fp16_moe
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [DEBUG] Cleanup fp16_moe: {time.time() - tick_cleanup:.4f}s")

        print(f"  [DEBUG] from_build_block total time: {time.time() - tick_start:.4f}s")
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
        time_triton_total = 0.0
        active_experts_count = 0
        active_bits_count = 0

        for expert_idx in range(self.num_experts):
            # print(f"  [DEBUG] expert_idx: {expert_idx}", flush=True)
            # t0 = time.time()
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
                t_triton_start = time.time()
                bit = int(bit_str)
                active_bits_count += 1

                wxa16_weights = self.bit_weights[bit_str]
                expert_offsets = self.expert_offsets[bit_str]

                start = int(expert_offsets[expert_idx].item())
                end = int(expert_offsets[expert_idx + 1].item())
                actual_inter_size = end - start

                if actual_inter_size == 0:
                    continue

                # ========== WxA16: Triton Fused + 部分反量化 ==========

                # 使用 Triton Fused Kernel 处理 gate_up
                gate_up_packed = wxa16_weights.gate_up_packed
                gate_up_out = triton_fused_matmul_grouped_slice_rows(
                    expert_tokens,
                    gate_up_packed["indices_packed"],
                    gate_up_packed["codebook"].to(x.device),
                    gate_up_packed["norms"],
                    gate_up_packed["seed"],
                    gate_up_packed["group_size"],
                    gate_up_packed["shape"][1],  # in_features = hidden_size
                    2*start, 2*end,  # row slice
                    bit
                )

                gate_out = gate_up_out[:, :actual_inter_size]
                up_out = gate_up_out[:, actual_inter_size:]
                del gate_up_out

                act_out = F.silu(gate_out) * up_out
                del gate_out, up_out

                # 使用 Triton Fused Kernel 处理 down (in_features slicing)
                down_packed = wxa16_weights.down_packed
                down_out = triton_fused_matmul_grouped_slice_in_features(
                    act_out,
                    down_packed["indices_packed"],
                    down_packed["codebook"].to(x.device),
                    down_packed["norms"],
                    down_packed["seed"],
                    down_packed["group_size"],
                    start, end,  # original_start, original_end
                    down_packed["shape"][1],  # full_in_features
                    bit
                )
                del act_out

                expert_out += down_out
                t_triton_end = time.time()
                time_triton_total += t_triton_end - t_triton_start
                # print(f"    [DEBUG] bit: {bit}, time: {t_triton_end - t_triton_start:.4f}s", flush=True)
                # ==========================================

            # 累加回最终结果
            expert_out.mul_(expert_weights)
            final_hidden_states.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce='sum'
            )

            del expert_out, expert_tokens, expert_weights, exp_token_idx
            # t1 = time.time()
            # print(f"    [DEBUG] expert_idx: {expert_idx}, time: {t1 - t0:.4f}s", flush=True)

        # Cleanup
        del flat_expert_indices, flat_expert_weights, flat_token_indices
        del idxs, sorted_experts, sorted_weights, sorted_tokens, tokens_per_expert

        # 及时清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t_compute_end = time.time()
        t4 = t_compute_end

        result = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        t5 = time.time()

        # 打印详细时间（仅第一次）
        if not hasattr(self, '_log_printed'):
            self._log_printed = True
            print(f"  [WxA16BitPartitionedGroupMoE] forward total: {t5 - t0:.4f}s", flush=True)
            print(f"    init: {t1 - t0:.4f}s, shared: {t_shared_end - t_shared_start:.4f}s, router: {t_router_end - t_router_start:.4f}s", flush=True)
            print(f"    compute: {t_compute_end - t_compute_start:.4f}s (triton: {time_triton_total:.4f}s, dequant: {time_dequant_total:.4f}s, gemm: {time_gemm_total:.4f}s)", flush=True)
            print(f"    reshape: {t5 - t4:.4f}s, active_experts: {active_experts_count}, active_bits: {active_bits_count}", flush=True)

        return result
