#!/usr/bin/env python3
"""
Grouped_GEMM_MoE 适配层
简单直接转换：直接替换为传统 MoE 格式供 DartMoQ 量化
"""

import torch
import torch.nn as nn
import sys
import time

sys.path.insert(0, '..')


class TraditionalExpertMLP(nn.Module):
    """单个专家的传统 MLP 格式，完全兼容 DartMoQ 重建流程"""
    def __init__(self, config, dtype=None, device=None):
        super().__init__()
        self.config = config

        # 从 config 获取维度
        self.hidden_size = config.hidden_size
        # 尝试 intermediate_size，没有就用 moe_intermediate_size
        if hasattr(config, 'intermediate_size'):
            self.intermediate_size = config.intermediate_size
        elif hasattr(config, 'moe_intermediate_size'):
            self.intermediate_size = config.moe_intermediate_size
        else:
            raise ValueError("config must have either intermediate_size or moe_intermediate_size")

        # 创建线性层，显式指定 dtype 和 device
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=dtype, device=device)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TraditionalMoEWrapper(nn.Module):
    """把 Grouped_GEMM_MoE 替换为传统 MoE 格式"""
    def __init__(self, original_mlp):
        super().__init__()

        # 保存原始类，用于后续重建
        self._original_mlp_class = original_mlp.__class__

        # 复制 gate
        self.gate = original_mlp.gate

        # 复制 shared_expert
        if hasattr(original_mlp, 'shared_expert'):
            self.shared_expert = original_mlp.shared_expert
        if hasattr(original_mlp, 'shared_expert_gate'):
            self.shared_expert_gate = original_mlp.shared_expert_gate

        # 解析权重形状
        gate_up_proj = original_mlp.experts.gate_up_proj
        down_proj = original_mlp.experts.down_proj

        self.num_experts = gate_up_proj.shape[0]
        self.intermediate_size = gate_up_proj.shape[1] // 2
        self.hidden_size = gate_up_proj.shape[2]

        # 创建临时 config 供 TraditionalExpertMLP 使用
        class TempConfig:
            def __init__(self, hidden_size, intermediate_size):
                self.hidden_size = hidden_size
                self.intermediate_size = intermediate_size
                self.moe_intermediate_size = intermediate_size

        self.config = TempConfig(self.hidden_size, self.intermediate_size)

        # 把合并的权重拆分成独立专家
        self.experts = nn.ModuleList()
        dtype = gate_up_proj.dtype
        device = gate_up_proj.device
        for i in range(self.num_experts):
            expert = TraditionalExpertMLP(self.config, dtype=dtype, device=device)
            expert.gate_proj.weight.data.copy_(gate_up_proj[i, :self.intermediate_size, :])
            expert.up_proj.weight.data.copy_(gate_up_proj[i, self.intermediate_size:, :])
            expert.down_proj.weight.data.copy_(down_proj[i, :, :])
            self.experts.append(expert)

        self._gate_up_shape = tuple(gate_up_proj.shape)
        self._down_shape = tuple(down_proj.shape)
        self._dtype = str(dtype)

        # 显式删除原始权重，释放 CPU 内存
        del gate_up_proj
        del down_proj
        if hasattr(original_mlp.experts, 'gate_up_proj'):
            del original_mlp.experts.gate_up_proj
        if hasattr(original_mlp.experts, 'down_proj'):
            del original_mlp.experts.down_proj

    def forward(self, x, *args, **kwargs):
        batch_size, seq_len, hidden_dim = x.shape
        hidden_states = x.reshape(-1, hidden_dim)

        final_hidden_states = torch.zeros_like(hidden_states)

        # Shared expert 路径
        if hasattr(self, 'shared_expert') and hasattr(self, 'shared_expert_gate'):
            shared_out = self.shared_expert(hidden_states)
            shared_out = shared_out * torch.sigmoid(self.shared_expert_gate(hidden_states))
            final_hidden_states += shared_out

        # Router
        gate_output = self.gate(hidden_states)
        if isinstance(gate_output, tuple):
            _, topk_weights, topk_indices = gate_output
        else:
            top_k = getattr(self, 'top_k', 6)
            router_logits = gate_output.softmax(dim=-1)
            topk_weights, topk_indices = router_logits.topk(top_k, dim=-1)

        top_k = topk_weights.shape[1]

        # 路由到 experts
        for i in range(top_k):
            expert_idx = topk_indices[:, i]
            weight = topk_weights[:, i].unsqueeze(-1)
            for e_idx in range(len(self.experts)):
                mask = expert_idx == e_idx
                if mask.any():
                    expert_input = hidden_states[mask]
                    expert_out = self.experts[e_idx](expert_input)
                    final_hidden_states[mask] += weight[mask] * expert_out

        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)


class MoEBuildBlock(nn.Module):
    """简单的 MoE 块，用于量化后重建，只需要支持基本 forward"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = getattr(config, 'num_experts', None)
        self.top_k = getattr(config, 'num_experts_per_tok', 6)
        # 其他属性会在重建后被覆盖

    def forward(self, x, *args, **kwargs):
        batch_size, seq_len, hidden_dim = x.shape
        hidden_states = x.reshape(-1, hidden_dim)

        final_hidden_states = torch.zeros_like(hidden_states)

        # Shared expert 路径
        if hasattr(self, 'shared_expert') and hasattr(self, 'shared_expert_gate'):
            shared_out = self.shared_expert(hidden_states)
            shared_out = shared_out * torch.sigmoid(self.shared_expert_gate(hidden_states))
            final_hidden_states += shared_out

        # Router
        gate_output = self.gate(hidden_states)
        if isinstance(gate_output, tuple):
            _, topk_weights, topk_indices = gate_output
        else:
            top_k = getattr(self, 'top_k', 6)
            router_logits = gate_output.softmax(dim=-1)
            topk_weights, topk_indices = router_logits.topk(top_k, dim=-1)

        top_k = topk_weights.shape[1]

        # 路由到 experts
        for i in range(top_k):
            expert_idx = topk_indices[:, i]
            weight = topk_weights[:, i].unsqueeze(-1)
            for e_idx in range(len(self.experts)):
                mask = expert_idx == e_idx
                if mask.any():
                    expert_input = hidden_states[mask]
                    expert_out = self.experts[e_idx](expert_input)
                    final_hidden_states[mask] += weight[mask] * expert_out

        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)


def convert_grouped_gemm_to_traditional(model, use_gpu_acceleration=True):
    """把模型中所有 Grouped_GEMM_MoE 层转换为传统格式"""
    converted_count = 0
    first_layer = None
    total_start_time = time.time()

    # 检查是否有可用 GPU
    has_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if has_gpu and use_gpu_acceleration else 'cpu')

    if has_gpu and use_gpu_acceleration:
        print(f"  Using GPU acceleration for conversion (device: {device})")

    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'experts') and hasattr(layer.mlp.experts, 'gate_up_proj'):
            if first_layer is None:
                first_layer = layer.mlp

            layer_start_time = time.time()

            # 如果用 GPU 加速，且层在 CPU 上，先临时移到 GPU
            layer_was_on_cpu = next(layer.parameters()).device.type == 'cpu'
            moved_to_gpu = False
            if has_gpu and use_gpu_acceleration and layer_was_on_cpu:
                layer = layer.to(device)
                moved_to_gpu = True

            # 获取原始信息
            original_gate_up = layer.mlp.experts.gate_up_proj
            original_down = layer.mlp.experts.down_proj

            print(f"  Converting layer {layer_idx}...")
            # print(f"    gate_up_proj shape: {tuple(original_gate_up.shape)}, dtype: {original_gate_up.dtype}")
            # print(f"    down_proj shape: {tuple(original_down.shape)}, dtype: {original_down.dtype}")
            # print(f"    num_experts: {original_gate_up.shape[0]}, intermediate_size: {original_gate_up.shape[1] // 2}")
            # if moved_to_gpu:
            #     print(f"    (Converting on GPU)")

            # 转换
            layer.mlp = TraditionalMoEWrapper(layer.mlp)

            # 如果之前在 CPU，移回去
            if moved_to_gpu:
                layer = layer.to('cpu')
                torch.cuda.empty_cache()

            layer_time = time.time() - layer_start_time
            print(f"    Layer {layer_idx} converted in {layer_time:.3f}s")
            converted_count += 1

    # 确保 model.config 有必要的属性供后续代码使用
    if converted_count > 0 and first_layer is not None:
        gate_up_proj = first_layer.experts.gate_up_proj
        num_experts = gate_up_proj.shape[0]
        intermediate_size = gate_up_proj.shape[1] // 2
        hidden_size = gate_up_proj.shape[2]

        # 设置专家数量
        if not hasattr(model.config, 'num_experts') and not hasattr(model.config, 'n_routed_experts'):
            model.config.num_experts = num_experts

        # 设置中间层大小
        if not hasattr(model.config, 'moe_intermediate_size'):
            if hasattr(model.config, 'intermediate_size'):
                model.config.moe_intermediate_size = model.config.intermediate_size
            else:
                model.config.moe_intermediate_size = intermediate_size

        # 确保 intermediate_size 也有
        if not hasattr(model.config, 'intermediate_size'):
            model.config.intermediate_size = model.config.moe_intermediate_size

        # 设置 top_k
        if hasattr(first_layer, 'top_k'):
            if not hasattr(model.config, 'num_experts_per_tok'):
                model.config.num_experts_per_tok = first_layer.top_k

    total_time = time.time() - total_start_time
    print(f"Converted {converted_count} Grouped_GEMM_MoE layers to traditional format in {total_time:.3f}s")
    return model


def is_grouped_gemm_moe_layer(layer):
    """检查是否是 Grouped_GEMM_MoE 层"""
    return hasattr(layer.mlp, 'experts') and hasattr(layer.mlp.experts, 'gate_up_proj')


def convert_single_layer(layer):
    """转换单层"""
    if is_grouped_gemm_moe_layer(layer):
        original_mlp = layer.mlp
        layer.mlp = TraditionalMoEWrapper(original_mlp)
        # 清理原始 mlp 的引用以帮助释放内存
        for attr_name in dir(original_mlp):
            if not attr_name.startswith('_') and hasattr(original_mlp, attr_name):
                try:
                    delattr(original_mlp, attr_name)
                except:
                    pass
        return layer, original_mlp
    return layer, None
