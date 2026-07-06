import torch
import torch.nn as nn
import time
import gc
import sys

# Add parent directory to path to import original utils
sys.path.insert(0, '..')

from transformers import AutoModelForCausalLM, AutoTokenizer

# Device configuration - match original DartMoQ
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Qwen35SingleExpert(nn.Module):
    """
    单个 Qwen3.5 专家的代理，包装 gate_up_proj 和 down_proj 的切片
    看起来像传统的 MLP 模块
    """
    def __init__(self, gate_up_proj_weight, down_proj_weight, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # 创建三个线性层，共享权重数据（不复制）
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # 直接赋值 weight 张量
        self.gate_proj.weight = nn.Parameter(gate_up_proj_weight[:intermediate_size, :])
        self.up_proj.weight = nn.Parameter(gate_up_proj_weight[intermediate_size:, :])
        self.down_proj.weight = nn.Parameter(down_proj_weight)

        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen35ExpertsProxy(nn.Module):
    """
    Qwen3.5 Experts 的代理，支持下标访问
    看起来像传统的 ModuleList，但不修改原始结构
    """
    def __init__(self, original_experts):
        super().__init__()
        self.original_experts = original_experts

        # 缓存形状信息
        self.gate_up_proj = original_experts.gate_up_proj  # (num_experts, 2*intermediate_size, hidden_size)
        self.down_proj = original_experts.down_proj        # (num_experts, hidden_size, intermediate_size)

        self.num_experts = self.gate_up_proj.shape[0]
        self.hidden_size = self.gate_up_proj.shape[2]
        self.intermediate_size = self.gate_up_proj.shape[1] // 2

        # 缓存单个专家对象（延迟创建）
        self._expert_cache = {}

    def __len__(self):
        return self.num_experts

    def __getitem__(self, idx):
        if idx not in self._expert_cache:
            # 动态创建单个专家代理
            self._expert_cache[idx] = Qwen35SingleExpert(
                self.gate_up_proj[idx],
                self.down_proj[idx],
                self.hidden_size,
                self.intermediate_size
            )
        return self._expert_cache[idx]

    def __iter__(self):
        for i in range(self.num_experts):
            yield self[i]


def wrap_qwen35_layer_for_quant(layer):
    """
    临时包装 Qwen3.5 层，让它看起来像传统 MoE 层
    """
    if not hasattr(layer.mlp, 'experts'):
        return layer, None

    if not hasattr(layer.mlp.experts, 'gate_up_proj'):
        return layer, None

    # 创建代理
    original_experts = layer.mlp.experts
    layer.mlp.experts = Qwen35ExpertsProxy(original_experts)

    return layer, original_experts


def unwrap_qwen35_layer(layer, original_experts):
    """恢复原始层结构"""
    if original_experts is not None:
        layer.mlp.experts = original_experts


def get_qwen35_model(model_path, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True
    )
    model.seqlen = 2048
    return model, tokenizer


def load_model(model_path, standby_cpu=False):
    print(f"Loading Qwen3.5 model from: {model_path}")
    device_map = "cpu" if standby_cpu else "auto"
    model, tokenizer = get_qwen35_model(model_path, device_map=device_map)
    model.eval()

    model_id = str(model_path).split('/')[-1]
    model.model_id = model_id
    if not model.model_id:
        model.model_id = getattr(model.config, '_name_or_path', None) or model_path
        model.model_id = str(model.model_id).split('/')[-1]

    print(f"model_id: {model.model_id}, model_type: {model.config.model_type}")
    model._standby_cpu = standby_cpu
    model._model_path = model_path

    return model, tokenizer


def is_qwen35_merged_weights(model):
    if not hasattr(model.model, 'layers'):
        return False
    layer = model.model.layers[0]
    if not hasattr(layer, 'mlp'):
        return False
    if not hasattr(layer.mlp, 'experts'):
        return False
    if hasattr(layer.mlp.experts, 'gate_up_proj') and hasattr(layer.mlp.experts, 'down_proj'):
        return True
    return False


def inspect_qwen35_layer(layer, verbose=False):
    info = {}
    if hasattr(layer.mlp, 'experts'):
        experts = layer.mlp.experts
        if hasattr(experts, 'gate_up_proj'):
            info['gate_up_proj_shape'] = experts.gate_up_proj.shape
            info['gate_up_proj_dtype'] = experts.gate_up_proj.dtype
        if hasattr(experts, 'down_proj'):
            info['down_proj_shape'] = experts.down_proj.shape
            info['down_proj_dtype'] = experts.down_proj.dtype
        if hasattr(layer.mlp, 'gate'):
            info['has_router'] = True
            info['router_type'] = type(layer.mlp.gate).__name__
    if verbose:
        print("Qwen3.5 Layer Info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
    return info


class TraditionalExpertMLP(nn.Module):
    """单个专家的传统 MLP 格式，用于 Qwen3.5 合并权重的适配"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def convert_qwen35_to_traditional(layer):
    """临时把 Qwen3.5 层转换为传统格式"""

    # 从 gate_up_proj 形状中推断尺寸
    gate_up_proj = layer.mlp.experts.gate_up_proj
    down_proj = layer.mlp.experts.down_proj

    # gate_up_proj 形状: (num_experts, 2*intermediate_size, hidden_size)
    num_experts = gate_up_proj.shape[0]
    hidden_size = gate_up_proj.shape[2]
    intermediate_size = gate_up_proj.shape[1] // 2

    experts = nn.ModuleList()

    for expert_idx in range(num_experts):
        expert = TraditionalExpertMLP(hidden_size, intermediate_size)
        expert.gate_proj.weight.data = gate_up_proj[expert_idx, :intermediate_size, :].clone()
        expert.up_proj.weight.data = gate_up_proj[expert_idx, intermediate_size:, :].clone()
        expert.down_proj.weight.data = down_proj[expert_idx, :, :].clone()
        experts.append(expert)

    # 创建一个临时 wrapper，继承原始 MLP 的所有属性
    class TraditionalMoEWrapper(nn.Module):
        def __init__(self, original_mlp, new_experts, hidden_size, intermediate_size):
            super().__init__()
            self.gate = original_mlp.gate
            self.experts = new_experts

            # 复制原始 MLP 的所有数值属性
            for attr_name in dir(original_mlp):
                if not attr_name.startswith('_') and not callable(getattr(original_mlp, attr_name)):
                    try:
                        setattr(self, attr_name, getattr(original_mlp, attr_name))
                    except:
                        pass

            # 确保关键属性存在
            if not hasattr(self, 'num_experts'):
                self.num_experts = len(new_experts)
            if not hasattr(self, 'top_k'):
                self.top_k = 6

            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size

        def forward(self, x):
            batch_size, seq_len, hidden_dim = x.shape
            hidden_states = x.reshape(-1, hidden_dim)

            final_hidden_states = torch.zeros_like(hidden_states)

            if hasattr(self, 'shared_expert') and hasattr(self, 'shared_expert_gate'):
                shared_out = self.shared_expert(hidden_states)
                shared_out = shared_out * torch.sigmoid(self.shared_expert_gate(hidden_states))
                final_hidden_states += shared_out

            gate_output = self.gate(hidden_states)
            if isinstance(gate_output, tuple):
                _, topk_weights, topk_indices = gate_output
            else:
                router_logits = gate_output.softmax(dim=-1)
                topk_weights, topk_indices = router_logits.topk(self.top_k, dim=-1)

            for i in range(self.top_k):
                expert_idx = topk_indices[:, i]
                weight = topk_weights[:, i].unsqueeze(-1)
                for e in range(len(self.experts)):
                    mask = expert_idx == e
                    if mask.any():
                        expert_input = hidden_states[mask]
                        expert_out = self.experts[e](expert_input)
                        final_hidden_states[mask] += weight[mask] * expert_out

            return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

    return TraditionalMoEWrapper(layer.mlp, experts, hidden_size, intermediate_size)

