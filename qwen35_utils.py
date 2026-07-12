import torch
import time
import gc
import sys
import psutil

# Add parent directory to path to import original utils
sys.path.insert(0, '..')

from transformers import AutoModelForCausalLM, AutoTokenizer

# Device configuration - match original DartMoQ
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




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

    import os
    model_id = os.path.basename(model_path.rstrip('/'))
    model.model_id = model_id
    if not model.model_id:
        fallback = getattr(model.config, '_name_or_path', None) or model_path
        model.model_id = os.path.basename(str(fallback).rstrip('/'))

    print(f"{model_path}, model_id: {model.model_id}, model_type: {model.config.model_type}", flush=True)
    model._standby_cpu = standby_cpu
    model._model_path = model_path

    return model, tokenizer


def is_grouped_gemm_moe(model):
    """检测是否是 Grouped_GEMM_MoE mode (Qwen3.5 风格)"""
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


def inspect_grouped_gemm_moe_layer(layer, verbose=False):
    """检查 Grouped_GEMM_MoE 层结构"""
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
        print("Grouped_GEMM_MoE Layer Info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
    return info


def get_memory_info_str():
    """Get memory info string (CUDA + CPU) for debugging"""
    mem_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            resvd = torch.cuda.memory_reserved(i) / 1024**3
            mem_info.append(f"CUDA {i}: {alloc:.2f}GB/{resvd:.2f}GB")

    try:
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / 1024**3
        mem_info.append(f"CPU: {cpu_mem:.2f}GB")
    except ImportError:
        pass

    return " | ".join(mem_info)


def print_memory_info(prefix="  [Memory] "):
    """Print memory info (CUDA + CPU) for debugging"""
    mem_str = get_memory_info_str()
    print(f"{prefix}{mem_str}", flush=True)



