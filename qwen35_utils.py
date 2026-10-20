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




def get_qwen35_model(model_path, device_map="auto", model_class=None):
    """外部 HF 模型通用加载（bench / 基线用）。

    model_class: 结构类，默认 AutoModelForCausalLM。官方 Qwen3.5 发布的
    ConditionalGeneration 结构（含 visual/mtp，权重 key 带 language_model
    前缀）需传 AutoModelForImageTextToText 才能对上权重名。
    """
    if model_class is None:
        model_class = AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model_class.from_pretrained(
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


def load_model_streaming(model_path):
    """meta 空结构 + 流式逐层反量化载入（--standby-layer-meta，大模型专用）。

    与 load_model 的区别：不在 CPU 上驻留整模型——层保持 meta，由量化
    主循环逐层从盘读取（int8 → GPU 反量化 bf16）。此处只载入小件：
    embed_tokens → GPU（Catcher 阶段需要），norm/lm_head → CPU（eval 用）。
    reader 挂在 model._stream_reader 上供主循环使用。
    """
    print(f"Loading Qwen3.5 model (streaming meta mode) from: {model_path}")
    from qwen35_stream_io import (StreamModelReader, build_meta_structure,
                                  materialize_small_pieces)

    model = build_meta_structure(model_path)
    reader = StreamModelReader(model_path)
    materialize_small_pieces(model, reader, embed_device=DEV)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.seqlen = 2048
    model.eval()

    import os
    model_id = os.path.basename(model_path.rstrip('/'))
    if not model_id:
        fallback = getattr(model.config, '_name_or_path', None) or model_path
        model_id = os.path.basename(str(fallback).rstrip('/'))
    model.model_id = model_id

    print(f"{model_path}, model_id: {model.model_id}, model_type: {model.config.model_type}", flush=True)
    model._standby_cpu = False
    model._model_path = model_path
    model._stream_reader = reader

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


# ---------------------------------------------------------------------------
# 临时泄漏诊断（DARTMOQ_DEBUG_CUDA_LEAK=1 时启用，定位完毕后由本人确认删除）
# ---------------------------------------------------------------------------

import gc as _gc


def cuda_tensor_census():
    """统计当前所有存活 CUDA 张量：(shape, dtype) -> (count, element_size)。

    只保存计数不保存张量引用，避免诊断代码本身阻止回收。
    """
    census = {}
    for obj in _gc.get_objects():
        if type(obj) is torch.Tensor and obj.device.type == "cuda":
            key = (tuple(obj.shape), str(obj.dtype))
            cnt, _ = census.get(key, (0, obj.element_size()))
            census[key] = (cnt + 1, obj.element_size())
    return census


def print_cuda_census_delta(prev, cur, layer_idx, top_k=10, min_bytes=50 * 1024**2):
    """打印相邻两层存活 CUDA 张量的增量（按字节排序），用于定位逐层泄漏。"""
    import math
    deltas = []
    for key in set(prev) | set(cur):
        pcnt = prev.get(key, (0, 0))
        ccnt, csz = cur.get(key, (0, 0))
        dc = ccnt - pcnt[0]
        if dc > 0:
            nbytes = dc * math.prod(key[0]) * csz
            if nbytes >= min_bytes:
                deltas.append((key, dc, nbytes))
    deltas.sort(key=lambda x: -x[2])
    total = sum(d[2] for d in deltas)
    print(f"  [leak-debug] layer {layer_idx} CUDA tensor growth: "
          f"{total / 1024**3:.2f}GB ({len(deltas)} groups >= {min_bytes//1024**2}MB)")
    for key, dc, nbytes in deltas[:top_k]:
        print(f"    {key} x{dc}: +{nbytes / 1024**2:.1f}MB")



