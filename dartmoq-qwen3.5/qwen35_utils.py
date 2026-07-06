"""
Utilities for Qwen3.5 MoE.
Phase 1+: Shared utilities across phases.
"""

import torch
import torch.nn as nn
import time
import gc
import sys

# Add parent directory to path to import original utils
sys.path.insert(0, '..')

# Device configuration - match original DartMoQ
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_qwen35_model(model_path, device_map="auto"):
    """Load Qwen3.5 MoE model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True
    )

    model.seqlen = 2048  # Can be made configurable if needed
    return model, tokenizer


def load_model(model_path, standby_cpu=False):
    """
    Load model with optional CPU standby mode for very large models.

    Args:
        model_path: Path to model
        standby_cpu: If True, loads model to CPU first for memory efficiency
    """
    print(f"Loading Qwen3.5 model from: {model_path}")

    device_map = "cpu" if standby_cpu else "auto"
    model, tokenizer = get_qwen35_model(model_path, device_map=device_map)

    model.eval()

    # Set model_id
    model_id = str(model_path).split('/')[-1]
    model.model_id = model_id
    if not model.model_id:
        model.model_id = getattr(model.config, '_name_or_path', None) or model_path
        model.model_id = str(model.model_id).split('/')[-1]

    print(f"model_id: {model.model_id}, model_type: {model.config.model_type}")

    # Mark if we're in CPU standby mode
    model._standby_cpu = standby_cpu
    model._model_path = model_path

    return model, tokenizer


def is_qwen35_merged_weights(model):
    """Check if model uses Qwen3.5's merged weight format."""
    if not hasattr(model.model, 'layers'):
        return False

    layer = model.model.layers[0]
    if not hasattr(layer, 'mlp'):
        return False
    if not hasattr(layer.mlp, 'experts'):
        return False

    # Check for Qwen3.5 style merged weights
    if hasattr(layer.mlp.experts, 'gate_up_proj') and hasattr(layer.mlp.experts, 'down_proj'):
        return True

    return False


def inspect_qwen35_layer(layer, verbose=False):
    """Inspect a Qwen3.5 layer's structure."""
    info = {}

    if hasattr(layer.mlp, 'experts'):
        experts = layer.mlp.experts

        if hasattr(experts, 'gate_up_proj'):
            info['gate_up_proj_shape'] = experts.gate_up_proj.shape
            info['gate_up_proj_dtype'] = experts.gate_up_proj.dtype
        if hasattr(experts, 'down_proj'):
            info['down_proj_shape'] = experts.down_proj.shape
            info['down_proj_dtype'] = experts.down_proj.dtype

        # Check for router
        if hasattr(layer.mlp, 'gate'):
            info['has_router'] = True
            info['router_type'] = type(layer.mlp.gate).__name__

    if verbose:
        print("Qwen3.5 Layer Info:")
        for k, v in info.items():
            print(f"  {k}: {v}")

    return info
