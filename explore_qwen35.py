#!/usr/bin/env python3
"""
Script to explore Qwen3.5 MoE architecture.
Phase 1: Understand Qwen3.5's structure and weight format.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def explore_model(model, tokenizer, detailed=False):
    """Explore Qwen3.5 MoE model architecture."""
    print("=" * 80)
    print("Qwen3.5 MoE Architecture Exploration")
    print("=" * 80)

    print(f"\nModel config:")
    print(f"  Model type: {model.config.model_type}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print(f"  Num attention heads: {model.config.num_attention_heads}")

    if hasattr(model.config, 'num_experts'):
        print(f"  Num experts: {model.config.num_experts}")
    if hasattr(model.config, 'n_routed_experts'):
        print(f"  Num routed experts: {model.config.n_routed_experts}")
    if hasattr(model.config, 'num_shared_experts'):
        print(f"  Num shared experts: {model.config.num_shared_experts}")
    if hasattr(model.config, 'top_k'):
        print(f"  Top K: {model.config.top_k}")
    if hasattr(model.config, 'intermediate_size'):
        print(f"  Intermediate size: {model.config.intermediate_size}")

    print("\n" + "=" * 80)
    print("Model Structure")
    print("=" * 80)

    # Print model structure
    print(model)

    # Explore first layer in detail
    if hasattr(model.model, 'layers'):
        layer = model.model.layers[0]
        print("\n" + "=" * 80)
        print("First Layer Structure")
        print("=" * 80)
        print(layer)

        if hasattr(layer, 'mlp'):
            print("\n" + "=" * 80)
            print("MLP Structure")
            print("=" * 80)
            print(layer.mlp)

            if hasattr(layer.mlp, 'experts'):
                print("\n" + "=" * 80)
                print("Experts Structure")
                print("=" * 80)
                print(f"Type of experts: {type(layer.mlp.experts)}")

                # Check if experts has gate_up_proj and down_proj (Qwen3.5 style)
                if hasattr(layer.mlp.experts, 'gate_up_proj'):
                    print(f"  gate_up_proj shape: {layer.mlp.experts.gate_up_proj.shape}")
                    print(f"  gate_up_proj dtype: {layer.mlp.experts.gate_up_proj.dtype}")
                if hasattr(layer.mlp.experts, 'down_proj'):
                    print(f"  down_proj shape: {layer.mlp.experts.down_proj.shape}")
                    print(f"  down_proj dtype: {layer.mlp.experts.down_proj.dtype}")

                # Check if experts is a ModuleList (traditional style)
                if isinstance(layer.mlp.experts, nn.ModuleList):
                    print(f"  Number of experts (ModuleList): {len(layer.mlp.experts)}")
                    if len(layer.mlp.experts) > 0:
                        print(f"  Expert 0 structure: {layer.mlp.experts[0]}")

    print("\n" + "=" * 80)
    print("Parameter Summary")
    print("=" * 80)

    total_params = 0
    expert_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'expert' in name.lower():
            expert_params += param.numel()
        if detailed:
            print(f"  {name:<60} {str(param.shape):<20} {param.dtype}")

    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Expert parameters: {expert_params:,} ({expert_params / 1e9:.2f}B)")

    # Test a small forward pass
    print("\n" + "=" * 80)
    print("Testing Forward Pass")
    print("=" * 80)

    if tokenizer:
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")
        test_input = {k: v.to(model.device) for k, v in test_input.items()}

        with torch.no_grad():
            output = model(**test_input)

        print(f"Input shape: {test_input['input_ids'].shape}")
        print(f"Output logits shape: {output.logits.shape}")
        print("Forward pass successful!")


def main():
    parser = argparse.ArgumentParser(description="Explore Qwen3.5 MoE architecture")
    parser.add_argument('model', type=str, help="Path to Qwen3.5 model")
    parser.add_argument('--device', type=str, default='cpu', help="Device to load model on")
    parser.add_argument('--detailed', action='store_true', help="Print detailed parameter list")

    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    print(f"Device: {args.device}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=args.device,
        trust_remote_code=True
    )

    model.eval()

    # Explore
    explore_model(model, tokenizer, detailed=args.detailed)


if __name__ == "__main__":
    main()
