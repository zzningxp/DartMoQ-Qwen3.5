#!/usr/bin/env python3
"""Inspect Qwen3.5 model layer structure"""

import sys
import torch
import argparse

sys.path.insert(0, '..')
from qwen35_utils import load_model, DEV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.model, standby_cpu=True)

    print("\nModel config:")
    print(model.config)

    print("\n--- Layer 0 inspection ---")
    layer = model.model.layers[0]

    print("\nLayer attributes:")
    print(dir(layer))

    print("\nMLP attributes:")
    print(dir(layer.mlp))

    print(f"\nhasattr(layer, 'self_attn') = {hasattr(layer, 'self_attn')}")
    print(f"hasattr(layer, 'linear_attn') = {hasattr(layer, 'linear_attn')}")

    print(f"\nhasattr(layer.mlp, 'num_experts') = {hasattr(layer.mlp, 'num_experts')}")
    print(f"hasattr(layer.mlp, 'top_k') = {hasattr(layer.mlp, 'top_k')}")

    print(f"\nhasattr(layer.mlp, 'hidden_size') = {hasattr(layer.mlp, 'hidden_size')}")
    print(f"hasattr(layer.mlp, 'intermediate_size') = {hasattr(layer.mlp, 'intermediate_size')}")

    print("\nExperts:")
    if hasattr(layer.mlp, 'experts'):
        print(f"  gate_up_proj shape: {layer.mlp.experts.gate_up_proj.shape}")
        print(f"  down_proj shape: {layer.mlp.experts.down_proj.shape}")

    print("\nShared expert:")
    print(f"  hasattr(layer.mlp, 'shared_expert') = {hasattr(layer.mlp, 'shared_expert')}")
    print(f"  hasattr(layer.mlp, 'shared_expert_gate') = {hasattr(layer.mlp, 'shared_expert_gate')}")

    if hasattr(layer.mlp, 'gate'):
        print(f"\nGate: {type(layer.mlp.gate)}")
        if hasattr(layer.mlp.gate, 'weight'):
            print(f"  weight shape: {layer.mlp.gate.weight.shape}")


if __name__ == "__main__":
    main()

