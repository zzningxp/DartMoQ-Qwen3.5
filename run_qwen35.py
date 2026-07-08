#!/usr/bin/env python3
"""Qwen3.5 MoE 量化主脚本，Phase 2"""

import time
import torch
import torch.nn as nn
import argparse
import sys
import os

sys.path.insert(0, '..')

from data_utils import get_loaders
from qwen35_utils import load_model
from qwen35_simple_wrapper import dartmoq_quant_grouped_gemm_moe


def main():
    parser = argparse.ArgumentParser(description="DartMoQ for Qwen3.5 MoE")
    parser.add_argument("model", type=str, help="Path to Qwen3.5 model")
    parser.add_argument("dataset", type=str, choices=["wikitext2", "ptb", "c4"], help="Calibration dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--slices", type=int, default=4, help="Number of sub experts to slice")
    parser.add_argument("--quant-scheme", type=str, default="a8s4m2233", help="Quantization scheme")
    parser.add_argument("--rank-mode", type=str, default="turboquant_innerproduct", help="Rank mode for neuron ordering")
    parser.add_argument("--standby-layer-cpu", action="store_true", default=False, help="Use CPU standby for layers")
    parser.add_argument("--sequential-eval", action="store_true", default=False, help="Use sequential PPL evaluation")
    parser.add_argument("--quantmode", type=str, default="turboquant", choices=["gptq", "turboquant"], help="Quantization mode")

    args = parser.parse_args()

    print("DartMoQ for Qwen3.5 MoE (Hybrid Mode Only)")
    print(f"Model: {args.model}")
    print(f"Calibration dataset: {args.dataset}")
    print(f"Quant scheme: {args.quant_scheme}")
    print(f"Rank mode: {args.rank_mode}")
    print(f"Slices per expert: {args.slices}")
    print(f"Hybrid MoE: Yes (always enabled)")
    print(f"Quant mode: {args.quantmode}")
    print(f"CPU standby: {'Yes' if args.standby_layer_cpu else 'No'}")

    print(f"\nCurrent time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    print("\nLoading model...")
    model, tokenizer = load_model(args.model, standby_cpu=args.standby_layer_cpu)

    print("Loading calibration data...")
    dataloader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        tokenizer=tokenizer,
        seqlen=model.seqlen
    )
    print(f"Number of calibration samples: {args.nsamples}")

    print("\nStarting quantization...")
    with torch.no_grad():
        quant_model = dartmoq_quant_grouped_gemm_moe(
            model, tokenizer, dataloader, args, test_ppl=True
        )

    print(f"\nFinish time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    if args.save_model:
        save_dir = f"models/dartmoq_qwen35_{args.rank_mode}_{args.quant_scheme}"
        print(f"\nSaving quantized model to: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        print("(Model saving not fully implemented yet - needs custom handling)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

