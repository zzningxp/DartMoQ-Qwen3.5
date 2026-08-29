#!/usr/bin/env python3
"""Qwen3.5 MoE 量化主脚本，Phase 2"""

import time
import torch
import torch.nn as nn
import argparse
import sys

sys.path.insert(0, '..')

from data_utils import get_loaders, get_git_hash
from qwen35_utils import load_model
from qwen35_simple_wrapper import dartmoq_quant_grouped_gemm_moe


def main():
    parser = argparse.ArgumentParser(description="DartMoQ for Qwen3.5 MoE")
    parser.add_argument("model", type=str, nargs='?', default=None, help="Path to Qwen3.5 model (optional with --load-quantized)")
    parser.add_argument("dataset", type=str, nargs='?', default=None, help="Calibration dataset (optional with --load-quantized)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--slices", type=int, default=4, help="Number of sub experts to slice")
    parser.add_argument("--quant-scheme", type=str, default="a8s4m2233", help="Quantization scheme")
    parser.add_argument("--rank-mode", type=str, default="turboquant_innerproduct", help="Rank mode for neuron ordering")
    parser.add_argument("--standby-layer-cpu", action="store_true", default=False, help="Use CPU standby for layers")
    parser.add_argument("--sequential-eval", action="store_true", default=False, help="Use sequential PPL evaluation")
    parser.add_argument("--quantmode", type=str, default="turboquant", choices=["gptq", "turboquant"], help="Quantization mode")
    parser.add_argument("--quant-layers", type=str, default=None, help="Only quantize specific layers (e.g., '0-5,8,10' for layers 0-5, 8, and 10; default: all layers)")
    parser.add_argument("--wxa16", action="store_true", default=False, help="Use WxA16 real quantization (stored packed format, not fake quant)")
    parser.add_argument("--inference-quant-mode", type=str, default="wxa16",
                        choices=["wxa16", "wxa8"],
                        help="推理量化模式：wxa16 (FP16 激活+FP16 计算，默认) / "
                             "wxa8 (INT8 激活+INT8 Tensor Core)。WxA8 暂未实现。")
    parser.add_argument("--save-quantized", type=str, default=None, help="Save quantized checkpoint (packed format) to this directory after quantization")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Batch size for normal (non-sequential) PPL evaluation")
    parser.add_argument("--load-quantized", type=str, default=None, help="Load quantized checkpoint from this directory, skip calibration & quantization, directly run PPL eval")

    args = parser.parse_args()

    if args.inference_quant_mode == 'wxa8':
        raise NotImplementedError("WxA8 推理模式尚未实现，敬请期待。详见 roadmaps/wxa8-plan-260829.md")

    assert not (args.save_quantized and args.load_quantized), \
        "save-quantized 与 load-quantized 不能同时使用"
    if not args.load_quantized:
        if not args.model:
            parser.error("model path is required when not using --load-quantized")
        if not args.dataset or args.dataset not in ("wikitext2", "ptb", "c4"):
            parser.error("dataset is required (wikitext2/ptb/c4) when not using --load-quantized")

    print("DartMoQ for Qwen3.5 MoE (Hybrid Mode Only)")
    git_hash = get_git_hash()
    print(f"Git HEAD: {git_hash}")
    if args.load_quantized:
        print(f"Quantized checkpoint: {args.load_quantized}")
    else:
        print(f"Model: {args.model}")
        print(f"Calibration dataset: {args.dataset}")
    print(f"Quant scheme: {args.quant_scheme}")
    print(f"Rank mode: {args.rank_mode}")
    print(f"Slices per expert: {args.slices}")
    print(f"Hybrid MoE: Yes (always enabled)")
    print(f"Quant mode: {args.quantmode}")
    print(f"WxA16 real quantization: {'Yes' if args.wxa16 else 'No (fake quant)'}")
    print(f"Inference quant mode: {args.inference_quant_mode}")
    print(f"CPU standby: {'Yes' if args.standby_layer_cpu else 'No'}")
    if args.save_quantized:
        print(f"Save quantized checkpoint: {args.save_quantized}")
    if args.load_quantized:
        print(f"Load quantized checkpoint: {args.load_quantized} (skip calibration & quantization)")
    if args.quant_layers:
        print(f"Quantize layers: {args.quant_layers}")
    else:
        print(f"Quantize layers: All")

    print(f"\nCurrent time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    if args.load_quantized:
        print("\nLoading quantized checkpoint (skip calibration & quantization)...")
        print("(Note: --quant-layers/--nsamples/--slices/--quant-scheme/--rank-mode/--quantmode are ignored in load mode)")
        from qwen35_quant_io import load_quantized_model
        from eval_qwen35 import run_ppl_evaluation
        model, tokenizer = load_quantized_model(
            args.model, args.load_quantized, standby_cpu=args.standby_layer_cpu
        )
        print("\nStarting PPL evaluation...")
        run_ppl_evaluation(model, tokenizer, args)
        print(f"\nFinish time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        return 0

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
            model, tokenizer, dataloader, args, test_ppl=True,
            save_quantized_dir=args.save_quantized
        )

    print(f"\nFinish time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # if args.save_model:
    #     save_dir = f"models/dartmoq_qwen35_{args.rank_mode}_{args.quant_scheme}"
    #     print(f"\nSaving quantized model to: {save_dir}")
    #     os.makedirs(save_dir, exist_ok=True)
    #     print("(Model saving not fully implemented yet - needs custom handling)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

