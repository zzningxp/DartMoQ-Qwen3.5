#!/usr/bin/env python3
"""
Simple test script for Qwen3.5 MoE: no external dataset needed.
"""

import time
import torch
import torch.nn as nn
import argparse
import sys

sys.path.insert(0, '..')
from qwen35_utils import load_model, DEV


@torch.no_grad()
def simple_test(model, tokenizer, args):
    """Simple test with generated dummy data."""
    print(f"\nSimple Test (no external dataset)")

    # Generate dummy input data
    print("Generating dummy input...")
    seqlen = model.seqlen
    vocab_size = tokenizer.vocab_size

    # Create dummy input ids
    dummy_input_ids = torch.randint(0, vocab_size, (1, args.val_samples * seqlen))

    # Wrap like get_loaders does
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    testloader = TokenizerWrapper(dummy_input_ids)

    print(f"Dummy test data shape: {testloader.input_ids.shape}")

    # Now run evaluation - reuse same sequential eval function but
    # we'll define a simpler one here for testing
    return simple_ppl_eval(model, testloader, "dummy", args)


@torch.no_grad()
def simple_ppl_eval(model, testloader, eval_set, args):
    """Simplified PPL evaluation for testing."""
    tick0 = time.time()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    testenc = testloader.input_ids
    nsamples = testenc.shape[1] // model.seqlen
    print(f'ppl evaluation samples: {nsamples}')
    print(f'Testenc shape: {testenc.shape}')

    # Move whole model to DEV for simple test
    print(f"Moving model to {DEV}...")
    model = model.to(DEV)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"CUDA {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")

    nlls = []

    print(f"Running {nsamples} samples...")
    for i in range(nsamples):
        if i % 10 == 0:
            print(f"  Sample {i}/{nsamples}")

        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(DEV)
        target_ids = batch.clone()

        with torch.no_grad():
            outputs = model(batch)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    tick1 = time.time()
    print(f'ppl on {eval_set}: {ppl.item():.4f} time: {tick1 - tick0:.2f}')
    model.config.use_cache = use_cache

    return ppl.item()


def main():
    parser = argparse.ArgumentParser(description="Simple test for Qwen3.5 MoE (no external dataset)")
    parser.add_argument('model', type=str, help="Path to Qwen3.5 model")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--val-samples', type=int, default=4, help="Number of evaluation samples")
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use: auto, cpu, cuda:0, etc.")

    args = parser.parse_args()

    print("Qwen3.5 MoE Simple Test (Phase 1)")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.val_samples}")

    # Override DEV if specified
    global DEV
    if args.device != 'auto':
        DEV = torch.device(args.device)
        print(f"Overriding DEV to: {DEV}")

    # Load model - default to device_map="auto" for simple test
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device_map = "auto" if args.device == 'auto' else args.device
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True
    )
    model.seqlen = 2048
    model.eval()
    model.model_id = str(args.model).split('/')[-1]
    print("Model loaded!")

    # Print model info
    print(f"\nModel config:")
    print(f"  Model type: {model.config.model_type}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")

    # Inspect MLP structure
    if hasattr(model.model, 'layers'):
        layer = model.model.layers[0]
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            experts = layer.mlp.experts
            print(f"\nMLP experts structure:")
            if hasattr(experts, 'gate_up_proj'):
                print(f"  gate_up_proj shape: {experts.gate_up_proj.shape}")
                print(f"  gate_up_proj dtype: {experts.gate_up_proj.dtype}")
            if hasattr(experts, 'down_proj'):
                print(f"  down_proj shape: {experts.down_proj.shape}")
                print(f"  down_proj dtype: {experts.down_proj.dtype}")

    # Simple test
    try:
        ppl = simple_test(model, tokenizer, args)
        print(f"\n✓ Test passed! Dummy ppl: {ppl:.4f}")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import gc
    sys.exit(main())
