#!/usr/bin/env python3
"""
Qwen3.5 MoE Evaluation Script.
Phase 1: Original FP16 perplexity evaluation with CPU standby mode support.
"""

import time
import torch
import torch.nn as nn
import argparse
import sys

# Add parent directory to import data_utils
sys.path.insert(0, '..')

from qwen35_utils import load_model, DEV
from data_utils import get_loaders


@torch.no_grad()
def qwen35_ppl_eval_sequential(model, testloader, eval_set, args):
    """
    Sequential PPL evaluation for Qwen3.5 MoE: keeps layers on CPU and moves
    them to GPU one by one. Processes samples in batches to save memory.
    """
    tick0 = time.time()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    testenc = testloader.input_ids
    nsamples = testenc.shape[1] // model.seqlen
    print(f'ppl evaluation samples (sequential mode): {nsamples}')

    # Save original device for each layer
    layers = model.model.layers
    original_devices = []
    for layer in layers:
        if hasattr(layer, 'parameters') and len(list(layer.parameters())) > 0:
            original_devices.append(next(layer.parameters()).device)
        else:
            original_devices.append(torch.device('cpu'))

    # Move all layers to CPU first
    print("Moving transformer layers to CPU for sequential evaluation...")
    for layer in layers:
        layer.to('cpu')

    # Keep embed_tokens, norm, lm_head on GPU permanently
    model.model.embed_tokens = model.model.embed_tokens.to(DEV)
    if hasattr(model.model, 'norm'):
        model.model.norm = model.model.norm.to(DEV)
    if hasattr(model, 'lm_head'):
        model.lm_head = model.lm_head.to(DEV)

    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            print(f"CUDA {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")

    # First, capture correct attention_mask, position_ids using Catcher
    # Temporarily move layer 0 to DEV for capturing
    layers[0] = layers[0].to(DEV)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            self.captured_kwargs = kwargs
            raise ValueError
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])

    # Get first batch to capture the kwargs
    first_batch = testenc[:, :model.seqlen].to(DEV)
    try:
        model(first_batch)
    except ValueError:
        pass

    # Get the captured kwargs
    captured_kwargs = layers[0].captured_kwargs
    attention_mask = captured_kwargs.get('attention_mask')
    position_ids = captured_kwargs.get('position_ids')

    # Restore layer 0
    layers[0] = layers[0].module
    layers[0] = layers[0].to('cpu')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get all target ids first - keep on GPU!
    all_target_ids = testenc[:, :nsamples * model.seqlen].clone().to(DEV)

    # Batch sizes - use smaller batch for memory constrained GPUs
    batch_size_transformer = 32
    batch_size_lm_head = 4

    # Precompute embeddings for all samples and keep on CPU to save GPU memory!
    print("Processing embeddings and caching on CPU...")
    all_embeddings = []
    for sample_idx in range(nsamples):
        batch = testenc[:, (sample_idx * model.seqlen):((sample_idx + 1) * model.seqlen)].to(DEV)
        with torch.no_grad():
            hidden = model.model.embed_tokens(batch)
        all_embeddings.append(hidden.cpu())  # Move to CPU immediately
        del batch, hidden
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Process each transformer layer sequentially, with batches
    all_hidden_states = [emb for emb in all_embeddings]  # Start with embeddings

    import inspect
    for layer_idx, layer in enumerate(layers):
        if layer_idx % 5 == 0:
            print(f"Processing layer {layer_idx}/{len(layers)}...", flush=True)

        # Move layer to GPU
        layer = layer.to(DEV)

        # Process samples in batches
        new_hidden_states = []
        for batch_start in range(0, nsamples, batch_size_transformer):
            batch_end = min(batch_start + batch_size_transformer, nsamples)
            actual_batch_size = batch_end - batch_start

            # Get this batch's hidden states and move to GPU
            batch_hidden = torch.cat(all_hidden_states[batch_start:batch_end], dim=0).to(DEV)

            # Prepare kwargs for this batch size - create on demand to save memory
            layer_kwargs = {}
            forward_signature = inspect.signature(layer.forward)

            if 'attention_mask' in forward_signature.parameters and attention_mask is not None:
                # Create attention mask for this specific batch size
                if attention_mask.dim() == 4:
                    layer_kwargs['attention_mask'] = attention_mask.repeat(actual_batch_size, 1, 1, 1)
                elif attention_mask.dim() == 3:
                    layer_kwargs['attention_mask'] = attention_mask.repeat(actual_batch_size, 1, 1)
                elif attention_mask.dim() == 2:
                    layer_kwargs['attention_mask'] = attention_mask.repeat(actual_batch_size, 1)

            if 'position_ids' in forward_signature.parameters and position_ids is not None:
                # Create position_ids for this specific batch size
                layer_kwargs['position_ids'] = position_ids.repeat(actual_batch_size, 1)

            # Forward pass
            with torch.no_grad():
                layer_outputs = layer(batch_hidden, **layer_kwargs)

            # Save output
            if isinstance(layer_outputs, tuple):
                batch_output = layer_outputs[0]
            else:
                batch_output = layer_outputs

            # Split batch into individual samples and move to CPU
            for i in range(actual_batch_size):
                new_hidden_states.append(batch_output[i:i+1].cpu())

            # Cleanup
            del batch_hidden, layer_outputs, batch_output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update hidden states for next layer
        all_hidden_states = new_hidden_states

        # Move layer back to CPU
        layer = layer.to('cpu')

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if layer_idx % 5 == 0:
                for i in range(torch.cuda.device_count()):
                    print(f"  Layer {layer_idx}, CUDA {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")

    # Final norm and lm_head - process in batches of 4
    print("Processing final norm and lm_head...")
    nlls = []

    for batch_start in range(0, nsamples, batch_size_lm_head):
        batch_end = min(batch_start + batch_size_lm_head, nsamples)
        actual_batch_size = batch_end - batch_start

        # Get this batch's hidden states and move to GPU
        batch_hidden = torch.cat(all_hidden_states[batch_start:batch_end], dim=0).to(DEV)

        # Process this batch through norm and lm_head
        with torch.no_grad():
            batch_hidden = model.model.norm(batch_hidden)
            batch_logits = model.lm_head(batch_hidden)

            # Get labels for this batch - already on DEV
            batch_labels = all_target_ids[:, batch_start * model.seqlen : batch_end * model.seqlen]
            batch_labels = batch_labels.reshape(actual_batch_size, model.seqlen)

            # Calculate loss for this batch
            shift_logits = batch_logits[:, :-1, :].contiguous()
            shift_labels = batch_labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.reshape(actual_batch_size, model.seqlen - 1)
            neg_log_likelihood = loss.float().sum(dim=1)
            nlls.extend(list(neg_log_likelihood))

        # Clean up
        del batch_hidden, batch_logits, batch_labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Restore original devices
    print("Restoring layers to original devices...")
    for layer_idx, layer in enumerate(layers):
        layer.to(original_devices[layer_idx])

    # Calculate final ppl
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    tick1 = time.time()
    print(f'ppl on {eval_set} (sequential mode): {ppl.item():.4f} time: {tick1 - tick0:.2f}')
    model.config.use_cache = use_cache

    return ppl.item()


@torch.no_grad()
def qwen35_ppl_eval(model, testloader, eval_set, args):
    """Normal PPL evaluation for Qwen3.5 MoE - no manual distribution needed."""
    # Check if we should use sequential mode
    use_sequential = getattr(args, 'sequential_eval', False)
    use_standby_cpu = getattr(args, 'standby_cpu', False)

    # If standby_cpu is True, force sequential_eval for stability
    if use_standby_cpu and not use_sequential:
        print("Warning: standby_cpu enabled, forcing sequential_eval for stability.")
        use_sequential = True

    if use_sequential:
        return qwen35_ppl_eval_sequential(model, testloader, eval_set, args)

    tick0 = time.time()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    testenc = testloader.input_ids
    nsamples = testenc.shape[1] // model.seqlen
    print(f'ppl evaluation samples: {nsamples}')

    nlls = []

    for i in range(nsamples):
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
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5 MoE perplexity")
    parser.add_argument('model', type=str, help="Path to Qwen3.5 model")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--val-samples', type=int, default=256, help="Number of evaluation samples")
    parser.add_argument('--sequential-eval', action='store_true', default=False,
                        help="Use sequential PPL evaluation (keeps layers on CPU, moves one by one)")
    parser.add_argument('--standby-cpu', action='store_true', default=False,
                        help="Use CPU standby mode (load model to CPU first for large models)")
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['wikitext2', 'c4'], help="Datasets to evaluate on")

    args = parser.parse_args()

    print("=" * 80)
    print("Qwen3.5 MoE Evaluation (Phase 1: FP16 Baseline)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Sequential eval: {args.sequential_eval}")
    print(f"Standby CPU: {args.standby_cpu}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.model, standby_cpu=args.standby_cpu)

    # If in CPU standby mode, make sure everything is on CPU
    if args.standby_cpu:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Run evaluation on each dataset
    ppl_results = {}
    for dataset in args.datasets:
        print(f"\n{'=' * 80}")
        print(f"Evaluating on {dataset}")
        print(f"{'=' * 80}")

        dataloader, testloader = get_loaders(
            dataset, nsamples=args.val_samples, seed=args.seed,
            tokenizer=tokenizer, seqlen=model.seqlen
        )

        ppl = qwen35_ppl_eval(model, testloader, dataset, args)
        ppl_results[dataset] = ppl

    # Print summary
    print(f"\n{'=' * 80}")
    print("Evaluation Summary")
    print(f"{'=' * 80}")
    for dataset, ppl in ppl_results.items():
        print(f"{dataset}: {ppl:.4f}")


if __name__ == "__main__":
    import gc
    main()
