#!/usr/bin/env python3
"""
Qwen3.5 MoE Evaluation Script.
Phase 1: Original FP16 perplexity evaluation with CPU standby mode support.
"""

import gc
import os
import time
import torch
import torch.nn as nn
import argparse
import sys

# Add parent directory to import data_utils
sys.path.insert(0, '..')

from qwen35_utils import load_model, DEV, print_memory_info, get_memory_info_str
from data_utils import get_loaders, get_git_hash


@torch.no_grad()
def qwen35_ppl_eval_sequential(model, testloader, eval_set, args):
    """
    Sequential PPL evaluation for Qwen3.5 MoE: keeps layers on CPU and moves
    them to GPU one by one. Hidden states stay on GPU the entire time.
    """
    tick0 = time.time()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    testenc = testloader.input_ids
    nsamples = testenc.shape[1] // model.seqlen
    print(f'ppl evaluation samples (sequential mode): {nsamples}')
    print(f'Testenc shape: {testenc.shape}')

    # Save original device for each layer
    layers = model.model.layers

    # P8 接线：把 chunk gated delta rule 换成仓库内 fast 版
    # （P8-2a pad 跳过 + P8-3 WY 闭式解，详见 roadmaps/wxa16-optimization-backlog-260824.md P8 节）
    from turboquant_utils.delta_rule import patch_delta_rule
    patch_delta_rule(model)

    # Norm 融合接线（2026-09-03）：RMSNorm / gated RMSNorm 换成 Triton 单 pass 版
    # （每层省 ~15ms GPU，对拍与集成验证见 test/test_norm_fusion.py，全部 PASS；
    #   A/B 对比时注释掉下一行即可，类级 patch 可用 unpatch_norms() 复原）
    from turboquant_utils.norm_kernels import patch_norms
    patch_norms(model)

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

    # Keep embed_tokens, norm, lm_head, and rotary_emb on GPU permanently
    print(f"Moving embed_tokens/norm/lm_head/rotary_emb to {DEV}...")
    model.model.embed_tokens = model.model.embed_tokens.to(DEV)
    if hasattr(model.model, 'norm'):
        model.model.norm = model.model.norm.to(DEV)
    if hasattr(model, 'lm_head'):
        model.lm_head = model.lm_head.to(DEV)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(DEV)

    # Verify devices
    print(f"embed_tokens device: {next(model.model.embed_tokens.parameters()).device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"CUDA {i} memory after moving: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")

    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            print(f"CUDA {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")

    # First, capture correct attention_mask, position_ids, position_embeddings using Catcher
    # Temporarily move necessary components to DEV for capturing
    print("Moving temporary components to GPU for kwargs capture...")
    layers[0] = layers[0].to(DEV)
    # Also move rotary embeddings if they exist
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(DEV)

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
    position_embeddings = captured_kwargs.get('position_embeddings')

    print(f"Captured kwargs: {list(captured_kwargs.keys())}")
    if attention_mask is not None:
        print(f"  attention_mask shape: {attention_mask.shape}")
    if position_ids is not None:
        print(f"  position_ids shape: {position_ids.shape}")
    if position_embeddings is not None:
        print(f"  position_embeddings type: {type(position_embeddings)}")

    # Restore layer 0 and move back to CPU
    layers[0] = layers[0].module
    layers[0] = layers[0].to('cpu')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get all target ids first - keep on GPU!
    all_target_ids = testenc[:, :nsamples * model.seqlen].clone().to(DEV)

    # Precompute embeddings for all samples and KEEP ON GPU!
    print("Processing embeddings and caching on GPU...")
    all_hidden_states = []
    for sample_idx in range(nsamples):
        batch = testenc[:, (sample_idx * model.seqlen):((sample_idx + 1) * model.seqlen)].to(DEV)
        with torch.no_grad():
            hidden = model.model.embed_tokens(batch)
        all_hidden_states.append(hidden)  # Keep on GPU
        del batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Concatenate into a single tensor on GPU
    hidden_states = torch.cat(all_hidden_states, dim=0)
    del all_hidden_states
    gc.collect()

    import inspect
    # Debug: Print layer types at the beginning
    print("\n=== Layer Type Debug Info ===")
    for layer_idx, layer in enumerate(layers):
        if layer_idx % 5 == 0:  # Print every 5th layer to avoid spam
            mlp_type = type(layer.mlp).__name__ if hasattr(layer, 'mlp') else 'N/A'
            print(f"  Layer {layer_idx}: mlp_type={mlp_type}, full_type={type(layer).__name__}")
    print("============================\n")

    # Batch sizes
    batch_size_transformer = 32
    batch_size_lm_head = 4

    # 统计全局时间
    total_time_forward = 0.0
    total_time_move = 0.0
    total_time_attn = 0.0
    total_time_moe = 0.0

    for layer_idx, layer in enumerate(layers):
        if layer_idx % 10 == 0:
            print(f"Processing layer {layer_idx}/{len(layers)}...", flush=True)
            # Print memory usage every 10 layers
            print_memory_info("  [Memory] ")

        tick_layer = time.time()

        # Move layer to GPU
        t_move_start = time.time()
        layer = layer.to(DEV)
        t_move_end = time.time()
        time_move = t_move_end - t_move_start
        total_time_move += time_move

        # --- Patch attn & mlp forward for per-component timing ---
        # Qwen3.5 has multiple attention types: self_attn (full_attention), linear_attn (linear_attention)
        # Auto-detect the attention attribute name
        attn_attr_name = None
        for _name in ('self_attn', 'linear_attn', 'attn', 'attention'):
            if hasattr(layer, _name):
                attn_attr_name = _name
                break
        attn_accum = [0.0]
        moe_accum = [0.0]
        _orig_attn_forward = getattr(layer, attn_attr_name).forward if attn_attr_name else None
        _orig_mlp_forward = layer.mlp.forward

        def _make_timed_forward(_orig_fn, _accum):
            def _timed_forward(*args, **kwargs):
                t0 = time.time()
                try:
                    return _orig_fn(*args, **kwargs)
                finally:
                    _accum[0] += time.time() - t0
            return _timed_forward

        if attn_attr_name:
            getattr(layer, attn_attr_name).forward = _make_timed_forward(_orig_attn_forward, attn_accum)
        layer.mlp.forward = _make_timed_forward(_orig_mlp_forward, moe_accum)
        # --- End patch ---

        # Triton kernel 预热（仅第一层做，编译缓存全局复用）
        # 提前触发所有 MoE Triton kernel 的 JIT 编译，消除第一层的冷启动开销
        if layer_idx == 0 and hasattr(layer.mlp, 'warmup_kernels'):
            _t_warmup_start = time.time()
            layer.mlp.warmup_kernels(seq_len=model.seqlen, batch_size=batch_size_transformer)
            torch.cuda.synchronize()
            _t_warmup = time.time() - _t_warmup_start
            print(f"  [Warmup] MoE Triton kernel 预编译完成: {_t_warmup:.2f}s", flush=True)

        # Process samples in batches - hidden states stay on GPU
        new_hidden_states = torch.empty_like(hidden_states)

        # --- CPU 侧瓶颈定位插桩（--cpu-profile 时启用）---
        _cpu_prof = getattr(args, 'cpu_profile', False)
        _t_kwargs_accum = 0.0
        if _cpu_prof:
            _ev0 = torch.cuda.Event(enable_timing=True)
            _ev1 = torch.cuda.Event(enable_timing=True)
            _ev0.record()

        t_forward_start = time.time()

        for batch_start in range(0, nsamples, batch_size_transformer):
            batch_end = min(batch_start + batch_size_transformer, nsamples)
            actual_batch_size = batch_end - batch_start

            # Get this batch's hidden states - already on GPU
            batch_hidden = hidden_states[batch_start:batch_end]

            # Prepare kwargs for this batch size - create on demand to save memory
            if _cpu_prof:
                _t_kw = time.time()
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

            if 'position_embeddings' in forward_signature.parameters and position_embeddings is not None:
                # Create position_embeddings for this specific batch size
                if isinstance(position_embeddings, tuple):
                    layer_kwargs['position_embeddings'] = tuple(
                        pe.repeat(actual_batch_size, 1, 1) if pe is not None else None
                        for pe in position_embeddings
                    )
                else:
                    layer_kwargs['position_embeddings'] = position_embeddings.repeat(actual_batch_size, 1, 1)

            if _cpu_prof:
                _t_kwargs_accum += time.time() - _t_kw

            # Forward pass
            with torch.no_grad():
                layer_outputs = layer(batch_hidden, **layer_kwargs)

            # Save output
            if isinstance(layer_outputs, tuple):
                batch_output = layer_outputs[0]
                # Cleanup any additional outputs in the tuple
                for extra_output in layer_outputs[1:]:
                    del extra_output
            else:
                batch_output = layer_outputs

            # Directly copy to new_hidden_states - everything on GPU
            new_hidden_states[batch_start:batch_end].copy_(batch_output)

            # Cleanup batch variables aggressively
            del batch_hidden, layer_outputs, batch_output
            del layer_kwargs

        t_forward_end = time.time()
        time_forward = t_forward_end - t_forward_start
        total_time_forward += time_forward

        # Restore original forward methods and collect per-component times
        if attn_attr_name:
            getattr(layer, attn_attr_name).forward = _orig_attn_forward
        layer.mlp.forward = _orig_mlp_forward
        time_attn = attn_accum[0]
        time_moe = moe_accum[0]
        total_time_attn += time_attn
        total_time_moe += time_moe

        # Swap hidden states pointers
        del hidden_states
        hidden_states = new_hidden_states
        del new_hidden_states

        # --- CPU 侧瓶颈定位插桩：逐层打印 ---
        if _cpu_prof:
            _ev1.record()
            torch.cuda.synchronize()  # 测量模式才 sync，正式 eval 不要开此 flag
            _gpu_ms = _ev0.elapsed_time(_ev1)
            _wall_ms = (time.time() - tick_layer) * 1000
            print(f"  [CPU-PROF] layer {layer_idx}: wall {_wall_ms:7.1f} ms | "
                  f"gpu {_gpu_ms:7.1f} ms | gap {_wall_ms - _gpu_ms:7.1f} ms | "
                  f"move {time_move * 1000:6.1f} ms | kwargs {_t_kwargs_accum * 1000:6.1f} ms | "
                  f"forward {time_forward * 1000:7.1f} ms | "
                  f"attn {time_attn * 1000:6.1f} ms | moe {time_moe * 1000:6.1f} ms",
                  flush=True)

        # Move layer back to CPU to free GPU memory (in-place, modifies layers[layer_idx])
        layer = layer.to('cpu')

        # Debug: Print layer timing and memory info every 10 layers or first 10 layers
        if layer_idx < 50 or layer_idx % 10 == 0:
            layer_time = time.time() - tick_layer
            mlp_type = type(layer.mlp).__name__ if hasattr(layer, 'mlp') else 'N/A'

            attn_type = attn_attr_name if attn_attr_name else 'N/A'
            mem_str = get_memory_info_str()
            print(f"  [Layer {layer_idx}] time: {layer_time:.2f}s (move: {time_move:.2f}s, forward: {time_forward:.2f}s, attn: {time_attn:.2f}s, moe: {time_moe:.2f}s), attn_type={attn_type}, mlp_type={mlp_type} | Memory: {mem_str}", flush=True)

        del layer
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 打印全局时间统计
    print(f"\n  [Eval Global Stats]", flush=True)
    print(f"    Total move time: {total_time_move:.2f}s", flush=True)
    print(f"    Total forward time: {total_time_forward:.2f}s", flush=True)
    print(f"    Total attn time: {total_time_attn:.2f}s", flush=True)
    print(f"    Total moe time: {total_time_moe:.2f}s", flush=True)

    # Final norm and lm_head - process in batches of 4
    print("Processing final norm and lm_head...")
    nlls = []

    for batch_start in range(0, nsamples, batch_size_lm_head):
        batch_end = min(batch_start + batch_size_lm_head, nsamples)
        actual_batch_size = batch_end - batch_start

        # Get this batch's hidden states - already on GPU
        batch_hidden = hidden_states[batch_start:batch_end]

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

    # 全模型在 GPU 上的 batch 化 eval：
    #   transformer 层用较大 batch（隐状态小，~16MB/sample），
    #   lm_head 分批算（logits 巨大，~600MB/sample 会 OOM）。
    # 直接调用 model.model(input_ids)，不传 attention_mask/position_ids，
    # 让内部自动生成，与 model(batch) 行为完全一致，保证 PPL 正确。
    # 输入是规整的 batch × seqlen（无 padding），不需要 mask。
    batch_size = getattr(args, 'eval_batch_size', 32)
    lm_head_batch_size = 4
    nlls = []
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    for batch_start in range(0, nsamples, batch_size):
        batch_end = min(batch_start + batch_size, nsamples)
        actual_bs = batch_end - batch_start
        if batch_start % 10 == 0 or batch_start == 0:
            print(f"  sample {batch_start}/{nsamples} (bs={actual_bs})...", flush=True)

        # [1, B*seqlen] → [B, seqlen]
        batch = testenc[:, (batch_start * model.seqlen):(batch_end * model.seqlen)].to(DEV)
        batch = batch.view(actual_bs, model.seqlen)

        with torch.no_grad():
            # 走 model.model（不含 lm_head），mask 和 pos_emb 内部自动生成
            transformer_out = model.model(batch)
            hidden = transformer_out[0] if isinstance(transformer_out, tuple) else transformer_out.last_hidden_state

            # lm_head 分批算 loss，避免 logits OOM
            shift_labels_all = batch[:, 1:].contiguous()
            for lm_start in range(0, actual_bs, lm_head_batch_size):
                lm_end = min(lm_start + lm_head_batch_size, actual_bs)
                h = hidden[lm_start:lm_end, :-1, :].contiguous()
                logits = model.lm_head(h)
                shift_labels = shift_labels_all[lm_start:lm_end, :].contiguous()
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = loss.view(lm_end - lm_start, model.seqlen - 1)
                neg_log_likelihood = loss.float().sum(dim=1)
                nlls.extend(neg_log_likelihood.unbind())
                del logits, h

            del hidden, transformer_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    tick1 = time.time()
    print(f'ppl on {eval_set}: {ppl.item():.4f} time: {tick1 - tick0:.2f}')
    model.config.use_cache = use_cache

    return ppl.item()


def run_ppl_evaluation(model, tokenizer, args):
    """
    wikitext2/c4 双数据集 PPL eval：sequential 判断 + 结果打印。

    量化路径（qwen35_simple_wrapper）与 checkpoint 加载路径共用，
    避免同一段 eval 逻辑重复实现。
    """
    use_sequential = getattr(args, 'sequential_eval', False)
    use_standby = getattr(args, 'standby_layer_cpu', False)

    # If standby is enabled, force sequential eval for stability
    if use_standby and not use_sequential:
        print("Warning: standby-layer-cpu enabled, forcing sequential-eval for stability.")
        use_sequential = True

    if use_sequential:
        print("Will use sequential PPL evaluation (layers stay on CPU)")
    else:
        print("Will use normal PPL evaluation")

    tick_ppl_start = time.time()
    ppl_results = {}
    for dataset in ['wikitext2', 'c4']:
        print(f"\nEvaluating on {dataset}")
        _, testloader = get_loaders(
            dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen,
            eval_only=True,
        )

        if use_sequential:
            ppl = qwen35_ppl_eval_sequential(model, testloader, dataset, args)
        else:
            ppl = qwen35_ppl_eval(model, testloader, dataset, args)

        ppl_results[dataset] = ppl
        print(f"{dataset}: {ppl:.4f}")

    tick_ppl_end = time.time()
    print(f"\nEvaluation summary")
    for dataset, ppl in ppl_results.items():
        print(f"{dataset}: {ppl:.4f}")
    print(f"PPL evaluation total time: {tick_ppl_end - tick_ppl_start:.2f}s")

    return ppl_results


def _is_quant_dir(path: str) -> bool:
    """判断一个目录是否为量化 checkpoint 目录（含 meta.json + model.safetensors）。"""
    if not path or not os.path.isdir(path):
        return False
    return (os.path.isfile(os.path.join(path, "meta.json"))
            and os.path.isfile(os.path.join(path, "model.safetensors")))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5 MoE perplexity")
    parser.add_argument('model', type=str, nargs='?', default=None,
                        help="Path to Qwen3.5 model (optional when --load-quantized is used)")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--val-samples', type=int, default=256, help="Number of evaluation samples")
    parser.add_argument('--sequential-eval', action='store_true', default=False,
                        help="Use sequential PPL evaluation (keeps layers on CPU, moves one by one)")
    parser.add_argument('--standby-cpu', action='store_true', default=False,
                        help="Use CPU standby mode (load model to CPU first for large models)")
    parser.add_argument('--load-quantized', type=str, default=None,
                        help='Load quantized checkpoint from this directory (skip fp16 model loading)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['wikitext2', 'c4'], help="Datasets to evaluate on")
    parser.add_argument('--eval-batch-size', type=int, default=32,
                        help="Batch size for normal (non-sequential) PPL evaluation")
    parser.add_argument('--cpu-profile', action='store_true', default=False,
                        help="CPU 侧瓶颈定位：逐层打印 wall/GPU/kwargs/move 拆分"
                             "（会在每层末尾 synchronize，只用于测量，别开它跑正式 eval）")
    parser.add_argument('--inference-quant-mode', type=str, default='wxa16',
                        choices=['wxa16', 'wxa8'],
                        help="推理量化模式：wxa16 (FP16 激活+FP16 计算，默认) / "
                             "wxa8 (INT8 激活+INT8 Tensor Core，MoE 部分；"
                             "attention 保持 W8A16 直到 P3)")

    args = parser.parse_args()

    # 自动判断：单位置参数若为量化目录，则等价于 --load-quantized
    if args.model and not args.load_quantized and _is_quant_dir(args.model):
        args.load_quantized = args.model
        args.model = None

    if not args.load_quantized and not args.model:
        parser.error("model path is required when not using --load-quantized")

    print("Qwen3.5 MoE Evaluation")
    git_hash = get_git_hash()
    print(f"Git HEAD: {git_hash}")
    if args.load_quantized:
        print(f"Quantized checkpoint: {args.load_quantized}")
        print(f"Inference quant mode: {args.inference_quant_mode}")
        if args.model:
            print(f"Base model (config/tokenizer): {args.model}")
    else:
        print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Sequential eval: {args.sequential_eval}")
    print(f"Standby CPU: {args.standby_cpu}")

    # Load model（fp16 原模型 或 量化 checkpoint，加载逻辑复用 qwen35_quant_io）
    if args.load_quantized:
        print("\nLoading quantized checkpoint (skip fp16 model loading)...")
        from qwen35_quant_io import load_quantized_model
        model, tokenizer = load_quantized_model(
            args.model, args.load_quantized, standby_cpu=args.standby_cpu,
            inference_quant_mode=args.inference_quant_mode)
    else:
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
        print(f"\nEvaluating on {dataset}")

        print(f"Loading {dataset} dataset... (this may take a while)")
        tick_data = time.time()
        _, testloader = get_loaders(
            dataset, nsamples=args.val_samples, seed=args.seed,
            tokenizer=tokenizer, seqlen=model.seqlen, eval_only=True,
        )
        print(f"Dataset loaded in {time.time() - tick_data:.2f}s")
        print(f"Test data shape: {testloader.input_ids.shape}")

        print("Starting PPL evaluation...")
        ppl = qwen35_ppl_eval(model, testloader, dataset, args)
        ppl_results[dataset] = ppl

    # Print summary
    print(f"\nEvaluation Summary")
    for dataset, ppl in ppl_results.items():
        print(f"{dataset}: {ppl:.4f}")


if __name__ == "__main__":
    import gc
    main()
