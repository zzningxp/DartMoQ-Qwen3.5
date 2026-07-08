#!/usr/bin/env python3
"""
Grouped_GEMM_MoE 量化 - 最小改动版
直接复制原 dartmoq_sequential，只做两处修改：
1. 开头把 Grouped_GEMM_MoE 层转换成传统格式
2. 注意力层支持 self_attn/linear_attn 交替
"""

import torch
import torch.nn as nn
import os
import time
import re
import gc
import sys

sys.path.insert(0, '..')

from dartmoq_utils import *
from data_utils import *
from eval_qwen35 import qwen35_ppl_eval as cmoe_ppl_eval
from qwen35_layer_reconstruct import reconstruct_moe_from_existing
from qwen35_utils import DEV, load_model

from grouped_gemm_moe_adapter import convert_grouped_gemm_to_traditional, TraditionalMoEWrapper


@torch.no_grad()
def construct_moe(model, moe_model_flag, layer, layer_idx, inp,
                    attention_mask, position_ids, position_embeddings,
                    n_experts, n_activated, slice_expert_num, ori_activated,
                    qscheme, args):
    import inspect

    modeltype = model.config.model_type
    batchsize = inp.shape[0]

    device = next(layer.parameters()).device

    # Forward attention
    inp = inp.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if position_ids is not None:
        position_ids = position_ids.to(device)

    residual = inp
    with torch.no_grad():
        hidden_states_inorm = layer.input_layernorm(inp)

    attn_out = torch.zeros_like(hidden_states_inorm)
    for b_i in range(0, batchsize):
        # ============ 改动1：支持 self_attn/linear_attn 交替 ============
        if modeltype == 'olmoe' or modeltype == 'llama' or modeltype == 'qwen3' or modeltype == 'qwen3_moe' or modeltype == 'qwen3_5' or modeltype == 'deepseek_v3':
            with torch.no_grad():
                if hasattr(layer, 'self_attn'):
                    # Use inspect to check what arguments the attention layer accepts
                    attn_forward = layer.self_attn.forward
                    forward_signature = inspect.signature(attn_forward)
                    attn_kwargs = {
                        'hidden_states': hidden_states_inorm[b_i:b_i+1]
                    }
                    if 'attention_mask' in forward_signature.parameters:
                        attn_kwargs['attention_mask'] = attention_mask
                    if 'position_ids' in forward_signature.parameters:
                        attn_kwargs['position_ids'] = position_ids
                    if 'position_embeddings' in forward_signature.parameters:
                        attn_kwargs['position_embeddings'] = position_embeddings
                    attn_out[b_i:b_i+1] = layer.self_attn(**attn_kwargs)[0]
                elif hasattr(layer, 'linear_attn'):
                    # Use inspect to check what arguments the attention layer accepts
                    attn_forward = layer.linear_attn.forward
                    forward_signature = inspect.signature(attn_forward)
                    attn_kwargs = {
                        'hidden_states': hidden_states_inorm[b_i:b_i+1]
                    }
                    if 'attention_mask' in forward_signature.parameters:
                        attn_kwargs['attention_mask'] = attention_mask
                    if 'position_ids' in forward_signature.parameters:
                        attn_kwargs['position_ids'] = position_ids
                    if 'position_embeddings' in forward_signature.parameters:
                        attn_kwargs['position_embeddings'] = position_embeddings
                    attn_out[b_i:b_i+1] = layer.linear_attn(**attn_kwargs)[0]
        else:
            with torch.no_grad():
                if hasattr(layer, 'self_attn'):
                    # Use inspect to check what arguments the attention layer accepts
                    attn_forward = layer.self_attn.forward
                    forward_signature = inspect.signature(attn_forward)
                    attn_kwargs = {
                        'hidden_states': hidden_states_inorm[b_i:b_i+1]
                    }
                    if 'attention_mask' in forward_signature.parameters:
                        attn_kwargs['attention_mask'] = attention_mask
                    if 'position_ids' in forward_signature.parameters:
                        attn_kwargs['position_ids'] = position_ids
                    if 'position_embeddings' in forward_signature.parameters:
                        attn_kwargs['position_embeddings'] = position_embeddings
                    attn_out[b_i:b_i+1] = layer.self_attn(**attn_kwargs)[0]
                elif hasattr(layer, 'linear_attn'):
                    # Use inspect to check what arguments the attention layer accepts
                    attn_forward = layer.linear_attn.forward
                    forward_signature = inspect.signature(attn_forward)
                    attn_kwargs = {
                        'hidden_states': hidden_states_inorm[b_i:b_i+1]
                    }
                    if 'attention_mask' in forward_signature.parameters:
                        attn_kwargs['attention_mask'] = attention_mask
                    if 'position_ids' in forward_signature.parameters:
                        attn_kwargs['position_ids'] = position_ids
                    if 'position_embeddings' in forward_signature.parameters:
                        attn_kwargs['position_embeddings'] = position_embeddings
                    attn_out[b_i:b_i+1] = layer.linear_attn(**attn_kwargs)[0]

    hidden_states = residual + attn_out
    residual = hidden_states
    with torch.no_grad():
        hidden_states = layer.post_attention_layernorm(hidden_states)

    is_moe_layer = hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')

    tick0 = time.time()
    use_hybrid_moe = True  # Always use hybrid mode
    quantmode = getattr(args, 'quantmode', 'gptq')
    global_mode = "global" in args.quant_scheme

    if moe_model_flag:
        if is_moe_layer:
            moe = reconstruct_moe_from_existing(model, layer, layer_idx, hidden_states,
                                                n_experts, n_activated, slice_expert_num, ori_activated, device,
                                                qscheme, use_hybrid_moe, global_mode, quantmode, args)
            layer.mlp = moe
    else:
        assert False, "Dense model is not supported"
    gc.collect()
    torch.cuda.empty_cache()
    tick1 = time.time()
    print(f"reconstruct_moe_from_existing layer {layer_idx} time: {tick1 - tick0}", flush=True)

    tick0 = time.time()
    if_quant_attn = True
    quant_layer_mix_precision(layer, layer_idx, if_quant_attn, n_experts, slice_expert_num,
                hidden_states_inorm, hidden_states, attention_mask, position_ids, position_embeddings,
                qscheme, use_hybrid_moe, quantmode, seed=args.seed)
    gc.collect()
    torch.cuda.empty_cache()
    tick1 = time.time()
    print(f"quant_layer_mix_precision layer {layer_idx} time: {tick1 - tick0}", flush=True)

    moe_out = torch.zeros_like(hidden_states)
    for b_i in range(0, batchsize):
        mlp_out = layer.mlp(hidden_states[b_i:b_i+1])
        if isinstance(mlp_out, tuple):
            moe_out[b_i:b_i+1] = mlp_out[0]
        else:
            moe_out[b_i:b_i+1] = mlp_out

    with torch.no_grad():
        moe_out = moe_out + residual

    # Clean up all intermediate tensors aggressively
    del hidden_states, hidden_states_inorm, residual, attn_out
    if 'attn_kwargs' in locals():
        del attn_kwargs
    if 'forward_signature' in locals():
        del forward_signature

    gc.collect()
    torch.cuda.empty_cache()

    return moe_out


@torch.no_grad()
def dartmoq_quant_grouped_gemm_moe(model, tokenizer, dataloader, args, test_ppl=True):
    print('Starting Grouped_GEMM_MoE quantization...')
    tick_quant_start = time.time()

    # ============ 改动2：先把 Grouped_GEMM_MoE 层转换成传统格式 ============
    print("Converting Grouped_GEMM_MoE layers to traditional format...")
    model = convert_grouped_gemm_to_traditional(model, use_gpu_acceleration=True)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = 1

    inps = torch.zeros(
        (args.nsamples//bsz, bsz, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    print(inps.shape)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    if args.standby_layer_cpu:
        model.model.embed_tokens = model.model.embed_tokens.to(DEV)
        layers[0] = layers[0].to(DEV)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])

    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(DEV))
            except ValueError:
                pass

    layers[0] = layers[0].module

    if args.standby_layer_cpu:
        layers[0] = layers[0].to('cpu')

    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    position_embeddings = cache.get('position_embeddings')
    if position_embeddings is None and hasattr(model.model, 'rotary_emb'):
        with torch.no_grad():
            for batch in dataloader:
                dummy_input = batch[0].to(DEV)
                dummy_hidden = model.model.embed_tokens(dummy_input)
                if position_ids is None:
                    position_ids = torch.arange(0, dummy_input.shape[1], device=DEV).unsqueeze(0)
                position_embeddings = model.model.rotary_emb(dummy_hidden, position_ids)
                break

    print('Ready.')

    moe_model_flag = False
    for layer in layers:
        moe_model_flag = moe_model_flag or hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')

    use_hybrid_moe = getattr(args, 'use_hybrid_moe', False)

    if moe_model_flag:
        slice_expert_num = args.slices

        if hasattr(model.config, 'num_experts'):
            ori_num_experts = model.config.num_experts
        elif hasattr(model.config, 'n_routed_experts'):
            ori_num_experts = model.config.n_routed_experts
        elif hasattr(model.model.layers[0].mlp, 'experts'):
            # 兜底方案：从第一层推断
            ori_num_experts = len(model.model.layers[0].mlp.experts)

        # 获取 top_k/num_experts_per_tok
        if hasattr(model.config, 'num_experts_per_tok'):
            ori_num_experts_per_tok = model.config.num_experts_per_tok
        elif hasattr(model.model.layers[0].mlp, 'top_k'):
            ori_num_experts_per_tok = model.model.layers[0].mlp.top_k
        else:
            ori_num_experts_per_tok = 6

        # Hybrid mode: keep original config
        new_num_expert = ori_num_experts
        new_num_experts_per_tok = ori_num_experts_per_tok

        print("The model is already a MoE model. Proceeding to create hybrid MoE structure.")
        print(f"Hybrid MoE: {ori_num_experts} experts with sub-experts sliced by {slice_expert_num}")
    else:
        assert False, "Dense model is not supported."

    inps = inps.squeeze(1)

    if args.standby_layer_cpu:
        layers_device = []
        for layer_idx, layer in enumerate(model.model.layers):
            params = list(layer.parameters())
            if params:
                dev = params[0].device
            else:
                dev = torch.device('cpu')
            layers_device.append(dev)
            if dev.type == 'cuda':
                layer = layer.to('cpu')
        for i in range(torch.cuda.device_count()):
            print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
            print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")

    qscheme_str = args.quant_scheme
    qscheme = {}
    try:
        match = re.search(r'a(\d)s(\d)m([\d.]+)', qscheme_str)
        aa = match.group(1)
        ss = match.group(2)
        ee = match.group(3)
        qscheme['attn'] = [int(aa)]
        qscheme['share'] = [int(ss)]
        if 'bpw' not in qscheme_str:
            assert len(ee) == slice_expert_num, f"Quant scheme {qscheme_str} should have {slice_expert_num} parts for expert quantization config."
            qscheme['econfig'] = [int(e) for e in ee]
            bpw = sum(qscheme['econfig']) * 1.0 / slice_expert_num
        else:
            bpw = float(ee)
            qscheme['target_bpw'] = bpw
        print(f"Quant expert scheme (ppl): {qscheme_str} with bpw {bpw} qscheme {qscheme}")
    except:
        assert False, f"Quant scheme {qscheme_str} is not valid."

    for layer_idx, layer in enumerate(layers):
        tick0 = time.time()
        print(f"\nProcessing layer {layer_idx}/{len(layers)}")

        if args.standby_layer_cpu:
            target_dev = layers_device[layer_idx] if layer_idx < len(layers_device) and layers_device[layer_idx].type == 'cuda' else DEV
            layer = layer.to(target_dev)
            if hasattr(model.model, 'rotary_emb'):
                model.model.rotary_emb = model.model.rotary_emb.to(DEV)

        moe_out = construct_moe(model,
            moe_model_flag,
            layer,
            layer_idx,
            inps,
            attention_mask,
            position_ids,
            position_embeddings,
            n_experts = new_num_expert,
            n_activated = new_num_experts_per_tok,
            slice_expert_num = slice_expert_num,
            ori_activated = ori_num_experts_per_tok,
            qscheme = qscheme,
            args = args
        )

        # Free memory from old inps before assigning new one
        del inps
        gc.collect()
        torch.cuda.empty_cache()

        inps = moe_out

        if args.standby_layer_cpu:
            layer = layer.to('cpu')

        for i in range(torch.cuda.device_count()):
            print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
            print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")

        tick1 = time.time()
        print(f"Layer {layer_idx} total reconstruct and quantization time: {tick1 - tick0:.2f} s", flush=True)
        print("." * 100, flush=True)

        # Aggressive memory cleanup after each layer
        gc.collect()
        torch.cuda.empty_cache()

    print("MoE reconstruction and quantization done.")
    tick_quant_end = time.time()
    time_quant = tick_quant_end - tick_quant_start
    print(f"Runtime of quantization only: {time_quant:.2f}")

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

    model.config.use_cache = use_cache

    if test_ppl:
        print("\nEvaluating perplexity...")
        tick_ppl_start = time.time()

        ppl_results = {}
        for dataset in ['wikitext2', 'c4']:
            print(f"\nEvaluating on {dataset}")
            _, testloader = get_loaders(
                dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen
            )

            if use_sequential:
                from eval_qwen35 import qwen35_ppl_eval_sequential
                ppl = qwen35_ppl_eval_sequential(model, testloader, dataset, args)
            else:
                ppl = cmoe_ppl_eval(model, testloader, dataset, args)

            ppl_results[dataset] = ppl
            print(f"{dataset}: {ppl:.4f}")

        tick_ppl_end = time.time()
        print(f"\nEvaluation summary")
        for dataset, ppl in ppl_results.items():
            print(f"{dataset}: {ppl:.4f}")

    return model
