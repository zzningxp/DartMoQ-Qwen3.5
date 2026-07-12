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
from qwen35_utils import DEV, load_model, print_memory_info, get_memory_info_str

from grouped_gemm_moe_adapter import convert_grouped_gemm_to_traditional, TraditionalMoEWrapper
from bit_partitioned_moe import BitPartitionedGroupMoE


def parse_quant_layers(quant_layers_str):
    """
    Parse layer selection string.
    Examples:
        '0-5,8,10' -> {0, 1, 2, 3, 4, 5, 8, 10}
        None or '' -> None (quantize all layers)
    """
    if quant_layers_str is None or quant_layers_str.strip() == '':
        return None
    layer_set = set()
    parts = quant_layers_str.strip().split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            try:
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                layer_set.update(range(start_idx, end_idx + 1))
            except ValueError:
                print(f"Warning: invalid layer range '{part}', skipping")
        else:
            try:
                layer_idx = int(part)
                layer_set.add(layer_idx)
            except ValueError:
                print(f"Warning: invalid layer index '{part}', skipping")
    return layer_set


def should_quantize_layer(layer_idx, quant_layers_set):
    """
    Determine if a layer should be quantized.
    Returns True if quant_layers_set is None (quantize all) or layer_idx is in the set.
    """
    if quant_layers_set is None:
        return True
    return layer_idx in quant_layers_set


@torch.no_grad()
def construct_moe(model, moe_model_flag, layer, layer_idx, inp,
                    attention_mask, position_ids, position_embeddings,
                    n_experts, n_activated, slice_expert_num, ori_activated,
                    qscheme, args):
    import inspect
    from grouped_gemm_moe_adapter import convert_single_layer, is_grouped_gemm_moe_layer

    # Convert this single layer first if needed
    if is_grouped_gemm_moe_layer(layer):
        print(f"  Converting layer {layer_idx} to traditional format...")
        tick_convert = time.time()
        layer, _ = convert_single_layer(layer)
        tick_convert_end = time.time()
        print(f"  Layer {layer_idx} converted in {tick_convert_end - tick_convert:.3f}s")

    # Normal quantization path - always quantize when this function is called
    modeltype = model.config.model_type
    batchsize = inp.shape[0]
    print(f"  [DEBUG] construct_moe: batchsize={batchsize}")

    device = DEV  # inp is already on DEV (GPU)

    # Forward attention
    tick_attn = time.time()
    # inp is already on GPU, no need to move!
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEV)

    if position_ids is not None:
        position_ids = position_ids.to(DEV)

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

    time_attn = time.time() - tick_attn
    print(f"  [DEBUG] Attention forward time: {time_attn:.2f}s")

    is_moe_layer = hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')

    tick0 = time.time()
    use_hybrid_moe = True  # Always use hybrid mode
    quantmode = getattr(args, 'quantmode', 'gptq')
    global_mode = "global" in args.quant_scheme

    if moe_model_flag:
        if is_moe_layer:
            moe, layer_metadata = reconstruct_moe_from_existing(model, layer, layer_idx,
                                                hidden_states,
                                                n_experts, n_activated, slice_expert_num,
                                                ori_activated, device,
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
    quant_layer_mix_precision(layer, layer_idx, if_quant_attn,
                              n_experts, slice_expert_num,
                              hidden_states_inorm, hidden_states,
                              attention_mask, position_ids, position_embeddings,
                              qscheme, use_hybrid_moe, quantmode, seed=args.seed)
    gc.collect()
    torch.cuda.empty_cache()
    tick1 = time.time()
    print(f"quant_layer_mix_precision layer {layer_idx} time: {tick1 - tick0:.4f}", flush=True)

    if moe_model_flag and is_moe_layer:
        # 重组为 BitPartitionedGroupMoE
        tick_restructure = time.time()
        print(f"Restructuring to BitPartitionedGroupMoE (layer {layer_idx})...")

        # 获取旧结构的引用，以便后续清理
        old_mlp = layer.mlp

        # 替换为新结构
        layer.mlp = BitPartitionedGroupMoE.from_build_block(old_mlp, layer_metadata)

        # 显式清理旧结构，释放内存
        if hasattr(old_mlp, 'experts'):
            for expert_wrapper in old_mlp.experts:
                if hasattr(expert_wrapper, 'sub_experts'):
                    # 清理每个子专家
                    for sub_expert in expert_wrapper.sub_experts:
                        if hasattr(sub_expert, 'gate_proj'):
                            del sub_expert.gate_proj
                        if hasattr(sub_expert, 'up_proj'):
                            del sub_expert.up_proj
                        if hasattr(sub_expert, 'down_proj'):
                            del sub_expert.down_proj
                    del expert_wrapper.sub_experts
                # 清理 wrapper 自身的属性
                if hasattr(expert_wrapper, 'bit_to_indices'):
                    del expert_wrapper.bit_to_indices
            del old_mlp.experts

        # 清理 old_mlp 的其他可能属性
        if hasattr(old_mlp, 'gate'):
            del old_mlp.gate  # 注意：gate 实际上被新结构复用了，所以我们只删除引用，不删除对象本身
        if hasattr(old_mlp, 'shared_expert'):
            del old_mlp.shared_expert  # 同理，shared_expert 也被复用
        if hasattr(old_mlp, 'shared_expert_gate'):
            del old_mlp.shared_expert_gate

        # 删除 old_mlp 引用
        del old_mlp

        # 强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache()

        tick_restructure_end = time.time()
        print(f"Restructured in {tick_restructure_end - tick_restructure:.4f}s")

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

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = 1

    # Hidden states stay on GPU!
    inps = torch.zeros(
        (args.nsamples//bsz, bsz, model.seqlen, model.config.hidden_size), dtype=dtype, device=DEV
    )
    # print(inps.shape)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    if args.standby_layer_cpu:
        model.model.embed_tokens = model.model.embed_tokens.to(DEV)
        layers[0] = layers[0].to(DEV)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp  # Keep on GPU!
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

    attention_mask = cache['attention_mask'].to(DEV) if cache['attention_mask'] is not None else None
    position_ids = cache['position_ids'].to(DEV) if cache['position_ids'] is not None else None

    position_embeddings = cache.get('position_embeddings')
    if position_embeddings is not None:
        if isinstance(position_embeddings, tuple):
            position_embeddings = tuple(pe.to(DEV) if pe is not None else None for pe in position_embeddings)
        else:
            position_embeddings = position_embeddings.to(DEV)
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

        # 从第一层获取必要的 config 信息（不需要 convert，直接读）
        first_mlp = model.model.layers[0].mlp
        if hasattr(model.config, 'num_experts'):
            ori_num_experts = model.config.num_experts
        elif hasattr(model.config, 'n_routed_experts'):
            ori_num_experts = model.config.n_routed_experts
        elif hasattr(first_mlp, 'experts'):
            if hasattr(first_mlp.experts, 'gate_up_proj'):
                ori_num_experts = first_mlp.experts.gate_up_proj.shape[0]
            else:
                ori_num_experts = len(first_mlp.experts)

        # 获取 top_k/num_experts_per_tok
        if hasattr(model.config, 'num_experts_per_tok'):
            ori_num_experts_per_tok = model.config.num_experts_per_tok
        elif hasattr(first_mlp, 'top_k'):
            ori_num_experts_per_tok = first_mlp.top_k
        else:
            ori_num_experts_per_tok = 6

        # 确保 config 有必要的属性
        if not hasattr(model.config, 'num_experts') and not hasattr(model.config, 'n_routed_experts'):
            model.config.num_experts = ori_num_experts
        if not hasattr(model.config, 'moe_intermediate_size'):
            if hasattr(model.config, 'intermediate_size'):
                model.config.moe_intermediate_size = model.config.intermediate_size
            elif hasattr(first_mlp.experts, 'gate_up_proj'):
                model.config.moe_intermediate_size = first_mlp.experts.gate_up_proj.shape[1] // 2
        if not hasattr(model.config, 'intermediate_size'):
            model.config.intermediate_size = model.config.moe_intermediate_size
        if not hasattr(model.config, 'num_experts_per_tok'):
            model.config.num_experts_per_tok = ori_num_experts_per_tok

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
        # for i in range(torch.cuda.device_count()):
        #     print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
        #     print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")

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

    # Parse quant layers selection
    quant_layers_set = parse_quant_layers(getattr(args, 'quant_layers', None))
    if quant_layers_set is not None:
        print(f"Quantizing layers: {sorted(quant_layers_set)}")
        # Ensure quant layers are continuous starting from 0
        sorted_layers = sorted(quant_layers_set)
        assert sorted_layers[0] == 0, f"Quant layers must start from 0, got {sorted_layers[0]}"
        for i in range(1, len(sorted_layers)):
            assert sorted_layers[i] == sorted_layers[i-1] + 1, f"Quant layers must be continuous, got gap between {sorted_layers[i-1]} and {sorted_layers[i]}"
    else:
        print("Quantizing all layers")

    # Ensure model.model_id exists (protect against missing)
    if not hasattr(model, 'model_id') or not model.model_id:
        if hasattr(model.config, '_name_or_path') and model.config._name_or_path:
            model.model_id = str(model.config._name_or_path).split('/')[-1]
        elif hasattr(args, 'model') and args.model:
            model.model_id = str(args.model).split('/')[-1]
        else:
            model.model_id = "qwen35_model"
        print(f"Set model.model_id to: {model.model_id}")

    for layer_idx, layer in enumerate(layers):
        # Determine if this layer should be quantized, skip layer for fast algorithm tests
        quantize_this_layer = should_quantize_layer(layer_idx, quant_layers_set)

        if not quantize_this_layer:
            print(f"Skipping layer {layer_idx} and all remaining layers...", flush=True)
            # Debug: Check the state of this layer and next layer before breaking
            if layer_idx < len(layers):
                mlp_type = type(layer.mlp).__name__ if hasattr(layer, 'mlp') else 'N/A'
                has_gate_up = hasattr(layer.mlp, 'experts') and hasattr(layer.mlp.experts, 'gate_up_proj') if hasattr(layer, 'mlp') else False
                print(f"  [DEBUG] Layer {layer_idx} before break: mlp_type={mlp_type}, has_gate_up_proj={has_gate_up}")
            if layer_idx + 1 < len(layers):
                next_layer = layers[layer_idx + 1]
                mlp_type_next = type(next_layer.mlp).__name__ if hasattr(next_layer, 'mlp') else 'N/A'
                has_gate_up_next = hasattr(next_layer.mlp, 'experts') and hasattr(next_layer.mlp.experts, 'gate_up_proj') if hasattr(next_layer, 'mlp') else False
                print(f"  [DEBUG] Layer {layer_idx + 1} (untouched): mlp_type={mlp_type_next}, has_gate_up_proj={has_gate_up_next}")
            break

        tick0 = time.time()
        print(f"\nProcessing layer {layer_idx}/{len(layers)}")
        if layer_idx % 10 == 0:
            print_memory_info("  [Memory] ")

        if args.standby_layer_cpu:
            target_dev = layers_device[layer_idx] if layer_idx < len(layers_device) and layers_device[layer_idx].type == 'cuda' else DEV
            layers[layer_idx] = layers[layer_idx].to(target_dev)  # Directly modify layers list
            layer = layers[layer_idx]
            if hasattr(model.model, 'rotary_emb'):
                model.model.rotary_emb = model.model.rotary_emb.to(DEV)

        # Quantize this layer - use construct_moe
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
            layers[layer_idx] = layers[layer_idx].to('cpu')  # Directly modify layers list
            del layer  # Clean up reference to help GPU memory release
            gc.collect()
            torch.cuda.empty_cache()

        # for i in range(torch.cuda.device_count()):
        #     print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
        #     print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")

        tick1 = time.time()
        mem_str = get_memory_info_str()
        print(f"Layer {layer_idx} total reconstruct and quantization time: {tick1 - tick0:.2f} s | Memory: {mem_str}", flush=True)
        print("." * 100, flush=True)

        # Aggressive memory cleanup after each layer
        gc.collect()
        torch.cuda.empty_cache()

    print("MoE reconstruction and quantization done.")

    # Debug: Print final layer states
    print("\n=== Final Layer States Debug Info ===")
    for layer_idx, layer in enumerate(layers):
        if layer_idx < 10 or layer_idx % 5 == 0:  # Print first 10 and every 5th
            mlp_type = type(layer.mlp).__name__ if hasattr(layer, 'mlp') else 'N/A'
            has_gate_up = hasattr(layer.mlp, 'experts') and hasattr(layer.mlp.experts, 'gate_up_proj') if hasattr(layer, 'mlp') else False
            print(f"  Layer {layer_idx}: mlp_type={mlp_type}, has_gate_up_proj={has_gate_up}")
    print("=====================================\n")

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
