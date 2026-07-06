#!/usr/bin/env python3
"""
Qwen3.5 最简单的适配方案：
直接复制原 DartMoQ 的核心函数，只修改必要的几行代码
"""

import torch
import torch.nn as nn
import sys
import os
import re
import time
import gc
from collections import Counter

sys.path.insert(0, '..')

from data_utils import get_loaders
from eval_qwen35 import qwen35_ppl_eval as cmoe_ppl_eval
from qwen35_utils import DEV, load_model


# ============ 直接复制原 DartMoQ 的 dartmoq_sequential.py，只做最小修改 ============

from dartmoq_utils import quant_layer_mix_precision
from dartmoq_utils import analyze_experts_activation
from dartmoq_utils import construct_experts_by_rates
from dartmoq_utils import analyze_neuron_activations
from dartmoq_utils import analyze_gptq_quant_outlier
from dartmoq_utils import analyze_turboquant_outlier_activation_aware
from camera_utils import analyze_expert_energy
from dp_utils import enum_optimal_m_scheme_separate_fast
from dp_utils import enum_optimal_m_scheme_global_fast
from dp_utils import enum_optimal_m_scheme_energy_global_fast
from dp_utils import extrapolate_0bit_loss_fix

from dartmoq_hybridmoe import DartMoQHybridWrapper
from dartmoq_hybridmoe import restructure_hybrid_qscheme


INTERMEDIATE_RESULT_DIR = "intermediate_result"


@torch.no_grad()
def qwen35_reconstruct_moe_from_existing(model, layer, layer_idx, inps,
                                         n_experts, n_activated, slice_expert_num,
                                         ori_activated, device, qscheme,
                                         use_hybrid_moe, global_mode, quantmode, args):
    """
    Qwen3.5 版本的 reconstruct_moe_from_existing
    主要改动：
    1. 支持 Qwen3.5 的 self_attn/linear_attn 交替
    2. 支持 Qwen3.5 的合并权重格式 gate_up_proj/down_proj
    """

    if global_mode:
        cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, "expert_activate", model.model_id)
        cache_path = os.path.join(cache_dir, f"{model.model_id}_L{layer_idx}.pt")
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(cache_path):
            try:
                expert_activation_rates = torch.load(cache_path, map_location="cpu")
                expert_activation_rates = torch.as_tensor(expert_activation_rates).detach().cpu()
                print(f"Loading cached expert activation rates for layer {layer_idx}", flush=True)
            except Exception as e:
                print(f"Failed to load cached expert activation rates {e}")
                expert_activation_rates = analyze_experts_activation(layer, layer_idx, inps, ori_activated, model.config.model_type)
                torch.save(expert_activation_rates.detach().cpu(), cache_path)
                print(f"Saved expert activation rates to {cache_path}")
        else:
            expert_activation_rates = analyze_experts_activation(layer, layer_idx, inps, ori_activated, model.config.model_type)
            torch.save(expert_activation_rates.detach().cpu(), cache_path)
            print(f"Saved expert activation rates to {cache_path}")

    # 获取专家数量 - 支持多种配置
    if hasattr(model.config, 'num_experts'):
        ori_expert_num = model.config.num_experts
    elif hasattr(model.config, 'n_routed_experts'):
        ori_expert_num = model.config.n_routed_experts
    elif hasattr(layer.mlp, 'experts'):
        # Qwen3.5: 从 gate_up_proj 形状推断
        if hasattr(layer.mlp.experts, 'gate_up_proj'):
            ori_expert_num = layer.mlp.experts.gate_up_proj.shape[0]
        else:
            ori_expert_num = len(layer.mlp.experts)
    else:
        ori_expert_num = 0

    if use_hybrid_moe:
        new_expert_num = ori_expert_num
    else:
        new_expert_num = ori_expert_num * slice_expert_num
        scaling_factor = slice_expert_num

    ori_router_gate = layer.mlp.gate.weight

    if use_hybrid_moe:
        all_new_experts = []
    else:
        if type(layer.mlp.gate) == nn.Linear:
            new_router = nn.Linear(model.config.hidden_size, new_expert_num, dtype=ori_router_gate.dtype, bias=False).to(device)
        else:
            new_router = layer.mlp.gate.__class__(model.config).to(device).to(layer.mlp.gate.weight.dtype)
        all_new_experts = nn.ModuleList()

    total_neurons_processed = 0
    gate_start_idx = 0

    sub_expert_bit_configs = []
    expert_to_subexperts = []

    probe_bit = 2
    dpscheme_list = None
    turboquant_outlier_modes = {
        "turboquant_iipl_fea": "iipl",
        "turboquant_iipl": "iipl",
        "turboquant_innerproduct_fea": "innerproduct",
        "turboquant_innerproduct": "innerproduct",
        "turboquant_diagonal": "diagonal",
        "turboquant_hessian": "hessian",
        "turboquant_qjl_sensitivity": "qjl_sensitivity",
        "turboquant_mse": "mse",
        "turboquant_mse_fea": "mse",
    }
    if args.rank_mode == "gptq_quant_outlier" or args.rank_mode in turboquant_outlier_modes:
        tick0 = time.time()
        turboquant_outlier_mode = turboquant_outlier_modes.get(args.rank_mode, "")

        q_rates = {}
        if 'target_bpw' not in qscheme:
            outlier_bits = {probe_bit}
        else:
            if getattr(args, 'disable_0bit_prune', False):
                outlier_bits = {1, 2, 3, 4}
            else:
                outlier_bits = {0, 1, 2, 3, 4}
        outlier_label = args.rank_mode if turboquant_outlier_mode else quantmode
        print(f"simulate {outlier_label} outlier_bits {outlier_bits}")

        cache_root = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{quantmode}")
        cache_dir = os.path.join(cache_root, args.rank_mode, model.model_id)
        os.makedirs(cache_dir, exist_ok=True)

        for x in sorted(outlier_bits, reverse=True):
            cache_path = os.path.join(cache_dir, f"{model.model_id}_L{layer_idx}_b{x}.pt")
            if os.path.exists(cache_path):
                try:
                    cached_data = torch.load(cache_path, map_location=device)
                    print(f"Loading cached {outlier_label} outlier data for layer {layer_idx}, wbits={x}", flush=True)
                    q_rates[x] = cached_data
                    continue
                except Exception as e:
                    print(f"Failed to load cached data {e}")
            if x == 0:
                print(f"Computing extrapolate 0 bit loss for layer {layer_idx}")
                q_rates[0] = extrapolate_0bit_loss_fix(q_rates, quant_type=outlier_label)
                q_rates[0] = [torch.from_numpy(q_rates[0][i]).to(device) for i in range(len(q_rates[0]))]
            else:
                print(f"Computing {outlier_label} outlier for layer {layer_idx}, wbits={x}, with inps shape {inps.shape}")
                if turboquant_outlier_mode:
                    q_rates[x] = analyze_turboquant_outlier_activation_aware(
                        layer, layer_idx, inps, ori_expert_num,
                        wbits=x, mode=turboquant_outlier_mode,
                        save_path=None, use_activation_hooks=not args.rank_mode.endswith("_fea"),
                        seed=args.seed)
                else:
                    q_rates[x] = analyze_gptq_quant_outlier(
                        layer, layer_idx, inps, ori_expert_num, wbits=x,
                        quantmode=quantmode, save_path=None, seed=args.seed)
            torch.save(q_rates[x], cache_path)
            print(f"Saved {outlier_label} outlier data to {cache_path}")

        if 'target_bpw' not in qscheme:
            all_rates = q_rates[probe_bit]
        else:
            if global_mode:
                expert_rates_list = []
                for expert_idx in range(ori_expert_num):
                    rates_x = {}
                    for x in outlier_bits:
                        rates_x[x] = q_rates[x][expert_idx].detach().cpu().float().numpy()
                    expert_rates_list.append(rates_x)

                dp_tick0 = time.time()
                dpscheme_list, all_rates_arr = enum_optimal_m_scheme_global_fast(
                    expert_rates_list,
                    expert_activation_rates,
                    slice_expert_num,
                    target_bpw=qscheme['target_bpw'],
                    enable_0bit_compensation=not getattr(args, 'disable_0bit_compensation', False)
                )
                dp_tick1 = time.time()
                print(f"enum_optimal_m_scheme_global_fast time {dp_tick1 - dp_tick0}", flush=True)

                all_rates = []
                for expert_idx in range(ori_expert_num):
                    rates_arr = all_rates_arr[expert_idx]
                    all_rates.append(torch.from_numpy(rates_arr).to(device))

                print(f"built dpscheme_list target_bpw {qscheme['target_bpw']} for {ori_expert_num} experts")
            else:
                all_rates = []
                dpscheme_list = []
                for expert_idx in range(ori_expert_num):
                    rates_x = {}
                    for x in outlier_bits:
                        rates_x[x] = q_rates[x][expert_idx].detach().cpu().float().numpy()
                    dpscheme, rates = enum_optimal_m_scheme_separate_fast(rates_x, slice_expert_num, target_bpw=qscheme['target_bpw'])
                    dpscheme_list.append(dpscheme)
                    rates = torch.from_numpy(rates).to(device)
                    all_rates.append(rates)

        tick1 = time.time()
        print(f"analyze quant outlier time {tick1 - tick0}", flush=True)
    elif args.rank_mode == "energy" and 'target_bpw' in qscheme and global_mode:
        tick0 = time.time()
        print(f"Energy mode with target_bpw {qscheme['target_bpw']}")

        expert_energy_list = []
        for expert_idx, expert in enumerate(layer.mlp.experts):
            energy = analyze_expert_energy(expert, inps)
            expert_energy_list.append(energy.detach().cpu().float().numpy())

        if getattr(args, 'disable_0bit_prune', False):
            energy_bits = [1, 2, 3, 4]
        else:
            energy_bits = [0, 1, 2, 3, 4]
        dpscheme_list, all_rates_arr = enum_optimal_m_scheme_energy_global_fast(
            expert_energy_list,
            expert_activation_rates,
            slice_expert_num,
            target_bpw=qscheme['target_bpw'],
            bits=energy_bits,
            enable_0bit_compensation=not getattr(args, 'disable_0bit_compensation', False)
        )

        print(f"built dpscheme_list for energy mode target_bpw {qscheme['target_bpw']} for {ori_expert_num} experts")
        tick1 = time.time()
        print(f"energy mode analyze time {tick1 - tick0}", flush=True)

    tick0 = time.time()

    all_new_expert_rates = []
    all_expert_groups = []

    # ============ Qwen3.5 关键改动：临时转换合并权重为传统格式 ============
    need_restore = False
    original_mlp = None
    if hasattr(layer.mlp.experts, 'gate_up_proj'):
        # Qwen3.5 合并权重格式
        from qwen35_utils import convert_qwen35_to_traditional
        original_mlp = layer.mlp
        layer.mlp = convert_qwen35_to_traditional(layer)
        need_restore = True

    for expert_idx, expert in enumerate(layer.mlp.experts):
        if args.rank_mode == "expert_activation":
            ori_gate_proj_weights = expert.gate_proj.weight
            ori_up_proj_weights = expert.up_proj.weight
            ori_down_proj_weights = expert.down_proj.weight

            analyze_sparsity = 0.1
            rates = analyze_neuron_activations(expert.act_fn, inps, ori_gate_proj_weights, ori_up_proj_weights, sparsity=analyze_sparsity)
        elif args.rank_mode == "energy":
            rates = analyze_expert_energy(expert, inps)
        elif args.rank_mode == "gptq_quant_outlier" or args.rank_mode in turboquant_outlier_modes:
            rates = all_rates[expert_idx]
        elif args.rank_mode == "random":
            rates = torch.randn(expert.gate_proj.weight.shape[0], device=device)
        elif args.rank_mode == "neuron_index":
            rates = torch.arange(expert.gate_proj.weight.shape[0], device=device)
        else:
            assert False, f"Unknown rank mode: {args.rank_mode}"

        expert_groups, expert_rates = construct_experts_by_rates(
            rates, num_experts=slice_expert_num
        )
        expert_groups = expert_groups[1:]
        all_expert_groups.append(expert_groups)

        if global_mode:
            _rates = [e * expert_activation_rates[expert_idx] for e in expert_rates[1:]]
            all_new_expert_rates.extend(_rates)
        else:
            all_new_expert_rates.extend(expert_rates[1:])

    if 'target_bpw' in qscheme and dpscheme_list is not None:
        qscheme['expert'] = dpscheme_list
    elif global_mode:
        ee = qscheme['econfig']
        e_bits = [int(e) for e in ee]

        if all_new_expert_rates is not None:
            _, sorted_index = torch.sort(torch.tensor(all_new_expert_rates), descending=True)
            qscheme['expert'] = [[0] * slice_expert_num for i in range(ori_expert_num)]
            for i, idx in enumerate(sorted_index):
                xi = int(idx // slice_expert_num)
                xj = int(idx % slice_expert_num)
                qscheme['expert'][xi][xj] = e_bits[i // ori_expert_num]
    else:
        qscheme['expert'] = [qscheme['econfig'] for i in range(ori_expert_num)]

    counter = Counter(tuple(s) for s in qscheme['expert'])
    print(f"layer {layer_idx} scheme type count: {counter}")

    if use_hybrid_moe:
        qscheme['slice_expert'] = qscheme['expert']
        qscheme['expert'] = restructure_hybrid_qscheme(qscheme['slice_expert'], slice_expert_num)

    for expert_idx, expert in enumerate(layer.mlp.experts):
        ori_gate_proj_weights = expert.gate_proj.weight
        ori_up_proj_weights = expert.up_proj.weight
        ori_down_proj_weights = expert.down_proj.weight

        expert_groups = all_expert_groups[expert_idx]

        if use_hybrid_moe:
            expert_sub_experts = []
            expert_sub_sizes = []

            orig_bit_config = qscheme['slice_expert'][expert_idx]
            restructured_config = qscheme['expert'][expert_idx]

            bit_to_indices = {}
            for bit, group_indices in zip(orig_bit_config, expert_groups):
                if bit not in bit_to_indices:
                    bit_to_indices[bit] = []
                bit_to_indices[bit].extend(group_indices)

            for bit in restructured_config:
                if bit == 0:
                    continue

                indices = bit_to_indices[bit]
                n_neurons = len(indices)

                new_config = model.config
                new_config.intermediate_size = n_neurons
                expert_mlp = expert.__class__(new_config).to(device)

                with torch.no_grad():
                    indices_tensor = torch.tensor(indices, dtype=torch.long, device=ori_gate_proj_weights.device)
                    expert_mlp.gate_proj.weight.data = ori_gate_proj_weights[indices_tensor, :].detach().clone()
                    expert_mlp.up_proj.weight.data = ori_up_proj_weights[indices_tensor, :].detach().clone()
                    expert_mlp.down_proj.weight.data = ori_down_proj_weights[:, indices_tensor].detach().clone()

                expert_mlp._quant_bit = bit

                expert_sub_experts.append(expert_mlp)
                expert_sub_sizes.append(n_neurons)
                total_neurons_processed += n_neurons

            all_new_experts.append(expert_sub_experts)
            sub_expert_bit_configs.append(tuple([bit for bit in restructured_config if bit != 0]))
            expert_to_subexperts.append(list(range(len(expert_sub_experts))))
        else:
            for ii, group_indices in enumerate(expert_groups):
                n_neurons = len(group_indices)

                new_config = model.config
                new_config.intermediate_size = n_neurons
                expert_mlp = expert.__class__(new_config).to(device)

                with torch.no_grad():
                    group_indices_tensor = torch.tensor(group_indices, dtype=torch.long, device=ori_gate_proj_weights.device)
                    expert_mlp.gate_proj.weight.data = ori_gate_proj_weights[group_indices_tensor, :].detach().clone()
                    expert_mlp.up_proj.weight.data = ori_up_proj_weights[group_indices_tensor, :].detach().clone()
                    expert_mlp.down_proj.weight.data = ori_down_proj_weights[:, group_indices_tensor].detach().clone() * scaling_factor

                all_new_experts.append(expert_mlp)
                new_expert_intermediate_size = expert_mlp.up_proj.weight.shape[0]
                total_neurons_processed += new_expert_intermediate_size

            expanded_gate = ori_router_gate.data[expert_idx, :].unsqueeze(0).repeat(slice_expert_num, 1).to(device).detach().clone()
            new_router.weight.data[gate_start_idx: gate_start_idx + slice_expert_num, :] = expanded_gate
            gate_start_idx += slice_expert_num

        del ori_gate_proj_weights, ori_up_proj_weights, ori_down_proj_weights
        gc.collect()
        torch.cuda.empty_cache()

    # ============ 临时恢复，后面会用 quant 后的覆盖 ============
    if need_restore and original_mlp is not None:
        layer.mlp = original_mlp

    tick1 = time.time()
    print(f"Layer {layer_idx}, {args.rank_mode} expert re- sort time: {tick1 - tick0}", flush=True)
    print("all_new_expert_rates:", len(all_new_expert_rates))

    if use_hybrid_moe:
        moe = layer.mlp.__class__(model.config).to(device)
        moe.gate = layer.mlp.gate
        moe.num_experts = len(all_new_experts)
        moe.experts = nn.ModuleList([DartMoQHybridWrapper(sub_experts) for sub_experts in all_new_experts])
        counter = Counter(sub_expert_bit_configs)
        print("reconstruct moe with sub_expert_bit_configs: ", counter)
        if hasattr(layer.mlp, 'shared_expert'):
            moe.shared_expert = layer.mlp.shared_expert
        if hasattr(layer.mlp, 'shared_expert_gate'):
            moe.shared_expert_gate = layer.mlp.shared_expert_gate
        moe.training = False
    else:
        moe = layer.mlp.__class__(model.config).to(device)
        moe.num_experts = len(all_new_experts)
        moe.top_k = n_activated
        moe.gate = new_router
        moe.experts = all_new_experts
        if hasattr(layer.mlp, 'shared_expert'):
            moe.shared_expert = layer.mlp.shared_expert
        if hasattr(layer.mlp, 'shared_expert_gate'):
            moe.shared_expert_gate = layer.mlp.shared_expert_gate

    gc.collect()
    torch.cuda.empty_cache()
    return moe


@torch.no_grad()
def qwen35_construct_moe(model, moe_model_flag, layer, layer_idx, inp,
                         attention_mask, position_ids, position_embeddings,
                         n_experts, n_activated, slice_expert_num, ori_activated,
                         qscheme, args):
    """
    Qwen3.5 版本的 construct_moe
    主要改动：
    1. 支持 self_attn/linear_attn 两种注意力层
    2. 支持 Qwen3.5 的 model_type
    """

    modeltype = model.config.model_type
    batchsize = inp.shape[0]
    device = next(layer.parameters()).device

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
        # ============ Qwen3.5 关键改动：支持 qwen3.5 model_type ============
        if modeltype == 'olmoe' or modeltype == 'llama' or modeltype == 'qwen3' or modeltype == 'qwen3_moe' or modeltype == 'qwen3_5' or modeltype == 'deepseek_v3':
            with torch.no_grad():
                if hasattr(layer, 'self_attn'):
                    attn_out[b_i:b_i+1] = layer.self_attn(
                        hidden_states=hidden_states_inorm[b_i:b_i+1],
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings)[0]
                elif hasattr(layer, 'linear_attn'):
                    attn_out[b_i:b_i+1] = layer.linear_attn(
                        hidden_states=hidden_states_inorm[b_i:b_i+1],
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings)[0]
        else:
            with torch.no_grad():
                if hasattr(layer, 'self_attn'):
                    attn_out[b_i:b_i+1] = layer.self_attn(
                        hidden_states=hidden_states_inorm[b_i:b_i+1],
                        attention_mask=attention_mask,
                        position_ids=position_ids)[0]
                elif hasattr(layer, 'linear_attn'):
                    attn_out[b_i:b_i+1] = layer.linear_attn(
                        hidden_states=hidden_states_inorm[b_i:b_i+1],
                        attention_mask=attention_mask,
                        position_ids=position_ids)[0]

    hidden_states = residual + attn_out
    residual = hidden_states
    with torch.no_grad():
        hidden_states = layer.post_attention_layernorm(hidden_states)

    is_moe_layer = hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')

    tick0 = time.time()
    use_hybrid_moe = getattr(args, 'use_hybrid_moe', False)
    quantmode = getattr(args, 'quantmode', 'gptq')
    global_mode = "global" in args.quant_scheme

    if moe_model_flag:
        if is_moe_layer:
            moe = qwen35_reconstruct_moe_from_existing(model, layer, layer_idx, hidden_states,
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

    del hidden_states, hidden_states_inorm, residual, attn_out

    gc.collect()
    torch.cuda.empty_cache()

    return moe_out


@torch.no_grad()
def qwen35_dartmoq_sequential_simple(model, tokenizer, dataloader, args, test_ppl=True):
    """
    Qwen3.5 最简单的顺序量化入口
    基于原 dartmoq_sequential.py，只做最小改动
    """
    print('Starting Qwen3.5 quantization ...')
    tick_quant_start = time.time()

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

        # ============ Qwen3.5 关键改动：支持多种配置获取专家数量 ============
        if hasattr(model.config, 'num_experts'):
            ori_num_experts = model.config.num_experts
        elif hasattr(model.config, 'n_routed_experts'):
            ori_num_experts = model.config.n_routed_experts
        elif hasattr(model.model.layers[0].mlp, 'num_experts'):
            ori_num_experts = model.model.layers[0].mlp.num_experts
        else:
            # 从 gate_up_proj 形状推断
            ori_num_experts = model.model.layers[0].mlp.experts.gate_up_proj.shape[0]

        if use_hybrid_moe:
            new_num_expert = ori_num_experts
        else:
            new_num_expert = slice_expert_num * ori_num_experts
            if hasattr(model.config, 'num_experts'):
                model.config.num_experts = new_num_expert
            elif hasattr(model.config, 'n_routed_experts'):
                model.config.n_routed_experts = new_num_expert

        # ============ Qwen3.5 关键改动：支持多种配置获取 top_k ============
        if hasattr(model.config, 'num_experts_per_tok'):
            ori_num_experts_per_tok = model.config.num_experts_per_tok
        elif hasattr(model.model.layers[0].mlp, 'top_k'):
            ori_num_experts_per_tok = model.model.layers[0].mlp.top_k
        else:
            ori_num_experts_per_tok = 6  # Qwen3.5 默认

        if use_hybrid_moe:
            new_num_experts_per_tok = ori_num_experts_per_tok
        else:
            new_num_experts_per_tok = slice_expert_num * ori_num_experts_per_tok
            if hasattr(model.config, 'num_experts_per_tok'):
                model.config.num_experts_per_tok = new_num_experts_per_tok

        if not use_hybrid_moe:
            if hasattr(model.config, 'moe_intermediate_size'):
                model.config.moe_intermediate_size = model.config.moe_intermediate_size // slice_expert_num
            elif hasattr(model.config, 'intermediate_size'):
                model.config.intermediate_size = model.config.intermediate_size // slice_expert_num

        if use_hybrid_moe:
            print("The model is already a MoE model. Proceeding to create hybrid MoE structure.")
            print(f"Hybrid MoE: {ori_num_experts} experts with sub-experts sliced by {slice_expert_num}")
        else:
            print("The model is already a MoE model. Proceeding to split experts. ")
            print(f"Slice expert by {slice_expert_num}: to {new_num_expert}, with {new_num_experts_per_tok} activated experts.")
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
        match = re.search(r'a(\d+)s(\d+)m([\d.]+)', qscheme_str)
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

        moe_out = qwen35_construct_moe(model,
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

        inps = moe_out

        if args.standby_layer_cpu:
            layer = layer.to('cpu')

        for i in range(torch.cuda.device_count()):
            print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
            print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")

        tick1 = time.time()
        print(f"Layer {layer_idx} total reconstruct and quantization time: {tick1 - tick0:.2f} s", flush=True)
        print("." * 100, flush=True)

    print("MoE reconstruction and quantization done.")
    tick_quant_end = time.time()
    time_quant = tick_quant_end - tick_quant_start
    print(f"Runtime of quantization only: {time_quant:.2f}")

    if getattr(args, 'sequential_eval', False):
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

            if args.sequential_eval:
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
