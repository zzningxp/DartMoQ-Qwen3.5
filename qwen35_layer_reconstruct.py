#!/usr/bin/env python3
"""Qwen3.5 MoE 层重构，基于原 DartMoQ，只做最小改动"""

import time
import torch
import torch.nn as nn
import os
import time
import gc
import numpy as np
import sys

sys.path.insert(0, '..')

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
from collections import Counter

from dartmoq_hybridmoe import DartMoQHybridWrapper
from dartmoq_hybridmoe import restructure_hybrid_qscheme
from grouped_gemm_moe_adapter import MoEBuildBlock


INTERMEDIATE_RESULT_DIR = "intermediate_result"


@torch.no_grad()
def reconstruct_moe_from_existing(model, layer, layer_idx, inps,
                                  n_experts, n_activated, slice_expert_num,
                                  ori_activated, device, qscheme,
                                  use_hybrid_moe, global_mode, quantmode, args, is_qwen35_merged=False):
    if global_mode:
        cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, "expert_activate", model.model_id)
        cache_path = os.path.join(cache_dir, f"{model.model_id}_L{layer_idx}.pt")
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(cache_path):
            try:
                expert_activation_rates = torch.load(cache_path, map_location="cpu")
                expert_activation_rates = torch.as_tensor(expert_activation_rates).detach().cpu()
                print(f"Loading cached expert activation rates for layer {layer_idx} from {cache_path}", flush=True)
            except Exception as e:
                print(f"Failed to load cached expert activation rates {e} from {cache_path}", flush=True)
                expert_activation_rates = analyze_experts_activation(layer, layer_idx, inps, ori_activated, model.config.model_type)
                torch.save(expert_activation_rates.detach().cpu(), cache_path)
                print(f"Saved expert activation rates to {cache_path}")
        else:
            expert_activation_rates = analyze_experts_activation(layer, layer_idx, inps, ori_activated, model.config.model_type)
            torch.save(expert_activation_rates.detach().cpu(), cache_path)
            print(f"Saved expert activation rates to {cache_path}")

    if hasattr(model.config, 'num_experts'):
        ori_expert_num = model.config.num_experts
    elif hasattr(model.config, 'n_routed_experts'):
        ori_expert_num = model.config.n_routed_experts
    else:
        ori_expert_num = len(layer.mlp.experts)

    # Hybrid mode only
    new_expert_num = ori_expert_num
    ori_router_gate = layer.mlp.gate.weight
    all_new_experts = []

    total_neurons_processed = 0

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
            # Always include 0bit
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

                # Free q_rates early since we don't need it after this point
                for x in outlier_bits:
                    del q_rates[x]
                del q_rates
                gc.collect()

                dp_tick0 = time.time()
                dpscheme_list, all_rates_arr = enum_optimal_m_scheme_global_fast(
                    expert_rates_list,
                    expert_activation_rates,
                    slice_expert_num,
                    target_bpw=qscheme['target_bpw'],
                    enable_0bit_compensation=True
                )
                dp_tick1 = time.time()
                print(f"enum_optimal_m_scheme_global_fast time {dp_tick1 - dp_tick0}", flush=True)

                # Free expert_rates_list after use
                del expert_rates_list
                gc.collect()

                all_rates = []
                for expert_idx in range(ori_expert_num):
                    rates_arr = all_rates_arr[expert_idx]
                    all_rates.append(torch.from_numpy(rates_arr).to(device))

                # Free all_rates_arr after use
                del all_rates_arr
                gc.collect()

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

        # Always include 0bit
        energy_bits = [0, 1, 2, 3, 4]
        dpscheme_list, all_rates_arr = enum_optimal_m_scheme_energy_global_fast(
            expert_energy_list,
            expert_activation_rates,
            slice_expert_num,
            target_bpw=qscheme['target_bpw'],
            bits=energy_bits,
            enable_0bit_compensation=True
        )

        print(f"built dpscheme_list for energy mode target_bpw {qscheme['target_bpw']} for {ori_expert_num} experts")
        tick1 = time.time()
        print(f"energy mode analyze time {tick1 - tick0}", flush=True)

    tick0 = time.time()

    all_new_expert_rates = []
    all_expert_groups = []

    # 保存元数据：每个专家的 bit_to_indices
    layer_expert_bit_indices = []

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

    # Hybrid mode only
    qscheme['slice_expert'] = qscheme['expert']
    qscheme['expert'] = restructure_hybrid_qscheme(qscheme['slice_expert'], slice_expert_num)

    # 保存原始的 bit config
    layer_orig_bit_config = qscheme['slice_expert']

    for expert_idx, expert in enumerate(layer.mlp.experts):
        ori_gate_proj_weights = expert.gate_proj.weight
        ori_up_proj_weights = expert.up_proj.weight
        ori_down_proj_weights = expert.down_proj.weight

        expert_groups = all_expert_groups[expert_idx]

        expert_sub_experts = []
        expert_sub_sizes = []

        orig_bit_config = qscheme['slice_expert'][expert_idx]
        restructured_config = qscheme['expert'][expert_idx]

        bit_to_indices = {}
        for bit, group_indices in zip(orig_bit_config, expert_groups):
            if bit not in bit_to_indices:
                bit_to_indices[bit] = []
            bit_to_indices[bit].extend(group_indices)

        # 保存这个专家的 bit_to_indices
        layer_expert_bit_indices.append(bit_to_indices)

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

        del ori_gate_proj_weights, ori_up_proj_weights, ori_down_proj_weights
        gc.collect()
        torch.cuda.empty_cache()

    tick1 = time.time()
    print(f"Layer {layer_idx}, {args.rank_mode} expert re- sort time: {tick1 - tick0}", flush=True)
    print("all_new_expert_rates:", len(all_new_expert_rates))

    # 收集所有使用的 bit
    all_bits = set()
    for expert_bit_idx in layer_expert_bit_indices:
        all_bits.update(expert_bit_idx.keys())
    bit_list = sorted(list(all_bits))

    # 获取维度信息
    hidden_size = model.config.hidden_size
    if hasattr(model.config, 'intermediate_size'):
        intermediate_size = model.config.intermediate_size
    else:
        intermediate_size = layer.mlp.experts[0].gate_proj.weight.shape[0]

    # 构建 layer_metadata
    layer_metadata = {
        'layer_idx': layer_idx,
        'expert_bit_indices': layer_expert_bit_indices,
        'expert_groups': all_expert_groups,
        'orig_bit_config': layer_orig_bit_config,
        'num_experts': ori_expert_num,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'bit_list': bit_list
    }

    # Clean up intermediate data structures
    del all_new_expert_rates
    if 'all_rates' in locals():
        del all_rates
    gc.collect()

    # Hybrid mode only
    # 用简单的 MoE 块，不依赖原始类
    moe = MoEBuildBlock(model.config).to(device)
    moe.gate = layer.mlp.gate
    moe.num_experts = len(all_new_experts)
    moe.experts = nn.ModuleList([
        DartMoQHybridWrapper(sub_experts, bit_to_indices=layer_expert_bit_indices[expert_idx])
        for expert_idx, sub_experts in enumerate(all_new_experts)
    ])

    # Clean up all_new_experts since we've transferred ownership to the ModuleList
    del all_new_experts
    gc.collect()

    counter = Counter(sub_expert_bit_configs)
    print("reconstruct moe with sub_expert_bit_configs: ", counter)
    if hasattr(layer.mlp, 'shared_expert'):
        moe.shared_expert = layer.mlp.shared_expert
    if hasattr(layer.mlp, 'shared_expert_gate'):
        moe.shared_expert_gate = layer.mlp.shared_expert_gate
    moe.training = False

    # Clean up sub_expert_bit_configs
    del sub_expert_bit_configs, expert_to_subexperts
    if 'dpscheme_list' in locals():
        del dpscheme_list

    gc.collect()
    torch.cuda.empty_cache()
    return moe, layer_metadata

