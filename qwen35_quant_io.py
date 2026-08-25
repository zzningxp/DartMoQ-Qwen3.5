#!/usr/bin/env python3
"""
量化后 checkpoint 的保存/加载（wxa16 packed 格式落盘）。

保存：量化完成后把 packed 权重与剩余 fp16 参数写入 safetensors，
      非张量元数据（seed/group_size/rotation/expert_offsets 等）写入 meta.json。
加载：在原模型路径上以 device_map="meta" 建空结构，替换量化模块后
      load_state_dict(assign=True) 回填，跳过校准与现场量化直接推理。

保存的种子是量化时的实际值（MoE 固定 42+bit / 42+bit+1000，Linear 为 args.seed+layer_idx），
加载不依赖公式，QR 旋转矩阵在推理时按种子重生成，与保存前逐 bit 一致。
"""

import gc
import json
import os
import shutil
import time

import torch

from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen35_utils import DEV
from wxa16_bit_partitioned_moe import WxA16BitPartitionedGroupMoE, WxA16Weights
from wxa16_linear import WxA16Linear


# ---------------------------------------------------------------------------
# 元数据收集（保存侧）
# ---------------------------------------------------------------------------

def _linear_meta(linear: WxA16Linear) -> dict:
    """收集 WxA16Linear 的非 state_dict 元数据 + buffer 形状。"""
    return {
        "type": "wxa16_linear",
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "bit_width": linear.bit_width,
        "group_size": linear.group_size,
        "seed": linear.seed,
        "rotation": linear.rotation,
        "orig_dtype": str(linear.orig_dtype),
        "has_bias": linear.bias is not None,
        "indices_packed_shape": list(linear.packed_indices.shape),
        "codebook_shape": list(linear.codebook.shape),
        "norms_shape": list(linear.norms.shape),
        "bias_shape": list(linear.bias.shape) if linear.bias is not None else None,
    }


def _packed_meta(packed: dict) -> dict:
    """收集单个 packed 字典（gate_up 或 down）的非张量元数据 + 张量形状。"""
    return {
        "seed": packed.get("seed"),
        "group_size": packed.get("group_size"),
        "shape": list(packed["shape"]),
        "bit_width": packed.get("bit_width"),
        "rotation": packed.get("rotation"),
        "orig_dtype": packed.get("orig_dtype"),
        "indices_packed_shape": list(packed["indices_packed"].shape),
        "codebook_shape": list(packed["codebook"].shape),
        "norms_shape": list(packed["norms"].shape),
    }


def _moe_meta(moe: WxA16BitPartitionedGroupMoE) -> dict:
    """收集 WxA16BitPartitionedGroupMoE 的非 state_dict 元数据。

    expert_offsets 是普通 dict 里的 LongTensor（不在 state_dict，
    且 layer.to('cpu') 不会移动它），必须转 CPU list 存 JSON。
    """
    bits = {}
    for bit_str, weights in moe.bit_weights.items():
        bits[bit_str] = {
            "gate_up": _packed_meta(weights.gate_up_packed),
            "down": _packed_meta(weights.down_packed),
        }

    expert_offsets = {}
    for bit_str, offsets in moe.expert_offsets.items():
        expert_offsets[bit_str] = offsets.cpu().tolist()

    return {
        "type": "wxa16_moe",
        "num_experts": moe.num_experts,
        "hidden_size": moe.hidden_size,
        "intermediate_size": moe.intermediate_size,
        "top_k": moe.top_k,
        "bit_list": list(moe.bit_list),
        "inter_size_by_bit": {str(k): v for k, v in moe.inter_size_by_bit.items()},
        "expert_offsets": expert_offsets,
        "bits": bits,
        "enable_timing": moe.enable_timing,
    }


def collect_quant_metadata(model, quant_args: dict = None, model_id: str = None,
                           base_model: str = None) -> dict:
    """遍历模型，按 dotted path（= state_dict key 前缀）收集量化模块的普通属性。"""
    meta = {
        "base_model": base_model,
        "model_class": model.__class__.__name__,
        "quant_args": quant_args or {},
        "modules": {},
        "layers": {},
    }
    if model_id:
        meta["model_id"] = model_id

    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, WxA16Linear):
            meta["modules"][name] = _linear_meta(module)
        elif isinstance(module, WxA16BitPartitionedGroupMoE):
            meta["layers"][name] = _moe_meta(module)

    return meta


# ---------------------------------------------------------------------------
# qmeta 种子（safetensors 冗余副本 + 交叉核对）
# ---------------------------------------------------------------------------

def _build_qmeta_tensors(meta: dict) -> dict:
    """把各量化模块的种子转为 qmeta. 前缀的标量 int64 张量。

    safetensors 只能存张量（rotation/orig_dtype 字符串与 expert_offsets 列表
    仍由 meta.json 承载），种子作为冗余副本写入，加载时与 meta.json 交叉核对。
    """
    qmeta = {}
    for path, m in meta.get("modules", {}).items():
        if m.get("seed") is not None:
            qmeta[f"qmeta/{path}/seed"] = torch.tensor(m["seed"], dtype=torch.int64)
    for path, lm in meta.get("layers", {}).items():
        for bit_str, bm in lm["bits"].items():
            for which in ("gate_up", "down"):
                seed_val = bm[which].get("seed")
                if seed_val is not None:
                    qmeta[f"qmeta/{path}/bits/{bit_str}/{which}/seed"] = torch.tensor(seed_val, dtype=torch.int64)
    qa_seed = meta.get("quant_args", {}).get("seed")
    if qa_seed is not None:
        qmeta["qmeta/quant_args/seed"] = torch.tensor(qa_seed, dtype=torch.int64)
    return qmeta


def _check_qmeta_seeds(meta: dict, qmeta_tensors: dict):
    """交叉核对 safetensors 中的 qmeta 种子与 meta.json（防文件错配）。"""
    expected = {}
    for path, m in meta.get("modules", {}).items():
        if m.get("seed") is not None:
            expected[f"qmeta/{path}/seed"] = m["seed"]
    for path, lm in meta.get("layers", {}).items():
        for bit_str, bm in lm["bits"].items():
            for which in ("gate_up", "down"):
                seed_val = bm[which].get("seed")
                if seed_val is not None:
                    expected[f"qmeta/{path}/bits/{bit_str}/{which}/seed"] = seed_val
    qa_seed = meta.get("quant_args", {}).get("seed")
    if qa_seed is not None:
        expected["qmeta/quant_args/seed"] = qa_seed

    if not qmeta_tensors:
        print("  [WARN] safetensors 中无 qmeta 种子（旧格式或未写入）")
        return

    mismatched = []
    missing = []
    for key, want in expected.items():
        got = qmeta_tensors.get(key)
        if got is None:
            missing.append(key)
        elif int(got.item()) != want:
            mismatched.append((key, want, int(got.item())))

    if mismatched:
        print(f"  [WARN] qmeta 种子与 meta.json 不一致 {len(mismatched)} 处: {mismatched[:5]}...（文件可能错配）")
    elif missing:
        print(f"  [WARN] {len(missing)} qmeta seed keys missing: {missing[:5]}...")
    else:
        print(f"  [OK] safetensors qmeta seeds match meta.json ({len(expected)} keys)")


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------

def _print_size_breakdown(sd: dict, total_bytes: int):
    """按 key 前缀统计 checkpoint 各部分大小（qmeta 标量 key 不计入分类）。"""
    cats = {
        "MoE packed (bit_weights)": 0,
        "Attn/Shared packed (indices/codebook/norms)": 0,
        "fp16 剩余参数": 0,
    }
    for name, t in sd.items():
        if name.startswith("qmeta/"):
            continue
        nbytes = t.numel() * t.element_size()
        if "bit_weights" in name:
            cats["MoE packed (bit_weights)"] += nbytes
        elif name.endswith(("packed_indices", "codebook", "norms")):
            cats["Attn/Shared packed (indices/codebook/norms)"] += nbytes
        else:
            cats["fp16 剩余参数"] += nbytes

    print("\n  [Quantized Checkpoint Size Breakdown]")
    for cat, nbytes in cats.items():
        print(f"    {cat}: {nbytes / 1024**3:.2f}GB")
    print(f"    合计: {total_bytes / 1024**3:.2f}GB")


def save_quantized_model(model, save_dir: str, base_model_path: str = None,
                         quant_args: dict = None, verbose: bool = True) -> dict:
    """把量化后的模型保存为 safetensors + meta.json。

    只做 CPU 拷贝，不修改模型自身的设备状态（保存后 eval 不受影响）。
    """
    tick0 = time.time()
    os.makedirs(save_dir, exist_ok=True)

    # 拷贝原模型 config.json（量化流程原地改过 model.config，不能用 save_pretrained）
    # 同时拷贝 tokenizer 和 modeling 文件，让 checkpoint 目录自给自足
    if base_model_path:
        for fname in ("config.json", "generation_config.json",
                      "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "vocab.json", "merges.txt"):
            src = os.path.join(base_model_path, fname)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(save_dir, fname))
        if verbose:
            print(f"Copied config/tokenizer files from {base_model_path}")

    # 拷贝项目本地的 modeling 文件（trust_remote_code 离线加载需要）
    local_modeling = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "modeling_qwen3_5_moe.py")
    if os.path.isfile(local_modeling):
        shutil.copy(local_modeling, os.path.join(save_dir, "modeling_qwen3_5_moe.py"))
        if verbose:
            print(f"Copied modeling_qwen3_5_moe.py from project")

    # 收集元数据并写 meta.json
    meta = collect_quant_metadata(
        model,
        quant_args=quant_args,
        model_id=getattr(model, "model_id", None),
        base_model=base_model_path,
    )
    meta_path = os.path.join(save_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"Saved metadata: {meta_path} (modules={len(meta['modules'])}, layers={len(meta['layers'])})")

    # state_dict 逐 key 转 CPU（safetensors 要求 CPU + contiguous）
    print("Collecting state_dict to CPU...")
    sd = {}
    seen_ids = set()
    for name, t in model.state_dict().items():
        if id(t) in seen_ids:
            raise AssertionError(f"state_dict 出现共享张量 {name}（当前模型不应有 tie）")
        seen_ids.add(id(t))
        sd[name] = t.detach().cpu() if t.device.type != "cpu" else t.detach()

    # 种子冗余写入 safetensors（qmeta. 前缀标量 int64 张量）
    sd.update(_build_qmeta_tensors(meta))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sd_path = os.path.join(save_dir, "model.safetensors")
    print(f"Writing safetensors to {sd_path} (this may take a while)...")
    save_file(sd, sd_path)
    total_bytes = os.path.getsize(sd_path)

    if verbose:
        _print_size_breakdown(sd, total_bytes)

    del sd
    gc.collect()
    print(f"Saved quantized checkpoint to {save_dir} in {time.time() - tick0:.2f}s, "
          f"total {total_bytes / 1024**3:.2f}GB")
    return meta


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def _split_path(root, path: str):
    """按 dotted path 定位 (父模块, 属性名)。"""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _materialize_meta_buffers(model) -> list:
    """物化 assign 加载后残留的 meta 非持久 buffer。

    rotary_emb.inv_freq / original_inv_freq 是 persistent=False buffer，
    load_state_dict 不参与 strict 匹配也不会回填，forward 里 .to(device)
    会对 meta 张量报错，必须用确定性公式重算（与 meta init 时计算一致）。
    """
    handled = []
    unknown = []
    for name, buf in model.named_buffers():
        if not buf.is_meta:
            continue
        parent_name, _, buf_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if hasattr(parent, "compute_default_rope_parameters") and buf_name in ("inv_freq", "original_inv_freq"):
            inv_freq, _ = parent.compute_default_rope_parameters(parent.config, None)
            parent.register_buffer("inv_freq", inv_freq, persistent=False)
            parent.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
            handled.append(name)
        else:
            unknown.append(name)
    if handled:
        print(f"Materialized meta buffers: {handled}")
    if unknown:
        print(f"  [WARN] 仍有 meta buffer 未物化: {unknown}（首次 forward 可能报错）", flush=True)
    return handled


def restore_quant_metadata(model, meta: dict, state_dict: dict = None):
    """按元数据替换量化模块并回填张量。

    顺序要求：
      1. 先替换 shared_expert 内部的 WxA16Linear（此时还在原始 mlp 结构下）
      2. 再整体替换 mlp 为 WxA16BitPartitionedGroupMoE（复用替换后的 shared_expert）
      3. 最后 load_state_dict(strict=True, assign=True) 回填
    """
    # 第一步：替换 WxA16Linear（attention 与 shared_expert 内部）
    for path, m in meta.get("modules", {}).items():
        if m.get("type") != "wxa16_linear":
            continue
        parent, attr = _split_path(model, path)
        setattr(parent, attr, WxA16Linear.from_metadata(m))

    # 第二步：整体替换 MoE
    for path, m in meta.get("layers", {}).items():
        parent, attr = _split_path(model, path)
        old_mlp = getattr(parent, attr)
        new_moe = WxA16BitPartitionedGroupMoE.from_metadata(
            m,
            gate=old_mlp.gate,
            shared_expert=old_mlp.shared_expert,
            shared_expert_gate=old_mlp.shared_expert_gate,
        )
        setattr(parent, attr, new_moe)

        # 清理旧结构引用（与量化路径 qwen35_simple_wrapper 的清理一致）
        if hasattr(old_mlp, "gate"):
            del old_mlp.gate
        if hasattr(old_mlp, "shared_expert"):
            del old_mlp.shared_expert
        if hasattr(old_mlp, "shared_expert_gate"):
            del old_mlp.shared_expert_gate
        del old_mlp

    # 第三步：回填张量（严格匹配：key 必须一一对应）
    # qmeta. 前缀的种子标量不属于模块结构，剥掉后与 meta.json 交叉核对
    if state_dict is not None:
        qmeta_tensors = {k: v for k, v in state_dict.items() if k.startswith("qmeta/")}
        sd_weights = {k: v for k, v in state_dict.items() if not k.startswith("qmeta/")}
        res = model.load_state_dict(sd_weights, strict=True, assign=True)
        if res.missing_keys:
            raise AssertionError(f"load_state_dict missing keys: {res.missing_keys[:10]}...")
        if res.unexpected_keys:
            raise AssertionError(f"load_state_dict unexpected keys: {res.unexpected_keys[:10]}...")
        _materialize_meta_buffers(model)
        _check_qmeta_seeds(meta, qmeta_tensors)

    return model


def load_quantized_model(base_model_path: str = None, quant_dir: str = None,
                         standby_cpu: bool = False, seqlen: int = 2048):
    """加载已量化的 checkpoint，跳过校准与现场量化。

    Args:
        base_model_path: 原模型路径（缺省时用 quant_dir，checkpoint 保存时已自包含
                         config/tokenizer/modeling 文件）
        quant_dir: 保存目录（含 model.safetensors + meta.json）
        standby_cpu: 加载后保持 CPU（逐层搬移的 sequential eval 用）
        seqlen: 模型序列长度（与原加载路径一致，固定 2048）
    """
    if base_model_path is None:
        # 优先从 meta.json 读 base_model 路径（旧 checkpoint 没拷 tokenizer 时也能用）
        meta_path_tmp = os.path.join(quant_dir, "meta.json")
        if os.path.isfile(meta_path_tmp):
            with open(meta_path_tmp, "r", encoding="utf-8") as f:
                meta_tmp = json.load(f)
            saved_base = meta_tmp.get("base_model")
            if saved_base and os.path.isdir(saved_base):
                base_model_path = saved_base
        if base_model_path is None:
            base_model_path = quant_dir
    print(f"Loading quantized checkpoint from: {quant_dir}")
    print(f"Base model path (config/tokenizer/remote code): {base_model_path}")
    tick0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print("Building model structure on meta device (no weights loaded)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="meta",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"Meta structure built in {time.time() - tick0:.2f}s")

    meta_path = os.path.join(quant_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sd_path = os.path.join(quant_dir, "model.safetensors")
    print(f"Loading quantized tensors: {sd_path} ({os.path.getsize(sd_path) / 1024**3:.2f}GB)...")
    sd = load_file(sd_path)
    print(f"Loaded {len(sd)} tensors in {time.time() - tick0:.2f}s")

    restore_quant_metadata(model, meta, sd)

    del sd
    gc.collect()

    # 对齐 qwen35_utils.load_model 设置的属性
    model.seqlen = seqlen
    model.model_id = meta.get("model_id") or os.path.basename(base_model_path.rstrip("/"))
    model._standby_cpu = standby_cpu
    model._model_path = base_model_path
    model.eval()

    if not standby_cpu:
        print(f"Moving quantized model to {DEV}...")
        model.to(DEV)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Quantized model loaded in {time.time() - tick0:.2f}s")
    return model, tokenizer
