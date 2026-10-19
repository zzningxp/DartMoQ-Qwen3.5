#!/usr/bin/env python3
"""
多模态（VL）量化模型加载：把纯文本量化 checkpoint 装回视觉塔。

背景：35B/122B 基座本身是 Qwen3_5MoeForConditionalGeneration（VL），量化
流水线用 AutoModelForCausalLM 只取语言塔（model.visual.* / mtp.* 权重不进入
量化 checkpoint，见 qwen35_stream_io 的存档说明）。本模块把量化 checkpoint
的 CausalLM 命名 key 重映射回 VL 命名（model.layers.* →
model.language_model.layers.*），并从原始 fp16 模型目录读回 model.visual.*
权重（bf16 不量化、原样读回），组装成完整 VL 模型。

复用关系（不重复实现）：
  - build_meta_structure（qwen35_stream_io，model_class 参数建 VL 结构）
  - restore_quant_metadata / load_quantized_state_dict / convert_model_to_*
    （qwen35_quant_io，前者经 key_prefix_map 参数做结构无关重映射）
  - remap_prefix（qwen35_quant_io）

视觉塔采用「正常实例化 + load_state_dict」而非 meta 结构：视觉塔不量化，
且其 rotary_emb.inv_freq 是 persistent=False buffer（meta 结构下需要额外
回填公式），正常实例化直接得到真实 buffer，省掉回填路径。
"""

import gc
import json
import os
import time

import torch

from safetensors import safe_open

from qwen35_utils import DEV


# ---------------------------------------------------------------------------
# 语言塔 key 重映射（CausalLM 命名 → VL 命名）
# ---------------------------------------------------------------------------

def vl_language_prefix_remap(key: str) -> str:
    """只把语言塔专属前缀重映射到 VL 命名，视觉塔 key 原样保留。

    量化 checkpoint（CausalLM 命名）的语言塔 key 与 meta.json 模块 path 都是
    model.layers.* / model.embed_tokens.* / model.norm.*，VL 结构下在
    model.language_model.*；model.visual.*（原 fp16 目录读回）本身就是 VL
    命名，不能动。
    """
    for p in ("model.layers.", "model.embed_tokens.", "model.norm.",
              "model.rotary_emb."):
        if key.startswith(p):
            return "model.language_model." + key[len("model."):]
    return key


# ---------------------------------------------------------------------------
# 视觉塔权重读取（原 fp16 模型目录）
# ---------------------------------------------------------------------------

def load_visual_weights(visual_source: str) -> dict:
    """从原 fp16 模型目录读 model.visual.* 权重（bf16 不量化，原样读回）。

    支持分 shard（model.safetensors.index.json）与单文件两种形态。
    只触碰 visual key 所在的 shard 数据段，不加载整模型。
    """
    vis_sd = {}
    idx_path = os.path.join(visual_source, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        vis_keys = [k for k in wm if k.startswith("model.visual.")]
        for fname in sorted({wm[k] for k in vis_keys}):
            with safe_open(os.path.join(visual_source, fname), framework="pt") as h:
                for k in vis_keys:
                    if wm[k] == fname:
                        vis_sd[k] = h.get_tensor(k)
    else:
        sf_path = os.path.join(visual_source, "model.safetensors")
        with safe_open(sf_path, framework="pt") as h:
            for k in h.keys():
                if k.startswith("model.visual."):
                    vis_sd[k] = h.get_tensor(k)

    if not vis_sd:
        raise FileNotFoundError(
            f"No model.visual.* weights found in {visual_source}")
    nbytes = sum(t.numel() * t.element_size() for t in vis_sd.values())
    print(f"[vl] visual tower weights: {len(vis_sd)} tensors, {nbytes / 1024**3:.2f}GB")
    return vis_sd


# ---------------------------------------------------------------------------
# VL 量化模型组装
# ---------------------------------------------------------------------------

def load_quantized_vl_model(quant_dir: str, visual_source: str,
                            inference_quant_mode: str = "wxa16",
                            device=None):
    """加载带视觉塔的量化 VL 模型。

    Args:
        quant_dir: 量化 checkpoint 目录（model.safetensors + meta.json，
            文本结构命名，语言塔量化）
        visual_source: 原始 fp16 模型目录（model.visual.* 权重 +
            preprocessor_config.json 等）
        inference_quant_mode: "wxa16" / "wxa8" / "wxfp8" / "wxfp4"，
            语义同 qwen35_quant_io.load_quantized_model
        device: 加载后搬移目标（默认 qwen35_utils.DEV）

    Returns:
        完整 Qwen3_5MoeForConditionalGeneration：视觉塔 bf16 原权重 +
        语言塔量化模块（wxa16/wxa8/wxfp8/wxfp4 任一推理模式）。
        processor/tokenizer 由调用方从 visual_source 用 AutoProcessor 加载。
    """
    if device is None:
        device = DEV
    tick0 = time.time()

    # ---- 1. VL meta 结构（visual + language_model + lm_head，零权重内存） ----
    from transformers import AutoModelForImageTextToText
    from qwen35_stream_io import build_meta_structure
    print(f"[vl] building VL meta structure (checkpoint: {quant_dir})...")
    model = build_meta_structure(quant_dir, model_class=AutoModelForImageTextToText)

    # ---- 2. 视觉塔：正常实例化（真实 buffer）+ 原 fp16 权重回填 ----
    # 视觉塔不量化；正常实例化避免 meta buffer 回填（rotary inv_freq 非持久）
    from transformers import AutoModel
    visual = AutoModel.from_config(model.config.vision_config)
    visual_sd = load_visual_weights(visual_source)
    # 模块级 load_state_dict 用相对 key（剥掉 model.visual. 前缀）
    vis_rel = {k[len("model.visual."):]: v for k, v in visual_sd.items()}
    res = visual.load_state_dict(vis_rel, strict=True)
    if res.missing_keys or res.unexpected_keys:
        raise AssertionError(f"visual tower load mismatch: missing={res.missing_keys[:5]}, "
                             f"unexpected={res.unexpected_keys[:5]}")
    model.model.visual = visual
    del vis_rel
    gc.collect()
    print(f"[vl] visual tower restored in {time.time() - tick0:.2f}s")

    # ---- 3. 读量化 state_dict + meta，重映射到 VL 命名 ----
    from qwen35_quant_io import (load_quantized_state_dict,
                                 restore_quant_metadata)
    meta_path = os.path.join(quant_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    sd = load_quantized_state_dict(quant_dir)

    sd_vl = {vl_language_prefix_remap(k): v for k, v in sd.items()}
    # 视觉权重并入（全名 key）：strict 匹配要求 sd 覆盖结构全部 key
    # （qmeta/* 单独核对，不进结构匹配）
    sd_vl.update(visual_sd)
    del sd, visual_sd
    gc.collect()

    # ---- 4. 解除 tie：VL 结构声明 lm_head↔embed 共享，但量化 checkpoint
    # 中两者独立保存（值不同），assign 加载前拆开 ----
    embed = model.model.language_model.embed_tokens.weight
    if model.lm_head.weight is embed:
        model.lm_head.weight = torch.nn.Parameter(
            torch.empty_like(embed, device="meta"))
        print("[vl] untied lm_head (quantized checkpoint stores it separately)")

    # ---- 5. 量化模块恢复 + 严格回填（重映射只作用于语言塔前缀） ----
    restore_quant_metadata(model, meta, sd_vl,
                           key_prefix_map=vl_language_prefix_remap)
    del sd_vl
    gc.collect()

    # ---- 6. 推理模式切换（checkpoint 不变，原地换 __class__） ----
    from qwen35_quant_io import (convert_model_to_wxa8, convert_model_to_wxfp8,
                                 convert_model_to_wxfp4)
    if inference_quant_mode == "wxa8":
        convert_model_to_wxa8(model)
    elif inference_quant_mode == "wxfp8":
        convert_model_to_wxfp8(model, attn="wxa8")
    elif inference_quant_mode == "wxfp4":
        convert_model_to_wxfp4(model)
    elif inference_quant_mode != "wxa16":
        raise ValueError(f"未知 inference_quant_mode: {inference_quant_mode}")

    # ---- 7. 属性对齐与设备搬移 ----
    model.seqlen = 2048
    model.model_id = meta.get("model_id") or os.path.basename(quant_dir.rstrip("/"))
    model._model_path = visual_source
    model._vl_visual_source = visual_source
    model.eval()
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[vl] VL quantized model loaded in {time.time() - tick0:.1f}s "
          f"(mode={inference_quant_mode}, device={device})")
    return model
