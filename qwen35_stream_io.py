#!/usr/bin/env python3
"""
standby-layer-meta 模式的流式 IO（122B 级 W8A8 int8 输入 checkpoint 专用）。

背景：compressed-tensors int8 checkpoint 全量反量化成 bf16 约 244GB，
超过本机 RAM+swap（~202GB），from_pretrained 路径必然 OOM。量化主流程
本身是逐层的，因此改为 meta 空结构 + 逐层从盘流式反量化载入，CPU RAM 全程
只短暂经过单层 int8 字节。

输入侧（StreamModelReader + materialize_module）：
  - index.json / shard header 常驻缓存（只存 dtype 元数据，不占数据内存）；
  - 结构 key → checkpoint key 前缀重映射（原生 transformers 建的是文本专属
    结构 model.layers.*，checkpoint 是多模态命名 model.language_model.layers.*，
    from_pretrained 内部有同样的重映射）；
  - I8 + '_scale' 在 GPU 上反量化成 bf16（fp32 中间态，与 compressed-tensors
    在 from_pretrained 里的数值路径一致）。

输出侧（StreamingSaver）：
  - 逐层追加、按层边界切 shard 写 safetensors，不累积整模型 state_dict；
  - finalize 补写小件（embed/lm_head/norm）、未量化层兜底（--quant-layers
    部分量化的场景，从 checkpoint 反量化直接写盘不经过模块）、meta.json、
    qmeta 种子张量、index.json；
  - visual/mtp 从源 checkpoint 字节级原样搬运到独立文件 visual_mtp.safetensors
    （不进 index.json：load 路径 strict 匹配只认文本结构 key，多模态部分纯存档）。
"""

import json
import os
import struct
import time

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from qwen35_utils import DEV


# ---------------------------------------------------------------------------
# meta 结构构建
# ---------------------------------------------------------------------------

def build_meta_structure(model_path, verbose=True, model_class=None):
    """剥掉 checkpoint 量化配置后建纯 bf16 的 meta 空结构（零权重内存）。

    W8A8 输入的 config.json 带 compression_config，不剥掉的话 from_config
    会走 compressed-tensors 量化模块替换，结构 key 与手动反量化路径不匹配。

    model_class: 结构类。默认 AutoModelForCausalLM（纯文本结构）；
        VL 评测传 AutoModelForImageTextToText 建 Qwen3_5MoeForConditionalGeneration。
    """
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    had_quant_cfg = (getattr(cfg, "compression_config", None) is not None
                     or getattr(cfg, "quantization_config", None) is not None)
    if had_quant_cfg:
        cfg.compression_config = None
        cfg.quantization_config = None
        print("[stream] Stripped quantization config from checkpoint "
              "(int8 input, manual dequant)")
    if model_class is None:
        model_class = AutoModelForCausalLM
    with init_empty_weights():
        model = model_class.from_config(cfg, trust_remote_code=True)
    if verbose:
        # 层数兼容两种结构：CausalLM 的 model.layers 与 VL 的
        # model.language_model.layers
        base = getattr(model, "model", model)
        lm = getattr(base, "language_model", base)
        n_layers = len(lm.layers) if hasattr(lm, "layers") else None
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[stream] meta structure: {n_layers} layers, {n_params / 1e9:.1f}B params, "
              f"dtype={next(iter(model.parameters())).dtype}")
    return model


# ---------------------------------------------------------------------------
# 输入侧：checkpoint 流式读取
# ---------------------------------------------------------------------------

class StreamModelReader:
    """按 key 流式读 checkpoint：header 常驻缓存，数据按需 get_tensor。

    只支持带 model.safetensors.index.json 的分 shard checkpoint（122B W8A8
    输入即此形态）；单文件 checkpoint 走原有 from_pretrained 路径即可。
    """

    def __init__(self, model_path):
        self.model_path = model_path
        idx_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.isfile(idx_path):
            raise FileNotFoundError(f"Streaming read requires a sharded index: {idx_path}")
        with open(idx_path, "r", encoding="utf-8") as f:
            self.weight_map = json.load(f)["weight_map"]

        # 只读各 shard header（dtype 元数据），不触碰数据段
        self._dtypes = {}
        self._sizes = {}
        self._handles = {}
        for fname in sorted(set(self.weight_map.values())):
            with open(os.path.join(model_path, fname), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            for k, v in header.items():
                if k != "__metadata__":
                    self._dtypes[k] = v["dtype"]
                    self._sizes[k] = v["data_offsets"][1] - v["data_offsets"][0]
        print(f"[stream] index ready: {len(self.weight_map)} keys, "
              f"{len(set(self.weight_map.values()))} shards")

    def to_checkpoint_key(self, struct_key):
        """结构 key → checkpoint key：model.X → model.language_model.X。"""
        if struct_key.startswith("model."):
            return "model.language_model." + struct_key[len("model."):]
        return struct_key

    def dtype_of(self, ckpt_key):
        return self._dtypes[ckpt_key]

    def get(self, ckpt_key):
        fname = self.weight_map[ckpt_key]
        handle = self._handles.get(fname)
        if handle is None:
            handle = safe_open(os.path.join(self.model_path, fname), framework="pt")
            self._handles[fname] = handle
        return handle.get_tensor(ckpt_key)

    def visual_mtp_keys(self):
        """多模态存档 key（文本结构之外的部分，原样搬运不参与量化）。"""
        return [k for k in self.weight_map
                if k.startswith("model.visual.") or k.startswith("mtp.")]

    def archive_bytes(self):
        """visual/mtp 存档的总字节数（header 直读，不碰数据段）。"""
        return sum(self._sizes[k] for k in self.visual_mtp_keys())

    def small_pieces_bytes(self):
        """embed/lm_head/final_norm 的字节数（BF16 直读，输出大小≈输入大小）。"""
        keys = ("model.language_model.embed_tokens.weight",
                "model.language_model.norm.weight",
                "lm_head.weight")
        return sum(self._sizes[k] for k in keys)

    def close(self):
        self._handles.clear()


def dequant_int8(w_int8, scale, device):
    """int8 权重 × per-channel scale → bf16（fp32 中间态，在 device 上执行）。

    scale 形状（实测）：2D weight (out,1) / 3D experts (E,out,1)，
    尾部补 1 维后与 weight 对齐广播。
    """
    s = scale.to(device=device, dtype=torch.float32)
    t = w_int8.to(device).to(torch.float32)
    t = t * s.reshape(s.shape + (1,) * (t.dim() - s.dim()))
    return t.to(torch.bfloat16)


def materialize_module(module, struct_prefix, reader, device):
    """把 meta 模块按 checkpoint 流式反量化载入到 device 并 assign 回模块。

    int8 反量化在 device（GPU）上做，CPU 只短暂经过单 tensor 的 int8 字节；
    strict=True 校验结构 key 与 checkpoint 一一对应，不齐会立刻报错。
    """
    sd = {}
    for name in module.state_dict().keys():
        ckpt_key = reader.to_checkpoint_key(struct_prefix + name)
        t = reader.get(ckpt_key)
        if reader.dtype_of(ckpt_key) == "I8":
            t = dequant_int8(t, reader.get(ckpt_key + "_scale"), device)
        else:
            t = t.to(device)
        sd[name] = t
    module.load_state_dict(sd, strict=True, assign=True)
    return module


def materialize_small_pieces(model, reader, embed_device, verbose=True):
    """载入层之外的小件（BF16 直读）：embed_tokens → embed_device，norm/lm_head → CPU。

    Catcher 阶段需要 embed + layer0 真实驻留 GPU；lm_head/final_norm 只有
    eval 用，留 CPU（与 standby 模式的设备布局一致）。
    """
    tick0 = time.time()
    materialize_module(model.model.embed_tokens, "model.embed_tokens.", reader, embed_device)
    materialize_module(model.model.norm, "model.norm.", reader, torch.device("cpu"))
    materialize_module(model.lm_head, "lm_head.", reader, torch.device("cpu"))

    # rotary 的 inv_freq 是非持久 buffer（不在 checkpoint 里），
    # meta 建模后缺失或为 meta，用确定性公式重算（与量化 checkpoint 的
    # load 路径 _materialize_meta_buffers 同一套逻辑）
    from qwen35_quant_io import _materialize_meta_buffers
    _materialize_meta_buffers(model)
    rotary = getattr(model.model, "rotary_emb", None)
    inv = getattr(rotary, "inv_freq", None) if rotary is not None else None
    if inv is None or inv.is_meta:
        if rotary is not None and hasattr(rotary, "compute_default_rope_parameters"):
            new_inv, _ = rotary.compute_default_rope_parameters(rotary.config, None)
            rotary.register_buffer("inv_freq", new_inv, persistent=False)
            if verbose:
                print("[stream] inv_freq missing/meta, recomputed and registered")
        else:
            print("  [WARN] rotary inv_freq missing and no recompute entry "
                  "(first forward may fail)", flush=True)
    if verbose:
        print(f"[stream] small pieces loaded (embed@{embed_device}, "
              f"norm/lm_head@cpu) in {time.time() - tick0:.2f}s")


# ---------------------------------------------------------------------------
# 输出侧：逐层流式保存
# ---------------------------------------------------------------------------

class StreamingSaver:
    """逐层流式写分 shard safetensors，RAM 只驻留当前 shard 缓冲。

    与 save_quantized_model 的区别：不收集整模型 state_dict，量化完一层
    立即落盘；finalize 时补小件/兜底层/qmeta/index，保证产出 checkpoint
    与整存路径字段一致（load_quantized_model 可直接加载）。
    """

    SHARD_BYTES = 4 * 1024**3  # 单 shard 目标大小（层边界对齐，允许略超）

    def __init__(self, save_dir, base_model_path=None, verbose=True,
                 n_planned_layers=None, extra_bytes=0):
        self.save_dir = save_dir
        self.verbose = verbose
        os.makedirs(save_dir, exist_ok=True)

        from qwen35_quant_io import _copy_checkpoint_aux_files
        if base_model_path:
            _copy_checkpoint_aux_files(save_dir, base_model_path, verbose)

        self._buf = {}
        self._buf_bytes = 0
        self._shards = []        # [(tmp_name, {key: ...}, nbytes)]
        self._total_bytes = 0
        self._added_layers = set()
        self._tick0 = time.time()
        # 磁盘预检参数：首层产物入缓冲时按实测大小外推全流程占用
        self._n_planned_layers = n_planned_layers
        self._extra_bytes = extra_bytes

    def _preflight_disk(self, layer_bytes):
        """首层实测外推总占用，磁盘不足立即报错（而非等第一次 flush 才失败）。"""
        if self._n_planned_layers is None:
            return
        import shutil
        projected = layer_bytes * self._n_planned_layers + self._extra_bytes
        free = shutil.disk_usage(self.save_dir).free
        if projected * 1.05 > free:
            raise RuntimeError(
                f"Projected checkpoint size {projected / 1024**3:.1f}GB "
                f"({layer_bytes / 1024**3:.2f}GB/layer x {self._n_planned_layers} layers "
                f"+ {self._extra_bytes / 1024**3:.1f}GB aux) exceeds free disk "
                f"{free / 1024**3:.1f}GB. Clean up disk or change --save-quantized path.")

    def add(self, key, tensor):
        assert tensor.device.type == "cpu", f"{key} must be a CPU tensor"
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        self._buf[key] = tensor
        self._buf_bytes += tensor.numel() * tensor.element_size()

    def add_layer(self, layer_idx, layer):
        """量化完成的层（已在 CPU）追加进缓冲，按层边界决定是否切 shard。"""
        layer_sd = layer.state_dict()
        layer_bytes = sum(t.numel() * t.element_size() for t in layer_sd.values())
        if not self._added_layers:
            self._preflight_disk(layer_bytes)
        if self._buf and self._buf_bytes + layer_bytes > self.SHARD_BYTES:
            self._flush()
        for name, t in layer_sd.items():
            full = f"model.layers.{layer_idx}.{name}"
            tt = t.detach().cpu() if t.device.type != "cpu" else t.detach()
            self.add(full, tt)
        self._added_layers.add(layer_idx)

    def _flush(self):
        if not self._buf:
            return
        idx = len(self._shards) + 1
        tmp_name = f".shard-tmp-{idx:05d}.safetensors"
        from safetensors.torch import save_file
        save_file(self._buf, os.path.join(self.save_dir, tmp_name))
        self._shards.append((tmp_name, dict(self._buf), self._buf_bytes))
        self._total_bytes += self._buf_bytes
        if self.verbose:
            print(f"  [stream-save] shard {idx}: {self._buf_bytes / 1024**3:.2f}GB, "
                  f"{len(self._buf)} tensors", flush=True)
        self._buf = {}
        self._buf_bytes = 0

    def _archive_layer(self, reader, layer_idx, struct_layer):
        """未量化层兜底（--quant-layers 部分量化）：从 checkpoint 反量化直接写盘。

        不经过模块 assign——跳过的层保持 meta 零内存，逐 tensor 流式处理。
        """
        names = list(struct_layer.state_dict().keys())
        est = sum(struct_layer.state_dict()[n].numel() * 2 for n in names)
        if self._buf and self._buf_bytes + est > self.SHARD_BYTES:
            self._flush()
        for name in names:
            ckpt_key = reader.to_checkpoint_key(f"model.layers.{layer_idx}.{name}")
            t = reader.get(ckpt_key)
            if reader.dtype_of(ckpt_key) == "I8":
                t = dequant_int8(t, reader.get(ckpt_key + "_scale"), DEV).cpu()
            self.add(f"model.layers.{layer_idx}.{name}", t)
        self._added_layers.add(layer_idx)
        if self.verbose:
            print(f"  [stream-save] layer {layer_idx} not quantized, "
                  f"archived from checkpoint directly", flush=True)

    def _dump_visual_mtp(self, reader):
        """visual/mtp 字节级原样搬运到独立存档文件（不进 index.json）。"""
        keys = reader.visual_mtp_keys()
        if not keys:
            return
        aux = {}
        total = 0
        for k in keys:
            t = reader.get(k)
            aux[k] = t
            total += t.numel() * t.element_size()
        from safetensors.torch import save_file
        path = os.path.join(self.save_dir, "visual_mtp.safetensors")
        save_file(aux, path)
        del aux
        print(f"[stream-save] visual/mtp archived as-is: {path} "
              f"({total / 1024**3:.2f}GB, {len(keys)} tensors)")

    def finalize(self, model, reader, quant_args=None, model_id=None, base_model=None):
        """收尾：小件 + 兜底层 + visual/mtp 存档 + meta/qmeta + index.json。"""
        # 1. 小件：embed/lm_head/final_norm（活模型上已载入）
        for name, t in model.state_dict().items():
            if name.startswith("model.layers."):
                continue
            if t.is_meta:
                raise RuntimeError(f"Small piece not loaded, cannot save: {name}")
            self.add(name, t.detach().cpu() if t.device.type != "cpu" else t.detach())

        # 2. 未量化层兜底（整模型量化时为空）
        for i in range(len(model.model.layers)):
            if i not in self._added_layers:
                self._archive_layer(reader, i, model.model.layers[i])

        # 3. visual/mtp 存档（独立文件，不进 index）
        self._dump_visual_mtp(reader)

        # 4. meta.json + qmeta 种子张量（收集自活模型，不重复写权重）
        from qwen35_quant_io import collect_quant_metadata, _build_qmeta_tensors
        meta = collect_quant_metadata(model, quant_args=quant_args,
                                      model_id=model_id, base_model=base_model)
        for k, v in _build_qmeta_tensors(meta).items():
            self.add(k, v)
        self._flush()

        # 5. shard 重命名为 HF 惯例 + index.json + meta.json
        total = len(self._shards)
        weight_map = {}
        for i, (tmp_name, keys, _) in enumerate(self._shards, 1):
            final_name = f"model-{i:05d}-of-{total:05d}.safetensors"
            os.rename(os.path.join(self.save_dir, tmp_name),
                      os.path.join(self.save_dir, final_name))
            for k in keys:
                weight_map[k] = final_name
        index = {"metadata": {"total_size": self._total_bytes}, "weight_map": weight_map}
        with open(os.path.join(self.save_dir, "model.safetensors.index.json"),
                  "w", encoding="utf-8") as f:
            json.dump(index, f, indent=1)
        with open(os.path.join(self.save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)

        print(f"[stream-save] done: {total} shards, {self._total_bytes / 1024**3:.2f}GB, "
              f"{time.time() - self._tick0:.2f}s "
              f"(modules={len(meta['modules'])}, layers={len(meta['layers'])})")
        return meta
