#!/usr/bin/env python3
"""
Prefill/decode 吞吐基准（评测计划 P0：decode 通路 + 35B tok/s 基准）。

对照口径：4090 上 Qwen3.5-35B-A3B W4A16 的公开数字（prefill 774.6 tok/s /
decode 103.8 tok/s）。本脚本在 batch=1 下分 prefill 段与 decode 段报
tok/s（serving 式单 prompt 前向），与现有整数据集批量吞吐口径互补。

支持两种模型加载：
  --quant-dir：本项目 2bpw checkpoint（--mode wxa16|wxa8|wxfp8|wxfp4）
  --model-path：任意 HF 模型目录（fp16 / w4a8 / w4a16 / W8A8 等外部模型，
                标准 from_pretrained 加载）

口径纪律（对齐 benchmarks/20261010_论文构思_评测方法与基准对照计划.md）：
  - JIT 预热 + 热跑计时（每档 prefill 长度先 warmup_kernels，再 3 次取中位数）
  - prefill 只走 transformer（不含 lm_head，对齐 eval_qwen35.py 省显存模式）；
    TTFT 在 decode 循环里按 prefill 段 + 第一步完整报
  - decode 手动逐 token 顶层循环（DynamicCache + use_cache=True，greedy），
    机制与 eval_tasks.py generate_until 已验证路径一致
  - wall（serving 真实口径，含 Python 开销）与 CUDA event（纯 GPU 口径）双报
  - 2bpw 模式顺带聚合各层 mlp.last_timings（init/kernel 拆分，对齐 2.4 口径）

用法:
  conda run -n dart312-t38 python bench_throughput.py \
      --model-path /home/daodao/models/<外部模型目录> \
      --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
  conda run -n dart312-t38 python bench_throughput.py \
      --quant-dir models/qwen3.5-2bpw-260930-s1 --mode wxa16

注意：HF 数据下载在本机可能挂起，数据集已缓存时建议加环境变量
HF_DATASETS_OFFLINE=1 运行。
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime

import torch

# 2026-10-23 从 benchmarks/ 移到项目根（主路径脚本）：脚本目录即项目根，
# 单层 dirname 即可（原双 dirname 会把 ~/Develop 插到 sys.path 首位，
# 有兄弟项目同名模块遮蔽风险）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import get_c4, get_git_hash, get_wikitext2
from qwen35_utils import DEV, get_qwen35_model

# MoE last_timings 里参与聚合的时间字段（active_experts 等计数类字段排除）
MOE_TIMING_KEYS = ("total", "init", "shared", "router", "sort_scatter_prep",
                   "compute", "triton", "dequant", "gemm", "cleanup_reshape")


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_bench_model(args, device_map=None, max_memory=None):
    """按参数加载模型，返回 (model, tokenizer, model_id, mode_label)。
    device_map=None 走默认 "auto"；max_memory 随 auto 布置透传（graph 分段
    capture 用它控制层切分点）。"""
    if args.quant_dir:
        from qwen35_quant_io import load_quantized_model
        model, tokenizer = load_quantized_model(
            base_model_path=args.base_model,
            quant_dir=args.quant_dir,
            inference_quant_mode=args.mode,
        )
        model_id = getattr(model, "model_id", os.path.basename(args.quant_dir.rstrip("/")))
        return model, tokenizer, model_id, args.mode

    assert args.model_path, "need --quant-dir or --model-path"
    model_path = args.model_path
    # 官方 FP8 的 finegrained-fp8 走 hub kernel（transformers 5.20 默认开启），
    # kernels 包每次惰性加载都要联网查 publisher 信任位；hf-mirror 不稳时该调用
    # 会挂死（1023.profile.decode 首跑，无超时）或 SSL 断（三跑 ConnectError）。
    # 离线又撞部分快照校验（IncompleteSnapshotError，1018 第三轮同坑）。此处按
    # env 显式放行信任校验使其短路（kernel 本体仍走本地缓存/镜像按需取）。
    # 注意 ALLOW_ALL_KERNELS=True 语义是放行全部 hub kernel 仓库；实际会被
    # 加载的仍只有 transformers 内置映射引用的 kernels-community/* 仓库。
    if os.environ.get("DARTMOQ_TRUST_HUB_KERNELS") == "1":
        from transformers.integrations import hub_kernels
        hub_kernels.ALLOW_ALL_KERNELS = True
        print("[ext] hub kernel trust check bypassed (DARTMOQ_TRUST_HUB_KERNELS=1)")
    try:
        # 官方发布的 ConditionalGeneration 结构（FP8/Int4 基线）权重 key 带
        # model.language_model 前缀，必须用 ImageTextToText 类加载才能对上；
        # 纯文本结构（外部 w4a8/w4a16 等）走 CausalLM
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        _qc = getattr(cfg, "quantization_config", None)
        _method = (_qc.get("quant_method") if isinstance(_qc, dict)
                   else getattr(_qc, "quant_method", None))
        if _method == "gptq":
            # transformers/optimum 路径加载不了官方 Int4：transformers 5.20 的
            # GPTQConfig 丢掉了 dynamic 的 -: 排除规则（BF16 的 shared_expert_gate
            # N=1 被误包成量化层），且 Qwen3.5 融合 MoE 的 per-expert qweight 在
            # get_layers 里根本没有对应模块（1018.thrput.2/3 多轮实测）。
            # 走 gptqmodel 原生加载（认识 dynamic 排除与融合 MoE），kernel 强制
            # TritonV2：marlin/exllama 在 sm_120 不可用、MarlinLinear 还有
            # pack_factor=8 整除约束，TritonV2 是唯一合法后端
            from gptqmodel import GPTQModel
            from gptqmodel.utils.backend import BACKEND
            from transformers import AutoTokenizer
            print("[ext] gptq checkpoint -> GPTQModel.load "
                  "(backend=gptq_triton)", flush=True)
            model = GPTQModel.load(
                model_path, backend=BACKEND.GPTQ_TRITON,
                device_map="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)
            model.eval()
            model_id = os.path.basename(model_path.rstrip("/"))
            return model, tokenizer, model_id, _detect_external_quant(model)
        archs = getattr(cfg, "architectures", None) or []
        if any("ConditionalGeneration" in a for a in archs):
            from transformers import AutoModelForImageTextToText
            model, tokenizer = get_qwen35_model(
                model_path, device_map=device_map or "auto",
                max_memory=max_memory,
                model_class=AutoModelForImageTextToText,
                experts_implementation=getattr(args, "experts_impl", None))
        else:
            model, tokenizer = get_qwen35_model(
                model_path, device_map=device_map or "auto",
                max_memory=max_memory,
                experts_implementation=getattr(args, "experts_impl", None))
    except Exception as e:
        raise SystemExit(
            f"Failed to load external model from {model_path}:\n  {type(e).__name__}: {e}\n"
            f"Check: 1) dir contains config.json + safetensors; "
            f"2) custom modeling file exists (trust_remote_code); "
            f"3) if compressed-tensors checkpoint, its config is intact; "
            f"4) official FP8 needs fbgemm-gpu, GPTQ-Int4 needs gptqmodel "
            f"(official baseline env, see roadmaps/throughput-bench-261019.md)."
        ) from e
    model.eval()
    model_id = os.path.basename(model_path.rstrip("/"))
    return model, tokenizer, model_id, _detect_external_quant(model)


def _detect_external_quant(model):
    """从 config 探测外部模型的量化方式（用于结果标注，不改变加载行为）。"""
    qc = getattr(model.config, "quantization_config", None)
    if qc:
        if isinstance(qc, dict):
            method, bits = qc.get("quant_method", ""), qc.get("bits", "")
        else:
            method = getattr(qc, "quant_method", "")
            bits = getattr(qc, "bits", "")
        tag = f"{method}-{bits}bit" if method else "quantized"
        return f"external({tag})"
    return "external"


def load_sample_tokens(dataset, tokenizer):
    """从 wiki/c4 测试集取真实文本 token 序列（复用 data_utils 加载器）。

    返回 CPU 上的一维 LongTensor；全部测量（prefill 各档/decode/一致性检查）
    从同一个序列切片，固定 offset 保证可复现。
    """
    print(f"[data] loading {dataset} test set (tokenizing)...", flush=True)
    t0 = time.time()
    if dataset == "wiki":
        _, enc = get_wikitext2(1, 0, 2048, tokenizer, eval_only=True)
    elif dataset == "c4":
        _, enc = get_c4(1, 0, 2048, tokenizer, eval_only=True)
    else:
        raise SystemExit(f"unknown dataset: {dataset} (expected wiki|c4)")
    ids = enc.input_ids[0].long()
    print(f"[data] {len(ids)} tokens ready in {time.time() - t0:.1f}s")
    return ids


def _text_tower(model):
    """文本塔解析：CausalLM → model.model；ConditionalGeneration → model.model.language_model。

    与 qwen35_stream_io.build_meta_structure 的层数解析同一套规则
    （base = model.model，lm = base.language_model 兜底 base）。
    """
    base = getattr(model, "model", model)
    lm = getattr(base, "language_model", None)
    return lm if lm is not None else base


# ---------------------------------------------------------------------------
# MoE stage 聚合（2bpw 模式；外部模型无此属性则跳过）
# ---------------------------------------------------------------------------

def sum_moe_timings(model):
    """聚合所有 MoE 层 last_timings（本 forward 的 init/compute/triton/... 拆分）。

    无量化模块的模型返回 None。timings 是 wall-clock（MoE 模块内置），
    每 step 之后调用一次即可得到该 step 的 MoE 内部拆分。
    """
    try:
        layers = _text_tower(model).layers
    except AttributeError:
        return None
    acc = {}
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        lt = getattr(mlp, "last_timings", None)
        if lt is None:
            continue
        for k, v in lt.items():
            if k in MOE_TIMING_KEYS and isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0.0) + float(v)
    return acc or None


# ---------------------------------------------------------------------------
# 预热
# ---------------------------------------------------------------------------

def warmup(model, prefill_lens, sample, dev):
    """JIT 热跑：Triton kernels 按各 prefill 长度编译 + 微型前向 + 短 decode。"""
    print("[warmup] compiling Triton kernels (first run may look stuck; "
          "unseen shapes still compile lazily)...", flush=True)
    t0 = time.time()
    for module in model.modules():
        if not hasattr(module, "warmup_kernels"):
            continue
        for L in sorted(set(prefill_lens)):
            try:
                module.warmup_kernels(seq_len=L, batch_size=1)
            except Exception as e:  # 过小/过大的形状 kernel 可能不支持，热跑时再暴露
                print(f"  [warmup] warmup_kernels(seq_len={L}) skipped: {e}")

    # 微型真前向 + cache 路径短 decode（尽早暴露整模型通路问题）
    ids = torch.ones((1, 16), dtype=torch.long, device=dev)
    # 2026-10-23：分段打印（1023.profile.decode 首跑卡死在 warmup 后无进展，
    # 钉死卡点在微型前向还是 decode 循环；调试打印，确认无碍后待本人确认再删）
    print("[warmup] micro forward (1x16) ...", flush=True)
    with torch.no_grad():
        model(input_ids=ids)
        print("[warmup] micro forward done; short decode loop (4 steps) ...",
              flush=True)
        decode_loop(model, sample[:32], 4, dev)
    print("[warmup] decode loop done; final sync ...", flush=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"[warmup] done in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# 测量核心
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_loop(model, prompt_ids, n_steps, dev, collect_logits=False):
    """逐 token 顶层 decode 循环（P0 口径：batch=1，greedy，use_cache）。

    step 0 是 prefill（整段 prompt），step 1..n_steps 每步只喂 1 个 token。
    返回:
      gen: 生成的 token 列表（长度 n_steps+1，gen[0] 由 prefill 末尾位置预测）
      wall_times: 每步 wall（含 Python 开销），长度 n_steps+1
      event_times: 每步 CUDA event 时间（CPU 上为空），长度 n_steps+1
      moe_times: 每步 MoE 层 last_timings 聚合（无量化模块时为空）
      step_logits: collect_logits=True 时逐步 logits（长度 n_steps+1）
    """
    # use_cache 的保存/恢复：CausalLM 在顶层 config，官方 ConditionalGeneration
    # 在嵌套 text_config，均无则跳过（forward 显式传 use_cache=True 即可）
    _cfg = model.config
    cache_obj = (_cfg if hasattr(_cfg, "use_cache") else
                 getattr(_cfg, "text_config", None))
    if cache_obj is not None and hasattr(cache_obj, "use_cache"):
        use_cache = cache_obj.use_cache
        cache_obj.use_cache = True
    else:
        cache_obj, use_cache = None, None
    try:
        past = None
        ids = prompt_ids.unsqueeze(0).to(dev)
        mask = torch.ones(1, ids.shape[1], dtype=torch.long, device=dev)
        gen, wall_times, event_times, moe_times, step_logits = [], [], [], [], []
        for step in range(n_steps + 1):
            cuda = ids.device.type == "cuda"
            ev0 = torch.cuda.Event(enable_timing=True) if cuda else None
            ev1 = torch.cuda.Event(enable_timing=True) if cuda else None
            if ev0 is not None:
                ev0.record()
            t0 = time.time()
            out = model(input_ids=ids, attention_mask=mask,
                        past_key_values=past, use_cache=True)
            if ev1 is not None:
                ev1.record()
            t1 = time.time()
            wall_times.append(t1 - t0)
            if ev1 is not None:
                ev0.synchronize()
                ev1.synchronize()
                event_times.append(ev0.elapsed_time(ev1) / 1000.0)

            past = out.past_key_values
            if collect_logits:
                step_logits.append(out.logits[0, -1].detach().clone())
            next_tok = int(out.logits[0, -1].argmax().item())
            del out
            gen.append(next_tok)
            ids = torch.tensor([[next_tok]], dtype=torch.long, device=dev)
            mask = torch.cat(
                [mask, torch.ones(1, 1, dtype=torch.long, device=dev)], dim=1)

            mt = sum_moe_timings(model)
            if mt:
                moe_times.append(mt)
            if step > 0 and n_steps >= 128 and (step % 64 == 0 or step == n_steps):
                print(f"  [decode] step {step}/{n_steps}, "
                      f"wall so far {sum(wall_times[1:]):.1f}s", flush=True)
    finally:
        if cache_obj is not None:
            cache_obj.use_cache = use_cache
    return {
        "gen": gen,
        "wall_times": wall_times,
        "event_times": event_times,
        "moe_times": moe_times,
        "step_logits": step_logits,
    }


@torch.no_grad()
def time_prefill(model, sample, lens, reps, dev):
    """batch=1 prefill 计时：只走 transformer（不含 lm_head），每档 reps 次取中位数。

    外部模型若无 .model 文本子模型（非常规结构）则回退到顶层前向。
    注意必须 no_grad：2048+ 档的 autograd 图会驻留全部层中间张量
    （~20GB），2026-10-16 OOM 排查结论（1015/1016.thrput 日志）。
    """
    try:
        transformer = _text_tower(model)
    except AttributeError:
        transformer = model
    results = {}
    for L in lens:
        print(f"  [prefill] measuring len={L}...", flush=True)
        ids = sample[:L].unsqueeze(0).to(dev)
        walls, events = [], []
        for _ in range(reps):
            cuda = ids.device.type == "cuda"
            ev0 = torch.cuda.Event(enable_timing=True) if cuda else None
            ev1 = torch.cuda.Event(enable_timing=True) if cuda else None
            if ev0 is not None:
                ev0.record()
            t0 = time.time()
            out = transformer(ids)
            if ev1 is not None:
                ev1.record()
            t1 = time.time()
            walls.append(t1 - t0)
            if ev1 is not None:
                ev0.synchronize()
                ev1.synchronize()
                events.append(ev0.elapsed_time(ev1) / 1000.0)
            del out
        row = {"latency_s": statistics.median(walls), "tok_s": L / statistics.median(walls),
               "reps": reps, "wall_all": walls}
        if events:
            row["latency_event_s"] = statistics.median(events)
            row["tok_s_event"] = L / statistics.median(events)
        results[L] = row
    return results


def consistency_check(model, tokenizer, sample, dev):
    """cache 路径 vs 无 cache 全序列前向的 logits 一致性守卫。

    64 token prefill + 8 步 decode（含 B=1/S=1 量化 kernel 形状），逐步
    logits 与全序列前向对比。

    判据按 dtype 分档：
      - fp32：max diff < 1e-3 且 argmax 全对（精确对拍，CPU tiny 自测用）；
      - bf16/fp16：max diff < 8.0 且 argmax 一致率 >= 6/8。decode 步用
        recurrent kernel、全序列前向用 chunked kernel，两算法数学等价但
        bf16 数值有差，30 层线性注意力累积后 logits diff ~2 属正常噪声
        （2026-10-16 实测 2.14，文本通顺）；真 bug（cache 错位/掩码错）
        会让 diff 到几十且 argmax 几乎全错，仍会被抓住。
    """
    L, N = 64, 8
    if len(sample) < L:
        raise SystemExit(f"sample too short for consistency check: "
                         f"{len(sample)} < {L} (increase --sample-offset "
                         f"headroom or prompt len)")
    prompt = sample[:L]
    res = decode_loop(model, prompt, N, dev, collect_logits=True)
    gen = res["gen"]
    full = torch.cat([prompt, torch.tensor(gen[:N], dtype=torch.long)],
                     dim=0).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits_full = model(input_ids=full).logits  # [1, L+N, V]

    max_diff = 0.0
    n_agree = 0
    for i in range(N):
        d = (res["step_logits"][i].float()
             - logits_full[0, L + i - 1].float()).abs().max().item()
        max_diff = max(max_diff, d)
        if int(logits_full[0, L + i - 1].argmax().item()) == gen[i]:
            n_agree += 1
    text_head = tokenizer.decode(
        torch.cat([prompt, torch.tensor(gen[:N], dtype=torch.long)]).tolist(),
        skip_special_tokens=False)[:128]

    dtype = next(model.parameters()).dtype
    if dtype == torch.float32:
        tol, min_agree = 1e-3, N
    else:
        tol, min_agree = 8.0, 6
    return {"max_diff": max_diff, "tol": tol, "argmax_agree": f"{n_agree}/{N}",
            "min_agree": min_agree,
            "pass": max_diff < tol and n_agree >= min_agree,
            "text_head": text_head, "prompt_len": L, "n_steps": N}


# ---------------------------------------------------------------------------
# 统计与输出
# ---------------------------------------------------------------------------

def _per_step_stats(times):
    """均值/p50/p99（ms）。"""
    q = statistics.quantiles(times, n=100, method="inclusive")
    return {"mean_ms": statistics.fmean(times) * 1000,
            "p50_ms": q[49] * 1000, "p99_ms": q[98] * 1000}


def _mean_moe_times(moe_times):
    """decode 步 MoE 内部拆分均值（moe_times 非空时）。"""
    if not moe_times:
        return None
    keys = set().union(*(m.keys() for m in moe_times))
    return {k: statistics.fmean([m.get(k, 0.0) for m in moe_times])
            for k in keys}


def print_report(report):
    """终端表格（英文日志，对齐项目日志规范）。"""
    meta, prefill, decode, moe, check = (report["meta"], report["prefill"],
                                         report["decode"], report["moe_stage"],
                                         report["check"])
    line = "=" * 72
    print(f"\n{line}\nPrefill/Decode Throughput Benchmark")
    print(f"model: {meta['model_id']} | mode: {meta['mode']} | "
          f"device: {meta['gpu']} | dataset: {meta['dataset']}")
    # checkpoint 标识行：同一日志多次运行（不同 checkpoint 同 mode）时，
    # logs_parser 靠它消歧（1018.thrput.1.w4 实踩：两份 4bpw checkpoint 的
    # wxfp8 summary 头部完全同构，匹配错块）
    print(f"checkpoint: {meta['checkpoint']}")
    print(f"prompt_len: {meta['prompt_len']} | "
          f"decode_tokens: {','.join(map(str, meta['decode_tokens']))} | "
          f"git: {meta['githash']}")
    print("-" * 72)
    print("Prefill (batch=1, no cache, transformer only, median):")
    for L, r in prefill.items():
        ev = f" | event {r['tok_s_event']:.1f} tok/s" if "tok_s_event" in r else ""
        print(f"  len={L:<5} latency={r['latency_s']:.4f}s  "
              f"tok/s={r['tok_s']:.1f}{ev}")
    print("Decode (batch=1, greedy, use_cache):")
    for tier in decode:
        n = tier["n_tokens"]
        pw = tier["per_step_wall"]
        line = (f"  tokens={n:<5} wall tok/s = {tier['tok_s_wall']:.2f} "
                f"({n} tokens / {tier['wall_s']:.2f}s, excl. prefill)")
        if "tok_s_event" in tier:
            line += f" | event tok/s = {tier['tok_s_event']:.2f}"
        line += (f" | per-step wall mean/p50/p99 = "
                 f"{pw['mean_ms']:.1f}/{pw['p50_ms']:.1f}/{pw['p99_ms']:.1f} ms")
        print(line)
    first = decode[0]
    print(f"  TTFT = {first['ttft_s']:.3f}s "
          f"(prefill {first['wall_times'][0]:.3f}s + step1 {first['wall_times'][1]:.3f}s)")
    if moe:
        print("MoE stage (mean per decode step, all layers, s):")
        for k in MOE_TIMING_KEYS:
            if k in moe:
                print(f"  {k:<18} {moe[k]:.4f}")
    if check:
        verdict = "PASS" if check["pass"] else "FAIL"
        print(f"Consistency check: {verdict} "
              f"(max logits diff {check['max_diff']:.2e} < {check['tol']}, "
              f"argmax agree {check['argmax_agree']} >= {check['min_agree']})")
        print(f"  generated head: {check['text_head']!r}")
    print(f"Results: {meta['out_path']}\n{line}")


def save_report(report, args):
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    safe_id = report["meta"]["model_id"].replace("/", "_")
    safe_mode = report["meta"]["mode"].replace("(", "_").replace(")", "")
    if args.out:
        path = args.out
    else:
        path = os.path.join("results",
                            f"throughput_{safe_id}_{safe_mode}_"
                            f"{report['meta']['githash']}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    report["meta"]["out_path"] = path
    print(f"[out] saved to {path}", flush=True)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prefill/decode throughput benchmark (P0)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--quant-dir", default=None,
                   help="本项目 2bpw checkpoint 目录（配 --mode）")
    g.add_argument("--model-path", default=None,
                   help="外部 HF 模型目录（fp16/w4a8/w4a16/W8A8 等）")
    p.add_argument("--base-model", default=None,
                   help="quant-dir 模式的 base 模型路径（缺省读 meta.json）")
    p.add_argument("--mode", default="wxa16",
                   choices=["wxa16", "wxa8", "wxfp8", "wxfp4"],
                   help="quant-dir 的推理模式（默认 wxa16）")
    p.add_argument("--prefill-lens", default="16,128,2048,4096",
                   help="prefill 档位，逗号分隔（默认 16,128,2048,4096）")
    p.add_argument("--decode-tokens", default="256",
                   help="decode 长度档位，逗号分隔（默认 256）")
    p.add_argument("--prompt-len", type=int, default=128,
                   help="decode 段的 prompt 长度（默认 128）")
    p.add_argument("--prefill-reps", type=int, default=3)
    p.add_argument("--dataset", default="wiki", choices=["wiki", "c4"])
    p.add_argument("--sample-offset", type=int, default=0,
                   help="样本序列切片起点（默认 0，固定可复现）")
    p.add_argument("--no-check", action="store_true",
                   help="跳过 logits 一致性守卫")
    p.add_argument("--out", default=None, help="结果 JSON 路径（默认自动命名）")
    args = p.parse_args()
    args.prefill_lens = [int(x) for x in args.prefill_lens.split(",")]
    args.decode_tokens = [int(x) for x in args.decode_tokens.split(",")]
    return args


def main():
    args = parse_args()
    model, tokenizer, model_id, mode_label = load_bench_model(args)
    # 输入统一放 DEV（device_map=auto 的外部模型首个参数可能在 CPU 侧，
    # 不能作为输入设备依据；两卡切分时入口也在 cuda:0）
    dev = DEV
    print(f"[model] {model_id} on {dev} | mode: {mode_label}", flush=True)

    sample = load_sample_tokens(args.dataset, tokenizer)
    # 切片需覆盖 prefill 最大档、decode prompt 与一致性检查（64）；decode 生成
    # 的 token 由循环自己产生，不消耗样本
    need = max(max(args.prefill_lens), args.prompt_len, 64) + 16
    if args.sample_offset + need > len(sample):
        raise SystemExit(f"sample too short: offset {args.sample_offset} + "
                         f"{need} > {len(sample)}")
    sample = sample[args.sample_offset:args.sample_offset + need]

    warmup(model, args.prefill_lens, sample, dev)
    # 释放预热期的 JIT 临时显存，让测量相位拿到干净的分配池
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    check = None
    if not args.no_check:
        print("[check] cache-vs-full logits consistency...", flush=True)
        check = consistency_check(model, tokenizer, sample, dev)
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    print("[bench] prefill...", flush=True)
    prefill = time_prefill(model, sample, args.prefill_lens, args.prefill_reps, dev)

    # decode 多档：每档各自跑一遍 prefill + n 步 decode（每档是完整会话）
    decode_tiers = []
    all_moe_times = []
    for n in args.decode_tokens:
        print(f"[bench] decode {n} tokens...", flush=True)
        dec = decode_loop(model, sample[:args.prompt_len], n, dev)
        step_walls = dec["wall_times"][1:]  # 去掉 prefill 段
        stats = {
            "n_tokens": n,
            "wall_s": sum(step_walls),
            "tok_s_wall": n / sum(step_walls),
            "ttft_s": dec["wall_times"][0] + dec["wall_times"][1],
            "wall_times": dec["wall_times"],
            "per_step_wall": _per_step_stats(step_walls),
            "gen_tokens": dec["gen"][:n],
        }
        if dec["event_times"]:
            step_events = dec["event_times"][1:]
            stats["event_s"] = sum(step_events)
            stats["tok_s_event"] = n / sum(step_events)
            stats["per_step_event"] = _per_step_stats(step_events)
        decode_tiers.append(stats)
        all_moe_times.extend(dec["moe_times"])

    report = {
        "meta": {
            "model_id": model_id, "mode": mode_label,
            "checkpoint": args.quant_dir or args.model_path,
            "dataset": args.dataset, "sample_offset": args.sample_offset,
            "prompt_len": args.prompt_len, "decode_tokens": args.decode_tokens,
            "prefill_reps": args.prefill_reps,
            "githash": get_git_hash(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "date": datetime.now().isoformat(timespec="seconds"),
        },
        "prefill": prefill,
        "decode": decode_tiers,
        "moe_stage": _mean_moe_times(all_moe_times),
        "check": check,
    }
    try:
        import triton
        report["meta"]["triton"] = triton.__version__
    except Exception:
        pass
    try:
        import transformers
        report["meta"]["transformers"] = transformers.__version__
    except Exception:
        pass

    save_report(report, args)
    print_report(report)


if __name__ == "__main__":
    main()
