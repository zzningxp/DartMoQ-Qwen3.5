#!/usr/bin/env python3
"""pf 档单步分层 profile：wall/event + MoE stage 分解 + CUDA kernel 归类占比。

背景（1018.thrput.10 后）：P5-2 四模式合并路径落地，wxfp4 4bpw
pf2048 4110–4127，对官方 FP8（10009）仍差 2.42×（pf4096 1.85×、
decode 1.12×）。由官方 json latency 外推出两个独立问题：长档边际速率
125–131 vs 98.6 μs/tok（1.3×，kernel 效率）+ 每次 prefill 调用
~180–240ms 固定开销（官方 ~2–15ms，pf128 差距 3.18× 比 pf2048 还大）。
本脚本把每档 forward 的时间拆到 attention / dense / MoE / 搬运 / 空隙，
决定下一个优化杠杆往哪打（P8-1 同法：测量先行）。

三层数据，每档长度各跑一遍：
  1. 干净计时（profile 关）：与 bench_throughput.time_prefill 完全同协议
     （batch=1、只走 transformer、no_grad、wiki 样本、中位数）——数字
     直接可与吞吐表对照；
  2. MoE stage 分解（set_cuda_profile(True)）：逐层 CudaStageProfiler 的
     last_cuda_stats 聚合（merged_gate_up / silu_mul / merged_down /
     scatter / act_quant / gather_* 等）。注意 profile 开着每 MoE forward
     首尾各一次 synchronize，会串行化——此档只看拆分占比，不看绝对值；
  3. torch.profiler（profile 关）：CUDA kernel 按 W 读取名归类
     （moe_gemm / act_quant / attn / norm / dense_gemm / route_scatter /
     copy_misc / other）+ GPU busy 率（Σkernel / event）+ 每 forward 的
     kernel launch 数（短档固定开销归因的关键数字）。

用法（主流程，35B 全模型，本人手动跑）：
    conda activate dart312-t38
    python profile_prefill.py \
        --quant-dir models/qwen3.5-35B-4bpw-261016-turboquant --mode wxfp4 \
        --prefill-lens 16,128,2048,4096

    # 想看 decode 单步形态可加 1（prefill len=1 ≈ 无 cache 的单 token 前向）
    python profile_prefill.py --quant-dir ... --mode wxfp4 \
        --prefill-lens 1,16,128

    # 带 cache 的真 decode 单步分层 profile（2026-10-22：fla 后 decode 持平，
    # 开销账要按真 decode 步拆；口径与 bench decode_loop 一致：全模型含 lm_head、
    # bs=1 greedy、逐步 event。三层同款，重点看每步 launch 数与 idle gap）
    python profile_prefill.py --quant-dir ... --mode wxfp4 \
        --prefill-lens 128 --decode-steps 64

结果 JSON 落 results/profile_prefill_<model>_<mode>_<githash>_<ts>.json。
"""

import argparse
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

from bench_throughput import (  # noqa: E402
    _text_tower,
    load_bench_model,
    load_sample_tokens,
    warmup,
)
from data_utils import get_git_hash  # noqa: E402
from qwen35_utils import DEV  # noqa: E402

# CUDA kernel 名归类（按序优先匹配，小写子串；未命中归 other）。
# 名单按本仓库 kernel 命名 + torch 栈常见 kernel 维护，跑了第一档后
# 按 top-kernel 表回头补。
KERNEL_CATS = [
    # 我们的 Triton kernel（函数名即 kernel 名）
    ("moe_gemm", ("_wxfp4_merged", "_wxfp4_fused", "_wxa8_merged",
                  "_wxa8_matmul", "_wxfp8_matmul", "_wxa8_kernel",
                  "_wxfp8_kernel", "fused_matmul_kernel", "_wxa16_",
                  "fused_grouped", "_triton_fused")),
    ("act_quant", ("rotate_quantize", "quantize_act", "_rq_fused")),
    ("attn", ("flash", "sdpa", "scaled_dot", "attention", "attn",
              "delta_rule", "wy_prep", "chunk_")),
    ("norm", ("layer_norm", "rmsnorm", "rms_norm", "_norm_kernel")),
    ("dense_gemm", ("gemm", "cutlass", "cublas", "nvjet", "matmul_kernel",
                    "s16816", "ampere_", "sm90_", "sm100_", "wgmma")),
    ("route_scatter", ("sort", "topk", "scatter", "index_", "segment",
                       "cumsum", "gather", "radix")),
    ("copy_misc", ("memcpy", "elementwise", "vectorized", "copy_",
                   "fill", "cat_", "triu", "where", "arange")),
]


def classify_kernel(name: str) -> str:
    low = name.lower()
    for cat, pats in KERNEL_CATS:
        for p in pats:
            if p in low:
                return cat
    return "other"


def set_moe_profile(model, enabled: bool) -> int:
    """拨所有 MoE 层的 CUDA stage profile 开关，返回命中的层数。"""
    n = 0
    for _, m in model.named_modules():
        if hasattr(m, "set_cuda_profile"):
            m.set_cuda_profile(enabled)
            n += 1
    return n


def moe_layers(model):
    """所有带 last_cuda_stats 的 MoE 层。"""
    lm = _text_tower(model)
    return [l.mlp for l in lm.layers if hasattr(getattr(l, "mlp", None),
                                                "last_cuda_stats")]


def harvest_stages(mlps):
    """跨层聚合一次 forward 的 stage CUDA 时间。返回 {stage: ms}。"""
    acc = {}
    for mlp in mlps:
        for name, v in (mlp.last_cuda_stats or {}).items():
            acc[name] = acc.get(name, 0.0) + v["ms"]
    return acc


@torch.no_grad()
def pass_clean(transformer, ids, reps):
    """口径 1：干净 event/wall 计时（与 time_prefill 同协议）。"""
    walls, events = [], []
    for _ in range(reps):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        t0 = time.time()
        out = transformer(ids)
        ev1.record()
        t1 = time.time()
        walls.append(t1 - t0)
        ev0.synchronize()
        ev1.synchronize()
        events.append(ev0.elapsed_time(ev1) / 1000.0)
        del out
    return {"wall_s": statistics.median(walls),
            "event_s": statistics.median(events),
            "wall_all": walls, "event_all": events}


@torch.no_grad()
def pass_stages(transformer, ids, mlps, reps=3):
    """口径 2：stage profile 开着跑，取后 reps-1 次平均（首轮含扩容噪声）。"""
    samples = []
    for _ in range(reps):
        transformer(ids)
        samples.append(harvest_stages(mlps))
    acc, n = {}, len(samples) - 1
    for s in samples[1:]:
        for k, v in s.items():
            acc[k] = acc.get(k, 0.0) + v / n
    return acc


@torch.no_grad()
def pass_profiler(transformer, ids, iters):
    """口径 3：torch.profiler 聚合 CUDA kernel 分类 + launch 数。

    返回 {"per_iter_s": event 均值, "cats": {cat: ms/iter},
          "launches": 每 forward kernel 数, "busy": Σkernel/event,
          "top": [(name, cat, ms/iter, count/iter)]}
    """
    from torch.profiler import ProfilerActivity, profile
    # 先捂热（profiler 自身有固定开销，进 profile 前先跑一遍同形状）
    transformer(ids)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            transformer(ids)
        torch.cuda.synchronize()
    ev1.record()
    ev0.synchronize()
    ev1.synchronize()
    per_iter_s = ev0.elapsed_time(ev1) / 1000.0 / iters

    cats, launches, rows = {}, 0, []
    for evt in prof.key_averages():
        dev_t = getattr(evt, "self_device_time_total", 0.0) or 0.0   # µs
        is_kernel = (str(getattr(evt, "device_type", "")).endswith("CUDA")
                     and dev_t > 0)
        if not is_kernel:
            continue
        ms = dev_t / 1000.0 / iters          # 每 forward 毫秒
        count = evt.count / iters
        launches += count
        cat = classify_kernel(evt.key)
        cats[cat] = cats.get(cat, 0.0) + ms
        rows.append((evt.key, cat, ms, count))
    rows.sort(key=lambda r: -r[2])
    total_kernel_ms = sum(cats.values())
    return {
        "per_iter_s": per_iter_s,
        "cats": cats,
        "launches": launches,
        "busy": (total_kernel_ms / 1000.0 / per_iter_s) if per_iter_s > 0 else 0.0,
        "top": rows[:15],
    }


@torch.no_grad()
def decode_run_steps(model, sample, prompt_len, n_steps, dev, on_step=None):
    """按 bench decode_loop 同口径连跑 n_steps 个单 token 步（bs=1 greedy + cache）。

    step 0 是 prompt prefill（建 cache，不计时）；每步 CUDA event 计时
    （含逐步 argmax/.item() 的同步，与 bench 完全一致）。
    on_step(out, ms) 在每步计时后回调（stages/profiler 聚合用）。"""
    ids = sample[:prompt_len].unsqueeze(0).to(dev)
    mask = torch.ones(1, prompt_len, dtype=torch.long, device=dev)
    out = model(input_ids=ids, attention_mask=mask,
                past_key_values=None, use_cache=True)
    past = out.past_key_values
    next_tok = int(out.logits[0, -1].argmax().item())
    del out
    ids = torch.tensor([[next_tok]], dtype=torch.long, device=dev)
    mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=dev)], dim=1)

    step_ms = []
    for _ in range(n_steps):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        out = model(input_ids=ids, attention_mask=mask,
                    past_key_values=past, use_cache=True)
        ev1.record()
        past = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax().item())
        ev0.synchronize()
        ev1.synchronize()
        ms = ev0.elapsed_time(ev1)
        step_ms.append(ms)
        if on_step is not None:
            on_step(out, ms)
        del out
        ids = torch.tensor([[next_tok]], dtype=torch.long, device=dev)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=dev)], dim=1)
    del past
    return step_ms


def decode_pass_clean(model, sample, prompt_len, n_steps, reps, dev):
    """decode 干净计时：reps 轮（每轮重新 prefill 建 cache），中位每步毫秒。"""
    all_ms = []
    for _ in range(reps):
        all_ms.extend(decode_run_steps(model, sample, prompt_len, n_steps, dev))
        torch.cuda.empty_cache()
    return {"median_ms": statistics.median(all_ms), "all_ms": all_ms}


def decode_pass_stages(model, mlps, sample, prompt_len, n_steps, dev):
    """decode 的 MoE stage 分解：逐步 harvest 取平均（profile 开着只看占比）。"""
    acc, n = {}, 0

    def on_step(out, ms):
        nonlocal n
        for k, v in harvest_stages(mlps).items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1

    decode_run_steps(model, sample, prompt_len, n_steps, dev, on_step=on_step)
    return {k: v / n for k, v in acc.items()}


def decode_pass_profiler(model, sample, prompt_len, n_steps, dev):
    """decode 的 torch.profiler 口径：kernel 分类/launch 数按每步归一。"""
    from torch.profiler import ProfilerActivity, profile

    decode_run_steps(model, sample, prompt_len, 2, dev)  # 捂热
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step_ms = decode_run_steps(model, sample, prompt_len, n_steps, dev)
        torch.cuda.synchronize()
    ev1.record()
    ev0.synchronize()
    ev1.synchronize()

    cats, launches, rows = {}, 0, []
    for evt in prof.key_averages():
        dev_t = getattr(evt, "self_device_time_total", 0.0) or 0.0
        if not (str(getattr(evt, "device_type", "")).endswith("CUDA") and dev_t > 0):
            continue
        ms = dev_t / 1000.0 / n_steps
        count = evt.count / n_steps
        launches += count
        cat = classify_kernel(evt.key)
        cats[cat] = cats.get(cat, 0.0) + ms
        rows.append((evt.key, cat, ms, count))
    rows.sort(key=lambda r: -r[2])
    return {
        "median_ms": statistics.median(step_ms),
        "cats": cats,
        "launches": launches,
        "busy": (sum(cats.values()) / statistics.median(step_ms)) if step_ms else 0.0,
        "top": rows[:15],
    }


def report_decode(prompt_len, n_steps, clean, stages, prof, n_layers):
    """打印 decode 档的合并报告（运行期输出英文）。"""
    med = clean["median_ms"]
    print(f"\n===== decode (prompt {prompt_len}, {n_steps} steps)  "
          f"median {med:.1f}ms/step  ({1000.0 / med:.2f} tok/s) =====")

    stage_total = sum(stages.values())
    if stages:
        print(f"[MoE stages] (CUDA-event, {n_layers} layers, avg per step)")
        for k, v in sorted(stages.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<22}{v:>8.2f} ms  {v / stage_total * 100:>5.1f}% of MoE")
        print(f"  {'SUM(stages)':<22}{stage_total:>8.2f} ms"
              f"  ({stage_total / med * 100:.1f}% of step)")

    cats = prof["cats"]
    kernel_total = sum(cats.values())
    print(f"[kernel cats] (profiler, ms/step; launches={prof['launches']:.0f}, "
          f"GPU busy {prof['busy'] * 100:.1f}% of median)")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14}{v:>8.2f} ms  {v / med * 100:>5.1f}% of step")
    print(f"  {'SUM(kernels)':<14}{kernel_total:>8.2f} ms  "
          f"(gap {med - kernel_total:.1f} ms idle/unattributed)")
    print("[top kernels]")
    for name, cat, ms, cnt in prof["top"]:
        print(f"  {ms:>8.2f} ms  x{cnt:>6.1f}  [{cat}]  {name[:96]}")


def report_len(L, clean, stages, prof, n_layers):
    """打印单档的合并报告（运行期输出英文）。"""
    ev_s = clean["event_s"]
    print(f"\n===== len={L}  wall {clean['wall_s'] * 1000:.1f}ms  "
          f"event {ev_s * 1000:.1f}ms  ({L / ev_s:.1f} tok/s) =====")

    stage_total = sum(stages.values())
    print(f"[MoE stages] (CUDA-event, {n_layers} layers, avg of profiled reps)")
    for k, v in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22}{v:>8.2f} ms  {v / stage_total * 100:>5.1f}% of MoE")
    print(f"  {'SUM(stages)':<22}{stage_total:>8.2f} ms")

    cats = prof["cats"]
    kernel_total = sum(cats.values())
    print(f"[kernel cats] (profiler, ms/forward; launches={prof['launches']:.0f}, "
          f"GPU busy {prof['busy'] * 100:.1f}% of event)")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14}{v:>8.2f} ms  {v / ev_s * 100:>5.1f}% of event")
    print(f"  {'SUM(kernels)':<14}{kernel_total:>8.2f} ms  "
          f"(gap {ev_s * 1000 - kernel_total:.1f} ms idle/unattributed)")
    print("[top kernels]")
    for name, cat, ms, cnt in prof["top"]:
        print(f"  {ms:>8.2f} ms  x{cnt:>6.1f}  [{cat}]  {name[:96]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--model-path", default=None,
                    help="官方/外部模型路径（bench --model-path 同款直通；"
                         "给官方 FP8/Int4 出 decode 分层账，MoE stage 段自动为空）")
    ap.add_argument("--mode", default="wxa16",
                    choices=["wxa16", "wxa8", "wxfp8", "wxfp4"])
    ap.add_argument("--prefill-lens", default="16,128,2048,4096")
    ap.add_argument("--decode-steps", type=int, default=0,
                    help=">0 时追加带 cache 的真 decode 单步分层 profile"
                         "（bench decode_loop 同口径，含 lm_head）")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--profile-iters", type=int, default=3)
    ap.add_argument("--dataset", default="wiki", choices=["wiki", "c4"])
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--out", default=None, help="结果 JSON 路径（默认自动命名）")
    args = ap.parse_args()
    lens = [int(x) for x in args.prefill_lens.split(",")]

    model, tokenizer, model_id, mode_label = load_bench_model(args)
    sample = load_sample_tokens(args.dataset, tokenizer)
    warmup(model, lens, sample, DEV)
    transformer = _text_tower(model)
    mlps = moe_layers(model)
    n_layers = len(mlps)
    print(f"[profile] MoE layers with stage stats: {n_layers}", flush=True)

    report = {"meta": {
        "model_id": model_id, "mode": mode_label,
        "dataset": args.dataset, "sample_offset": args.sample_offset,
        "githash": get_git_hash(),
        "gpu": torch.cuda.get_device_name(0),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
    }, "lens": {}}

    for L in lens:
        print(f"\n[profile] len={L}: clean timing ({args.reps} reps)...",
              flush=True)
        ids = sample[:L].unsqueeze(0).to(DEV)

        set_moe_profile(model, False)
        clean = pass_clean(transformer, ids, args.reps)

        set_moe_profile(model, True)
        stages = pass_stages(transformer, ids, mlps)
        set_moe_profile(model, False)

        print(f"[profile] len={L}: torch.profiler ({args.profile_iters} iters)...",
              flush=True)
        prof = pass_profiler(transformer, ids, args.profile_iters)

        report_len(L, clean, stages, prof, n_layers)
        report["lens"][str(L)] = {
            "clean": {k: v for k, v in clean.items() if not k.endswith("_all")},
            "stages": stages, "profiler": {
                "per_iter_s": prof["per_iter_s"], "cats": prof["cats"],
                "launches": prof["launches"], "busy": prof["busy"],
                "top": [{"name": n, "cat": c, "ms_per_fwd": m, "count": ct}
                        for n, c, m, ct in prof["top"]],
            },
        }
        del ids
        torch.cuda.empty_cache()

    if args.decode_steps > 0:
        print(f"\n[profile] decode: {args.decode_steps} steps "
              f"(prompt {args.prompt_len}) clean timing ({args.reps} reps)...",
              flush=True)
        set_moe_profile(model, False)
        clean = decode_pass_clean(model, sample, args.prompt_len,
                                  args.decode_steps, args.reps, DEV)
        set_moe_profile(model, True)
        stages = decode_pass_stages(model, mlps, sample, args.prompt_len,
                                    args.decode_steps, DEV)
        set_moe_profile(model, False)
        print(f"[profile] decode: torch.profiler ...", flush=True)
        prof = decode_pass_profiler(model, sample, args.prompt_len,
                                    args.decode_steps, DEV)
        report_decode(args.prompt_len, args.decode_steps, clean, stages, prof,
                      n_layers)
        report["decode"] = {
            "prompt_len": args.prompt_len, "steps": args.decode_steps,
            "median_ms": clean["median_ms"],
            "stages": stages,
            "profiler": {
                "median_ms": prof["median_ms"], "cats": prof["cats"],
                "launches": prof["launches"], "busy": prof["busy"],
                "top": [{"name": n, "cat": c, "ms_per_step": m, "count": ct}
                        for n, c, m, ct in prof["top"]],
            },
        }
        torch.cuda.empty_cache()

    out = args.out or os.path.join(
        "results", f"profile_prefill_{model_id}_{mode_label}_"
                   f"{get_git_hash()}_{datetime.now().strftime('%y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[profile] saved: {out}")


if __name__ == "__main__":
    main()
