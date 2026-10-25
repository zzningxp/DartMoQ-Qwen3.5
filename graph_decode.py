#!/usr/bin/env python3
"""CUDA Graph decode PoC（P9-5 官方 FP8 先行，2026-10-23 立项，backlog §P9-5）。

背景：官方 FP8 decode 实测 4614 launch/步、GPU busy 12.6%、idle gap 57.5ms
（84% of 68.4ms，1023.profile.decode 双跑逐位一致）——dispatch 主导，kernel
优化碰不到那 57.5ms，只有把整步捕获成 CUDA graph 重放才能吃掉。本脚本在
transformers 框架内做最小闭环验证（mini decode 引擎雏形）：

  1) 正确性（硬判据）：closed-loop greedy token 序列与 eager decode_loop
     完全一致；teacher-forced 逐步 logits 对拍为信息性软判据；
  2) 速度：replay-only 事件计时（理论上限参考）+ 带 .item() 同步的整循环
     （bench decode_loop 同口径，可直接对 68.4ms）；
     --steps / --prompt-len 均接受逗号多档（steps 64,256,512,1024 =
     profile decode 档 + bench decode 三档；prompt 16,128,2048,4096 =
     bench prefill 四档），全交叉出表；capture 是单步图，与 prompt 长度
     无关，一次捕获全档复用。eager 参考每档 prompt 只跑最长 steps 档——
     greedy 逐 token 生成，短档序列必然是最长档前缀，前缀对拍即可；
     logits 对拍只在首档 prompt（capture 保真度与上下文长度无关）。

静态化依赖（transformers 5.20 cache_utils 源码已核实）：
  - StaticCache：full-attn KV 预分配槽位，cumulative_length 是设备张量
    in-place 递增（lazy_initialization 里 .to(device) + mark_static_address，
    注释明言为 cudagraphs），写入走 index_copy_；
  - linear attention 层（GDN）：conv/recurrent state copy_ 原地更新 +
    静态地址（同为 cudagraphs 设计）；
  - position_ids 传 (4,1,1) 设备 buffer，逐步 fill_ 更新（绕开框架内
    get_seq_length 的 host 分支）；
  - mask 两模式：native（框架自建，StaticCache 下形状静态，值若走设备
    张量则 capture 正确——PoC 直接检验）/ buffer（自持 (1,1,1,max_len)
    加性 mask 逐步原地更新，语义完全自控）。默认 native，capture 报错或
    对拍 FAIL 时换 buffer。

风险点（PoC 直接暴露，失败即信息）：fla recurrent kernel 捕获、fbgemm
w8a8 hub kernel 捕获、双卡 device_map 跨设备拷贝的 capture 合法性
（accelerate hook 是 host 代码，capture 期照常执行，只要逐步路径恒定）。

用法（35B 全模型，本人手动跑）：
    conda activate dart312
    export DARTMOQ_TRUST_HUB_KERNELS=1 HF_ENDPOINT=https://hf-mirror.com \
           HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1
    python graph_decode.py --model-path ~/models/Qwen3.5-35B-A3B-FP8/ \
        --strategy segment --prompt-len 16,128,2048,4096 --steps 64,256,512,1024

    # 双卡官方 FP8 专用说明（2026-10-23 判定）：全步 capture 与 device_map 的
    # 跨设备拷贝结构性互斥（探针 test/test_graph_capture_crossdev.py），单卡
    # 又装不下（language+lm_head 33.26GiB > 32GB）→ 唯一框架内路线是分段：
    # --strategy segment 按层切两段各自捕获，跨设备拷贝留图外执行。

    # 我方模型（后续对照）：
    conda activate dart312-t38
    export CUDA_VISIBLE_DEVICES=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
    python graph_decode.py \
        --quant-dir models/qwen3.5-35B-4bpw-261016-gptq --mode wxfp4 \
        --prompt-len 16,128,2048,4096 --steps 64,256,512,1024

结果 JSON 落 results/graph_decode_<model>_<mode>_<githash>_<ts>.json。
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback
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
)
from data_utils import get_git_hash  # noqa: E402

# nsys 剖析窗口开关（run.nsys.graph.sh 配套）：图重放与 host 路径共用同一批
# merged kernel（名字相同），kern_sum 按名字聚合无法区分来源，须用
# cudaProfilerApi 时间窗只录 pass B 计时循环；nsys 侧
# --capture-range=cudaProfilerApi --capture-range-end=stop 在首个 stop 截止，
# 故只采第一个 (plen, steps, rep) 档的窗口
_PROF_WINDOW = os.environ.get("DARTMOQ_PROF_WINDOW", "0") == "1"


def emb_device(model):
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def eager_reference(model, sample, prompt_len, steps):
    """bench decode_loop 同款 greedy（DynamicCache 默认路径），收 token/logits/计时。

    返回 tokens（steps+1 个）、logits（steps+1 份 cpu）、decode 段每步 ms。"""
    dev = emb_device(model)
    ids = sample[:prompt_len].unsqueeze(0).to(dev)
    mask = torch.ones(1, prompt_len, dtype=torch.long, device=dev)
    out = model(input_ids=ids, attention_mask=mask,
                past_key_values=None, use_cache=True)
    past = out.past_key_values
    tokens, logits, step_ms = [int(out.logits[0, -1].argmax().item())], \
        [out.logits[0, -1].float().cpu()], []
    ids = torch.tensor([[tokens[0]]], dtype=torch.long, device=dev)
    mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=dev)], dim=1)

    for _ in range(steps):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        out = model(input_ids=ids, attention_mask=mask,
                    past_key_values=past, use_cache=True)
        ev1.record()
        past = out.past_key_values
        tok = int(out.logits[0, -1].argmax().item())
        ev0.synchronize()
        ev1.synchronize()
        step_ms.append(ev0.elapsed_time(ev1))
        tokens.append(tok)
        logits.append(out.logits[0, -1].float().cpu())
        del out
        ids = torch.tensor([[tok]], dtype=torch.long, device=dev)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=dev)], dim=1)
    del past
    torch.cuda.empty_cache()
    return tokens, logits, step_ms


def dump_exc_chain(tag, e, max_depth=8, tail_frames=4):
    """capture 失败聚合报错的解包打印。

    capture 期真实非法调用若在 fn() 内部先抛，graph.__exit__ 的 capture_end
    二次报错会把它链到 __context__ 上，表层只剩「previous error during
    capture」。沿 __context__/__cause__ 链逐层打印类型+首行消息+最内层
    traceback 帧（含模型文件行号），可直接看到是哪个 op、哪一行触发。"""
    seen, cur, depth = set(), e, 0
    while cur is not None and id(cur) not in seen and depth < max_depth:
        seen.add(id(cur))
        msg = str(cur).splitlines()[0] if str(cur) else ""
        print(f"[{tag}] exc[{depth}]: {type(cur).__name__}: {msg}", flush=True)
        for fr in traceback.extract_tb(cur.__traceback__)[-tail_frames:]:
            print(f"[{tag}]   at {os.path.basename(fr.filename)}:"
                  f"{fr.lineno} in {fr.name}", flush=True)
        cur = cur.__context__ or cur.__cause__
        depth += 1


def reset_cache(cache, dev):
    """把 StaticCache 复位到「未写」状态（复用同一批 buffer，不重新分配）。

    full-attn 层：cumulative_length.zero_()（设备张量 in-place）+ KV 清零；
    linear-attn 层：优先走自带 reset()，否则 conv/recurrent state 清零。
    has_previous_state / is_initialized 等 python 标志保持 True（正合意：
    懒初始化与首写分支已在 prefill 走过，decode 路径恒定）。"""
    n_full = n_lin = 0
    for layer in cache.layers:
        if hasattr(layer, "keys"):  # StaticLayer（含滑动/索引变体）
            if layer.keys is None:
                # 惰性层尚未分配：首写 update 时按层所在设备建 buffer，
                # 无位可复位（seg 路径 reset 在首次 prefill 之前会走到）
                continue
            layer.cumulative_length.zero_()
            layer.keys.zero_()
            layer.values.zero_()
            n_full += 1
        elif hasattr(layer, "conv_states"):  # LinearAttentionLayer
            try:
                layer.reset()
            except Exception:
                for i in range(len(layer.conv_states)):
                    if layer.conv_states[i] is not None:
                        layer.conv_states[i].zero_()
                    if layer.recurrent_states[i] is not None:
                        layer.recurrent_states[i].zero_()
            n_lin += 1
    torch.cuda.synchronize()
    return n_full, n_lin


@torch.no_grad()
def static_prefill(model, sample, prompt_len, cache):
    """静态 cache 下的 eager prefill（图外），返回末位 logits。"""
    dev = emb_device(model)
    ids = sample[:prompt_len].unsqueeze(0).to(dev)
    out = model(input_ids=ids, attention_mask=None,
                past_key_values=cache, use_cache=True)
    logits = out.logits[0, -1].float().cpu()
    del out
    return logits


def build_layer_device_map(model_path, split_at):
    """官方 FP8 双卡分段加载参数：device_map 固定 auto + max_memory 限卡0
    容量，由 accelerate 顺序贪心落卡。手写字典路线已实测否决（1023.graph.
    seg.off：层键虽匹配 hook 检查，量化权重仍按 checkpoint 顺序全堆 GPU1，
    OOM 在 lm_head 512MiB），auto 路径 bench 长期验证有效。实际切分层不
    在此定——加载后按每层参数真实设备反推（seg_main），split_at 仅用于
    估算卡0容量上限（意图切分点）。"""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True).get_text_config()
    n = cfg.num_hidden_layers
    split = split_at if 0 < split_at < n else n // 2
    # 单层均摊 0.81GiB（1023 权重扫描：language 32.3GiB/40 层，fp8 口径）；
    # 1.2GiB = embed+rotary+余量；卡1 cap 给足（norm/lm_head/visual/mtp 都落卡1）
    cap0 = 1.2 + split * 0.81
    max_memory = {0: f"{cap0:.1f}GiB", 1: "23GiB"}
    return "auto", max_memory, split


def run_eager_refs(model, sample, plen_list, steps_list, max_steps):
    """eager 参考（DynamicCache 默认路径，每档 prompt 只跑最长 steps 档）。
    greedy 逐 token 生成，短档序列必然是最长档前缀，前缀对拍即可；logits
    只留首档 prompt（对拍是软判据，capture 保真度与上下文长度无关）。"""
    refs = {}        # plen -> (tokens, step_ms)
    ref_logits = None
    eager_med = {}   # (plen, steps) -> ms
    for plen in plen_list:
        print(f"[graph] eager reference decode (plen={plen}, {max_steps} steps, "
              f"covers {steps_list} as prefixes) ...", flush=True)
        toks, lgs, ms = eager_reference(model, sample, plen, max_steps)
        refs[plen] = (toks, ms)
        if ref_logits is None:
            ref_logits = lgs
        else:
            del lgs
        for s in steps_list:
            eager_med[(plen, s)] = statistics.median(ms[:s])
            print(f"[graph] eager plen={plen:4d} {s:4d} steps: median "
                  f"{eager_med[(plen, s)]:.1f}ms/step "
                  f"({1000.0 / eager_med[(plen, s)]:.2f} tok/s)", flush=True)
    return refs, ref_logits, eager_med


def seg_main(args, model, model_id, mode_label, sample):
    """分段 capture 主流程：手写逐层 decode step（embed→rotary→40 层→norm→
    lm_head，绕开 wrapper——wrapper 内的跨设备 .to() 一进 capture 即非法），
    按设备边界切两段各自捕获，跨设备拷贝留图外。每步 host 调用从 4614 次
    launch 塌缩为 ~15 次（feed 填充 + rotary + 两段 replay + 拷贝 + argmax）。
    正确性两道门：capture 前手写 eager 对 wrapper 的漂移有界门（closed-loop 允许
    在真平局步翻转 + teacher-forced logits max_abs 有界，StaticCache 满槽 vs
    DynamicCache 逐步 cat 的 SDPA 布局差异必然带来 bf16 末位漂移）；capture 后
    graph 重放对手写 eager 的 bitwise 保真门（同 buffer 重放应逐位一致，硬判据）。"""
    from transformers import StaticCache

    text = _text_tower(model)
    types = list(getattr(text.config, "layer_types", []) or [])
    n = len(text.layers)
    devs = [next(l.parameters()).device for l in text.layers]
    # 切分层由加载后的真实设备分布反推（auto+max_memory 只保证近似切在
    # 意图位置）；分段 capture 的前提是恰好一个设备边界
    boundaries = [i for i in range(1, n) if devs[i] != devs[i - 1]]
    if len(boundaries) != 1:
        from collections import Counter
        print(f"[seg] BAD placement: boundaries at {boundaries}, "
              f"layer devices: {Counter(str(d) for d in devs)}", flush=True)
        print(f"[seg] hf_device_map: {getattr(model, 'hf_device_map', None)}",
              flush=True)
        raise SystemExit("[seg] need exactly one device boundary between the "
                         "two GPUs (tune cap0 in build_layer_device_map)")
    split = boundaries[0]
    dev0, dev1 = devs[0], devs[-1]
    norm_dev, lm_dev = (next(text.norm.parameters()).device,
                        next(model.lm_head.parameters()).device)
    if norm_dev != dev1 or lm_dev != dev1:
        raise SystemExit(f"[seg] norm/lm_head must sit on the second device "
                         f"(norm={norm_dev}, lm_head={lm_dev})")
    emb_dev = text.embed_tokens.weight.device
    if emb_dev != dev0:
        raise SystemExit(f"[seg] embed_tokens on {emb_dev}, expected {dev0}")
    # rotary 的 inv_freq 小 buffer 若被 auto 放到卡1，显式挪回卡0
    # （seg_a 图内在卡0算位置编码；拷贝图外一次性，不进 capture）
    inv = getattr(text.rotary_emb, "inv_freq", None)
    if inv is not None and inv.device != dev0:
        old_dev = inv.device
        inv.data = inv.data.to(dev0)
        print(f"[seg] moved rotary inv_freq {old_dev} -> {dev0}", flush=True)
    want = args.split_at if 0 < args.split_at < n else None
    note = f" (intended {want})" if want is not None and want != split else ""
    print(f"[seg] {n} layers: [0:{split}] on {dev0}, [{split}:{n}] on {dev1}, "
          f"norm/lm_head on {dev1}{note}", flush=True)

    steps_list = [int(s) for s in str(args.steps).split(",") if s.strip()]
    plen_list = [int(p) for p in str(args.prompt_len).split(",") if p.strip()]
    max_steps = max(steps_list)
    max_cache_len = max(plen_list) + max_steps + args.warmup_steps + 16
    p0 = plen_list[0]

    # ---- eager 参考（与 whole 策略同一份口径）----
    refs, ref_logits, eager_med = run_eager_refs(
        model, sample, plen_list, steps_list, max_steps)

    # ---- 静态 cache + 两卡各自的静态 buffer ----
    static_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    # 激活 dtype 必须取 embed 权重 dtype（= wrapper 激活 dtype）：官方多模态
    # checkpoint 参数迭代序 vision 塔在前，next(model.parameters()).dtype 可能
    # 是 fp32/fp16——h_mid_in 若按它建 fp32，seg_b 整段激活走 fp32，norm 输出
    # fp32 直量 fp8（单次舍入）vs wrapper bf16 中转（双次舍入），fp8 仅 3 位
    # 尾数，大量元素差 1 ulp → 每步恒定 ~0.4 的系统性 logits 漂移（1023.seg.3
    # 定位，gate drift 曲线基线 0.4 不随步增长即此特征）
    dt = text.embed_tokens.weight.dtype
    H = text.config.hidden_size
    ids_buf = torch.zeros(1, 1, dtype=torch.long, device=dev0)
    pos_buf = torch.zeros(4, 1, 1, dtype=torch.long, device=dev0)
    pos_buf1 = torch.zeros(4, 1, 1, dtype=torch.long, device=dev1)
    dummy1 = torch.zeros(1, 1, H, dtype=text.embed_tokens.weight.dtype,
                         device=dev1)
    has_fa0 = any(types[i] == "full_attention" for i in range(split))
    has_fa1 = any(types[i] == "full_attention" for i in range(split, n))
    mask_f0 = (torch.zeros(1, 1, 1, max_cache_len, dtype=dt, device=dev0)
               if has_fa0 else None)
    mask_f1 = (torch.zeros(1, 1, 1, max_cache_len, dtype=dt, device=dev1)
               if has_fa1 else None)
    h_mid_in = torch.zeros(1, 1, H, dtype=dt, device=dev1)
    # 两卡各自的 rotary cos/sin 静态 buffer：rotary 模块经 bisect 判明是
    # capture 非法源（embed 单独可捕、+rotary 即 invalidated，1023.seg.4），
    # 故整体提图外——feed() 每步预计算后拷入，段内只消费静态 buffer
    # rotary 预计算用 dummy 的 dtype 必须与 embed 输出一致（= embed 权重
    # dtype）：wrapper 路径的 cos/sin 从 embed 输出推导，若 dummy 用 fp8
    # 模型首个参数的 dtype（可能 fp32）会产生精度级漂移，gate 会在近似
    # 平局步翻转（1023.seg.dbg：DIFF@14）
    rdt = text.embed_tokens.weight.dtype
    dummy0 = torch.zeros(1, 1, H, dtype=rdt, device=dev0)
    # rotary 的 AlignDevicesHook 把调用路由到 dispatch 时记下的
    # execution_device（cuda:1）：即使 inv_freq 已搬到 cuda:0，hook 仍把输入
    # 搬到 cuda:1 执行 → hooked 调用产出的 pe 实际在 cuda:1（1024 第四轮
    # 实锤：hookless capture 时 layer3 apply_rotary 的 q(cuda:0)×cos(cuda:1)
    # 设备不匹配才暴露）。cos/sin 不落本段消费侧设备则两条路都死：
    # ① hooked capture 时逐层 hook 的 send_to_device 跨设备拷贝判死（首轮
    # 异常链的 tuple genexpr 即此）；② hookless 时直接设备不匹配。
    # 故 rotary 一律走未包装 forward（_old_forward，在 dummy 所在卡本地算），
    # 再显式 .to(段设备) 兜底；feed() 在计时循环内，同路径避免每步两次 D2D。
    rotary_raw = getattr(text.rotary_emb, "_old_forward", None) or \
        text.rotary_emb.forward
    pe0 = rotary_raw(dummy0, pos_buf[1:])
    cos0_buf, sin0_buf = pe0[0].to(dev0).clone(), pe0[1].to(dev0).clone()
    del pe0
    pe1 = rotary_raw(dummy1, pos_buf1[1:])
    cos1_buf, sin1_buf = pe1[0].to(dev1).clone(), pe1[1].to(dev1).clone()
    del pe1

    layers = text.layers

    def seg_a():
        """卡0 段：embed + layers[0:split]；rotary 已提图外，吃静态 cos/sin。"""
        h = text.embed_tokens(ids_buf)
        pe = (cos0_buf, sin0_buf)
        for i in range(split):
            m = mask_f0 if types[i] == "full_attention" else None
            h = layers[i](h, position_embeddings=pe, attention_mask=m,
                          past_key_values=static_cache, use_cache=True)
        return h

    def seg_b(h):
        """卡1 段：layers[split:] + norm + lm_head。"""
        pe1_ = (cos1_buf, sin1_buf)
        for i in range(split, n):
            m = mask_f1 if types[i] == "full_attention" else None
            h = layers[i](h, position_embeddings=pe1_, attention_mask=m,
                          past_key_values=static_cache, use_cache=True)
        return model.lm_head(text.norm(h))

    def manual_step():
        """eager 手写整步（warmup + 正确性门用）。"""
        h = seg_a()
        h_mid_in.copy_(h)
        return seg_b(h_mid_in)

    def feed(pos, token):
        """图外静态 buffer 更新：两卡各自的 mask/位置与 rotary cos/sin。

        rotary 走 rotary_raw（未包装 forward）：hook 会把调用路由到
        cuda:1 再 D2D 搬回，本函数在计时循环内，必须本卡直算。"""
        ids_buf.fill_(token)
        pos_buf.fill_(pos)
        pos_buf1.fill_(pos)
        for m in (mask_f0, mask_f1):
            if m is not None:
                m.fill_(float("-inf"))
                m[..., : pos + 1].zero_()
        pe0_ = rotary_raw(dummy0, pos_buf[1:])
        cos0_buf.copy_(pe0_[0])
        sin0_buf.copy_(pe0_[1])
        pe1_ = rotary_raw(dummy1, pos_buf1[1:])
        cos1_buf.copy_(pe1_[0])
        sin1_buf.copy_(pe1_[1])

    def _locate_drift():
        """漂移超界时定位首个分歧层：wrapper（DynamicCache）逐层 hook 快照 vs
        手写循环（StaticCache）逐层快照，teacher-forced 喂同一 token，逐层 max-abs。
        layer -1 = embed 输出，layer 0..n-1 = 各 decoder layer 输出。"""
        diag_steps = 4
        print(f"[seg] locating drift source ({diag_steps} teacher-forced steps) ...",
              flush=True)
        print(f"[seg] dtypes: first_param={next(model.parameters()).dtype} "
              f"embed={text.embed_tokens.weight.dtype} "
              f"mask_f0={mask_f0.dtype if mask_f0 is not None else None} "
              f"h_mid_in={h_mid_in.dtype} cos0={cos0_buf.dtype}", flush=True)

        # --- 手写循环侧：逐层快照（先复位到与 wrapper 同一起点：prefill 后 decode）---
        reset_cache(static_cache, dev0)
        static_prefill(model, sample, p0, static_cache)
        manual_hs = []  # [step] -> {layer: fp32 cpu}, layer in -1..n-1
        for s in range(diag_steps):
            feed(p0 + s, refs[p0][0][s])
            h = text.embed_tokens(ids_buf)
            hs = {-1: h.detach().float().cpu()}
            pe = (cos0_buf, sin0_buf)
            for i in range(split):
                m = mask_f0 if types[i] == "full_attention" else None
                h = layers[i](h, position_embeddings=pe, attention_mask=m,
                              past_key_values=static_cache, use_cache=True)
                hs[i] = h.detach().float().cpu()
            h_mid_in.copy_(h)
            pe1_ = (cos1_buf, sin1_buf)
            for i in range(split, n):
                m = mask_f1 if types[i] == "full_attention" else None
                h = layers[i](h_mid_in, position_embeddings=pe1_, attention_mask=m,
                              past_key_values=static_cache, use_cache=True)
                hs[i] = h.detach().float().cpu()
            manual_hs.append(hs)

        # --- wrapper 侧：逐层 hook 快照（prefill 一次 + diag_steps 次 decode）---
        dev = emb_device(model)
        rec = {i: [] for i in range(-1, n)}
        handles = []

        def _hook(i):
            def h(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                rec[i].append(t.detach().float().cpu())
            return h

        handles.append(text.embed_tokens.register_forward_hook(_hook(-1)))
        for i, l in enumerate(layers):
            handles.append(l.register_forward_hook(_hook(i)))
        ids = sample[:p0].unsqueeze(0).to(dev)
        mask = torch.ones(1, p0, dtype=torch.long, device=dev)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask,
                        past_key_values=None, use_cache=True)
        past = out.past_key_values
        for s in range(diag_steps):
            ids = torch.tensor([[refs[p0][0][s]]], dtype=torch.long, device=dev)
            mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=dev)],
                             dim=1)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask,
                            past_key_values=past, use_cache=True)
            past = out.past_key_values
        for hd in handles:
            hd.remove()
        # rec[i] = [prefill, decode0, decode1, ...]，decode 步 s -> rec[i][s+1]
        for s in range(diag_steps):
            diffs = [(manual_hs[s][i] - rec[i][s + 1]).abs().max().item()
                     for i in range(-1, n)]
            first = next((i for i, d in enumerate(diffs) if d > 1e-2), None)
            print(f"[seg] drift step {s}: first-diverging layer = "
                  f"{first if first is not None else 'none'}", flush=True)
            if s == 0:
                print(f"[seg]   per-layer diff (embed..last) = "
                      f"{['%.1e' % d for d in diffs]}", flush=True)

    # ---- warmup（teacher-forced 手写 eager，触发惰性编译与 GDN decode 分支）----
    print(f"[seg] warmup {args.warmup_steps} manual-eager steps ...", flush=True)
    reset_cache(static_cache, dev0)
    static_prefill(model, sample, p0, static_cache)
    for k in range(args.warmup_steps):
        feed(p0 + k, refs[p0][0][k])
        manual_step()
    torch.cuda.synchronize()

    # ---- 正确性门（漂移有界版）：16 步 closed-loop 对拍 + teacher-forced 漂移 ----
    # 机制已实证（1023.seg.4 逐层定位）：手写循环（StaticCache 满槽 buffer，
    # softmax 归约覆盖 max_cache_len 个 key，mask 屏蔽尾槽）与 wrapper decode
    #（DynamicCache 精确长度 + SDPA is_causal 捷径）归约分组不同 → 偶发
    # 1 bf16 ulp（大部分步在 bf16 分辨率下逐位一致，layer0-3=0.0）→ 1 ulp 在
    # prefill 期进入 GDN conv/recurrent 状态（decode 显形，首见于 layer4）→
    # 逐层累积 ~1e-2 → 256-expert top-8 路由翻转离散放大（layer20 处
    # 1.6e-2→0.33）→ logits 漂移 ~1.5（plen=128/16 步实测）。此为静态重实现
    # 的固有实现差，wrapper 亦只是众多等价 bf16 实现之一；故对 wrapper 只做
    # 漂移有界 + 平局翻转放行，bitwise 硬判据留给 capture 后的保真门
    #（graph 重放 vs 手写 eager 同 buffer，应逐位一致）。
    gate_n = min(16, min(steps_list))
    reset_cache(static_cache, dev0)
    pf = static_prefill(model, sample, p0, static_cache)
    gate_tokens = [int(pf.argmax())]
    top2_gaps = []   # 每步 manual logits 的 top-2 间隙
    ref_gaps = []    # 每步 wrapper ref logits 的 top-2 间隙（同一 decode step）
    for i in range(gate_n):
        feed(p0 + i, gate_tokens[-1])
        lg = manual_step()[0, -1].float()
        tv = torch.topk(lg, 2).values
        top2_gaps.append(float((tv[0] - tv[1]).detach()))
        rv = torch.topk(ref_logits[i + 1], 2).values
        ref_gaps.append(float((rv[0] - rv[1]).detach()))
        gate_tokens.append(int(lg.argmax().item()))
    gate_ok = gate_tokens == refs[p0][0][: gate_n + 1]
    diff_at, g_m, g_r, tie_flip = -1, None, None, None

    # teacher-forced 漂移测量：手写循环喂 wrapper 参考 token，logits 逐步对拍
    # （与 closed-loop 是否翻转无关，独立量化两条路径的数值距离）
    reset_cache(static_cache, dev0)
    static_prefill(model, sample, p0, static_cache)
    parity = []
    for i in range(gate_n):
        feed(p0 + i, refs[p0][0][i])
        lg = manual_step()[0, -1].float().cpu()
        parity.append((lg - ref_logits[i + 1]).abs().max().item())
    drift_max = max(parity) if parity else 0.0
    print(f"[seg] gate drift (teacher-forced vs wrapper, {gate_n} steps): "
          f"max_abs={drift_max:.3e}  per-step="
          f"{[f'{p:.2e}' for p in parity]}", flush=True)

    if gate_ok:
        print(f"[seg] manual-eager gate ({gate_n} steps closed-loop): MATCH", flush=True)
    else:
        diff_at = next((i for i, (a, b) in enumerate(
            zip(gate_tokens, refs[p0][0])) if a != b), -1)
        step = diff_at - 1  # 翻转发生在 manual step diff_at-1（其 argmax 即 token[diff_at]）
        g_m = top2_gaps[step]
        g_r = ref_gaps[step]
        tie_flip = min(g_m, g_r) <= args.gate_tie_gap
        print(f"[seg] manual-eager gate: DIFF at step {diff_at}", flush=True)
        print(f"[seg] gate detail: manual={gate_tokens[diff_at:diff_at + 3]} "
              f"ref={refs[p0][0][diff_at:diff_at + 3]}", flush=True)
        print(f"[seg] gate detail: manual top-2 logit gaps near DIFF "
              f"{[f'{g:.4f}' for g in top2_gaps[max(0, step - 1):step + 3]]}",
              flush=True)
        print(f"[seg] gate detail: flip step {step} gaps manual={g_m:.4f} "
              f"ref={g_r:.4f} (min <= {args.gate_tie_gap} -> tie flip, "
              f"else real bug)", flush=True)
        if tie_flip and drift_max <= args.gate_drift_max:
            print(f"[seg] gate verdict: benign tie-flip (bounded drift "
                  f"{drift_max:.3e} <= {args.gate_drift_max}) — proceed to capture",
                  flush=True)
        else:
            print(f"[seg] gate verdict: NOT benign (tie_flip={tie_flip}, "
                  f"drift_max={drift_max:.3e} vs limit {args.gate_drift_max})",
                  flush=True)
            _locate_drift()
            raise SystemExit("[seg] manual layer loop mismatches wrapper decode — "
                             "fix loop semantics before capture")

    # ---- capture：两段各自捕获（每卡一图；gB replay 须切到其捕获设备）----
    # 每设备独立 capture 流：torch.cuda.graph 的 default_capture_stream 是
    # 类级惰性单例（首次使用时在当前设备创建——本流程是 cuda:0）。B 段在
    # cuda:1 上复用它时，capture 流与工作设备不同卡：cuda:1 的工作落到该卡
    # legacy 默认流（不进图）→ 空图（capture_end 警告 "The CUDA Graph is
    # empty"）→ replay 无操作、g_logits 恒为 capture 期值。global 模式判死
    # 该状态，relaxed 放行成假成功（1024 第五轮保真门 MISMATCH tf=23.2 /
    # graph=[264,264,264] 实锤）。必须显式传 per-device stream。
    cap_streams = {}

    def _try_capture(dev, fn, mode=None):
        """单次 capture 尝试；失败时清掉粘滞错误后返回异常（不抛出）。"""
        g = torch.cuda.CUDAGraph()
        s = cap_streams.get(dev) or cap_streams.setdefault(
            dev, torch.cuda.Stream(device=dev))
        try:
            with torch.cuda.device(dev):
                with torch.cuda.graph(g, stream=s, capture_error_mode=mode or
                                      args.capture_error_mode):
                    fn()
            return g, None
        except Exception as e:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            return None, e

    def bisect_why(side, dev):
        """capture 失败时的二分定位：纯算子基线（确认本进程 capture 机制
        未被污染）→ embed → +rotary → 逐层加前缀 → norm+lm_head，打印首个
        非法探针（定位粒度=层/子块）。探针会推进 cache 状态，定位完即退出，
        不影响后续流程。"""
        t_sane = torch.ones(8, device=dev)
        probes = [("sanity: pure mul on " + str(dev), lambda: t_sane.mul(2.0))]
        if side == "A":
            probes.append(("embed", lambda: text.embed_tokens(ids_buf)))
            probes.append(("embed+static(cos0,sin0)",
                           _run_part(0, 0, True, False, side)))
            for k in range(1, split + 1):
                probes.append((f"layers[0:{k}]", _run_part(0, k, True, False, side)))
            # 监视探针放最后（首败即停，不能挡住真实路径定位）：rotary 原始
            # 模块 1023.seg.4 曾判非法，1024 查明真因是其 hook 把调用路由到
            # cuda:1 的跨设备搬运（算子本身无辜）；此处持续留档观察
            probes.append(("rotary module (raw, known-illegal monitor)",
                           lambda: text.rotary_emb(dummy0, pos_buf[1:])))
            _pos = pos_buf[1:]
            _inv = text.rotary_emb.inv_freq
            probes.append(("rot: pos expand+float",
                           lambda: _pos.expand(3, -1, -1)[..., None].float()))
            probes.append(("rot: mul inv_freq",
                           lambda: _pos.expand(3, -1, -1)[..., None].float() * _inv))
            probes.append(("rot: cos()",
                           lambda: (_pos.expand(3, -1, -1)[..., None].float() * _inv).cos()))
        else:
            probes.append(("norm+lm_head(h_mid_in)",
                           lambda: model.lm_head(text.norm(h_mid_in))))
            for k in range(split + 1, n + 1):
                probes.append((f"layers[{split}:{k}]", _run_part(split, k, False, False, side)))
            probes.append((f"layers[{split}:{n}]+norm+lm_head",
                           _run_part(split, n, False, True, side)))
        last_ok = None
        for tag, fn in probes:
            _, e = _try_capture(dev, fn)
            if e is not None:
                print(f"[seg] bisect[{side}]: '{tag}' first FAILS: "
                      f"{type(e).__name__}: {str(e).splitlines()[0]}", flush=True)
                dump_exc_chain(f"seg bisect[{side}]", e)
                print(f"[seg] bisect[{side}]: last OK prefix = {last_ok}", flush=True)
                m_ = re.match(r"layers\[(\d+):(\d+)\]", tag)
                if m_:
                    idx = int(m_.group(2)) - 1
                    kids = {name: type(cm).__name__
                            for name, cm in layers[idx].named_children()}
                    fp8_mods = sorted({type(sm).__name__
                                       for _, sm in layers[idx].named_modules()
                                       if "fp8" in type(sm).__name__.lower()})
                    print(f"[seg] bisect[{side}]: culprit layer {idx} "
                          f"({types[idx]}), children: {kids}\n"
                          f"[seg] bisect[{side}]: fp8 modules: "
                          f"{fp8_mods or 'none'}", flush=True)
                    # 子模块级 bisect（1023.seg.5 新增）：定位到 layer 内部哪个
                    # 子模块非法。合成非零输入（+0.01 防 fp8 激活量化除零 scale），
                    # layernorm 在图外预计算好，逐个探 attn / mlp。探针会推进
                    # GDN 状态 / KV 槽位，定位完即退出不影响后续流程。
                    h0 = ((text.embed_tokens(ids_buf) if side == "A" else h_mid_in)
                          + 0.01).clone()
                    ln = layers[idx].input_layernorm(h0)
                    sub = [("input_layernorm",
                            lambda: layers[idx].input_layernorm(h0))]
                    if "linear_attn" in kids:
                        sub.append(("linear_attn (GDN)",
                                    lambda: layers[idx].linear_attn(
                                        hidden_states=ln, cache_params=static_cache,
                                        attention_mask=None)))
                    elif "self_attn" in kids:
                        pe_sub = (cos0_buf, sin0_buf) if side == "A" \
                            else (cos1_buf, sin1_buf)
                        m_sub = (mask_f0 if side == "A" else mask_f1) \
                            if types[idx] == "full_attention" else None
                        sub.append(("self_attn (full)",
                                    lambda: layers[idx].self_attn(
                                        hidden_states=ln,
                                        position_embeddings=pe_sub,
                                        attention_mask=m_sub,
                                        past_key_values=static_cache)))
                    if "mlp" in kids:
                        sub.append(("mlp (MoE)",
                                    lambda: layers[idx].mlp(ln)))
                    for tag2, fn2 in sub:
                        _, e2 = _try_capture(dev, fn2)
                        verdict = "OK" if e2 is None else \
                            f"FAILS: {type(e2).__name__}: " \
                            f"{str(e2).splitlines()[0]}"
                        print(f"[seg] bisect[{side}]: layer {idx} submodule "
                              f"'{tag2}' -> {verdict}", flush=True)
                    # 裸 FP8Linear 探针：attn 与 mlp 双失败时区分「FP8Linear
                    # 公因子非法」vs「accelerate hook 非法」（子模块直调不经
                    # layer 级 hook，但 fp8 子模块自身可能挂有 hook）
                    try:
                        fp8_lin = next(sm for _, sm in layers[idx].named_modules()
                                       if type(sm).__name__ == "FP8Linear")
                        in_f = fp8_lin.in_features
                        x_lin = ln if ln.shape[-1] == in_f else \
                            torch.randn(1, 1, in_f, dtype=ln.dtype, device=dev)
                        _, e3 = _try_capture(dev, lambda: fp8_lin(x_lin))
                        print(f"[seg] bisect[{side}]: bare FP8Linear "
                              f"(in={in_f}) -> "
                              f"{'OK' if e3 is None else 'FAILS: ' + type(e3).__name__}",
                              flush=True)
                    except StopIteration:
                        print(f"[seg] bisect[{side}]: no FP8Linear found in "
                              f"layer {idx}", flush=True)
                    # 组合级探针（1023.seg.6 新增）：单模块各自 capture 合法但整层
                    # 非法 ⇒ 剩余差异只在「组合」。用 identity 替换一半子模块、
                    # 保留真实 DecoderLayer.forward 调用链（kwargs/residual/norm
                    # 全真），把层切成 token-mixer 半 / mlp 半 / 纯管道三种组合。
                    # 1024 源码核对：SparseMoeBlock.forward 单参数（探针已同款），
                    # GDN forward 无 use_cache 分支（torch_recurrent_gated_delta_
                    # rule 的 **kwargs 直接吞掉）——kwargs 差异排除，只剩组合。
                    tm_name = "linear_attn" if "linear_attn" in kids else "self_attn"
                    tm = getattr(layers[idx], tm_name)
                    pe_comp = (cos0_buf, sin0_buf) if side == "A" \
                        else (cos1_buf, sin1_buf)
                    m_comp = (mask_f0 if side == "A" else mask_f1) \
                        if types[idx] == "full_attention" else None

                    def layer_call(h_):
                        return layers[idx](h_, position_embeddings=pe_comp,
                                           attention_mask=m_comp,
                                           past_key_values=static_cache,
                                           use_cache=True)

                    # identity 必须能吃两种调用形态：mlp 走位置参数
                    # self.mlp(hidden_states)，token-mixer 走关键字
                    # self.linear_attn(hidden_states=...)；full-attention 的
                    # self_attn 返回 (hidden_states, _)，故按 kids 决定是否包 tuple
                    def _id_any(hidden_states=None, **kw):
                        z = torch.zeros_like(hidden_states)
                        return (z, None) if "self_attn" in kids else z

                    saved_mlp = layers[idx].mlp.forward
                    saved_tm = tm.forward
                    combos = [
                        (f"{tm_name} only (mlp=identity)", None, _id_any),
                        (f"mlp only ({tm_name}=identity)", _id_any, None),
                        ("pure norm/residual (both identity)", _id_any, _id_any),
                    ]
                    try:
                        for tag3, f_mlp, f_tm in combos:
                            layers[idx].mlp.forward = f_mlp or saved_mlp
                            tm.forward = f_tm or saved_tm
                            _, e4 = _try_capture(dev, lambda: layer_call(h0))
                            verdict = "OK" if e4 is None else \
                                f"FAILS: {type(e4).__name__}: " \
                                f"{str(e4).splitlines()[0]}"
                            print(f"[seg] bisect[{side}]: layer {idx} "
                                  f"composition '{tag3}' -> {verdict}",
                                  flush=True)
                    finally:
                        layers[idx].mlp.forward = saved_mlp
                        tm.forward = saved_tm
                    # 同一张图内连跑两次该层：真实段图是 18 层顺序执行，若存在
                    # 「单次合法、第二次非法」（capture 内 cache 状态分支切换 /
                    # 惰性 workspace 二次分配）此探针暴露；relaxed 探针则判定
                    # 失败是否只是 strict 模式对库内无害 API 过敏。
                    _, e5 = _try_capture(dev, lambda: (layer_call(h0),
                                                       layer_call(h0)))
                    print(f"[seg] bisect[{side}]: layer {idx} two sequential "
                          f"calls in one capture -> "
                          f"{'OK' if e5 is None else 'FAILS: ' + type(e5).__name__}",
                          flush=True)
                    _, e6 = _try_capture(dev, lambda: layer_call(h0), "relaxed")
                    v6 = ("OK (culprit is a strict-mode-illegal library call, "
                          "not a real error)" if e6 is None
                          else "FAILS: " + type(e6).__name__)
                    print(f"[seg] bisect[{side}]: layer {idx} one call under "
                          f"capture_error_mode=relaxed -> {v6}", flush=True)
                return
            last_ok = tag
        print(f"[seg] bisect[{side}]: ALL probes capture OK "
              f"(real capture failure was transient?)", flush=True)

    def _run_part(k_lo, k_hi, with_emb, with_head, side):
        """bisect 探针闭包：可选 embed+rotary（A 段）或 h_mid_in 起点（B 段），
        跑 layers[k_lo:k_hi]，可选 norm+lm_head。"""
        def fn():
            if with_emb:
                h = text.embed_tokens(ids_buf)
                pe = (cos0_buf, sin0_buf)
            else:
                h = h_mid_in
                pe = (cos1_buf, sin1_buf)
            for i in range(k_lo, k_hi):
                m = mask_f0 if side == "A" else mask_f1
                m = m if types[i] == "full_attention" else None
                h = layers[i](h, position_embeddings=pe, attention_mask=m,
                              past_key_values=static_cache, use_cache=True)
            if with_head:
                model.lm_head(text.norm(h))
        return fn

    cap_out = {}  # capture 期间产生的静态输出张量（h_mid / logits）
    cap_mode = args.capture_error_mode  # 实际生效的 capture 模式（记入报告）
    hooks_off = False  # capture 期间是否摘除了 accelerate hook（记入报告）
    saved_hooks = []  # (module, hook, hooked_forward) 恢复清单

    def disable_hooks():
        """临时摘除 accelerate hook：capture 期专用，capture 完必须 restore。

        1024 两轮实锤：① 异常链最内层是 accelerate operations.py
        send_to_device——DecoderLayer 级 AlignDevicesHook 的 pre_forward 对
        (cos,sin) 等 tuple 实参递归 .to(device)，跨设备拷贝在 capture 期
        判死（legacy stream vs capturing stream 依赖违规）；② 但同一批 hook
        恰是 wrapper 前向跨卡搬运的载体（剥掉后 static_prefill 在 layer18
        边界 cuda:0/cuda:1 不匹配）——所以只能「capture 期临时摘除、捕获完
        立即装回」。机制（accelerate/hooks.py add_hook_to_module 源码）：
        hook 通过替换 module.forward 生效，原始 forward 存在 _old_forward——
        摘除=换回 _old_forward + _hf_hook 置 None（GDN 的
        force_accelerate_hooks 也会因此跳过），不动 hook 对象本身。
        正确性：seg 手写循环所有输入已按层预置设备，hook 对齐本就是 no-op，
        且 gate 带 hook 通过已证 eager 语义；capture 后保真硬门兜底。"""
        nonlocal hooks_off
        for m in model.modules():
            hook = getattr(m, "_hf_hook", None)
            if hook is not None:
                saved_hooks.append((m, hook, m.forward))
                m.forward = getattr(m, "_old_forward", m.forward)
                m._hf_hook = None
        hooks_off = True
        print(f"[seg] accelerate hooks disabled for capture: "
              f"{len(saved_hooks)} modules", flush=True)

    def restore_hooks():
        nonlocal hooks_off
        for m, hook, hooked_fwd in saved_hooks:
            m.forward = hooked_fwd
            m._hf_hook = hook
        saved_hooks.clear()
        hooks_off = False
        print("[seg] accelerate hooks restored", flush=True)

    if args.strip_hooks == "always":
        disable_hooks()

    # hook 清单（capture 前留档）：哪个模块挂了哪类 hook、执行设备指向哪
    hooked = [(name, type(getattr(m, "_hf_hook")).__name__,
               str(getattr(getattr(m, "_hf_hook"), "execution_device", None)))
              for name, m in model.named_modules()
              if getattr(m, "_hf_hook", None) is not None]
    print(f"[seg] accelerate hooks: {len(hooked)} hooked modules; "
          f"sample: {[('.'.join(n.split('.')[-2:]), t, d)
                       for n, t, d in hooked[:6]]}", flush=True)

    print(f"[seg] capturing segment A (embed+rotary+layers[0:{split}]) "
          f"on {dev0} (capture_error_mode={cap_mode}) ...", flush=True)
    torch.cuda.synchronize()
    gA, eA = _try_capture(dev0, lambda: cap_out.update(h_mid=seg_a()), cap_mode)
    if eA is not None and cap_mode == "global":
        # strict(global) 模式会把库内无害 API（malloc/属性查询/同步探针）判死，
        # 报错经 capture_end 聚合后只剩「previous error during capture」，掩盖
        # 真实调用点。relaxed 允许这类调用（单线程 GPU 负载下安全），先当探针
        # 用：成功即沿用，失败则模式不是症结，继续 bisect 定位真实算子。
        print("[seg] capture A failed under global; retry with "
              "capture_error_mode=relaxed ...", flush=True)
        cap_out.clear()
        torch.cuda.empty_cache()
        gA, eA = _try_capture(dev0, lambda: cap_out.update(h_mid=seg_a()),
                              "relaxed")
        if eA is None:
            cap_mode = "relaxed"
            print("[seg] capture A OK under relaxed — the illegal call is "
                  "strict-mode-only (library malloc/query), not a real error",
                  flush=True)
    if eA is not None and not hooks_off and args.strip_hooks != "never":
        # 第三级重试：capture 期临时摘除 accelerate hook（机制见
        # disable_hooks 注释；1024 异常链实锤 hook pre_forward 的
        # send_to_device 跨设备 .to()）。摘除不重跑 sanity——gate 已带 hook
        # 验证过同一手写 eager 路径，hook 对齐在 seg 路径是 no-op，硬判据
        # 在 capture 后的保真门。
        print("[seg] capture A still failing; disable accelerate hooks "
              "during capture and retry ...", flush=True)
        disable_hooks()
        cap_out.clear()
        torch.cuda.empty_cache()
        gA, eA = _try_capture(dev0, lambda: cap_out.update(h_mid=seg_a()),
                              cap_mode)
        if eA is None:
            print("[seg] capture A OK with accelerate hooks disabled — "
                  "hook pre_forward cross-device .to() was the culprit",
                  flush=True)
    if eA is not None:
        print(f"[seg] CAPTURE A FAILED: {type(eA).__name__}: {eA}\n"
              f"[seg] hints: fp8 w8a8 / fla kernel / accelerate hook inside "
              f"segment —— running bisect ...", flush=True)
        dump_exc_chain("seg", eA)
        bisect_why("A", dev0)
        if hooks_off:
            restore_hooks()
        sys.exit(2)
    h_mid_out = cap_out["h_mid"]
    print(f"[seg] capturing segment B (layers[{split}:{n}]+norm+lm_head) "
          f"on {dev1} (capture_error_mode={cap_mode}) ...", flush=True)
    torch.cuda.synchronize()
    gB, eB = _try_capture(dev1, lambda: cap_out.update(logits=seg_b(h_mid_in)),
                          cap_mode)
    if eB is not None and cap_mode != "relaxed":
        print("[seg] capture B failed; retry with capture_error_mode=relaxed ...",
              flush=True)
        cap_out.pop("logits", None)
        torch.cuda.empty_cache()
        gB, eB = _try_capture(dev1, lambda: cap_out.update(logits=seg_b(h_mid_in)),
                              "relaxed")
        if eB is None:
            cap_mode = "relaxed"
            print("[seg] capture B OK under relaxed mode", flush=True)
    if eB is not None and not hooks_off and args.strip_hooks != "never":
        # 与 A 对称的第三级重试（A 成功不等于 B 免疫——B 段含 norm/lm_head）
        print("[seg] capture B still failing; disable accelerate hooks "
              "during capture and retry ...", flush=True)
        disable_hooks()
        cap_out.pop("logits", None)
        torch.cuda.empty_cache()
        gB, eB = _try_capture(dev1, lambda: cap_out.update(logits=seg_b(h_mid_in)),
                              cap_mode)
        if eB is None:
            print("[seg] capture B OK with accelerate hooks disabled",
                  flush=True)
    if eB is not None:
        print(f"[seg] CAPTURE B FAILED: {type(eB).__name__}: {eB}\n"
              f"[seg] running bisect ...", flush=True)
        dump_exc_chain("seg", eB)
        bisect_why("B", dev1)
        if hooks_off:
            restore_hooks()
        sys.exit(2)
    g_logits = cap_out["logits"]
    hooks_were_off = hooks_off  # 记入报告（restore 会把 hooks_off 清回 False）
    if hooks_off:
        # 捕获完成立即装回：后续 wrapper prefill（static_prefill/_run_n 的
        # 跨卡流）依赖 hook 搬运；replay 不跑 Python，与 hook 无关
        restore_hooks()
    print(f"[seg] both segments captured (capture_error_mode={cap_mode}, "
          f"hooks_disabled_for_capture={hooks_were_off})", flush=True)

    def graph_step():
        gA.replay()
        h_mid_in.copy_(h_mid_out)
        with torch.cuda.device(dev1):
            gB.replay()
        return g_logits

    # ---- capture 保真硬门：graph 重放 vs 手写 eager（同静态 buffer）----
    # CUDA graph 唯一真正威胁正确性的机制点（capture 期状态污染 / buffer 别名 /
    # 漏 feed / 重放读写错槽）都在这里暴露：同一批 kernel、同一批 buffer，重放
    # 与 eager 应逐位一致（确定性 kernel）。与 wrapper 的 bf16 末位漂移无关——
    # 那是两条不同 cache 布局路径的固有距离，见 gate 注释。
    def _run_n(run_step, teacher_forced):
        """复位→prefill→gate_n 步：teacher-forced 喂参考 token 或 closed-loop。"""
        reset_cache(static_cache, dev0)
        pf = static_prefill(model, sample, p0, static_cache)
        toks = [int(pf.argmax())]
        lgs = []
        for i in range(gate_n):
            tok = refs[p0][0][i] if teacher_forced else toks[-1]
            feed(p0 + i, tok)
            lgs.append(run_step()[0, -1].float().cpu().clone())
            toks.append(int(lgs[-1].argmax()))
        return toks, lgs

    print(f"[seg] capture fidelity check ({gate_n} steps, manual-eager vs "
          f"graph replay on same buffers) ...", flush=True)
    _, manual_lgs = _run_n(lambda: manual_step(), True)
    _, graph_lgs = _run_n(lambda: graph_step(), True)
    tf_max = max((a - b).abs().max().item()
                 for a, b in zip(manual_lgs, graph_lgs)) if gate_n else 0.0
    manual_cl, _ = _run_n(lambda: manual_step(), False)
    graph_cl, _ = _run_n(lambda: graph_step(), False)
    cl_ok = manual_cl == graph_cl
    if tf_max == 0.0 and cl_ok:
        print(f"[seg] capture fidelity: OK (teacher-forced bitwise max_abs="
              f"{tf_max:.3e}, closed-loop {gate_n} tokens MATCH)", flush=True)
    else:
        print(f"[seg] capture fidelity: MISMATCH (tf max_abs={tf_max:.3e}, "
              f"closed-loop match={cl_ok})", flush=True)
        if not cl_ok:
            dd = next((i for i, (a, b) in enumerate(zip(manual_cl, graph_cl))
                       if a != b), -1)
            print(f"[seg] fidelity closed-loop first diff at step {dd}: "
                  f"manual={manual_cl[dd:dd + 3]} graph={graph_cl[dd:dd + 3]}",
                  flush=True)
        raise SystemExit("[seg] capture replay does not reproduce manual eager — "
                         "fix capture before benchmarking")

    # ---- pass B：prompt×steps 逐档 closed-loop（wall 计时，bench 同口径）----
    runs = []
    for plen in plen_list:
        for s in steps_list:
            ref_tokens = refs[plen][0]
            loop_stats, seq_ok, first_diff = [], True, -1
            for rep in range(args.reps):
                reset_cache(static_cache, dev0)
                pf = static_prefill(model, sample, plen, static_cache)
                graph_tokens = [int(pf.argmax())]
                t0 = time.perf_counter()
                for i in range(s):
                    feed(plen + i, graph_tokens[-1])
                    graph_step()
                    graph_tokens.append(int(g_logits[0, -1].argmax().item()))
                loop_stats.append((time.perf_counter() - t0) / s * 1000)
                if seq_ok and graph_tokens != ref_tokens[: s + 1]:
                    seq_ok = False
                    first_diff = next(
                        (i for i, (a, b) in enumerate(zip(graph_tokens, ref_tokens))
                         if a != b), -1)
            med_loop = statistics.median(loop_stats)
            runs.append({
                "prompt_len": plen,
                "steps": s,
                "loop_ms": med_loop,
                "speedup": eager_med[(plen, s)] / med_loop,
                "tok_per_s": 1000.0 / med_loop,
                "token_match": seq_ok,
                "first_diff_step": first_diff,
            })

    print(f"\n===== graph decode (segment, split={split}) =====")
    print(f"{'plen':>4} {'steps':>5}  {'eager':>12}  {'graph':>12}  "
          f"{'speedup':>7}  {'graph tok/s':>11}  vs-eager")
    for r in runs:
        plen, s = r["prompt_len"], r["steps"]
        label = "MATCH" if r["token_match"] else f"DIFF@{r['first_diff_step']}"
        print(f"{plen:>4} {s:>5}  {eager_med[(plen, s)]:>10.1f}ms  "
              f"{r['loop_ms']:>10.1f}ms  {r['speedup']:>6.2f}x  "
              f"{r['tok_per_s']:>11.2f}  {label}")
    # vs-eager 列是软判据（graph vs wrapper，DynamicCache）：漂移有界时平局步
    # 翻转属预期，DIFF@k ≠ 错；硬判据 = capture fidelity（bitwise）+ gate 漂移
    print(f"(vs-eager is informative; hard criteria: capture fidelity bitwise "
          f"tf={tf_max:.3e} cl={cl_ok}, gate drift={drift_max:.3e})", flush=True)

    # ---- pass C：teacher-forced logits 对拍（首档 prompt × 最长 steps）----
    reset_cache(static_cache, dev0)
    static_prefill(model, sample, p0, static_cache)
    max_abs = 0.0
    for i in range(max_steps):
        feed(p0 + i, refs[p0][0][i])
        graph_step()
        d = (g_logits[0, -1].float().cpu() - ref_logits[i + 1]).abs().max().item()
        max_abs = max(max_abs, d)
    print(f"logits parity (teacher-forced, plen={p0}, {max_steps} steps): "
          f"max_abs={max_abs:.3e}")

    report = {
        "meta": {
            "model_id": model_id, "mode": mode_label, "strategy": "segment",
            "split_at": split, "prompt_len_list": plen_list,
            "steps_list": steps_list, "warmup_steps": args.warmup_steps,
            "capture_error_mode": cap_mode,
            "hooks_disabled_for_capture": hooks_were_off,
            "githash": get_git_hash(),
            "gpu": torch.cuda.get_device_name(0),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "eager_median_ms": {f"{p}:{s}": eager_med[(p, s)]
                            for p in plen_list for s in steps_list},
        "graph_runs": runs,
        "logits_max_abs": {"prompt_len": p0, "steps": max_steps, "max_abs": max_abs},
        "gate": {
            "steps": gate_n,
            "closed_loop_match": gate_ok,
            "first_diff_step": diff_at,
            "tie_flip": tie_flip,
            "flip_manual_gap": g_m,
            "flip_ref_gap": g_r,
            "drift_max_abs": drift_max,
            "drift_per_step": parity,
        },
        "fidelity": {
            "teacher_forced_max_abs": tf_max,
            "closed_loop_match": cl_ok,
        },
    }
    out = args.out or os.path.join(
        "results", f"graph_decode_{model_id}_{mode_label}_seg_"
                   f"{get_git_hash()}_{datetime.now().strftime('%y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[seg] saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--experts-impl", default=None, choices=[None, "eager", "grouped_mm"],
                    help="MoE experts 实现透传 from_pretrained；None=框架默认解析"
                         "（grouped_mm 不可用时静默回退 eager）。FP8Experts 的 eager"
                         " 路径有 nonzero()+host 专家循环，capture 必死；graph 场景"
                         "应显式 grouped_mm（检查不过会 raise，暴露真实原因）")
    ap.add_argument("--mode", default="wxa16",
                    choices=["wxa16", "wxa8", "wxfp8", "wxfp4"])
    ap.add_argument("--prompt-len", default="128",
                    help="逗号分隔多档 prompt 长度（如 16,128,2048,4096）；"
                         "与 --steps 全交叉，一次 capture 全档复用")
    ap.add_argument("--steps", default="64",
                    help="逗号分隔多档步数（如 64,256,512,1024）；eager 参考"
                         "只跑最长档，短档按前缀对拍")
    ap.add_argument("--warmup-steps", type=int, default=3,
                    help="capture 前侧流捂热步数（也是惰性编译的触发步；"
                         "消耗同名数量的 cache 槽位，已计入 max_cache_len）")
    ap.add_argument("--mask-mode", choices=["native", "buffer"], default="native")
    ap.add_argument("--gate-tie-gap", type=float, default=0.5,
                    help="seg 门平局阈值：closed-loop 翻转步双方 top-2 间隙的 "
                         "min ≤ 此值判定为平局翻转（benign）；bf16 logits 的 "
                         "1ulp≈0.0625-0.125，默认 0.5")
    ap.add_argument("--gate-drift-max", type=float, default=3.0,
                    help="seg 门漂移阈值：teacher-forced 逐步 logits 对拍的 "
                         "max_abs ≤ 此值判定漂移有界。机制已实证（1023.seg.4 "
                         "逐层定位）：StaticCache 满槽 buffer vs DynamicCache "
                         "精确长度的 softmax 归约差 → 偶发 1 bf16 ulp → GDN "
                         "prefill 状态吸收 → 256-expert top-8 路由翻转离散放大"
                         "（layer20 处 1.6e-2→0.33），plen=128/16 步实测 1.53，"
                         "默认 3.0 留余量；真语义 bug 特征=首步 O(1) 且集中于"
                         "特定层型（localizer 可辨）")
    ap.add_argument("--capture-error-mode", choices=["global", "thread_local",
                                                     "relaxed"], default="global",
                    help="cudaStreamCaptureMode：global 严格（默认），库内无害 "
                         "API（malloc/查询）也会判死 capture；relaxed 放行这类"
                         "调用（单线程 GPU 负载下安全）。segment capture 失败时"
                         "自动以 relaxed 重试一次当探针，成功即沿用并记录报告")
    ap.add_argument("--strip-hooks", choices=["auto", "always", "never"],
                    default="auto",
                    help="accelerate hook 的 capture 期摘除策略：auto=capture "
                         "失败时自动临时摘除重试（1024 异常链实锤 hook "
                         "pre_forward 的跨设备 send_to_device 判死 capture；"
                         "摘除仅覆盖 capture 窗口，捕获完立即装回——wrapper "
                         "prefill 的跨卡搬运依赖 hook）；always=首次 capture "
                         "前即摘除（A/B 对照用）；never=不摘除（留档复现）")
    ap.add_argument("--strategy", choices=["whole", "segment"], default="whole",
                    help="whole=整步一张图（仅单卡模型合法）；segment=按设备边界"
                         "切段逐层手跑，跨设备拷贝留图外（官方 FP8 双卡路线）")
    ap.add_argument("--split-at", type=int, default=0,
                    help="segment 意图切分层（估算卡0容量上限用）；实际切分"
                         "以加载后逐层真实设备反推为准；0=自动取半")
    ap.add_argument("--reps", type=int, default=1, help="graph 循环重复轮数（取中位）")
    ap.add_argument("--dataset", default="wiki", choices=["wiki", "c4"])
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import StaticCache

    device_map, max_mem = None, None
    if args.strategy == "segment":
        assert args.model_path, "--strategy segment needs --model-path (official)"
        device_map, max_mem, want_split = build_layer_device_map(args.model_path, args.split_at)
        print(f"[seg] load spec: device_map=auto, max_memory={max_mem} "
              f"(intended split near layer {want_split}; actual split is "
              f"derived from real per-layer devices after load)", flush=True)
    model, tokenizer, model_id, mode_label = load_bench_model(
        args, device_map=device_map, max_memory=max_mem)
    sample = load_sample_tokens(args.dataset, tokenizer)
    text_cfg = _text_tower(model).config
    layer_types = list(getattr(text_cfg, "layer_types", []) or [])
    n_lin = layer_types.count("linear_attention")
    n_full = layer_types.count("full_attention")
    print(f"[graph] layers: total={len(layer_types)} linear={n_lin} "
          f"full={n_full} other={len(layer_types) - n_lin - n_full}", flush=True)
    # experts 实现解析实锤（1024.graph 判定用）：eager=nonzero+host 专家循环
    # （capture 必死），grouped_mm=histc 静态路径（cuda-graph 安全设计）
    print(f"[graph] experts_implementation resolved: "
          f"{getattr(text_cfg, '_experts_implementation', None)} "
          f"(requested: {args.experts_impl or 'auto'})", flush=True)

    if args.strategy == "segment":
        seg_main(args, model, model_id, mode_label, sample)
        return

    dev = emb_device(model)
    steps_list = [int(s) for s in str(args.steps).split(",") if s.strip()]
    plen_list = [int(p) for p in str(args.prompt_len).split(",") if p.strip()]
    max_steps = max(steps_list)
    max_plen = max(plen_list)
    max_cache_len = max_plen + max_steps + args.warmup_steps + 16

    # ---- 1) eager 参考（DynamicCache 默认路径，每档 prompt 跑最长 steps 档）----
    refs, ref_logits, eager_med = run_eager_refs(
        model, sample, plen_list, steps_list, max_steps)

    # 我方量化模型的 MoE 静态派发（CUDA graph 前置）：eager 基线保持
    # host 编排口径，此处（基线之后、warmup 之前）打开；warmup 首个
    # decode 步以 eager 方式构建池状态，随后 capture/重放全程设备端定形。
    # 四模式（wxa16/wxa8/wxfp8/wxfp4）的静态池计算均已实现，全打开；
    # 未实现静态化的新子类会被 MoE forward 的 _STATIC_DISPATCH_READY 守卫挡掉
    if args.mode in ("wxa16", "wxa8", "wxfp8", "wxfp4"):
        os.environ["DARTMOQ_STATIC_DISPATCH"] = "1"
        print("[graph] DARTMOQ_STATIC_DISPATCH=1 "
              f"(MoE static dispatch enabled for {args.mode})", flush=True)

    # ---- 2) 静态 cache + prefill（槽位按 max(prompt)+max(steps) 预分配）----
    static_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    p0 = plen_list[0]  # 捂热/捕获/logits 对拍的基准档
    pf_logits = static_prefill(model, sample, p0, static_cache)
    tok0 = int(pf_logits.argmax())
    print(f"[graph] static prefill ok; first token matches eager: "
          f"{tok0 == refs[p0][0][0]}", flush=True)

    # ---- 3) 静态 buffer 与单步函数 ----
    ids_buf = torch.zeros(1, 1, dtype=torch.long, device=dev)
    pos_buf = torch.zeros(4, 1, 1, dtype=torch.long, device=dev)
    if args.mask_mode == "buffer":
        # 同 seg 路径：激活 dtype 取 embed 权重 dtype（多模态 checkpoint 首参
        # 是 vision 塔，dtype 可能 fp32/fp16，不能作激活 dtype 依据）
        dt = _text_tower(model).embed_tokens.weight.dtype
        mask_buf = torch.zeros(1, 1, 1, max_cache_len, dtype=dt, device=dev)
        mask_dict = {"full_attention": mask_buf, "linear_attention": None}
    else:
        mask_buf, mask_dict = None, None

    def feed(pos, token):
        """图外每步的静态 buffer 更新（in-place，无新分配）。"""
        ids_buf.fill_(token)
        pos_buf.fill_(pos)
        if mask_buf is not None:
            mask_buf.fill_(float("-inf"))
            mask_buf[..., : pos + 1].zero_()

    def step():
        out = model(input_ids=ids_buf, position_ids=pos_buf,
                    attention_mask=mask_dict if mask_dict is not None else None,
                    past_key_values=static_cache, use_cache=True)
        return out.logits

    # ---- 4) 侧流捂热（teacher-forced 喂参考 token，状态保持一致）----
    print(f"[graph] warmup {args.warmup_steps} steps on side stream "
          f"(triggers lazy kernel compile) ...", flush=True)
    reset_cache(static_cache, dev)
    static_prefill(model, sample, p0, static_cache)
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for k in range(args.warmup_steps):
            feed(p0 + k, refs[p0][0][k])
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize(dev)

    # ---- 5) 捕获 ----
    print("[graph] capturing decode step ...", flush=True)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            g_logits = step()
    except Exception as e:
        print(f"[graph] CAPTURE FAILED: {type(e).__name__}: {e}\n"
              f"[graph] hints: dual-GPU device_map cross-device copies / "
              f"mask-mode native host branch —— retry with --mask-mode buffer "
              f"or a single-GPU model", flush=True)
        dump_exc_chain("graph", e)
        sys.exit(2)
    print("[graph] capture ok", flush=True)

    # ---- 6) pass B：prompt×steps 逐档 复位 → prefill → 重放（计时 + 序列对拍）----
    runs = []
    for plen in plen_list:
        for s in steps_list:
            ref_tokens = refs[plen][0]
            loop_stats, replay_stats, seq_ok, first_diff = [], [], True, -1
            for rep in range(args.reps):
                reset_cache(static_cache, dev)
                pf = static_prefill(model, sample, plen, static_cache)
                graph_tokens = [int(pf.argmax())]
                # 剖析窗口开在计时循环外圈：reset/prefill（host 路径 kernel）
                # 不入窗，窗内只有 feed/replay/argmax 一步的真实构成
                if _PROF_WINDOW:
                    torch.cuda.profiler.start()
                t0 = time.time()
                for i in range(s):
                    feed(plen + i, graph_tokens[-1])
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    g.replay()
                    ev1.record()
                    tok = int(g_logits[0, -1].argmax().item())
                    ev0.synchronize()
                    ev1.synchronize()
                    replay_stats.append(ev0.elapsed_time(ev1))
                    graph_tokens.append(tok)
                loop_stats.append((time.time() - t0) / s * 1000)
                if _PROF_WINDOW:
                    torch.cuda.profiler.stop()
                if seq_ok and graph_tokens != ref_tokens[: s + 1]:
                    seq_ok = False
                    for i, (a, b) in enumerate(zip(graph_tokens, ref_tokens)):
                        if a != b:
                            first_diff = i
                            break
            med_loop = statistics.median(loop_stats)
            runs.append({
                "prompt_len": plen,
                "steps": s,
                "loop_ms": med_loop,
                "replay_ms": statistics.median(replay_stats),
                "speedup": eager_med[(plen, s)] / med_loop,
                "tok_per_s": 1000.0 / med_loop,
                "token_match": seq_ok,
                "first_diff_step": first_diff,
            })

    print(f"\n===== graph decode (mask={args.mask_mode}) =====")
    print(f"{'plen':>4} {'steps':>5}  {'eager':>12}  {'graph':>12}  {'speedup':>7}  "
          f"{'replay-only':>12}  {'graph tok/s':>11}  tokens")
    for r in runs:
        plen, s = r["prompt_len"], r["steps"]
        label = "MATCH" if r["token_match"] else f"DIFF@{r['first_diff_step']}"
        print(f"{plen:>4} {s:>5}  {eager_med[(plen, s)]:>10.1f}ms  "
              f"{r['loop_ms']:>10.1f}ms  {r['speedup']:>6.2f}x  "
              f"{r['replay_ms']:>10.1f}ms  {r['tok_per_s']:>11.2f}  {label}")

    # ---- 6b) pass B2：闭环分叉诊断（仅 token_match=False 的档重跑一遍闭环；
    #       独立于 pass B 计时循环，不污染 loop_ms）----
    # 261025 四模式首轮出现的矛盾（2bpw parity=0 却 DIFF@1、4bpw parity 13~17
    # 却有 MATCH）需要闭环路径上的逐步证据：
    #   - 分叉步 logits diff + 两侧 top-2 gap → 区分平局翻转（benign）/
    #     状态失配（drift 放大）/ logits 相同却 argmax 不同（门取数矛盾）
    #   - prefill parity（vs eager 参考的 prefill logits）= 每次 run 的状态
    #     指纹；诊断 run 复现不了 pass B 的分叉 ⇒ run 间非确定性（host 路径
    #     index_add_ 原子序等）的直接证据
    # logits 对拍仅对首档 prompt 有效（ref_logits 只留首档），其余档只看
    # token 复现性。
    gate_diag = []
    for r in runs:
        if r["token_match"]:
            continue
        plen, s = r["prompt_len"], r["steps"]
        reset_cache(static_cache, dev)
        pf = static_prefill(model, sample, plen, static_cache)
        pf_parity = ((pf - ref_logits[0]).abs().max().item()
                     if plen == p0 else -1.0)
        tok0 = int(pf.argmax())
        # prefill token 是否与 eager 参考一致：B2 的 prefill 与 setup 时的
        # prefill 是两次独立执行，不一致 ⇒ prefill 本身非确定（host 路径
        # 原子序等），闭环喂入不同 token 单独就能造成 DIFF@1
        pf_tok_match = (None if plen != p0 else (tok0 == refs[p0][0][0]))
        diag_tokens = [tok0]
        cl_parity, div, matching = 0.0, None, True
        for i in range(s):
            feed(plen + i, diag_tokens[-1])
            g.replay()
            # detach：step() 的输出带 requires_grad（capture 在 no_grad 外），
            # 后续 float()/topk() 会触发 UserWarning
            gl = g_logits[0, -1].detach().float().cpu()
            tok = int(gl.argmax().item())
            diag_tokens.append(tok)
            if plen != p0 or not matching:
                continue
            d = (gl - ref_logits[i + 1]).abs().max().item()
            # NaN 用 max(0.0, nan) 会被吞掉（nan>x 为 False），显式保留
            if d != d or d > cl_parity:
                cl_parity = d
            if tok != refs[p0][0][i + 1]:
                matching = False
                gt = gl.topk(2).values
                et = ref_logits[i + 1].topk(2).values
                div = {
                    "step": i + 1,
                    "logits_max_abs": d,
                    "graph_logits_has_nan": bool(torch.isnan(gl).any().item()),
                    "graph_top2_gap": float(gt[0] - gt[1]),
                    "eager_top2_gap": float(et[0] - et[1]),
                    "graph_tok": tok,
                    "eager_tok": refs[p0][0][i + 1],
                }
        rerun_diff = next(
            (i for i, (a, b) in enumerate(zip(diag_tokens, refs[plen][0]))
             if a != b), -1)
        if div is not None and div["graph_logits_has_nan"]:
            # 261025 实测（2bpw wxfp4）：闭环 step1 logits 含 NaN（argmax 落 0），
            # 非 tie-flip；同进程 teacher-forced pass C 却 bitwise 0 ⇒ NaN 只出
            # 现在闭环喂入序列上，与参考流分叉后逐步定位需逐层 NaN 溯源
            verdict = "NaN in closed-loop logits (state corruption, not a tie-flip)"
        elif div is not None:
            if div["logits_max_abs"] < 1e-6:
                verdict = "CONTRADICTION: identical logits but argmax differs (gate bug)"
            elif (div["logits_max_abs"] < 1e-2
                  and div["graph_top2_gap"] < 1e-2
                  and div["eager_top2_gap"] < 1e-2):
                verdict = "tie-flip (likely benign)"
            elif div["logits_max_abs"] > 1.0:
                verdict = "state divergence (drift amplified)"
            else:
                verdict = "ambiguous (inspect numbers below)"
        elif rerun_diff == -1:
            verdict = ("MATCH on rerun: pass B DIFF not reproduced "
                       "(run-to-run nondeterminism)")
        else:
            verdict = f"diverged again @step {rerun_diff} (non-p0 prompt, no logits evidence)"
        gate_diag.append({
            "prompt_len": plen, "steps": s,
            "prefill_parity_max_abs": pf_parity,
            "prefill_tok_match": pf_tok_match,
            "closed_loop_parity_max_abs": cl_parity,
            "rerun_first_diff_step": rerun_diff,
            "divergence": div,
            "verdict": verdict,
        })
        print(f"[graph] gate diag plen={plen} steps={s}: "
              f"prefill_parity={pf_parity:.3e} "
              f"prefill_tok_match={pf_tok_match} "
              f"closed_loop_parity={cl_parity:.3e} "
              f"rerun_diff@{rerun_diff} -> {verdict}", flush=True)
        if div is not None:
            print(f"[graph]   div step={div['step']} "
                  f"logits_diff={div['logits_max_abs']:.3e} "
                  f"graph_nan={div['graph_logits_has_nan']} "
                  f"graph_top2_gap={div['graph_top2_gap']:.3e} "
                  f"eager_top2_gap={div['eager_top2_gap']:.3e} "
                  f"tok graph={div['graph_tok']} eager={div['eager_tok']}",
                  flush=True)

    # ---- 7) pass C：teacher-forced logits 对拍（信息性；首档 prompt × 最长
    #      steps 档，覆盖该档全部前缀；capture 保真度与上下文长度无关）----
    reset_cache(static_cache, dev)
    static_prefill(model, sample, p0, static_cache)
    max_abs = 0.0
    for i in range(max_steps):
        feed(p0 + i, refs[p0][0][i])
        g.replay()
        d = (g_logits[0, -1].detach().float().cpu()
             - ref_logits[i + 1]).abs().max().item()
        # 同 B2：max(0.0, nan) 会吞 NaN，显式保留
        if d != d or d > max_abs:
            max_abs = d
    print(f"logits parity (teacher-forced, plen={p0}, {max_steps} steps): "
          f"max_abs={max_abs:.3e}")

    report = {
        "meta": {
            "model_id": model_id, "mode": mode_label, "mask_mode": args.mask_mode,
            "prompt_len_list": plen_list, "steps_list": steps_list,
            "warmup_steps": args.warmup_steps,
            "githash": get_git_hash(),
            "gpu": torch.cuda.get_device_name(0),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "eager_median_ms": {f"{p}:{s}": eager_med[(p, s)]
                            for p in plen_list for s in steps_list},
        "graph_runs": runs,
        "gate_diag": gate_diag,
        "logits_max_abs": {"prompt_len": p0, "steps": max_steps, "max_abs": max_abs},
    }
    out = args.out or os.path.join(
        "results", f"graph_decode_{model_id}_{mode_label}_"
                   f"{get_git_hash()}_{datetime.now().strftime('%y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[graph] saved: {out}")


if __name__ == "__main__":
    main()
