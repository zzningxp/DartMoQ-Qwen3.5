#!/usr/bin/env python3
"""
eval_tasks.py — lm-eval-harness 下游任务评测（量化 Qwen3.5 checkpoint 全模式）。

对齐 BitsMoE 七任务（five-shot）：hellaswag / arc_easy / mathqa / mmlu /
openbookqa / winogrande / gsm8k。与上游 DartMoQ eval_zero_shot 的 HFLM 透传不同，
这里用自定义 DartMoQLM（lm_eval TemplateLM 子类）直接驱动量化模型：

  - model.model(...) 只取 hidden，仅在 continuation 预测位过 lm_head
    （词表 ~248K，全量 logits 约 600MB/样本，不可整批算）；
  - 同输入的单 token 选项分组去重（mmlu/arc/obqa 每 doc 一次 forward）；
  - 评测路径不依赖 HFLM（HFLM 仅在 test/test_eval_tasks_lm.py 作 parity 基准）。

顶层 import 保持轻量（不拉 torch/量化依赖），重依赖全部在函数内延迟导入，
test/test_eval_tasks_lm.py 借此在无量化依赖环境下直接 import 本模块。

用法（主流程，占大量显存，请手动运行）：
  python eval_tasks.py models/qwen3.5-2bpw-260930-s1 \
      --inference-quant-mode wxa8 --tasks arc_easy --limit 20 --num-fewshot 5
  python eval_tasks.py models/qwen3.5-2bpw-260930-s1 \
      --inference-quant-mode wxa8 --num-fewshot 5   # 默认七任务，gsm8k 排最后
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

from lm_eval.api.model import TemplateLM

# BitsMoE 对齐的七任务；gsm8k 是 generate_until（KV-cache 生成），排最后单独暴露风险
DEFAULT_TASKS = [
    "hellaswag", "arc_easy", "mathqa", "mmlu",
    "openbookqa", "winogrande", "gsm8k",
]

# mathqa 本地 parquet 任务 yaml（fetch_task_datasets.py 生成：
# hub 的 math_qa.py 旧式 script 已被 datasets 4.x 禁用）
MATHQA_LOCAL_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "benchmarks", "lm_eval_tasks", "mathqa.yaml")


class DartMoQLM(TemplateLM):
    """直接驱动 DartMoQ 量化模型的 lm-eval LM。

    只实现 token 级原语（tok_encode / _loglikelihood_tokens / generate_until /
    loglikelihood_rolling），上下文编码与空上下文 prefix 处理继承自 TemplateLM。
    """

    def __init__(self, model, tokenizer, batch_size=8, gen_batch_size=1,
                 max_length=4096, max_gen_toks=256, max_score_positions=256,
                 device=None, sequential=False):
        super().__init__()
        self._model = model
        self.tokenizer = tokenizer
        self._batch_size = int(batch_size)
        self._gen_batch_size = int(gen_batch_size)
        self._max_length = int(max_length)
        self._max_gen_toks = int(max_gen_toks)
        # lm_head 分块的评分位数上限：248K 词表下 256 位约 380MB 峰值（含 fp32 副本）
        self._max_score_positions = int(max_score_positions)
        # sequential 层循环模式（fp16 基线：模型驻 CPU，逐层搬 GPU 前向）
        self._sequential = bool(sequential)

        # pad 兜底（Qwen checkpoint 自带 pad；以防万一）
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 输入侧设备（两卡切分时 embed 在 cuda:0，hidden/lm_head 在 cuda:1）
        import torch
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = next(model.model.embed_tokens.parameters()).device

        # 与 HFLM 的 add_bos_token=None 语义一致：用 tokenizer 自身默认
        self._add_bos_token = None

    # ---------------- 基础属性 ----------------

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # HFLM 语义：BOS 优先，无 BOS 用 EOT（Qwen 无 BOS → 用 eos）
        return self.tokenizer.bos_token_id or self.eot_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def config(self):
        # simple_evaluate 会读 config._name_or_path 记录模型名
        return self._model.config

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path

    def get_model_info(self):
        n_params = sum(p.numel() for p in self._model.parameters())
        return {
            "model_num_parameters": n_params,
            "model_dtype": str(next(self._model.parameters()).dtype),
        }

    # ---------------- tokenize ----------------

    def tok_encode(self, string, add_special_tokens=None, left_truncate_len=None,
                   **kwargs):
        # 镜像 HFLM.tok_encode：None 走 tokenizer 默认；BOS 前缀防重复添加
        from lm_eval.models.utils import _add_special_kwargs, has_bos_prefix

        special_tokens_kwargs = _add_special_kwargs(
            add_special_tokens, self._add_bos_token)
        if add_special_tokens is None and has_bos_prefix(
                string, self.tokenizer.decode(self.prefix_token_id)):
            special_tokens_kwargs["add_special_tokens"] = False
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    # ---------------- loglikelihood ----------------

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        """token 级打分核心。

        输入 inp = (ctx_enc + cont_enc)[-(max_length+1):][:-1]（HFLM 同款左截断算术）；
        scored 位置为 inp 的最后 m 位（m = len(cont_enc)），targets 即 cont_enc。
        """
        import torch
        from tqdm import tqdm

        bs = self._batch_size if override_bs is None else int(override_bs)

        # 1) 构输入 + 分组去重：key = tuple(inp)。
        #    mmlu/arc/obqa 的单 token 选项共享上下文 → 每 doc 一次 forward
        groups = {}   # key -> {"inp": [...], "hits": [(req_idx, m, cont_tokens, req_str), ...]}
        n_truncated = 0
        for idx, (req_str, ctx_enc, cont_enc) in enumerate(requests):
            assert len(ctx_enc) > 0, "empty context_enc"
            assert len(cont_enc) > 0, "empty continuation_enc"
            assert len(cont_enc) <= self._max_length, (
                f"continuation longer than max_length: {len(cont_enc)}")
            total = ctx_enc + cont_enc
            if len(total) > self._max_length + 1:
                n_truncated += 1
            inp = total[-(self._max_length + 1):][:-1]
            key = tuple(inp)
            g = groups.get(key)
            if g is None:
                groups[key] = {"inp": inp,
                               "hits": [(idx, len(cont_enc), cont_enc, req_str)]}
            else:
                g["hits"].append((idx, len(cont_enc), cont_enc, req_str))
        if n_truncated:
            print(f"[warn] left-truncated {n_truncated} request(s) "
                  f"exceeding max_length={self._max_length}")

        # 2) 长度降序排批（OOM 早暴露），记录分组顺序
        keys_sorted = sorted(groups, key=lambda k: -len(k))
        res = [None] * len(requests)

        use_cache = self._model.config.use_cache
        self._model.config.use_cache = False
        pbar = tqdm(total=len(requests), disable=disable_tqdm,
                    desc="Running loglikelihood requests")
        try:
            for start in range(0, len(keys_sorted), bs):
                batch_keys = keys_sorted[start:start + bs]
                self._forward_and_score([groups[k] for k in batch_keys],
                                        res, pbar, torch)
        finally:
            self._model.config.use_cache = use_cache
            pbar.close()
        return res

    def _forward_and_score(self, batch_groups, res, pbar, torch):
        """一批 group 的 forward + 分块打分；OOM 时对半降批重试（bs=1 再抛）。"""
        import torch.nn.functional as F

        try:
            pad_id = self.tokenizer.pad_token_id
            maxlen = max(len(g["inp"]) for g in batch_groups)
            B = len(batch_groups)
            input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((B, maxlen), dtype=torch.long)
            for b, g in enumerate(batch_groups):
                L = len(g["inp"])
                input_ids[b, :L] = torch.tensor(g["inp"], dtype=torch.long)
                attn[b, :L] = 1

            # 展平所有评分位（右填充下真实段位置不受 padding 影响）
            rows, cols, tgts, hit_ids = [], [], [], []
            for b, g in enumerate(batch_groups):
                L = len(g["inp"])
                for h_idx, m, cont, _req_str in g["hits"]:
                    for k in range(m):
                        rows.append(b)
                        cols.append(L - m + k)
                        tgts.append(cont[k])
                        hit_ids.append(h_idx)

            with torch.no_grad():
                if self._sequential:
                    hidden = self._forward_sequential(
                        input_ids.to(self._device), attn.to(self._device))
                else:
                    out = self._model.model(
                        input_ids=input_ids.to(self._device),
                        attention_mask=attn.to(self._device))
                    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
                # hidden 可能落在另一张卡（两卡切分时 lm_head 在 cuda:1），
                # 评分就地做，目标 token 搬到 hidden 所在设备
                hd = hidden.device
                targets_all = torch.tensor(tgts, dtype=torch.long, device=hd)
                ll_acc = {}
                greedy_acc = {}
                step = self._max_score_positions
                for c0 in range(0, len(rows), step):
                    c1 = min(c0 + step, len(rows))
                    r = torch.tensor(rows[c0:c1], dtype=torch.long, device=hd)
                    c = torch.tensor(cols[c0:c1], dtype=torch.long, device=hd)
                    t = targets_all[c0:c1]
                    h = hidden[r, c, :]
                    logits = self._model.lm_head(h).float()
                    logp = F.log_softmax(logits, dim=-1)
                    tok_lp = logp.gather(1, t.unsqueeze(1)).squeeze(1)
                    is_greedy = logits.argmax(dim=-1) == t
                    for j in range(c1 - c0):
                        hid = hit_ids[c0 + j]
                        ll_acc[hid] = ll_acc.get(hid, 0.0) + float(tok_lp[j])
                        greedy_acc[hid] = greedy_acc.get(hid, True) and bool(is_greedy[j])
                    del logits, logp, h
                del hidden
        except torch.cuda.OutOfMemoryError:
            if len(batch_groups) == 1:
                raise
            torch.cuda.empty_cache()
            print(f"[warn] CUDA OOM at batch={len(batch_groups)}, "
                  f"halving to {len(batch_groups) // 2}")
            mid = len(batch_groups) // 2
            self._forward_and_score(batch_groups[:mid], res, pbar, torch)
            self._forward_and_score(batch_groups[mid:], res, pbar, torch)
            return

        for g in batch_groups:
            for h_idx, m, cont, req_str in g["hits"]:
                answer = (ll_acc[h_idx], greedy_acc[h_idx])
                res[h_idx] = answer
                if req_str is not None:
                    self.cache_hook.add_partial("loglikelihood", req_str, answer)
                pbar.update(1)

    # ---------------- sequential 层循环前向（fp16 基线） ----------------

    def _forward_sequential(self, input_ids, attention_mask):
        """sequential 层循环前向：层驻 CPU，逐层搬 GPU。

        重放外层 forward 的 mask/position 生成逻辑（modeling_qwen3_5_moe.py
        Qwen3_5MoeTextModel.forward）：变长 batch 不能像 ppl 定长那样只捕获
        一次 kwargs，需按本批形状重建。
        """
        import torch
        from transformers.masking_utils import (
            create_causal_mask, create_recurrent_attention_mask)

        model = self._model
        dev = self._device
        B, L = input_ids.shape
        hidden = model.model.embed_tokens(input_ids)  # [B, L, d]，GPU

        # position_ids：纯文本路径 = arange 展开 [4, B, L]（text + mrope 3 维），
        # 与 attention_mask 无关（padding 靠 mask 挡），见 TextModel.forward
        position_ids = torch.arange(L, device=dev).view(1, 1, -1).expand(4, B, -1)
        text_position_ids = position_ids[0]
        mrope_position_ids = position_ids[1:]

        cfg = model.config
        mask_kwargs = {
            "config": cfg,
            "inputs_embeds": hidden,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": text_position_ids,
        }
        causal_masks = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }
        position_embeddings = model.model.rotary_emb(hidden, mrope_position_ids)
        layer_types = cfg.layer_types

        layers = model.model.layers
        with torch.no_grad():
            for i in range(len(layers)):
                layer = layers[i].to(dev)
                hidden = layer(
                    hidden,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_masks[layer_types[i]],
                    position_ids=text_position_ids,
                )
                layers[i] = layer.to("cpu")
        hidden = model.model.norm(hidden)
        return hidden

    # ---------------- generate_until（gsm8k 等生成任务） ----------------

    def generate_until(self, requests, disable_tqdm=False):
        """左填充批量生成，语义镜像 HFLM.generate_until / _model_generate。"""
        import torch
        from tqdm import tqdm
        from lm_eval.models.utils import (handle_stop_sequences,
                                          normalize_gen_kwargs,
                                          postprocess_generated_text)
        from lm_eval.models.utils_hf import stop_sequences_criteria

        # 按 gen_kwargs 分组（不同采样参数不能同批）
        kw_groups = {}
        for idx, req in enumerate(requests):
            ctx, gk = req.args
            key = json.dumps(gk, sort_keys=True, default=str)
            kw_groups.setdefault(key, {"gen_kwargs": gk, "items": []})
            kw_groups[key]["items"].append((idx, ctx))

        res = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=disable_tqdm,
                    desc="Running generate_until requests")
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        use_cache = self._model.config.use_cache
        self._model.config.use_cache = True
        try:
            for group in kw_groups.values():
                gen_kwargs = group["gen_kwargs"]
                items = group["items"]
                kwargs = normalize_gen_kwargs(gen_kwargs, self._max_gen_toks)
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
                max_gen_toks = kwargs.pop("max_gen_toks")
                # 语义与 HFLM 一致：max_length 由内部按 ctx+max_gen_toks 决定，
                # 任务侧若塞了 max_length 一律丢弃
                kwargs.pop("max_length", None)
                max_ctx_len = self._max_length - max_gen_toks
                assert max_ctx_len > 0, (
                    f"max_gen_toks ({max_gen_toks}) must be < "
                    f"max_length ({self._max_length})")

                # temperature/do_sample 规整（HFLM _model_generate 同款：
                # do_sample=False 时去掉 temperature，避免 HF 告警）
                temp = kwargs.get("temperature", 0.0)
                if temp == 0.0 and kwargs.get("do_sample") is None:
                    kwargs["do_sample"] = False
                if kwargs.get("do_sample", False) is False and temp == 0.0:
                    kwargs.pop("temperature", None)

                # 上下文编码 + 左截断 + 长度降序（OOM 早暴露）
                enc_items = [(idx, self.tok_encode(c)) for idx, c in items]
                enc_items.sort(key=lambda x: -len(x[1]))
                for start in range(0, len(enc_items), self._gen_batch_size):
                    batch = enc_items[start:start + self._gen_batch_size]
                    self._generate_batch(batch, kwargs, until, max_gen_toks,
                                         max_ctx_len, eos, res, pbar,
                                         stop_sequences_criteria, torch)
        finally:
            self._model.config.use_cache = use_cache
            pbar.close()
        return res

    def _generate_batch(self, batch, kwargs, until, max_gen_toks, max_ctx_len,
                        eos, res, pbar, stop_sequences_criteria, torch):
        """一批生成；OOM 对半重试（bs=1 再抛）。"""
        from lm_eval.models.utils import postprocess_generated_text

        try:
            pad_id = self.tokenizer.pad_token_id
            encs = [e[-max_ctx_len:] if len(e) > max_ctx_len else e
                    for _, e in batch]
            padlen = max(len(e) for e in encs)
            B = len(encs)
            input_ids = torch.full((B, padlen), pad_id, dtype=torch.long)
            attn = torch.zeros((B, padlen), dtype=torch.long)
            for b, e in enumerate(encs):        # 左填充：真实 token 靠右
                input_ids[b, padlen - len(e):] = torch.tensor(e, dtype=torch.long)
                attn[b, padlen - len(e):] = 1

            stopping = stop_sequences_criteria(
                self.tokenizer, until, padlen, B)
            with torch.no_grad():
                out = self._model.generate(
                    input_ids=input_ids.to(self._device),
                    attention_mask=attn.to(self._device),
                    max_length=padlen + max_gen_toks,
                    stopping_criteria=stopping,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                    **kwargs)
            gen = out.tolist()
            for (idx, ctx), row in zip(batch, gen):
                cont_toks = row[padlen:]        # 丢弃上下文 + 左填充
                s = self.tok_decode(cont_toks, skip_special_tokens=True)
                # 停止串后处理兜底（stop criteria 已尽量截断）
                s = postprocess_generated_text(s, until, None)
                res[idx] = s
                self.cache_hook.add_partial("generate_until", (ctx, None), s)
                pbar.update(1)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            print(f"[warn] CUDA OOM at gen batch={len(batch)}, "
                  f"halving to {len(batch) // 2}")
            mid = len(batch) // 2
            self._generate_batch(batch[:mid], kwargs, until, max_gen_toks,
                                 max_ctx_len, eos, res, pbar,
                                 stop_sequences_criteria, torch)
            self._generate_batch(batch[mid:], kwargs, until, max_gen_toks,
                                 max_ctx_len, eos, res, pbar,
                                 stop_sequences_criteria, torch)

    # ---------------- loglikelihood_rolling（抽象契约；七任务用不到） ----------------

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        import torch  # noqa: F401  (保持与 HFLM 相同的调用形态)
        from tqdm import tqdm
        from lm_eval.utils import (get_rolling_token_windows,
                                   make_disjoint_window)

        all_windows = []      # (req_idx, (request_str=None, ctx_enc, cont_enc))
        counts = []
        for req_idx, req in enumerate(tqdm(requests, disable=disable_tqdm,
                                           desc="Tokenizing rolling inputs")):
            windows = list(map(
                make_disjoint_window,
                get_rolling_token_windows(
                    token_list=self.tok_encode(req.args[0]),
                    prefix_token=self.prefix_token_id,
                    max_seq_len=self._max_length,
                    context_len=1)))
            all_windows.extend((req_idx, (None, w[0], w[1])) for w in windows)
            counts.append(len(windows))

        nlls = self._loglikelihood_tokens(
            [w for _, w in all_windows], disable_tqdm=disable_tqdm)
        out, cur = [], 0
        for cnt in counts:
            req_windows = nlls[cur:cur + cnt]
            total = sum(x[0] for x in req_windows)
            out.append(total)
            self.cache_hook.add_partial(
                "loglikelihood_rolling",
                (requests[len(out) - 1].args[0],), total)
            cur += cnt
        return out


def _apply_fast_patches_and_warmup(model, batch_size, max_length):
    """kernel patch + Triton JIT 预热（与 eval_qwen35.py 同款环境阀）。"""
    import torch

    if not os.environ.get("DARTMOQ_SKIP_PATCHES"):
        if not os.environ.get("DARTMOQ_SKIP_DELTA"):
            from turboquant_utils.delta_rule import patch_delta_rule
            patch_delta_rule(model)
        if not os.environ.get("DARTMOQ_SKIP_NORMS"):
            from turboquant_utils.norm_kernels import patch_norms
            patch_norms(model)

    print("[warmup] compiling Triton kernels (first run may look stuck; "
          "unseen shapes still compile lazily)...", flush=True)
    t0 = time.time()
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "warmup_kernels"):
            mlp.warmup_kernels(seq_len=min(1024, max_length), batch_size=batch_size)
            mlp.warmup_kernels(seq_len=min(2048, max_length), batch_size=batch_size)
    # 一次带 attention_mask 的微型真 forward：尽早暴露整模型通路问题
    dev = next(model.model.embed_tokens.parameters()).device
    ids = torch.ones((1, 16), dtype=torch.long, device=dev)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        model.model(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"[warmup] done in {time.time() - t0:.1f}s", flush=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="lm-eval-harness downstream tasks on quantized Qwen3.5 checkpoints")
    parser.add_argument('model', type=str, nargs='?', default=None,
                        help="Quantized checkpoint dir (auto-detected); "
                             "or base model path when used with --load-quantized")
    parser.add_argument('--load-quantized', type=str, default=None,
                        help="Load quantized checkpoint (packed safetensors + meta.json)")
    parser.add_argument('--inference-quant-mode', type=str, default='wxa16',
                        choices=['wxa16', 'wxa8', 'wxfp8', 'wxfp4'],
                        help="Inference quant mode (wxfp4 requires dart312-t38 env)")
    parser.add_argument('--tasks', type=str, nargs='+', default=DEFAULT_TASKS,
                        help=f"lm-eval tasks (default: {DEFAULT_TASKS}; gsm8k last)")
    parser.add_argument('--num-fewshot', type=int, default=5,
                        help="few-shot count (BitsMoE alignment: 5)")
    parser.add_argument('--batch-size', type=int, default=8,
                        help="loglikelihood forward batch size")
    parser.add_argument('--gen-batch-size', type=int, default=1,
                        help="generate batch size (batched decode untested; keep 1)")
    parser.add_argument('--max-score-positions', type=int, default=256,
                        help="lm_head scoring chunk size (positions per chunk)")
    parser.add_argument('--limit', type=int, default=None,
                        help="limit docs per task (smoke runs; per-subtask for groups)")
    parser.add_argument('--max-length', type=int, default=4096,
                        help="max sequence length (model supports 262144; clamp for memory)")
    parser.add_argument('--max-gen-toks', type=int, default=256,
                        help="max tokens to generate per request")
    parser.add_argument('--log-samples', action='store_true', default=False,
                        help="store per-sample outputs in result JSON")
    parser.add_argument('--output', type=str, default=None,
                        help="output JSON path (default: results/tasks_<ckpt>_<mode>_...json)")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fp16-mode', type=str, default='auto',
                        choices=['auto', 'sequential'],
                        help="fp16 baseline load mode: auto = device_map multi-GPU "
                             "+ CPU offload (gsm8k supported); sequential = layer "
                             "loop from CPU (loglikelihood only, gsm8k skipped)")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    from eval_qwen35 import _is_quant_dir
    if args.model and not args.load_quantized and _is_quant_dir(args.model):
        args.load_quantized = args.model

    # ---- 模型加载：量化 checkpoint 或 fp16 原始模型（sequential 层循环） ----
    import torch
    sequential = False
    if args.load_quantized:
        # 量化路径（复用 qwen35_quant_io，不重复实现）
        from qwen35_quant_io import load_quantized_model
        base_path = args.model if (args.model and not _is_quant_dir(args.model)) else None
        print(f"Load quantized checkpoint: {args.load_quantized} "
              f"(mode={args.inference_quant_mode})", flush=True)
        model, tokenizer = load_quantized_model(
            base_path, args.load_quantized, standby_cpu=False,
            inference_quant_mode=args.inference_quant_mode)

        # ---- standby 守卫：量化任务评测需要整模型驻留 GPU ----
        if getattr(model, "_standby_cpu", False) or not any(
                p.is_cuda for p in model.parameters()):
            raise SystemExit(
                "Model is not fully resident on GPU (standby-CPU fallback engaged). "
                "Downstream-task eval needs whole-model residency: free a GPU, or use "
                "two GPUs (auto split), or run a smaller checkpoint. For sequential "
                "PPL eval use eval_qwen35.py instead.")

        # 两卡切分：bs 收敛到 8（delta rule fp32 瞬时峰值随 bs 线性增长，见
        # eval_qwen35.clamp_two_gpu_eval_batch 的实测注释；该函数绑定 eval_batch_size
        # 属性名，这里等价内联）
        if getattr(model, "_two_gpu_split", False) and args.batch_size > 8:
            print(f"[auto] batch-size {args.batch_size} -> 8 (two-GPU split: "
                  f"attention fp32 transient peaks with batch size)")
            args.batch_size = 8
    else:
        # fp16 基线路径：原始 bf16 模型 ~70GB 装不进单卡。
        #   默认 device_map="auto"（双卡切分 + 少量 CPU offload，generate 原生可用，
        #   gsm8k 可跑；loglikelihood 六任务的 sequential 数字与 auto 数学等价，
        #   无需重跑）；--fp16-mode sequential 为旧的层驻 CPU 逐层搬路径
        #   （只支持一次前向形态，gsm8k 会 skip）
        if not args.model:
            raise SystemExit(
                "No model given. Pass a quantized checkpoint dir (auto-detected) "
                "or an fp16 base model path (device_map auto / sequential).")
        from qwen35_utils import load_model
        if args.fp16_mode == "sequential":
            print(f"Load fp16 baseline model: {args.model} (sequential layer-loop)",
                  flush=True)
            model, tokenizer = load_model(args.model, standby_cpu=True)
            sequential = True
            # sequential 层循环：batch 可以大（hidden 小、无量化 kernel fp32 瞬峰），
            # 权重搬运量与 batch 无关，batch 越大效率越高
            if args.batch_size < 16:
                print(f"[auto] batch-size {args.batch_size} -> 16 "
                      f"(sequential layer-loop; transfer cost is batch-independent)")
                args.batch_size = 16
            # 常驻 GPU 部件：embed/rotary/norm/lm_head（层循环反复用到，搬一次）
            model.model.embed_tokens = model.model.embed_tokens.to(DEV)
            model.model.rotary_emb = model.model.rotary_emb.to(DEV)
            model.model.norm = model.model.norm.to(DEV)
            model.lm_head = model.lm_head.to(DEV)
            print(f"Resident components on {DEV}: embed/rotary/norm/lm_head "
                  f"(layers streamed from CPU per batch)", flush=True)
        else:
            print(f"Load fp16 baseline model: {args.model} (device_map auto, "
                  f"multi-GPU + CPU offload)", flush=True)
            model, tokenizer = load_model(args.model, standby_cpu=False)

    if not sequential:
        _apply_fast_patches_and_warmup(model, args.batch_size, args.max_length)

    lm = DartMoQLM(
        model, tokenizer,
        batch_size=args.batch_size,
        gen_batch_size=args.gen_batch_size,
        max_length=args.max_length,
        max_gen_toks=args.max_gen_toks,
        max_score_positions=args.max_score_positions,
        sequential=sequential)

    # ---- 逐任务评测（错误隔离：单任务失败不阻塞其余任务） ----
    import lm_eval
    from lm_eval import evaluator
    from data_utils import get_git_hash
    import transformers

    ckpt_name = os.path.basename(os.path.normpath(
        args.load_quantized if args.load_quantized else args.model))
    mode_name = "fp16" if sequential else args.inference_quant_mode
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.output or os.path.join(
        "results", f"tasks_{ckpt_name}_{mode_name}_"
                   f"{args.num_fewshot}shot_{get_git_hash()}_{ts}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    t_start = time.time()
    task_results = {}
    n_failed = 0
    for task in args.tasks:
        print(f"\n=== Evaluating {task} (num_fewshot={args.num_fewshot}, "
              f"limit={args.limit}) ===", flush=True)
        t0 = time.time()
        # sequential 层循环不支持生成任务（KV cache 无法跨层搬运），提前拦截
        # 记 skipped（非 error）：不是失败，parser 会显示 SKIP
        if sequential and task == "gsm8k":
            print(f"[skip] gsm8k unsupported in sequential layer-loop mode "
                  f"(KV-cache generation needs whole-model residency)", flush=True)
            task_results[task] = {
                "metrics": None,
                "wall_s": 0.0,
                "error": None,
                "skipped": "generate_until unsupported in sequential mode",
                "traceback": "",
            }
            continue
        try:
            # mathqa：优先用本地 parquet yaml（hub 旧式 script 已被 datasets 4.x 禁用）
            task_ref = task
            if task == "mathqa" and os.path.isfile(MATHQA_LOCAL_YAML):
                task_ref = MATHQA_LOCAL_YAML
                print(f"  [mathqa] using local parquet task yaml: {MATHQA_LOCAL_YAML}",
                      flush=True)
            result = evaluator.simple_evaluate(
                model=lm,
                tasks=[task_ref],
                num_fewshot=args.num_fewshot,
                limit=args.limit,
                bootstrap_iters=0,
                log_samples=args.log_samples)
            metrics = result["results"][task]
            task_results[task] = {
                "metrics": metrics,
                "wall_s": round(time.time() - t0, 1),
                "error": None,
            }
            if args.log_samples:
                task_results[task]["samples"] = result.get("samples", {}).get(task)
            print(f"--- {task} ---")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
        except Exception as e:  # noqa: BLE001  隔离单任务失败
            n_failed += 1
            print(f"[error] task {task} failed: {e}", flush=True)
            traceback.print_exc()
            task_results[task] = {
                "metrics": None,
                "wall_s": round(time.time() - t0, 1),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- 结果落盘（含 git hash / 版本 / 口径，论文可直接引用） ----
    payload = {
        "git_hash": get_git_hash(),
        "lm_eval_version": lm_eval.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "quant_dir": os.path.abspath(args.load_quantized if args.load_quantized else args.model),
        "inference_quant_mode": "fp16" if sequential else args.inference_quant_mode,
        "sequential_eval": sequential,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "gen_batch_size": args.gen_batch_size,
        "max_length": args.max_length,
        "two_gpu_split": bool(getattr(model, "_two_gpu_split", False)),
        "tasks": args.tasks,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_s": round(time.time() - t_start, 1),
        "results": task_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n===== Summary ({ckpt_name}, {mode_name}, "
          f"{args.num_fewshot}-shot) =====")
    for task, r in task_results.items():
        if r["error"]:
            print(f"  {task}: FAILED ({r['error']})")
        else:
            # 通用浮点遍历：覆盖 acc / acc_norm / exact_match,<filter> 等
            # （gsm8k 的 key 形如 exact_match,strict-match）
            floats = {k: v for k, v in r["metrics"].items()
                      if isinstance(v, float)}
            line = "  " + " ".join(f"{k}={v:.4f}" for k, v in sorted(floats.items()))
            print(line + f"  [{r['wall_s']}s]")
    print(f"Results saved to: {out_path}")
    if n_failed:
        print(f"{n_failed} task(s) failed; see JSON for tracebacks")
        sys.exit(1)


if __name__ == "__main__":
    main()
