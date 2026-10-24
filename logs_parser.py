#!/usr/bin/env python3
"""Parse DartMoQ slurm logs into aligned benchmark rows.

The parser scans each log sequentially and starts a new row at every
"Loading model: (ppl) ..." line. Metrics that are missing because a run crashed
or skipped downstream evaluation are left empty instead of being shifted onto a
neighboring row.

另支持 bench_throughput.py 的 prefill/decode 吞吐日志（评测计划 P0）：
以 "[model] ... | mode: ..." 行起始一条记录（bench 脚本专属标记，其他
脚本日志无此格式），报告块内解析 prefill 各档 tok/s、decode 各档 tok/s、
TTFT、一致性守卫结论与 MoE 每步总耗时。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Iterable


# 自适应列分组：测什么显示什么。
# COMMON 恒显；量化/eval 组在批次里有数据才显（列集合与旧 DISPLAY_FIELDS
# 一致，不含 moe_struct/disable_0bit_prune 等几乎永远空着的字段）；
# throughput 组按 run 类型显，档位列（prefill_tokps_<len> /
# decode_tokps_<n>）由实际测过的档位动态生成。
COMMON_FIELDS = ["git_hash", "model_name", "quantmode", "status", "error"]
QUANT_EVAL_FIELDS = [
    "slices", "quant_scheme", "rank_mode", "quant_layers", "bpw",
    "ppl_wikitext2", "ppl_c4",
    "runtime_ppl", "runtime_quant", "runtime_ppl_eval",
    "runtime_ppl_wikitext2", "runtime_ppl_c4",
]

# New log format patterns
START_RUN_RE = re.compile(r"DartMoQ for Qwen3.5 MoE")
EVAL_START_RUN_RE = re.compile(r"Qwen3.5 MoE Evaluation")
NEW_MODEL_RE = re.compile(r"^Model:\s+(?P<path>\S+)")
NEW_GIT_HASH_RE = re.compile(r"^Git HEAD:\s+(?P<git_hash>[a-f0-9]+\+?)")
NEW_QUANT_SCHEME_RE = re.compile(r"^Quant scheme:\s+(?P<quant_scheme>\S+)")
NEW_RANK_MODE_RE = re.compile(r"^Rank mode:\s+(?P<rank_mode>\S+)")
NEW_SLICES_RE = re.compile(r"^Slices per expert:\s+(?P<slices>\S+)")
NEW_QUANTMODE_RE = re.compile(r"^Quant mode:\s+(?P<quantmode>\S+)")
NEW_STANDBY_CPU_RE = re.compile(r"^CPU standby:\s+(?P<standby_layer_cpu>\S+)")
NEW_WXA16_REAL_QUANT_RE = re.compile(r"^WxA16 real quantization:\s+(?P<wxa16_real_quant>\S+)")
NEW_QUANT_LAYERS_RE = re.compile(r"^Quantizing layers:\s+(?P<quant_layers>.+)")
NEW_START_TIME_RE = re.compile(r"^Current time:\s*(?P<value>.+)")
NEW_FINISH_TIME_RE = re.compile(r"^Finish time:\s*(?P<value>.+)")
NEW_PPL_RE = re.compile(r"ppl on (?P<dataset>wikitext2|c4)(?:\s+\([^)]+\))?:\s*(?P<value>[-+0-9.eE]+|nan|inf)(?:\s+time:\s*(?P<time_value>[-+0-9.eE]+))?")
LAYER_TIME_RE = re.compile(r"Layer (?P<layer>\d+) total reconstruct and quantization time:\s*(?P<value>[-+0-9.eE]+) s")
# Eval-specific patterns
EVAL_DATASETS_RE = re.compile(r"^Datasets:\s+(?P<datasets>.+)")
EVAL_SEQUENTIAL_RE = re.compile(r"^Sequential eval:\s+(?P<sequential>\S+)")
EVAL_STANDBY_CPU_RE = re.compile(r"^Standby CPU:\s+(?P<standby_cpu>\S+)")
# eval_qwen35.py 的 checkpoint 行有两代格式：旧 "Load quantized checkpoint:"
# 与现行参数回显 "Quantized checkpoint:"（吞吐日志的 "Loading quantized
# checkpoint from:" 带 Loading 前缀，不会被此正则误匹配）
EVAL_LOAD_QUANTIZED_RE = re.compile(
    r"^(?:Load )?[Qq]uantized checkpoint:\s+(?P<path>\S+)")
# WxA8 推理模式（eval_qwen35.py 打印，位于 Quantized checkpoint 之后）
EVAL_INFERENCE_QUANT_MODE_RE = re.compile(r"^Inference quant mode:\s+(?P<mode>\S+)")

# Throughput 日志（bench_throughput.py，2026-10-23 从 benchmarks/ 移到根）
THROUGHPUT_START_RE = re.compile(
    r"^\[model\]\s+(?P<model>\S+)\s+on\s+\S+\s+\|\s*mode:\s*(?P<mode>\S+)")
THROUGHPUT_MODEL_RE = re.compile(
    r"^model:\s+(?P<model>\S+)\s*\|\s*mode:\s*(?P<mode>\S+)"
    r".*dataset:\s*(?P<dataset>\S+)")
THROUGHPUT_GIT_RE = re.compile(r"\bgit:\s*(?P<git_hash>\S+)")
THROUGHPUT_PREFILL_RE = re.compile(
    r"^\s*len=(?P<len>\d+)\s+latency=[-+0-9.eE]+s\s+tok/s=(?P<tokps>[-+0-9.eE]+)")
THROUGHPUT_DECODE_RE = re.compile(
    r"^\s*tokens=(?P<n>\d+)\s+wall tok/s = (?P<tokps>[-+0-9.eE]+)")
THROUGHPUT_TTFT_RE = re.compile(r"^\s*TTFT = (?P<ttft>[-+0-9.eE]+)s")
THROUGHPUT_CHECK_RE = re.compile(r"^Consistency check:\s*(?P<verdict>PASS|FAIL)")
# checkpoint 标识：bench summary 的 checkpoint: 行。⚠ 只认 summary 行——
# "Loading quantized checkpoint from:" 打印在**下一次**运行的 [model] 行
# 之前，会泄漏进上一条（failed 无 summary 的）记录
THROUGHPUT_CKPT_RE = re.compile(r"^checkpoint:\s+(?P<path>\S+)")
CHECKPOINT_BPW_RE = re.compile(r"(\d+(?:\.\d+)?)\s*bpw", re.IGNORECASE)
# 报告里以 total 开头的行只有 MoE stage 块的 total 行（_clean_line 会 strip
# 行首空格，不能靠缩进判断块边界，直接匹配即可）
THROUGHPUT_MOE_TOTAL_RE = re.compile(r"^\s*total\s+(?P<value>[-+0-9.eE]+)")

# Old log format patterns (for backward compatibility)
LOADING_RE = re.compile(r"Loading model:\s*\(ppl\)\s*(?P<path>\S+)")
QUANTMODE_RE = re.compile(
    r"slices/quant-scheme/rank-mode/moe-struct/quantmode(?:/disable-0bit-prune)?(?:/standby-layer-cpu)?:\s*\(ppl\)\s+"
    r"(?P<slices>\S+)\s+(?P<quant_scheme>\S+)\s+(?P<rank_mode>\S+)\s+"
    r"(?P<moe_struct>\S+)\s+(?P<quantmode>\S+)(?:\s+(?P<disable_0bit_prune>\S+))?(?:\s+(?P<standby_layer_cpu>\S+))?"
)
BPW_RE = re.compile(r"\bwith bpw\s+(?P<bpw>[-+0-9.eE]+)")
PPL_RE = re.compile(r"ppl on (?P<dataset>wikitext2|c4)(?:\s+\([^)]+\))?:\s*(?P<value>[-+0-9.eE]+|nan|inf)")
RUNTIME_RE = re.compile(r"Runtime of training-free construction \(ppl\):\s*(?P<value>[-+0-9.eE]+)")
RUNTIME_QUANT_RE = re.compile(r"Runtime of quantization only:\s*(?P<value>[-+0-9.eE]+)")
RUNTIME_PPL_EVAL_RE = re.compile(r"Runtime of wiki/c4 validation:\s*(?P<value>[-+0-9.eE]+)")
START_TIME_RE = re.compile(r"Current start time:\s*(?P<value>.+)")
FATAL_RE = re.compile(r"Segmentation fault|Traceback|RuntimeError|CUDA out of memory|Killed", re.IGNORECASE)
MODEL_NAME_RE = re.compile(r"^model:\s+(?P<path>\S+)\s+(?P<name>\S+)")
NUMERIC_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")

# For backward compatibility with old logs
PPL_INDIVIDUAL_TIME_RE = re.compile(r"ppl on (?P<dataset>wikitext2|c4)(?:\s+\([^)]+\))?:\s*(?:[-+0-9.eE]+|nan|inf)\s*time:\s*(?P<value>[-+0-9.eE]+)")

# Pattern to parse bpw from quant_scheme (format like "w8a8" or "w4a4" etc.)
QUANT_SCHEME_BPW_RE = re.compile(r"w(?P<weight_bits>\d+)(?:a(?P<act_bits>\d+))?")


@dataclass
class RunRecord:
    source: str
    run_idx: int
    start_line: int
    end_line: int = 0
    status: str = "incomplete"
    start_time: str = ""
    model_path: str = ""
    model_name: str = ""
    slices: str = ""
    quant_scheme: str = ""
    rank_mode: str = ""
    moe_struct: str = ""
    quantmode: str = ""
    disable_0bit_prune: str = ""
    standby_layer_cpu: str = ""
    quant_layers: str = ""
    wxa16_real_quant: str = ""
    git_hash: str = ""
    bpw: str = ""
    ppl_wikitext2: str = ""
    ppl_c4: str = ""
    runtime_ppl_wikitext2: str = ""
    runtime_ppl_c4: str = ""
    runtime_ppl: str = ""
    runtime_quant: str = ""
    runtime_ppl_eval: str = ""
    error: str = ""
    # Throughput（bench_throughput.py）字段
    is_throughput: bool = False
    dataset: str = ""
    prefill_tokps: dict[int, str] = field(default_factory=dict, repr=False)
    decode_tokps: dict[int, str] = field(default_factory=dict, repr=False)
    ttft_s: str = ""
    check: str = ""
    moe_total_ms: str = ""
    ckpt_path: str = ""
    checkpoint_path: str = ""
    _fatal: bool = field(default=False, repr=False)
    # Temporary fields for accumulating time from old logs
    _layer_times: list[float] = field(default_factory=list, repr=False)
    _ppl_eval_times: list[float] = field(default_factory=list, repr=False)

    @property
    def ckpt(self) -> str:
        """checkpoint 目录名：吞吐批次混合多 checkpoint（gptq/turboquant/s1）
        时区分每行来源。优先吞吐 summary 的 checkpoint: 行，其次 eval 侧。"""
        for p in (self.checkpoint_path, self.ckpt_path, self.model_path):
            if p:
                return _last_path_component(p)
        return ""

    def finalize(self, end_line: int) -> None:
        self.end_line = end_line
        # Only calculate and show quant time if we actually have ppl results
        # (meaning the quantization process completed successfully)
        if not self.runtime_quant and self._layer_times and (self.ppl_wikitext2 or self.ppl_c4):
            total = sum(self._layer_times)
            self.runtime_quant = f"{total:.2f}"
        # Calculate ppl eval time from individual ppl times if not already set
        if not self.runtime_ppl_eval and self._ppl_eval_times:
            total = sum(self._ppl_eval_times)
            self.runtime_ppl_eval = f"{total:.2f}"
        # Calculate bpw from quant_scheme if not already set
        if not self.bpw and self.quant_scheme:
            bpw = self._calculate_bpw_from_scheme(self.quant_scheme)
            if bpw is not None:
                self.bpw = f"{bpw:.2f}"
        # Calculate total time: time = t_q + t_ppl
        if not self.runtime_ppl:
            t_quant = 0.0
            t_ppl = 0.0
            try:
                if self.runtime_quant:
                    t_quant = float(self.runtime_quant)
            except ValueError:
                pass
            try:
                if self.runtime_ppl_eval:
                    t_ppl = float(self.runtime_ppl_eval)
            except ValueError:
                pass
            if t_quant > 0 or t_ppl > 0:
                self.runtime_ppl = f"{t_quant + t_ppl:.2f}"
        if self._fatal:
            self.status = "failed"
        elif self.decode_tokps:
            # Throughput 记录：decode 数字齐全即 ok
            self.status = "ok"
        elif self.prefill_tokps:
            # 只有 prefill（decode 前崩了）——OOM 等 fatal 已在上面判 failed
            self.status = "partial"
        elif self.ppl_wikitext2 and self.ppl_c4:
            # Have both ppl results
            self.status = "ok"
        elif self.runtime_ppl:
            # Have other results but not both ppl
            self.status = "partial"
        else:
            # Have only one ppl or nothing at all
            self.status = "incomplete"

    @staticmethod
    def _calculate_bpw_from_scheme(quant_scheme: str) -> float | None:
        """Calculate bpw from quant_scheme string.

        Format examples: "w8a8", "w4a4", "w4", "w8a16", etc.
        Returns None if parsing fails.
        """
        match = QUANT_SCHEME_BPW_RE.search(quant_scheme)
        if match:
            weight_bits = int(match.group("weight_bits"))
            act_bits = match.group("act_bits")
            if act_bits:
                # Simple estimation: average of weight and activation bits
                # This can be adjusted based on actual calculation logic
                act_bits_val = int(act_bits)
                return (weight_bits + act_bits_val) / 2.0
            else:
                return float(weight_bits)
        return None

    def public_dict(self) -> dict[str, str | int]:
        row = asdict(self)
        row.pop("_fatal", None)
        return row


def _clean_line(line: str) -> str:
    return line.replace("\r", "").strip()


def _last_path_component(path: str) -> str:
    return os.path.basename(path.rstrip("/"))


def _append_error(record: RunRecord, line: str) -> None:
    message = line.strip()
    if not message:
        return
    if record.error:
        record.error += " | " + message
    else:
        record.error = message


def parse_log(path: str) -> list[RunRecord]:
    records: list[RunRecord] = []
    current: RunRecord | None = None
    pending_start_time = ""
    source = os.path.basename(path)
    last_line_no = 0
    # Track which format we're using
    using_new_format = False
    using_eval_format = False
    using_throughput_format = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, 1):
            last_line_no = line_no
            line = _clean_line(raw_line)
            if not line:
                continue

            # Check for eval format start
            eval_start_match = EVAL_START_RUN_RE.search(line)
            if eval_start_match:
                if current is not None:
                    current.finalize(line_no - 1)
                    records.append(current)
                current = RunRecord(
                    source=source,
                    run_idx=len(records) + 1,
                    start_line=line_no,
                    start_time="",
                    model_path="",
                )
                current.quantmode = "fp16"
                using_eval_format = True
                using_new_format = False
                continue

            # Check for new format start
            start_run_match = START_RUN_RE.search(line)
            if start_run_match:
                if current is not None:
                    current.finalize(line_no - 1)
                    records.append(current)
                current = RunRecord(
                    source=source,
                    run_idx=len(records) + 1,
                    start_line=line_no,
                    start_time="",
                    model_path="",
                )
                using_new_format = True
                using_eval_format = False
                continue

            # Check for throughput format start (bench_throughput.py 的
            # "[model] ... | mode: ..." 行，其他脚本日志无此格式)
            throughput_start_match = THROUGHPUT_START_RE.search(line)
            if throughput_start_match:
                if current is not None:
                    current.finalize(line_no - 1)
                    records.append(current)
                current = RunRecord(
                    source=source,
                    run_idx=len(records) + 1,
                    start_line=line_no,
                    start_time="",
                    model_path="",
                )
                current.model_name = throughput_start_match.group("model")
                current.quantmode = throughput_start_match.group("mode")
                current.is_throughput = True
                using_throughput_format = True
                using_new_format = False
                using_eval_format = False
                continue

            # If not using new format yet, check for old format patterns
            if not using_new_format and not using_eval_format and not using_throughput_format:
                start_time_match = START_TIME_RE.search(line)
                if start_time_match:
                    pending_start_time = start_time_match.group("value").strip()

                loading_match = LOADING_RE.search(line)
                if loading_match:
                    if current is not None:
                        current.finalize(line_no - 1)
                        records.append(current)
                    current = RunRecord(
                        source=source,
                        run_idx=len(records) + 1,
                        start_line=line_no,
                        start_time=pending_start_time,
                        model_path=loading_match.group("path"),
                    )
                    current.model_name = _last_path_component(current.model_path)
                    using_new_format = False
                    using_eval_format = False
                    continue

            if current is None:
                continue

            # Parse layer times - do this first for all formats!
            layer_time_match = LAYER_TIME_RE.search(line)
            if layer_time_match:
                try:
                    t = float(layer_time_match.group("value"))
                    current._layer_times.append(t)
                except ValueError:
                    pass
                # Don't continue, fall through to other matches

            # Throughput format parsing (bench_throughput.py 报告块)
            if using_throughput_format:
                tput_model_match = THROUGHPUT_MODEL_RE.search(line)
                if tput_model_match:
                    current.model_name = tput_model_match.group("model")
                    current.quantmode = tput_model_match.group("mode")
                    current.dataset = tput_model_match.group("dataset")
                    continue
                tput_ckpt_match = THROUGHPUT_CKPT_RE.search(line)
                if tput_ckpt_match:
                    current.checkpoint_path = tput_ckpt_match.group("path")
                    m = CHECKPOINT_BPW_RE.search(current.checkpoint_path)
                    if m and not current.bpw:
                        current.bpw = m.group(1)
                    continue
                tput_git_match = THROUGHPUT_GIT_RE.search(line)
                if tput_git_match:
                    current.git_hash = tput_git_match.group("git_hash")
                    continue
                tput_prefill_match = THROUGHPUT_PREFILL_RE.search(line)
                if tput_prefill_match:
                    current.prefill_tokps[int(tput_prefill_match.group("len"))] = \
                        tput_prefill_match.group("tokps")
                    continue
                tput_decode_match = THROUGHPUT_DECODE_RE.search(line)
                if tput_decode_match:
                    current.decode_tokps[int(tput_decode_match.group("n"))] = \
                        tput_decode_match.group("tokps")
                    continue
                tput_ttft_match = THROUGHPUT_TTFT_RE.search(line)
                if tput_ttft_match:
                    current.ttft_s = tput_ttft_match.group("ttft")
                    continue
                tput_check_match = THROUGHPUT_CHECK_RE.search(line)
                if tput_check_match:
                    current.check = tput_check_match.group("verdict")
                    continue
                tput_moe_total = THROUGHPUT_MOE_TOTAL_RE.search(line)
                if tput_moe_total:
                    current.moe_total_ms = tput_moe_total.group("value")
                    continue
                tput_ckpt_match = THROUGHPUT_CKPT_RE.search(line)
                if tput_ckpt_match:
                    current.ckpt_path = tput_ckpt_match.group("path")
                    if not current.bpw:
                        m = CHECKPOINT_BPW_RE.search(current.ckpt_path)
                        if m:
                            current.bpw = m.group(1)
                    continue
                if FATAL_RE.search(line):
                    current._fatal = True
                    _append_error(current, line)
                continue

            # Both new formats use model line
            model_match = NEW_MODEL_RE.search(line)
            if model_match:
                current.model_path = model_match.group("path")
                current.model_name = _last_path_component(current.model_path)
                continue

            # Parse git hash
            git_hash_match = NEW_GIT_HASH_RE.search(line)
            if git_hash_match:
                current.git_hash = git_hash_match.group("git_hash")
                continue

            # Eval format parsing
            if using_eval_format:
                # Eval-specific fields
                eval_datasets_match = EVAL_DATASETS_RE.search(line)
                if eval_datasets_match:
                    # Just note it, not a critical field
                    pass

                eval_sequential_match = EVAL_SEQUENTIAL_RE.search(line)
                if eval_sequential_match:
                    # Could map to something, not critical
                    pass

                eval_standby_cpu_match = EVAL_STANDBY_CPU_RE.search(line)
                if eval_standby_cpu_match:
                    current.standby_layer_cpu = eval_standby_cpu_match.group("standby_cpu")
                    continue

                # 量化 checkpoint 直接 eval（--load-quantized）
                eval_quant_match = EVAL_LOAD_QUANTIZED_RE.search(line)
                if eval_quant_match:
                    qpath = eval_quant_match.group("path")
                    current.model_path = qpath
                    current.model_name = _last_path_component(qpath)
                    current.wxa16_real_quant = "Yes"
                    current.quantmode = "wxa16_load"
                    continue

                # 推理量化模式：wxa8 覆盖默认的 wxa16_load 标记
                # （Quantized checkpoint 行之后打印，见 eval_qwen35.py）
                eval_inf_mode_match = EVAL_INFERENCE_QUANT_MODE_RE.search(line)
                if eval_inf_mode_match:
                    mode = eval_inf_mode_match.group("mode")
                    current.quantmode = f"{mode}_load"
                    continue

                # Eval uses the same PPL format
                new_ppl_match = NEW_PPL_RE.search(line)
                if new_ppl_match:
                    dataset = new_ppl_match.group("dataset")
                    setattr(current, f"ppl_{dataset}", new_ppl_match.group("value"))
                    time_value = new_ppl_match.groupdict().get("time_value")
                    if time_value:
                        try:
                            t = float(time_value)
                            current._ppl_eval_times.append(t)
                            setattr(current, f"runtime_ppl_{dataset}", f"{t:.2f}")
                        except ValueError:
                            pass
                    continue

            # New format parsing
            elif using_new_format:
                quant_scheme_match = NEW_QUANT_SCHEME_RE.search(line)
                if quant_scheme_match:
                    current.quant_scheme = quant_scheme_match.group("quant_scheme")
                    continue

                rank_mode_match = NEW_RANK_MODE_RE.search(line)
                if rank_mode_match:
                    current.rank_mode = rank_mode_match.group("rank_mode")
                    continue

                slices_match = NEW_SLICES_RE.search(line)
                if slices_match:
                    current.slices = slices_match.group("slices")
                    continue

                quantmode_match = NEW_QUANTMODE_RE.search(line)
                if quantmode_match:
                    current.quantmode = quantmode_match.group("quantmode")
                    continue

                standby_cpu_match = NEW_STANDBY_CPU_RE.search(line)
                if standby_cpu_match:
                    current.standby_layer_cpu = standby_cpu_match.group("standby_layer_cpu")
                    continue

                wxa16_real_quant_match = NEW_WXA16_REAL_QUANT_RE.search(line)
                if wxa16_real_quant_match:
                    current.wxa16_real_quant = wxa16_real_quant_match.group("wxa16_real_quant")
                    continue

                quant_layers_match = NEW_QUANT_LAYERS_RE.search(line)
                if quant_layers_match:
                    current.quant_layers = quant_layers_match.group("quant_layers")
                    continue
                # Handle "Quantize layers:" (without "ing") case from run_qwen35.py
                if line.startswith("Quantize layers: "):
                    value = line[len("Quantize layers: "):].strip()
                    if value.lower() == "all":
                        current.quant_layers = "all"
                    else:
                        current.quant_layers = value
                    continue
                # Handle "Quantizing all layers" case
                if "Quantizing all layers" in line:
                    current.quant_layers = "all"
                    continue

                start_time_match = NEW_START_TIME_RE.search(line)
                if start_time_match:
                    current.start_time = start_time_match.group("value").strip()
                    continue

                # New format PPL with time
                new_ppl_match = NEW_PPL_RE.search(line)
                if new_ppl_match:
                    dataset = new_ppl_match.group("dataset")
                    setattr(current, f"ppl_{dataset}", new_ppl_match.group("value"))
                    time_value = new_ppl_match.groupdict().get("time_value")
                    if time_value:
                        try:
                            t = float(time_value)
                            current._ppl_eval_times.append(t)
                            setattr(current, f"runtime_ppl_{dataset}", f"{t:.2f}")
                        except ValueError:
                            pass
                    continue

                # New format finish time - can calculate total runtime
                finish_time_match = NEW_FINISH_TIME_RE.search(line)
                if finish_time_match and current.start_time:
                    # Try to calculate runtime from start and finish times if available
                    pass
                continue

            # Old format parsing
            quantmode_match = QUANTMODE_RE.search(line)
            if quantmode_match:
                for key, value in quantmode_match.groupdict().items():
                    setattr(current, key, value)
                continue

            bpw_match = BPW_RE.search(line)
            if bpw_match and not current.bpw:
                current.bpw = bpw_match.group("bpw")
                continue

            model_name_match = MODEL_NAME_RE.search(line)
            if model_name_match:
                current.model_name = model_name_match.group("name")
                continue

            # Backward compatibility: parse individual ppl eval times for old logs
            # Need to check this before PPL_RE because PPL_RE matches a subset
            ppl_ind_time_match = PPL_INDIVIDUAL_TIME_RE.search(line)
            if ppl_ind_time_match:
                try:
                    t = float(ppl_ind_time_match.group("value"))
                    current._ppl_eval_times.append(t)
                except ValueError:
                    pass
                # Still need to extract the ppl value, so don't continue here

            ppl_match = PPL_RE.search(line)
            if ppl_match:
                dataset = ppl_match.group("dataset")
                setattr(current, f"ppl_{dataset}", ppl_match.group("value"))
                continue

            runtime_match = RUNTIME_RE.search(line)
            if runtime_match:
                current.runtime_ppl = runtime_match.group("value")
                continue

            runtime_quant_match = RUNTIME_QUANT_RE.search(line)
            if runtime_quant_match:
                current.runtime_quant = runtime_quant_match.group("value")
                continue

            runtime_ppl_eval_match = RUNTIME_PPL_EVAL_RE.search(line)
            if runtime_ppl_eval_match:
                current.runtime_ppl_eval = runtime_ppl_eval_match.group("value")
                continue

            if FATAL_RE.search(line):
                current._fatal = True
                _append_error(current, line)

    if current is not None:
        current.finalize(last_line_no)
        records.append(current)

    return records


def parse_logs(paths: Iterable[str]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for path in paths:
        records.extend(parse_log(path))
    return records


def _format_row(record: RunRecord, fields: list[str]) -> dict[str, str]:
    return {field: _format_value(field, _field_value(record, field))
            for field in fields}


def _format_export_row(record: RunRecord, fields: list[str]) -> dict[str, str]:
    return {_export_header(field): _format_value(field, _field_value(record, field))
            for field in fields}


def _csv_escape(value: str) -> str:
    if "," in value or '"' in value or "\n" in value:
        return f'"{value.replace(chr(34), chr(34)*2)}"'
    return value


def write_csv(records: list[RunRecord], out) -> None:
    fields = _select_fields(records)
    headers = [_export_header(f) for f in fields]
    out.write(",".join([_csv_escape(h) for h in headers]) + "\n")
    for record in records:
        row = _format_row(record, fields)
        out.write(",".join([_csv_escape(row.get(field, "")) for field in fields]) + "\n")


def _field_value(record: RunRecord, field: str) -> str:
    """按字段名取值；档位字段从 dict 动态解析（prefill_tokps_<len> 等）。"""
    if field.startswith("prefill_tokps_"):
        return record.prefill_tokps.get(int(field[len("prefill_tokps_"):]), "")
    if field.startswith("decode_tokps_"):
        return record.decode_tokps.get(int(field[len("decode_tokps_"):]), "")
    return str(getattr(record, field, "") or "")


def _select_fields(records: list[RunRecord]) -> list[str]:
    """自适应选择展示列：按批次内容决定列集合。"""
    fields = list(COMMON_FIELDS)
    # checkpoint 目录名列：混合批次（gptq/turboquant 等）必须能区分每行来源；
    # ckpt 名内已含模型名（qwen3.5-35B-...），任一记录有 ckpt 即用 ckpt 替换
    # model 列，避免包含关系冗余；未完成/无 ckpt 的行该列显示 ----
    if any(r.ckpt for r in records):
        fields[fields.index("model_name")] = "ckpt"
    # bpw 不触发 eval 列集：吞吐记录的 bpw 来自 checkpoint: 行（summary），
    # 不是量化元数据，不应把 sli/qsch/rank 等 eval 列带出来
    if any(any(getattr(r, f) for r in records)
           for f in QUANT_EVAL_FIELDS if f != "bpw"):
        fields += QUANT_EVAL_FIELDS
    if any(r.is_throughput for r in records):
        if any(r.bpw for r in records) and "bpw" not in fields:
            fields.insert(fields.index("quantmode"), "bpw")
        if any(r.dataset for r in records):
            fields.append("dataset")
        # 模型 bpw（checkpoint 路径提取，吞吐批次混合多 checkpoint 时也有意义）
        if any(r.bpw for r in records) and "bpw" not in fields:
            fields.insert(fields.index("quantmode"), "bpw")
        prefill_lens = sorted({L for r in records for L in r.prefill_tokps})
        decode_tiers = sorted({n for r in records for n in r.decode_tokps})
        fields += [f"prefill_tokps_{L}" for L in prefill_lens]
        fields += [f"decode_tokps_{n}" for n in decode_tiers]
        for f in ("ttft_s", "check", "moe_total_ms"):
            if any(getattr(r, f) for r in records):
                fields.append(f)
    return fields


def _plain_header(field: str) -> str:
    """plain 输出的短表头；档位字段生成 pf<len>/dec<n> 形式。"""
    if field.startswith("prefill_tokps_"):
        return "pf" + field.rsplit("_", 1)[1]
    if field.startswith("decode_tokps_"):
        return "dec" + field.rsplit("_", 1)[1]
    return PLAIN_HEADERS.get(field, field)


def _export_header(field: str) -> str:
    """csv/md/json 输出表头；档位字段直接用完整字段名。"""
    return EXPORT_HEADERS.get(field, field)

PLAIN_HEADERS = {
    "model_name": "model",
    "ckpt": "ckpt",
    "slices": "sli",
    "quant_scheme": "qsch",
    "rank_mode": "rank",
    "moe_struct": "moe",
    "quantmode": "qmode",
    "disable_0bit_prune": "no0prune",
    "standby_layer_cpu": "stdbycpu",
    "quant_layers": "qlayers",
    "git_hash": "git",
    "bpw": "bpw",
    "ppl_wikitext2": "wiki",
    "ppl_c4": "c4",
    "status": "status",
    "runtime_ppl": "time",
    "runtime_quant": "t_quant",
    "runtime_ppl_eval": "t_ppl",
    "runtime_ppl_wikitext2": "t_wiki",
    "runtime_ppl_c4": "t_c4",
    "error": "err",
    "dataset": "ds",
    "ttft_s": "ttft",
    "check": "check",
    "moe_total_ms": "moe_ms",
}

EXPORT_HEADERS = {
    "model_name": "model",
    "ckpt": "ckpt",
    "slices": "slices",
    "quant_scheme": "quant_scheme",
    "rank_mode": "rank_mode",
    "moe_struct": "moe_struct",
    "quantmode": "quantmode",
    "disable_0bit_prune": "disable_0bit_prune",
    "standby_layer_cpu": "standby_layer_cpu",
    "quant_layers": "quant_layers",
    "wxa16_real_quant": "wxa16_real_quant",
    "git_hash": "git_hash",
    "bpw": "bpw",
    "ppl_wikitext2": "ppl_wikitext2",
    "ppl_c4": "ppl_c4",
    "status": "status",
    "runtime_ppl": "total_time",
    "runtime_quant": "quant_time",
    "runtime_ppl_eval": "ppl_eval_time",
    "runtime_ppl_wikitext2": "t_wiki",
    "runtime_ppl_c4": "t_c4",
    "dataset": "dataset",
    "ttft_s": "ttft_s",
    "check": "check",
    "moe_total_ms": "moe_total_ms",
}


def _format_value(field: str, value) -> str:
    if field == "error":
        return ""
    # git_hash 和 wxa16_real_quant 等字段看起来像数字但不应被格式化
    if field in ("git_hash", "wxa16_real_quant"):
        return str(value).replace("\t", " ")
    text = str(value)
    if NUMERIC_RE.fullmatch(text):
        try:
            text = f"{float(text):.4f}".rstrip("0").rstrip(".")
        except (ValueError, TypeError):
            pass
    return text.replace("\t", " ")


def _format_plain_value(record: RunRecord, field: str, value) -> str:
    if record.status in ("failed", "partial", "incomplete"):
        if field == "error":
            return "----"
        if field != "status" and value == "":
            return "----"
    return _format_value(field, value)


def write_plain(records: list[RunRecord], out) -> None:
    fields = _select_fields(records)
    rows: list[list[str]] = []
    rows.append([_plain_header(f) for f in fields])
    for record in records:
        rows.append([_format_plain_value(record, f, _field_value(record, f))
                     for f in fields])

    widths = [max(len(row[i]) for row in rows) for i in range(len(fields))]
    for row in rows:
        pieces = []
        for i, value in enumerate(row):
            pieces.append(value)
            if i != len(row) - 1:
                padding = " " * (widths[i] - len(value) + 2)
                pieces.append(padding)
        print("".join(pieces), file=out)


def write_markdown(records: list[RunRecord], out) -> None:
    fields = _select_fields(records)
    print("<table>", file=out)
    print("  <thead>", file=out)
    print("    <tr>", file=out)
    for field in fields:
        print(f"      <th>{html.escape(_export_header(field))}</th>", file=out)
    print("    </tr>", file=out)
    print("  </thead>", file=out)
    print("  <tbody>", file=out)
    for record in records:
        print("    <tr>", file=out)
        for field in fields:
            value = _format_plain_value(record, field, _field_value(record, field))
            print(f"      <td>{html.escape(value)}</td>", file=out)
        print("    </tr>", file=out)
    print("  </tbody>", file=out)
    print("</table>", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse DartMoQ slurm logs without positional grep/awk alignment."
    )
    parser.add_argument("logs", nargs="+", help="slurm log files to parse")
    parser.add_argument(
        "--format", "-f", choices=("plain", "csv", "json", "md"), default="plain",
        help="output format. plain prints to stdout; csv/json/md writes to <logfile>.<format>",
    )
    parser.add_argument(
        "--complete-only", action="store_true",
        help="only emit rows whose status is ok",
    )
    args = parser.parse_args(argv)

    if args.format == "plain":
        records = parse_logs(args.logs)
        if args.complete_only:
            records = [record for record in records if record.status == "ok"]
        write_plain(records, sys.stdout)
        return 0

    for log_path in args.logs:
        records = parse_log(log_path)
        total_records = len(records)
        if args.complete_only:
            records = [record for record in records if record.status == "ok"]
        filtered_records = len(records)

        output_path = f"{log_path}.{args.format}"
        with open(output_path, "w", newline="", encoding="utf-8") as out:
            if args.format == "json":
                fields = _select_fields(records)
                json.dump([_format_export_row(record, fields)
                           for record in records], out, indent=2,
                          ensure_ascii=False)
                print(file=out)
            elif args.format == "md":
                write_markdown(records, out)
            else:
                write_csv(records, out)

        # Print output log
        print(f"[LOG] Parsed: {log_path}")
        print(f"[LOG] Output: {output_path}")
        print(f"[LOG] Total runs: {total_records}")
        if args.complete_only:
            print(f"[LOG] Filtered to {filtered_records} complete runs (status='ok')")
        else:
            status_counts = {}
            for r in records:
                status_counts[r.status] = status_counts.get(r.status, 0) + 1
            status_str = ", ".join([f"{k}: {v}" for k, v in sorted(status_counts.items())])
            print(f"[LOG] Status breakdown: {status_str}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
