#!/usr/bin/env python3
"""
Parse eval_tasks.py run logs / result JSONs into aligned task-eval benchmark rows.

One row per eval_tasks.py run. Primary data source is the result JSON written by
eval_tasks.py (located via the "Results saved to:" line in the log, or passed
directly); when a run died before writing its JSON, the parser falls back to
parsing the log text ("--- <task> ---" metric lines and "<task>: FAILED" lines).

Reuses helpers from logs_parser (same repo conventions for plain/csv/json/md
output and run splitting).

用法:
  python task_logs_parser.py logs/1010.0.task.log                  # plain table
  python task_logs_parser.py --format csv logs/1010.0.task.log     # -> <log>.csv
  python task_logs_parser.py results/tasks_*.json                  # direct JSON
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

# 复用 logs_parser 的通用工具（同一仓库，避免重复实现）
from logs_parser import _clean_line, _csv_escape, _format_value, _last_path_component

# BitsMoE 对齐的七任务展示顺序；未列出的任务按日志出现顺序追加
TASK_ORDER = [
    "hellaswag", "arc_easy", "mathqa", "mmlu",
    "openbookqa", "winogrande", "gsm8k",
]

RUN_START_RE = re.compile(
    r"^Load quantized checkpoint:\s+(?P<path>\S+)\s+\(mode=(?P<mode>\w+)\)")
# fp16 基线路径（sequential 层循环）：加载行格式不同
RUN_START_FP16_RE = re.compile(
    r"^Load fp16 baseline model:\s+(?P<path>\S+)\s+\(sequential layer-loop\)")
SAVED_TO_RE = re.compile(r"^Results saved to:\s+(?P<path>\S+)")
EVAL_TASK_RE = re.compile(
    r"^=== Evaluating (?P<task>\S+) \(num_fewshot=(?P<shot>\d+), "
    r"limit=(?P<limit>\S+)\) ===")
TASK_HEADER_RE = re.compile(r"^--- (?P<task>\S+) ---$")
# 注意：_clean_line 会 strip 掉行首空白，故这里用 \s*（真实日志行有缩进）
METRIC_LINE_RE = re.compile(r"^\s*(?P<name>\S+):\s+(?P<value>[-+0-9.eE]+)$")
FAILED_RE = re.compile(r"^\[error\] task (?P<task>\S+) failed:\s*(?P<err>.*)$")
SKIP_RE = re.compile(r"^\[skip\] (?P<task>\S+)\s")
SUMMARY_HDR_RE = re.compile(
    r"^===== Summary \((?P<ckpt>\S+), (?P<mode>\w+), (?P<shot>\d+)-shot\) =====")
SUMMARY_METRIC_RE = re.compile(
    r"^\s*(?P<metrics>.+?)\s+\[(?P<wall>[-+0-9.eE]+)s\]$")
STDERR_SUFFIX_RE = re.compile(r"^(?P<base>.*?)_stderr(?:,|$)")
NUMERIC_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")


@dataclass
class TaskResult:
    metric: str = ""       # 主指标名（acc_norm,none / acc,none / exact_match,strict-match）
    value: float | None = None
    stderr: float | None = None
    wall_s: float = 0.0
    error: str = ""
    skipped: str = ""      # 主动跳过（如 sequential 模式的 gsm8k），非失败


@dataclass
class TaskRun:
    source: str = ""
    ckpt: str = ""
    mode: str = ""
    num_fewshot: int = 0
    limit: str = ""
    git_hash: str = ""
    lm_eval_version: str = ""
    total_wall_s: float = 0.0
    json_path: str = ""
    tasks: dict[str, TaskResult] = field(default_factory=dict)
    _task_order: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        done = [t for t in self.tasks.values()
                if t.error == "" and not t.skipped and t.value is not None]
        failed = [t for t in self.tasks.values() if t.error]
        if not self.tasks or (not done and not failed):
            return "incomplete"  # 空 run 或所有任务都还没出分
        if failed and not done:
            return "failed"
        if failed:
            return "partial"
        return "ok"

    @property
    def mean_valid(self) -> str:
        """有效任务均值 + 覆盖数，如 '0.837(4/7)'；跳过任务不计分母；无有效任务为空。"""
        vals = [t.value for t in self.tasks.values()
                if t.error == "" and not t.skipped and t.value is not None]
        if not vals:
            return ""
        n_scored = len([t for t in self.tasks.values()
                        if not t.skipped])
        return f"{sum(vals) / len(vals):.4f}({len(vals)}/{n_scored})"


def _pick_primary_metric(metrics: dict) -> tuple[str, float | None, float | None]:
    """从 lm-eval 结果 dict 里挑主指标：em(strict) > acc_norm > acc > 首个浮点。"""
    def _stderr_of(base: str) -> float | None:
        for k, v in metrics.items():
            if k.startswith(base + "_stderr") and isinstance(v, float):
                return v
        return None

    if "exact_match,strict-match" in metrics:
        return ("exact_match,strict-match", metrics["exact_match,strict-match"],
                _stderr_of("exact_match,strict-match"))
    if "acc_norm,none" in metrics:
        return ("acc_norm,none", metrics["acc_norm,none"], _stderr_of("acc_norm"))
    if "acc,none" in metrics:
        return ("acc,none", metrics["acc,none"], _stderr_of("acc"))
    for k in sorted(metrics):
        v = metrics[k]
        if isinstance(v, float) and not isinstance(v, bool) and "stderr" not in k:
            return (k, v, _stderr_of(k))
    return ("", None, None)


def parse_result_json(path: str) -> TaskRun:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    run = TaskRun(source=os.path.basename(path), json_path=path)
    run.ckpt = _last_path_component(d.get("quant_dir", ""))
    run.mode = d.get("inference_quant_mode", "")
    run.num_fewshot = d.get("num_fewshot", 0)
    run.limit = str(d.get("limit") or "")
    run.git_hash = d.get("git_hash", "")
    run.lm_eval_version = d.get("lm_eval_version", "")
    run.total_wall_s = float(d.get("total_wall_s", 0.0))
    for task, r in d.get("results", {}).items():
        tr = TaskResult(wall_s=float(r.get("wall_s", 0.0)))
        if r.get("error"):
            tr.error = r["error"]
            # 兼容旧格式：error 以 "skipped:" 开头视为主动跳过
            if r["error"].startswith("skipped:"):
                tr.error = ""
                tr.skipped = r["error"][len("skipped:"):]
        elif r.get("skipped"):
            tr.skipped = r["skipped"]
        else:
            metric, value, stderr = _pick_primary_metric(r.get("metrics", {}))
            tr.metric, tr.value, tr.stderr = metric, value, stderr
        run.tasks[task] = tr
        run._task_order.append(task)
    return run


def _parse_log_fallback(lines: list[str], start: int, end: int) -> TaskRun:
    """JSON 缺失时的日志文本回退解析（run 死于写 JSON 之前）。"""
    run = TaskRun()
    task = None
    for line in lines[start:end]:
        line = _clean_line(line)
        if not line:
            continue
        m = RUN_START_RE.match(line)
        if m:
            run.ckpt = _last_path_component(m.group("path"))
            run.mode = m.group("mode")
            continue
        m = RUN_START_FP16_RE.match(line)
        if m:
            run.ckpt = _last_path_component(m.group("path"))
            run.mode = "fp16"
            continue
        m = SUMMARY_HDR_RE.match(line)
        if m:
            run.ckpt = run.ckpt or m.group("ckpt")
            run.mode = run.mode or m.group("mode")
            run.num_fewshot = run.num_fewshot or int(m.group("shot"))
            continue
        m = EVAL_TASK_RE.match(line)
        if m:
            task = m.group("task")
            run.num_fewshot = run.num_fewshot or int(m.group("shot"))
            run.limit = run.limit or m.group("limit")
            if task not in run.tasks:
                run.tasks[task] = TaskResult()
                run._task_order.append(task)
            continue
        m = TASK_HEADER_RE.match(line)
        if m:
            task = m.group("task")
            if task not in run.tasks:
                run.tasks[task] = TaskResult()
                run._task_order.append(task)
            continue
        m = FAILED_RE.match(line)
        if m and m.group("task") in run.tasks:
            run.tasks[m.group("task")].error = m.group("err")
            continue
        m = SKIP_RE.match(line)
        if m and m.group("task") in run.tasks:
            run.tasks[m.group("task")].skipped = "skipped in log"
            continue
        m = METRIC_LINE_RE.match(line)
        if m and task and task in run.tasks:
            tr = run.tasks[task]
            name, value = m.group("name"), float(m.group("value"))
            if "stderr" in name:
                tr.stderr = value
            else:
                tr.value = value
                tr.metric = name
            continue
        m = SUMMARY_METRIC_RE.match(line)
        if m:
            # Summary 行无任务名，按 "=== Evaluating" 出现顺序映射
            try:
                wall = float(m.group("wall"))
            except ValueError:
                continue
            candidates = [t for t in run._task_order
                          if run.tasks[t].wall_s == 0.0 and not run.tasks[t].error]
            if candidates:
                run.tasks[candidates[0]].wall_s = wall
            continue
    return run


def parse_log(path: str) -> list[TaskRun]:
    """一个 log 文件可含多个 run（smoke + 全量），逐 run 切分。

    每个 run 优先读其 "Results saved to:" 指向的 JSON；JSON 缺失/损坏时
    回退解析该 run 行区间内的日志文本。
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    spans = []  # (start_line_idx, json_path_or_None)
    cur_start = None
    cur_json = None
    for i, raw in enumerate(lines):
        line = _clean_line(raw)
        if RUN_START_RE.match(line) or RUN_START_FP16_RE.match(line):
            if cur_start is not None:
                spans.append((cur_start, cur_json))
            cur_start, cur_json = i, None
            continue
        if cur_start is not None:
            m = SAVED_TO_RE.match(line)
            if m:
                cur_json = m.group("path")
    if cur_start is not None:
        spans.append((cur_start, cur_json))

    runs = []
    for si, (start, json_path) in enumerate(spans):
        end = spans[si + 1][0] if si + 1 < len(spans) else len(lines)
        if json_path and os.path.isfile(json_path):
            try:
                run = parse_result_json(json_path)
                run.source = os.path.basename(path)
                runs.append(run)
                continue
            except (json.JSONDecodeError, KeyError, OSError) as e:
                print(f"[warn] {json_path} unreadable ({e}); "
                      f"falling back to log text", file=sys.stderr)
        run = _parse_log_fallback(lines, start, end)
        run.source = os.path.basename(path)
        runs.append(run)
    return runs


def parse_inputs(paths: list[str]) -> list[TaskRun]:
    runs = []
    for p in paths:
        if p.endswith(".json"):
            runs.append(parse_result_json(p))
        else:
            runs.extend(parse_log(p))
    return runs


# ---------------- output ----------------

def _task_columns(runs: list[TaskRun]) -> list[str]:
    seen = []
    for order in TASK_ORDER, :
        for t in order:
            if any(t in r.tasks for r in runs) and t not in seen:
                seen.append(t)
    for r in runs:
        for t in r._task_order:
            if t not in seen:
                seen.append(t)
    return seen


def _task_cell(run: TaskRun, task: str) -> str:
    tr = run.tasks.get(task)
    if tr is None:
        return "----"
    if tr.skipped:
        return "SKIP"
    if tr.error:
        return "FAIL"
    if tr.value is None:
        return "----"
    return f"{tr.value:.4f}"


def _row_dict(run: TaskRun, task_cols: list[str]) -> dict:
    row = {
        "model": run.ckpt,
        "mode": run.mode,
        "shot": str(run.num_fewshot) if run.num_fewshot else "",
        "limit": run.limit,
        "git": run.git_hash,
        "lm_eval": run.lm_eval_version,
        "wall_s": f"{run.total_wall_s:.1f}" if run.total_wall_s else "",
        "mean": run.mean_valid,
        "status": run.status,
    }
    for t in task_cols:
        tr = run.tasks.get(t)
        row[t] = {
            "value": tr.value if tr else None,
            "metric": tr.metric if tr else "",
            "stderr": tr.stderr if tr else None,
            "wall_s": f"{tr.wall_s:.1f}" if tr and tr.wall_s else "",
            "error": tr.error if tr else "",
            "skipped": tr.skipped if tr else "",
        }
    return row


PLAIN_HEADERS = ["model", "mode", "shot", "limit", "git", "lm_eval",
                 "wall_s", "mean", "status"]


def write_plain(runs: list[TaskRun], out) -> None:
    task_cols = _task_columns(runs)
    headers = PLAIN_HEADERS + task_cols
    rows = [headers]
    for run in runs:
        rd = _row_dict(run, task_cols)
        row = []
        for h in PLAIN_HEADERS:
            row.append(_format_value(h, rd[h]))
        for t in task_cols:
            tr = run.tasks.get(t)
            if tr is None:
                row.append("----")
            elif tr.skipped:
                row.append("SKIP")
            elif tr.error:
                row.append("FAIL")
            elif tr.value is None:
                row.append("----")
            else:
                row.append(f"{tr.value:.4f}")
        rows.append(row)

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    for row in rows:
        pieces = []
        for i, value in enumerate(row):
            pieces.append(value)
            if i != len(row) - 1:
                pieces.append(" " * (widths[i] - len(value) + 2))
        print("".join(pieces), file=out)


def write_csv(runs: list[TaskRun], out) -> None:
    task_cols = _task_columns(runs)
    headers = PLAIN_HEADERS
    for t in task_cols:
        headers += [f"{t}_value", f"{t}_metric", f"{t}_stderr",
                    f"{t}_wall_s", f"{t}_error", f"{t}_skipped"]
    out.write(",".join(_csv_escape(h) for h in headers) + "\n")
    for run in runs:
        rd = _row_dict(run, task_cols)
        cells = [_csv_escape(rd[h]) for h in PLAIN_HEADERS]
        for t in task_cols:
            d = rd[t]
            cells += [
                _csv_escape(str(d["value"]) if d["value"] is not None else ""),
                _csv_escape(d["metric"]),
                _csv_escape(str(d["stderr"]) if d["stderr"] is not None else ""),
                _csv_escape(d["wall_s"]),
                _csv_escape(d["error"]),
                _csv_escape(d["skipped"]),
            ]
        out.write(",".join(cells) + "\n")


def write_markdown(runs: list[TaskRun], out) -> None:
    task_cols = _task_columns(runs)
    headers = PLAIN_HEADERS + task_cols
    print("| " + " | ".join(headers) + " |", file=out)
    print("|" + "|".join(["---"] * len(headers)) + "|", file=out)
    for run in runs:
        rd = _row_dict(run, task_cols)
        cells = [_format_value(h, rd[h]) for h in PLAIN_HEADERS]
        cells += [_task_cell(run, t) for t in task_cols]
        print("| " + " | ".join(cells) + " |", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse eval_tasks.py run logs / result JSONs into task-eval rows.")
    parser.add_argument("inputs", nargs="+",
                        help="eval_tasks.py log files and/or results/tasks_*.json")
    parser.add_argument("--format", "-f", choices=("plain", "csv", "json", "md"),
                        default="plain",
                        help="plain prints to stdout; csv/json/md writes to <input>.<format>")
    parser.add_argument("--complete-only", action="store_true",
                        help="only emit runs whose status is ok")
    args = parser.parse_args(argv)

    if args.format == "plain":
        runs = parse_inputs(args.inputs)
        if args.complete_only:
            runs = [r for r in runs if r.status == "ok"]
        write_plain(runs, sys.stdout)
        return 0

    for path in args.inputs:
        runs = parse_inputs([path])
        total = len(runs)
        if args.complete_only:
            runs = [r for r in runs if r.status == "ok"]
        out_path = f"{path}.{args.format}"
        with open(out_path, "w", newline="", encoding="utf-8") as out:
            if args.format == "json":
                json.dump([_row_dict(r, _task_columns(runs)) for r in runs],
                          out, indent=2, ensure_ascii=False)
                print(file=out)
            elif args.format == "md":
                write_markdown(runs, out)
            else:
                write_csv(runs, out)
        print(f"[LOG] Parsed: {path}")
        print(f"[LOG] Output: {out_path}")
        print(f"[LOG] Total runs: {total}")
        status_counts = {}
        for r in runs:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
        print(f"[LOG] Status breakdown: "
              + ", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items())))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
