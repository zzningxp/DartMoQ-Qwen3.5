#!/usr/bin/env python3
"""
eval_vl_tasks.py — 多模态（VL）量化模型下游任务评测（lmms-eval harness）。

把量化 checkpoint 装回视觉塔（qwen35_vl_io.load_quantized_vl_model）后，用
lmms-eval 0.7.3 内置四任务评测（与学生 7B 多模态实验同款任务域）：

  - textvqa_val / textvqa_val_lite   TextVQA（VQA accuracy，exact match）
  - scienceqa_img                    ScienceQA-IMG（选项字母，exact match）
  - vizwiz_vqa_val / vizwiz_vqa_val_lite  VizWiz（VQA accuracy）
  - ocrbench                         OCRBench（记分规则内置）

DartMoQVL 继承 lmms-eval 内置 Qwen3_VL（Qwen3.5 家族适配器，generate_until /
chat template / 批处理 / batch_decode 全套现成），只覆写模型加载。fp16 基线
直接用内置 qwen3_5 适配器（--fp16，device_map=auto 双卡 + CPU offload）。

用法（主流程，占大量显存，请手动运行）：
  # 量化模型四任务 lite 子集
  python eval_vl_tasks.py models/qwen3.5-2bpw-260930-s1 \
      --visual-source ~/models/Qwen3.5-35B-A3B \
      --inference-quant-mode wxa8 \
      --tasks textvqa_val_lite vizwiz_vqa_val_lite scienceqa_img ocrbench --limit 100

  # fp16 基线（同口径对照）
  python eval_vl_tasks.py ~/models/Qwen3.5-35B-A3B --fp16 \
      --tasks ocrbench --limit 20

口径纪律（对齐 benchmarks/20261010 评测纪律）：
  - 结果 JSON 记录 git hash / 各库版本 / ckpt / mode / limit / pixels 配置；
  - 每任务记录 wall time（精度与速度成对出现）；
  - enable_thinking 默认关闭（量化与 fp16 两侧同口径）。
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

# lmms-eval 内置的四任务全量版；lite 版用于快速验证
# （沿用学生实验的子集纪律：先 lite/limit 子集，再全量）
DEFAULT_TASKS = ["textvqa_val", "scienceqa_img", "vizwiz_vqa_val", "ocrbench"]

# 本地数据集任务映射：内置任务名 -> (本地 yaml 文件名, 本地 task 名)。
# 数据源走 ModelScope 本地化（benchmarks/mm_datasets/，绕开 HF 网络与
# gated 仓库；fetch_mm_datasets.py 拉取）。_lite 版用 --limit 等价替代。
# yaml 与 <prefix>_utils.py 平铺在 benchmarks/lmms_tasks/（TaskManager
# include_path 只扫描一级目录）。
LMMS_TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "benchmarks", "lmms_tasks")
LOCAL_TASK_MAP = {
    "textvqa_val": ("textvqa_val_local.yaml", "textvqa_val_local"),
    "textvqa_val_lite": ("textvqa_val_local.yaml", "textvqa_val_local"),
    "scienceqa_img": ("scienceqa_img_local.yaml", "scienceqa_img_local"),
    "vizwiz_vqa_val": ("vizwiz_vqa_val_local.yaml", "vizwiz_vqa_val_local"),
    "vizwiz_vqa_val_lite": ("vizwiz_vqa_val_local.yaml", "vizwiz_vqa_val_local"),
    "ocrbench": ("ocrbench_local.yaml", "ocrbench_local"),
}


def _resolve_task_ref(task):
    """内置任务名 -> (simple_evaluate 用 ref, 结果 key)。

    TaskManager(include_path) 以 yaml 的 task 字段为索引 key，所以 ref 用
    本地任务名（xxx_local），配合 _build_task_manager() 的 include_path。
    """
    if task in LOCAL_TASK_MAP:
        fname, local_name = LOCAL_TASK_MAP[task]
        if os.path.isfile(os.path.join(LMMS_TASKS_DIR, fname)):
            return local_name, local_name
    return task, task


def _build_task_manager():
    """TaskManager（include_path 注册本地任务 yaml）。"""
    from lmms_eval.tasks import TaskManager
    return TaskManager(include_path=LMMS_TASKS_DIR)


class DartMoQVL:
    """DartMoQ 量化 VL 模型的 lmms-eval 适配器（延迟构造，避免顶层重依赖）。"""

    @staticmethod
    def build(quant_dir, visual_source, inference_quant_mode,
              device="cuda", batch_size=1, use_cache=True,
              min_pixels=256 * 28 * 28, max_pixels=1605632,
              total_pixels=None, max_num_frames=32, fps=None,
              system_prompt="You are a helpful assistant.",
              interleave_visuals=False, enable_thinking=False):
        # 重依赖延迟导入（对齐 eval_tasks.py 的顶层轻量纪律）
        import torch
        from accelerate import Accelerator
        from transformers import AutoProcessor, AutoTokenizer
        from lmms_eval.api.model import lmms
        from lmms_eval.models.simple.qwen3_vl import Qwen3_VL

        class _DartMoQVL(Qwen3_VL):
            """继承 Qwen3_VL 的 generate_until / chat template / 批处理，
            只覆写模型加载（量化 checkpoint + 视觉塔装回）。"""

            def __init__(self):
                # 复刻 Qwen3_VL.__init__ 的设备/属性骨架，模型加载换 DartMoQ
                lmms.__init__(self)
                accelerator = Accelerator()
                self.accelerator = accelerator
                self._device = torch.device(device)
                self.device_map = device  # 非 "auto"：generate_until 里 inputs 搬到 self.device

                from qwen35_vl_io import load_quantized_vl_model
                self._model = load_quantized_vl_model(
                    quant_dir=quant_dir,
                    visual_source=visual_source,
                    inference_quant_mode=inference_quant_mode)
                self._model.eval()

                self.min_pixels = min_pixels
                self.max_pixels = max_pixels
                self.total_pixels = total_pixels
                self.max_num_frames = max_num_frames
                self.fps = fps
                self.enable_thinking = enable_thinking

                self.processor = AutoProcessor.from_pretrained(
                    visual_source, max_pixels=max_pixels, min_pixels=min_pixels)
                self._tokenizer = AutoTokenizer.from_pretrained(visual_source)
                self.system_prompt = system_prompt
                self.interleave_visuals = interleave_visuals
                self.reasoning_prompt = None  # _preprocess_chunk 会访问

                self._config = self.model.config
                self._max_length = 2048
                self.batch_size_per_gpu = int(batch_size)
                self.use_cache = use_cache
                self._rank = 0
                self._world_size = 1

        return _DartMoQVL()


def _build_fp16_baseline(model_path, batch_size, enable_thinking=False):
    """fp16 基线：lmms-eval 内置 qwen3_5 适配器（双卡 auto + CPU offload）。"""
    from lmms_eval.models.simple.qwen3_5 import Qwen3_5
    return Qwen3_5(
        pretrained=model_path,
        device_map="auto",
        batch_size=batch_size,
        enable_thinking=enable_thinking)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="lmms-eval VL tasks on quantized Qwen3.5 checkpoints")
    parser.add_argument('model', type=str, nargs='?', default=None,
                        help="Quantized checkpoint dir; or fp16 model dir with --fp16")
    parser.add_argument('--fp16', action='store_true', default=False,
                        help="load fp16 baseline via built-in qwen3_5 adapter "
                             "(device_map auto, multi-GPU + CPU offload)")
    parser.add_argument('--visual-source', type=str, default=None,
                        help="original fp16 model dir with model.visual.* weights "
                             "+ preprocessor_config.json (required for quantized path)")
    parser.add_argument('--inference-quant-mode', type=str, default='wxa16',
                        choices=['wxa16', 'wxa8', 'wxfp8', 'wxfp4'],
                        help="inference quant mode (wxfp4 requires triton >= 3.8; "
                             "checked at runtime)")
    parser.add_argument('--tasks', type=str, nargs='+', default=DEFAULT_TASKS,
                        help=f"lmms-eval VL tasks (default: {DEFAULT_TASKS})")
    parser.add_argument('--limit', type=int, default=None,
                        help="limit docs per task (smoke runs)")
    parser.add_argument('--batch-size', type=int, default=1,
                        help="generate batch size (batched decode untested; keep 1)")
    parser.add_argument('--min-pixels', type=int, default=256 * 28 * 28,
                        help="processor min_pixels (Qwen3_VL default)")
    parser.add_argument('--max-pixels', type=int, default=1605632,
                        help="processor max_pixels (Qwen3_VL default)")
    parser.add_argument('--enable-thinking', action='store_true', default=False,
                        help="enable <think> mode in chat template (default off)")
    parser.add_argument('--system-prompt', type=str,
                        default="You are a helpful assistant.",
                        help="system prompt for chat template")
    parser.add_argument('--log-samples', action='store_true', default=False,
                        help="store per-sample outputs in result JSON")
    parser.add_argument('--output', type=str, default=None,
                        help="output JSON path (default: results/vl_tasks_<ckpt>_<mode>_...json)")
    parser.add_argument('--seed', type=int, default=0)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("No model given. Pass a quantized checkpoint dir, "
                         "or an fp16 base model dir with --fp16.")
    if args.fp16:
        if args.visual_source is not None:
            print("[warn] --fp16 ignores --visual-source (processor loaded from model dir)")
        print(f"Load fp16 baseline: {args.model} (built-in qwen3_5, device_map auto)", flush=True)
        lm = _build_fp16_baseline(args.model, args.batch_size,
                                  enable_thinking=args.enable_thinking)
        quant_dir = None
        mode_name = "fp16"
    else:
        if args.visual_source is None:
            raise SystemExit("--visual-source required for quantized path "
                             "(original fp16 model dir with visual weights)")
        if args.inference_quant_mode == "wxfp4":
            import triton
            if tuple(int(x) for x in triton.__version__.split(".")[:2]) < (3, 8):
                raise SystemExit(
                    f"wxfp4 requires triton >= 3.8 (current: {triton.__version__}); "
                    "use wxa8/wxfp8/wxa16 in this env")
        print(f"Load quantized VL model: {args.model} "
              f"(mode={args.inference_quant_mode}, visual={args.visual_source})", flush=True)
        lm = DartMoQVL.build(
            quant_dir=args.model,
            visual_source=args.visual_source,
            inference_quant_mode=args.inference_quant_mode,
            batch_size=args.batch_size,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            system_prompt=args.system_prompt,
            enable_thinking=args.enable_thinking)
        quant_dir = args.model
        mode_name = args.inference_quant_mode

        # kernel patch + Triton JIT 预热（复用 eval_tasks.py 的结构无关版本）
        from eval_tasks import _apply_fast_patches_and_warmup
        _apply_fast_patches_and_warmup(lm.model, args.batch_size, 2048)

    # ---- 逐任务评测（错误隔离：单任务失败不阻塞其余任务） ----
    import torch
    import lmms_eval
    import transformers
    from lmms_eval import evaluator
    from data_utils import get_git_hash

    ckpt_name = os.path.basename(os.path.normpath(args.model))
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.output or os.path.join(
        "results", f"vl_tasks_{ckpt_name}_{mode_name}_{get_git_hash()}_{ts}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    t_start = time.time()
    task_results = {}
    n_failed = 0
    for task in args.tasks:
        print(f"\n=== Evaluating {task} (limit={args.limit}) ===", flush=True)
        t0 = time.time()
        try:
            task_ref, result_key = _resolve_task_ref(task)
            if task_ref != task:
                print(f"  [{task}] using local dataset yaml: {task_ref}", flush=True)
            result = evaluator.simple_evaluate(
                model=lm,
                tasks=[task_ref],
                limit=args.limit,
                bootstrap_iters=0,
                log_samples=args.log_samples,
                random_seed=args.seed,
                task_manager=_build_task_manager())
            metrics = result["results"][result_key]
            task_results[task] = {
                "metrics": metrics,
                "wall_s": round(time.time() - t0, 1),
                "error": None,
            }
            if args.log_samples:
                task_results[task]["samples"] = result.get("samples", {}).get(task)
            print(f"--- {task} ---")
            for k, v in sorted(metrics.items()):
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v}")
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
        "lmms_eval_version": lmms_eval.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_path": os.path.abspath(args.model),
        "quant_dir": os.path.abspath(quant_dir) if quant_dir else None,
        "visual_source": os.path.abspath(args.visual_source) if args.visual_source else None,
        "inference_quant_mode": mode_name,
        "fp16_baseline": args.fp16,
        "enable_thinking": args.enable_thinking,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "tasks": args.tasks,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_s": round(time.time() - t_start, 1),
        "results": task_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n===== Summary ({ckpt_name}, {mode_name}) =====")
    for task, r in task_results.items():
        if r["error"]:
            print(f"  {task}: FAILED ({r['error']})")
        else:
            floats = {k: v for k, v in r["metrics"].items()
                      if isinstance(v, (int, float))}
            line = "  " + " ".join(f"{k}={v:.4f}" for k, v in sorted(floats.items()))
            print(line + f"  [{r['wall_s']}s]")
    print(f"Results saved to: {out_path}")
    if n_failed:
        print(f"{n_failed} task(s) failed; see JSON for tracebacks")
        sys.exit(1)


if __name__ == "__main__":
    main()
