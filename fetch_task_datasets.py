#!/usr/bin/env python3
"""
一次性拉取 lm-eval 七任务缺的 HF 数据集到本地缓存。

背景：日常跑评测用 HF_DATASETS_OFFLINE=1（本机 HF hub 访问会挂起）。
新增数据集需临时联网拉一次，之后继续离线跑。墙外网络不稳定，
脚本内置重试与逐数据集失败隔离，失败的单独重跑即可。

用法（在线模式，勿加 OFFLINE 环境变量）:
  conda run -n dart312-t38 python fetch_task_datasets.py
  conda run -n dart312-t38 python fetch_task_datasets.py --datasets openbookqa winogrande

拉取清单（dataset_path 与 lm-eval 任务 yaml 一致，全部 split 拉取）:
  openbookqa  allenai/openbookqa   （fewshot=train, eval=test）
  winogrande  allenai/winogrande   （fewshot=train, eval=test）
  mathqa      allenai/math_qa      （fewshot=train, eval=test）
  hellaswag   Rowan/hellaswag      （fewshot=train, eval=validation）
"""

import argparse
import os
import sys
import time

# 显式去掉 OFFLINE 变量（防止调用方 shell 已 export）
for var in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
    os.environ.pop(var, None)

# 默认走 hf-mirror（直连 huggingface.co 不通）；VPN 窗口内可外部覆盖：
#   HF_ENDPOINT=https://huggingface.co python fetch_task_datasets.py
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 网络探测请求 10s 超时进重试，避免单个请求挂起几分钟
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")

TASK_DATASETS = {
    "openbookqa": ("allenai/openbookqa", None),
    # dataset_name 对齐 lm-eval winogrande/default.yaml（多 config，必选）
    "winogrande": ("allenai/winogrande", "winogrande_xl"),
    "hellaswag": ("Rowan/hellaswag", None),
}

# mathqa 特殊处理：hub 上只有旧式 math_qa.py（datasets 4.x 已禁用 script），
# 从官方数据源下载 zip 在本地转 parquet（详见 fetch_mathqa_manual）
MATHQA_URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"
MATHQA_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "benchmarks", "mathqa_parquet")
MATHQA_YAML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "benchmarks", "lm_eval_tasks")


def fetch_mathqa_manual(retries=3, delay=10.0):
    """下载官方 MathQA.zip（7.3MB），转成本地 HF parquet 目录 + 自定义任务 yaml。"""
    import io
    import json
    import zipfile

    import pandas as pd
    import requests

    split_map = {"train": "train", "validation": "dev", "test": "test"}
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            print(f"Downloading {MATHQA_URL} ...", flush=True)
            r = requests.get(MATHQA_URL, timeout=60)
            r.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            names = zf.namelist()
            for split, fname in split_map.items():
                target = next((n for n in names if n.endswith(f"/{fname}.json")
                               or n == f"{fname}.json"), None)
                if target is None:
                    raise FileNotFoundError(
                        f"{fname}.json not in zip ({sorted(names)})")
                data = json.loads(zf.read(target))
                df = pd.DataFrame(data)
                out_split = os.path.join(MATHQA_OUT_DIR, split)
                os.makedirs(out_split, exist_ok=True)
                df.to_parquet(
                    os.path.join(out_split, "data-00000-of-00001.parquet"),
                    engine="pyarrow")
            print(f"[OK] mathqa: {split_map} parquet written to {MATHQA_OUT_DIR} "
                  f"[{time.time() - t0:.1f}s]", flush=True)
            _write_mathqa_yaml()
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[retry {attempt}/{retries}] mathqa: {type(e).__name__}: {e}",
                  flush=True)
            if attempt < retries:
                time.sleep(delay)
    print(f"[FAIL] mathqa after {retries} attempts", flush=True)
    return False


def _write_mathqa_yaml():
    """生成 lm-eval 自定义任务 yaml（dataset_path 指向本地 parquet 目录）。

    字段与 lm-eval 内置 mathqa.yaml 一致，doc_to_choice 复用同目录 utils.py
    （从内置任务复制，正则解析 options 字符串）。
    """
    os.makedirs(MATHQA_YAML_DIR, exist_ok=True)
    yaml_text = f"""# 由 fetch_task_datasets.py 自动生成：mathqa 数据集改走本地 parquet
# （hub 的 math_qa.py 旧式 script 已被 datasets 4.x 禁用）
task: mathqa
dataset_path: {MATHQA_OUT_DIR}
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
doc_to_text: "Question: {{{{Problem}}}}\\nAnswer:"
doc_to_target: "{{{{['a', 'b', 'c', 'd', 'e'].index(correct)}}}}"
doc_to_choice: !function utils.doc_to_choice
should_decontaminate: true
doc_to_decontamination_query: "Question: {{{{Problem}}}}\\nAnswer:"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""
    with open(os.path.join(MATHQA_YAML_DIR, "mathqa.yaml"), "w") as f:
        f.write(yaml_text)
    # utils.py 直接从 lm_eval 内置任务目录复制（与官方口径逐字符一致）
    import lm_eval
    src_utils = os.path.join(os.path.dirname(lm_eval.__file__),
                             "tasks", "mathqa", "utils.py")
    with open(src_utils, "r") as fsrc, \
            open(os.path.join(MATHQA_YAML_DIR, "utils.py"), "w") as fdst:
        fdst.write(fsrc.read())
    print(f"[OK] mathqa yaml written: {MATHQA_YAML_DIR}/mathqa.yaml")


def fetch_one(name, path, config=None, retries=3, delay=10.0):
    from datasets import load_dataset

    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            ds = load_dataset(path, config)
            info = ", ".join(f"{k}={len(v)}" for k, v in ds.items())
            print(f"[OK] {name} ({path}): {info} [{time.time() - t0:.1f}s]",
                  flush=True)
            return True
        except Exception as e:  # noqa: BLE001  网络不稳定，统一重试
            print(f"[retry {attempt}/{retries}] {name}: {type(e).__name__}: {e}",
                  flush=True)
            if attempt < retries:
                time.sleep(delay)
    print(f"[FAIL] {name} ({path}) after {retries} attempts", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Fetch missing lm-eval task datasets into the local HF cache.")
    parser.add_argument("--datasets", nargs="+",
                        default=["openbookqa", "winogrande", "mathqa", "hellaswag"],
                        choices=["openbookqa", "winogrande", "mathqa", "hellaswag"],
                        help="datasets to fetch (default: all four; mathqa goes "
                             "through the local parquet conversion path)")
    parser.add_argument("--endpoint", default="https://hf-mirror.com",
                        help="HF endpoint (default: hf-mirror; "
                             "VPN 窗口内可 --endpoint https://huggingface.co)")
    args = parser.parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint
    print(f"Using HF_ENDPOINT={args.endpoint}", flush=True)

    results = {}
    for name in args.datasets:
        if name == "mathqa":
            results[name] = fetch_mathqa_manual()
            continue
        path, config = TASK_DATASETS[name]
        results[name] = fetch_one(name, path, config)

    failed = [k for k, v in results.items() if not v]
    print("\n===== Fetch summary =====")
    for k, v in results.items():
        print(f"  {k}: {'ok' if v else 'FAILED'}")
    if failed:
        print(f"Rerun for failed ones: "
              f"python {os.path.basename(__file__)} --datasets {' '.join(failed)}")
        sys.exit(1)
    print("All datasets cached. Back to offline mode for eval runs.")


if __name__ == "__main__":
    main()
