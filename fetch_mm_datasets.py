#!/usr/bin/env python3
"""
拉取 lmms-eval 多模态四任务的数据集到本地 benchmarks/mm_datasets/。

数据源：ModelScope（国内直连稳定，绕开 HF 网络不稳与 scienceqa/ocrbench 的
gated 限制）。下载后用本地任务 yaml（benchmarks/lmms_tasks/*_local.yaml）
指向本地目录，评测全程离线。

下载后处理：
  - dataset_infos.json 重命名为 .bak：ModelScope 镜像多为旧版 datasets
    写出（Value 缺 dtype），datasets 4.8.5 解析会报错；移走后 datasets
    自动从 parquet schema 推断 split 与 features（已实测通过）。

拉取清单（ModelScope repo -> 本地目录）:
  textvqa     lmms-lab/textvqa       -> benchmarks/mm_datasets/textvqa
  vizwiz      lmms-lab/VizWiz-VQA    -> benchmarks/mm_datasets/vizwiz
  scienceqa   swift/ScienceQA        -> benchmarks/mm_datasets/scienceqa_img
  ocrbench    evalscope/OCRBench     -> benchmarks/mm_datasets/ocrbench

用法:
  conda run -n dart312-t38 python fetch_mm_datasets.py
  conda run -n dart312-t38 python fetch_mm_datasets.py --datasets ocrbench
"""

import argparse
import os
import shutil
import sys

import pyarrow.parquet as pq

MM_DATASETS = {
    # 任务名 -> (ModelScope repo, 本地目录名)
    "textvqa": ("lmms-lab/textvqa", "textvqa"),
    "vizwiz": ("lmms-lab/VizWiz-VQA", "vizwiz"),
    "scienceqa": ("swift/ScienceQA", "scienceqa_img"),
    "ocrbench": ("evalscope/OCRBench", "ocrbench"),
}

MM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "benchmarks", "mm_datasets")


def fetch_one(name, ms_repo, local_dir, retries=3):
    from modelscope.hub.snapshot_download import snapshot_download

    for attempt in range(1, retries + 1):
        try:
            print(f"[{name}] downloading {ms_repo} ...", flush=True)
            snapshot_download(ms_repo, repo_type="dataset", local_dir=local_dir)
            _postprocess(name, local_dir)
            print(f"[OK] {name} -> {local_dir}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001  网络不稳定，统一重试
            print(f"[retry {attempt}/{retries}] {name}: {type(e).__name__}: {e}",
                  flush=True)
            if attempt < retries:
                import time
                time.sleep(10)
    print(f"[FAIL] {name} ({ms_repo}) after {retries} attempts", flush=True)
    return False


def _postprocess(name, local_dir):
    """旧格式 dataset_infos.json 改 .bak；打印 parquet 行数供列检查。"""
    infos = os.path.join(local_dir, "dataset_infos.json")
    if os.path.isfile(infos) and not os.path.isfile(infos + ".bak"):
        shutil.move(infos, infos + ".bak")
        print(f"  [{name}] dataset_infos.json -> .bak (old format, "
              f"auto-infer from parquet)")
    for f in sorted(os.listdir(os.path.join(local_dir, "data"))):
        if f.endswith(".parquet"):
            n = pq.read_metadata(
                os.path.join(local_dir, "data", f)).num_rows
            print(f"  [{name}] {f}: {n} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch mm-eval datasets from ModelScope into "
                    "benchmarks/mm_datasets/.")
    parser.add_argument("--datasets", nargs="+",
                        default=list(MM_DATASETS.keys()),
                        choices=list(MM_DATASETS.keys()),
                        help="datasets to fetch (default: all four)")
    args = parser.parse_args()
    os.makedirs(MM_DIR, exist_ok=True)

    results = {}
    for name in args.datasets:
        ms_repo, local_name = MM_DATASETS[name]
        results[name] = fetch_one(
            name, ms_repo, os.path.join(MM_DIR, local_name))

    failed = [k for k, v in results.items() if not v]
    print("\n===== Fetch summary =====")
    for k, v in results.items():
        print(f"  {k}: {'ok' if v else 'FAILED'}")
    if failed:
        print(f"Rerun for failed ones: "
              f"python {os.path.basename(__file__)} --datasets {' '.join(failed)}")
        sys.exit(1)
    print("All datasets ready in benchmarks/mm_datasets/ (offline eval OK).")


if __name__ == "__main__":
    main()
