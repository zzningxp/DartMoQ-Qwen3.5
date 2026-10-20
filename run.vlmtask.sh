#!/bin/sh

# eval_vl_tasks.py 多模态评测（lmms-eval 四任务，量化 checkpoint 装回视觉塔）
# 数据集已 ModelScope 本地化（benchmarks/mm_datasets/），全程离线可跑

export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

modelname="$PWD/models/qwen3.5-2bpw-260930-s1/"
visualsrc="$HOME/models/Qwen3.5-35B-A3B/"

# 冒烟：单任务小样本（端到端链路先验证这个）
# python eval_vl_tasks.py $modelname --visual-source $visualsrc --inference-quant-mode wxa8 --tasks ocrbench --limit 20

# 四任务全量（textvqa_val/scienceqa_img/vizwiz_vqa_val/ocrbench；_lite 等价用 --limit）
# python eval_vl_tasks.py $modelname --visual-source $visualsrc --inference-quant-mode wxa8

# 其余模式（triton 3.8 已修复，wxfp4 可跑）
# python eval_vl_tasks.py $modelname --visual-source $visualsrc --inference-quant-mode wxa16
# python eval_vl_tasks.py $modelname --visual-source $visualsrc --inference-quant-mode wxfp8
# python eval_vl_tasks.py $modelname --visual-source $visualsrc --inference-quant-mode wxfp4

# fp16 基线（内置 qwen3_5 适配器，device_map auto 双卡 + CPU offload，慢）
# 注意：基线需要双卡，跑之前把上面的 CUDA_VISIBLE_DEVICES 改成 0,1
# python eval_vl_tasks.py $visualsrc --fp16 --tasks ocrbench --limit 20

# 结果 JSON 落 results/vl_tasks_<ckpt>_<mode>_<githash>_<ts>.json
python eval_vl_tasks.py $modelname --visual-source $visualsrc --inference-quant-mode wxa8 --tasks ocrbench --limit 20
