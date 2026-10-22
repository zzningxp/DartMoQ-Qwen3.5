#!/bin/sh

# MODEL_PATH=""
#       https://huggingface.co/allenai/OLMoE-1B-7B-0924

export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1 
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# python eval_qwen35.py models/qwen3.5-2bpw-260824/ --datasets wikitext2 c4 --sequential-eval
# modelname="$HOME/models/Qwen3.5-35B-A3B/"

# python eval_qwen35.py $modelname --datasets wikitext2 --sequential-eval --standby-cpu
# python eval_qwen35.py $modelname --datasets c4 --sequential-eval --standby-cpu

# int8 (W8A8) 122B baseline：分 shard 大模型 + standby 自动触发流式评测
# （meta 结构 + 算一层读一层 + 用完即弃，RAM ~15GB，无需 compressed-tensors）
# modelname_int8="$HOME/models/Qwen3.5-122B-A10B-W8A8"
# python eval_qwen35.py $modelname_int8 --sequential-eval --standby-cpu

# modelname="$PWD/models/qwen3.5-2bpw-260831-u8/"
modelname="$PWD/models/qwen3.5-2bpw-260930-s1/"
# modelname="$PWD/models/qwen3.5-112B-2bpw-260930-s1/"
# P5-2 重做验证（2026-10-20）：4bpw checkpoint，wxa8 默认走合并路径；
# 逐 expert 对照用 DARTMOQ_MERGED_MOE=0 sh run.evalppl.sh
# modelname="$PWD/models/qwen3.5-35B-4bpw-261016-turboquant/"
# modelname="$PWD/models/qwen3.5-35B-4bpw-261016-gptq/"

python eval_qwen35.py $modelname --inference-quant-mode wxa16 --sequential-eval --standby-cpu
python eval_qwen35.py $modelname --inference-quant-mode wxa8 --sequential-eval --standby-cpu
python eval_qwen35.py $modelname --inference-quant-mode wxfp8 --sequential-eval --standby-cpu
# wxfp4 在 4bpw（全 bit4 池）上 OOM（WFF-5 驻留副本，见 wxfp4-fold-plan）
# python eval_qwen35.py $modelname --inference-quant-mode wxfp4 --sequential-eval --standby-cpu

