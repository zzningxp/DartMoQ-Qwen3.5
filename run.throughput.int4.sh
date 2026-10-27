#!/bin/sh

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 DARTMOQ_TRUST_HUB_KERNELS=1
export CUDA_VISIBLE_DEVICES=1
cd ~/Develop/DartMoQ-Qwen3.5

# 主候选：marlin
DARTMOQ_EXT_GPTQ_BACKEND=gptq_marlin python bench_throughput.py \
    --model-path $HOME/models/Qwen3.5-35B-A3B-GPTQ-Int4/ \
    --prefill-lens 128,2048 --decode-tokens 256,1024

# 对照：exllama_v2（首次运行现场编译几分钟）
DARTMOQ_EXT_GPTQ_BACKEND=gptq_exllama_v2 python bench_throughput.py \
    --model-path $HOME/models/Qwen3.5-35B-A3B-GPTQ-Int4/ \
    --prefill-lens 128,2048 --decode-tokens 256,1024

# 基线重跑（默认 gptq_triton，与历史口径同款，便于同日对比）
python bench_throughput.py --model-path $HOME/models/Qwen3.5-35B-A3B-GPTQ-Int4/ \
    --prefill-lens 128,2048 --decode-tokens 256,1024

