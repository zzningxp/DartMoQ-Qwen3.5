#!/bin/sh

# MODEL_PATH=""
#       https://huggingface.co/allenai/OLMoE-1B-7B-0924

# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=1
export HF_DATASETS_OFFLINE=1 
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

modelname="$HOME/models/Qwen3.5-35B-A3B/"
# modelname="$HOME/models/Qwen3.5-122B-A10B-W8A8"

# python run_qwen35.py $modelname wikitext2 --inference-quant-mode wxa8 \
#         --nsamples 64 --slices 1 --quant-scheme global-a8s8m2bpw \
#         --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-meta \
#         --save-quantized models/qwen3.5-112B-2bpw-260930-s1/

python run_qwen35.py $modelname wikitext2 --inference-quant-mode wxa8 \
        --nsamples 64 --slices 1 --quant-scheme global-a4s4m4bpw \
        --rank-mode gptq_quant_outlier --quantmode gptq --standby-layer-meta \
        --save-quantized models/qwen3.5-35B-4bpw-261016-gptq/

python run_qwen35.py $modelname wikitext2 --inference-quant-mode wxa8 \
        --nsamples 64 --slices 1 --quant-scheme global-a4s4m4bpw \
        --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-meta \
        --save-quantized models/qwen3.5-35B-4bpw-261016-turboquant/


# python run_qwen35.py $modelname wikitext2 --quant-layers 0 --wxa16 \
#         --nsamples 64 --slices 4 --quant-scheme global-a8s8m2bpw \
#         --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-cpu

# python run_qwen35.py $modelname wikitext2 --quant-layers 0-4 --wxa16 \
#         --nsamples 64 --slices 4 --quant-scheme global-a8s8m2bpw \
#         --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-cpu

# python run_qwen35.py $modelname wikitext2 --quant-layers 0-4\
#         --nsamples 64 --slices 4 --quant-scheme global-a8s8m2bpw --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-cpu

# python run_qwen35.py $modelname wikitext2\
#         --nsamples 64 --slices 4 --quant-scheme global-a8s8m2bpw --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-cpu

# git checkout a46355e18df516a02d70a4351ad13999dd8c9d02
# python run_qwen35.py $modelname wikitext2 --wxa16\
#         --nsamples 64 --slices 4 --quant-scheme global-a8s8m2bpw --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-cpu


# python run_qwen35.py $modelname wikitext2 --wxa16 \
#         --nsamples 64 --slices 4 --quant-scheme global-a8s8m2bpw --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-cpu

# python run_qwen35.py $modelname wikitext2 \
#         --nsamples 64 --slices 16 --quant-scheme global-a8s8m2bpw --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-cpu

# python run_qwen35.py $modelname wikitext2 \
#         --nsamples 64 --slices 8 --quant-scheme global-a8s8m2bpw --rank-mode gptq_quant_outlier --quantmode gptq --standby-layer-cpu
