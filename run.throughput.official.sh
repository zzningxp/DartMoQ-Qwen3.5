#!/bin/sh

# 官方发布 FP8 / GPTQ-Int4 基线吞吐（bench_throughput.py P0 同口径，用他们的 kernel）
# ⚠ 用 dart312 环境运行（torch 2.9.0+cu128 栈，已不在主流程使用）。
#    官方 kernel 包的最新版都按 torch ≥ 2.11 编译，在 torch 2.9 下必须钉版本
#    （pip --dry-run 看不出来，是运行时 import 才暴露；1018.thrput.2 实踩）：
#      gptqmodel 7.6/7.5/7.4  → w4a_nvfp4 用 torch.nn.functional.ScalingType
#                                （torch 2.11 API）import 即挂 → 钉 7.3.6
#      fbgemm-gpu 1.9/1.5/1.3 → .so 链接 torch 2.11 符号，加载失败 → 钉 1.4.0
#      torchao 0.18 同理 → 0.17.0（cpp 扩展跳过警告无碍：两条基线不用 torchao 算子）
#    另两个 transformers 5.20 的硬依赖（1018.thrput.2 第二轮实踩）：
#      optimum          → GPTQ 加载经 optimum 胶水层（内部再调 gptqmodel）
#      kernels==0.17.2  → transformers finegrained-fp8 kernel（FP8 块量化路径），
#                         从 hub 镜像按需拉取（脚本已设 HF_ENDPOINT）
#    gptqmodel 7.3.6 的 cpp 扩展同样跳过，sm_120 走 tritonv2 后端（triton kernel）。
#    ⚠ 绝不能装进 dart312-t38（triton 3.8 + ptxas 12.8 是 sm_120a mxfp4 配方）。
#    conda activate dart312
#    pip install "gptqmodel==7.3.6" "fbgemm-gpu==1.4.0" "torchao==0.17.0" \
#                optimum "kernels==0.17.2"   # 一次性
# 官方结构是 ConditionalGeneration（含 visual/mtp，权重带 language_model 前缀），
# bench 已自动用 AutoModelForImageTextToText 加载；加载时由 transformers 按
# quantization_config 调用官方后端 kernel：FP8 → fbgemm-gpu，GPTQ-Int4 → gptqmodel。

# 本机 HF hub 直连挂起：kernels 包拉 finegrained-fp8 内核必须走镜像端点；
# 且 kernels 的离线快照校验不认「只下了选中 build」的部分缓存
# （第三轮实踩：HF_HUB_OFFLINE=1 + 预热缓存仍报 no local snapshot），
# 故本脚本走镜像在线，datasets 仍离线（缓存已有 wiki/c4）。
export HF_DATASETS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

int4name="$HOME/models/Qwen3.5-35B-A3B-GPTQ-Int4/"
fp8name="$HOME/models/Qwen3.5-35B-A3B-FP8/"

# Int4（磁盘 23GB）：单卡装得下
export CUDA_VISIBLE_DEVICES=0
python benchmarks/bench_throughput.py --model-path $int4name \
        --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024

# FP8（磁盘 35GB，attn/conv 等保持 bf16）：单卡 31.4GB 装不下，双卡 device_map=auto
# export CUDA_VISIBLE_DEVICES=0,1
# python benchmarks/bench_throughput.py --model-path $fp8name \
#         --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024

# 结果 JSON 落 results/throughput_<model>_external(<method>-Nbit)_<githash>_<ts>.json
