#!/bin/sh

# bench_throughput.py prefill/decode 吞吐基准（评测计划 P0）
# 对齐 4090 Qwen3.5-35B-A3B W4A16 公开数字（prefill 774.6 / decode 103.8 tok/s）
# 先激活环境：conda activate dart312-t38（wxfp4 必须此环境：triton 3.8 + ptxas 12.8）

# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# 外部 w4a8 Qwen3.5-35B（拷到 $HOME/models/ 下后改路径再开）
# python bench_throughput.py \
#     --model-path "$HOME/models/<w4a8目录>" \
#     --prefill-lens 16,128,2048,4096 --decode-tokens 256

# 我们的 2bpw checkpoint（四种 kernel 模式，checkpoint 同一份）
# modelname="$PWD/models/qwen3.5-2bpw-260930-s1/"
modelname="$PWD/models/qwen3.5-35B-4bpw-261016-gptq/"


python bench_throughput.py --quant-dir $modelname \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
python bench_throughput.py --quant-dir $modelname --mode wxa8 \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
python bench_throughput.py --quant-dir $modelname --mode wxfp8 \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
python bench_throughput.py --quant-dir $modelname --mode wxfp4 \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024

# 结果 JSON 落 results/throughput_<model>_<mode>_<githash>_<ts>.json

modelname="$PWD/models/qwen3.5-35B-4bpw-261016-turboquant/"

python bench_throughput.py --quant-dir $modelname \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
python bench_throughput.py --quant-dir $modelname --mode wxa8 \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
python bench_throughput.py --quant-dir $modelname --mode wxfp8 \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
python bench_throughput.py --quant-dir $modelname --mode wxfp4 \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024

# ---- CUDA graph decode PoC（P9-5，2026-10-23 官方侧不可行后转我方模型）----
# 官方 FP8 双卡 device_map 与 capture 结构性互斥（跨设备 .to() 判死 capture，
# 最小探针 test/test_graph_capture_crossdev.py 证实；单卡又装不下 31.4GB）
# → PoC 转我方单卡模型。判读同官方脚本：token MATCH 为硬判据；capture 失败
# 先换 --mask-mode buffer；DIFF 看首异步号。只想跑 PoC 的话把上面 bench 段
# 全注释（12 条），或直接单独跑这两行。
python graph_decode.py \
    --quant-dir "$PWD/models/qwen3.5-35B-4bpw-261016-gptq/" --mode wxfp4 \
    --prompt-len 16,128,2048,4096 --steps 64,256,512,1024

###

modelname="$PWD/models/qwen3.5-2bpw-260930-s1/"

python bench_throughput.py --quant-dir $modelname \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
python bench_throughput.py --quant-dir $modelname --mode wxa8 \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
python bench_throughput.py --quant-dir $modelname --mode wxfp8 \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
python bench_throughput.py --quant-dir $modelname --mode wxfp4 \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
