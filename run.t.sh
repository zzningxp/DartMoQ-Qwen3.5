#!/bin/sh

# eval_tasks.py 下游任务评测（lm-eval 七任务，对齐 BitsMoE，详见 benchmarks/roadmap.md）

export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1   # 122B checkpoint 需双卡（40GB 自动两卡切分，bs 收敛 8）
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

modelname="$PWD/models/qwen3.5-2bpw-260930-s1/"
# modelname="$PWD/models/qwen3.5-112B-2bpw-260930-s1/"

# openbookqa 数据集本地没有：首次需去掉上面两个 OFFLINE 跑一次小样本拉数据
# （hellaswag/mathqa 只有 hub 快照，一并拉全后再离线跑）
# python eval_tasks.py $modelname --inference-quant-mode wxa8 --tasks openbookqa hellaswag mathqa --limit 2

# 冒烟：单任务小样本
# python eval_tasks.py $modelname --inference-quant-mode wxa8 --tasks arc_easy --limit 20 --num-fewshot 5

# 六个 loglikelihood 任务 + gsm8k（默认七任务，gsm8k 排最后，逐任务失败隔离）
# python eval_tasks.py $modelname --inference-quant-mode wxa8 --num-fewshot 5 --batch-size 8

# gsm8k 单独跑（本仓库首次走 .generate：量化 MoE 的 T=1 解码形状未验证过）
# python eval_tasks.py $modelname --inference-quant-mode wxa8 --tasks gsm8k --num-fewshot 5

# 其余模式（wxfp4 必须 dart312-t38：triton 3.8 + ptxas 12.8）
# python eval_tasks.py $modelname --inference-quant-mode wxa16 --num-fewshot 5 --batch-size 8
# python eval_tasks.py $modelname --inference-quant-mode wxfp8 --num-fewshot 5 --batch-size 8
# python eval_tasks.py $modelname --inference-quant-mode wxfp4 --num-fewshot 5 --batch-size 8

# fp16 基线（原始 bf16 ~70GB 装不进单卡）：
#   默认 --fp16-mode auto = device_map 双卡切分 + CPU offload（generate 原生可用，gsm8k 可跑）
#   --fp16-mode sequential = 旧的层驻 CPU 逐层搬路径（慢 ~12x，仅 loglikelihood，gsm8k 跳过）
# 六任务 sequential 数字已出（logs/1010.2.task.fp16.log），无需重跑；只剩 gsm8k：
# python eval_tasks.py $HOME/models/Qwen3.5-35B-A3B/ --tasks gsm8k --num-fewshot 5
# python eval_tasks.py $HOME/models/Qwen3.5-35B-A3B/ --fp16-mode sequential --num-fewshot 5 --batch-size 32
python eval_tasks.py $HOME/models/Qwen3.5-35B-A3B/ --tasks gsm8k --num-fewshot 5

# 结果 JSON 落 results/tasks_<ckpt>_<mode>_5shot_<githash>_<ts>.json
