#!/bin/sh

# profile_prefill.py decode 分层 profile 批跑（P9-5 判定：launch 数 / idle gap 实测）
# 背景（2026-10-22/23）：官方 FP8 重跑后 decode 仅 16.43 tok/s（61ms/step），
# 与我方 14 tok/s 同陷 eager 框架开销；本批把两侧的每步账拆开对照：
#   判读标准——launch 千级/步 + GPU busy 低位 → 开销主导（P9-5 CUDA Graph 成立）；
#   busy >80% 且 launch 仅数百 → 开销叙事不成立，回头查 kernel。
# 跨两个 conda env，脚本内用 conda run 自包含，无需手动 activate。
# 用法：sh run.profile.sh
#   或：nohup sh run.profile.sh > logs/1023.profile.decode.log 2>&1 &
# 结果 JSON 落 results/profile_prefill_<model>_<mode>_<githash>_<ts>.json

# ---- 1/2 官方 FP8（dart312，双卡 device_map；HF 镜像与离线同官方吞吐脚本）----
echo "[batch] 1/2: official FP8 decode profile (dart312, dual GPU)"
export HF_DATASETS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"
export CUDA_VISIBLE_DEVICES=0,1
# 官方 FP8 hub kernel 的 publisher 信任校验联网会挂死/SSL 断（1023 首跑/三跑），
# 放行使其短路（见 bench_throughput.load_bench_model 同名注释）；
# kernels 在线版解析还有 list_repo_refs 联网调用，mirror 断连会崩进程
# （1019.thrput.4 实踩）——kernel 快照已补全量，HF_HUB_OFFLINE=1 纯缓存零网络
export DARTMOQ_TRUST_HUB_KERNELS=1
export HF_HUB_OFFLINE=1

fp8name="$HOME/models/Qwen3.5-35B-A3B-FP8/"

conda run --no-capture-output -n dart312 python profile_prefill.py \
    --model-path "$fp8name" \
    --prefill-lens 128 --decode-steps 64

# ---- 2/2 我方 4bpw wxfp4 对照（dart312-t38 主流程，单卡）----
# turboquant 版把 gptq 换成 261016-turboquant 即可（注意该档 check FAIL 待查）
# echo "[batch] 2/2: our 4bpw wxfp4 decode profile (dart312-t38, single GPU)"
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
# export CUDA_VISIBLE_DEVICES=1

# quantname="$PWD/models/qwen3.5-35B-4bpw-261016-gptq/"

# conda run --no-capture-output -n dart312-t38 python profile_prefill.py \
#     --quant-dir "$quantname" --mode wxfp4 \
#     --prefill-lens 128 --decode-steps 64
