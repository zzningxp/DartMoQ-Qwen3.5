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
#    2026-10-22：dart312 补装 fla-core/flash-linear-attention 0.5.2 +
#    causal-conv1d 1.7.0（本机 nvcc 12.8 自编 wheel，含 sm_120），GDN delta rule
#    与 causal conv 走库 kernel——warmup 日志第一行不再出现 "The fast path is
#    not available"。⚠ 之前所有官方基线数字（1018.thrput.* 系列）都是 attention
#    torch 慢路径口径，与装包后不可混表，需重跑。kernels hub 路径未启用
#    （use_kernels=True 没开），仍走「原包优先」，与 finegrained-fp8 互不冲突。
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
# 2026-10-23：官方 FP8 hub kernel（kernels-community/finegrained-fp8）惰性加载时
# kernels 包联网查 publisher 信任位，mirror 不稳会挂死/SSL 断；放行使校验短路
# （bench_throughput.load_bench_model 里有同款说明）
export DARTMOQ_TRUST_HUB_KERNELS=1
# 2026-10-23（晚）：trust 放行还不够——kernels 在线路径解析版本还要
# list_repo_refs 联网（kernels/_versions.py），mirror 断连直接崩进程
# （1019.thrput.4 bench+graph 双崩，httpx.RemoteProtocolError）。
# 已把 kernel 快照补成全量 35 文件，配合 HF_HUB_OFFLINE=1 走纯缓存：
# 版本解析读本地 refs/v4，信任校验自动跳过，全程零网络，mirror 抖动免疫。
export HF_HUB_OFFLINE=1

int4name="$HOME/models/Qwen3.5-35B-A3B-GPTQ-Int4/"
fp8name="$HOME/models/Qwen3.5-35B-A3B-FP8/"

# Int4（磁盘 23GB）：单卡装得下
export CUDA_VISIBLE_DEVICES=1
# python bench_throughput.py --model-path $int4name \
#         --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024

# FP8（磁盘 35GB，attn/conv 等保持 bf16）：单卡 31.4GB 装不下，双卡 device_map=auto
export CUDA_VISIBLE_DEVICES=0,1
# python bench_throughput.py --model-path $fp8name \
#         --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024

# ---- CUDA graph decode PoC（P9-5 官方 FP8，2026-10-23 分段路线）----
# 全步 capture 与双卡 device_map 结构性互斥（跨设备 .to() 判死 capture，探针
# test/test_graph_capture_crossdev.py 证实）；单卡又装不下（language+lm_head
# 33.26GiB > 32GB）→ --strategy segment：按层切两段各自捕获成图，跨设备
# 拷贝留图外执行，每步 host 调用 4614 → ~15。capture 前有 16 步手写 eager
# 正确性门（manual 层循环 vs wrapper 对拍，不过不捕）。
# 2026-10-24：FP8Experts eager 路径有 nonzero()+host 专家循环 capture 必死，
# 显式 --experts-impl grouped_mm（histc 静态路径，官方 cuda-graph 安全设计）。
python graph_decode.py --model-path $fp8name \
        --strategy segment --prompt-len 128 --steps 1024 \
        --experts-impl grouped_mm

# python graph_decode.py --model-path $fp8name \
#         --strategy segment --prompt-len 16,128 --steps 64,256,512,1024

# 结果 JSON 落 results/throughput_<model>_external(<method>-Nbit)_<githash>_<ts>.json
#          与 results/graph_decode_<model>_<mode>_<githash>_<ts>.json
