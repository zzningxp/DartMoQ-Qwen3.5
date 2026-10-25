#!/bin/sh

# nsys 剖析 graph decode 一步 18.5ms 的构成（graph 拆解第一步，261025）
# 原理：图重放与 host 路径共用同一批 merged kernel（名字相同），kern_sum 按
# 名字聚合无法区分来源 → 必须用 cudaProfilerApi 时间窗只录 pass B 计时循环
# （DARTMOQ_PROF_WINDOW=1 时 graph_decode.py 在循环外圈 profiler.start/stop；
# nsys 在首个 stop 截止，只采第一个 plen/steps/rep 档）
# 输出：/tmp/graphprof_<mode>_4bpw.nsys-rep + kernel 聚合报告（stdout 打印前 45 行）
# 测两档：wxa8 = 当前最快 18.5ms（主拆解对象）
#         wxfp4 = 20.1ms（验证 fp4 池 BLOCK_N=128 大 tile SM 填充不足假设）
# 先激活环境：conda activate dart312-t38（wxfp4 必须：triton 3.8 + ptxas 12.8）

export CUDA_VISIBLE_DEVICES=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"
export DARTMOQ_PROF_WINDOW=1

modelname="$PWD/models/qwen3.5-35B-4bpw-261016-turboquant/"

for mode in wxa8 wxfp4; do
    # 步数 256 已够统计（256 步 × ~19ms ≈ 5s 图内 kernel 样本）；
    # --sample=none --cpuctxsw=none：不看 CPU 采样，降剖析开销
    nsys profile -o /tmp/graphprof_${mode}_4bpw \
        --force-overwrite true \
        --trace=cuda --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        python graph_decode.py \
        --quant-dir $modelname --mode $mode \
        --prompt-len 128 --steps 256

    # 时间窗内 kernel 聚合：名字 × 次数 × 均值/最大时长（按总时长排序）
    nsys stats --report cuda_gpu_kern_sum \
        /tmp/graphprof_${mode}_4bpw.nsys-rep | head -45
done

# 解读要点（对照 roadmaps/graph-decode-retrospective-261024.md §四）：
#  1. MoE 静态 GEMM（*merged* kernel）合计 vs attention/lm_head/杂项的占比
#  2. dispatch 链小 kernel（softmax/排序/填充/expert_info 等）的条数 × 单条
#     时长 → 验证 "~1000 图内节点 ≈ 2–4ms" 估计
#  3. wxa8 vs wxfp4 的 gate_up/down kernel 均值差 → 验证 fp4 大 tile
#     （BLOCK_N=128 → 活跃 8 expert 仅 64/128 块 vs wxa8 256/512，5090 170 SM）
#     SM 填充不足假设；确认后 fp4 池 decode 档 tile 重扫（128→32/64）
#  4. 窗内还含每步 feed 填充/argmax/DtoH 拷贝，即 loop_ms 与 replay_ms 的差
# 备用报告（按需开）：
#  nsys stats --report cuda_gpu_sum /tmp/graphprof_wxa8_4bpw.nsys-rep
#  nsys-ui /tmp/graphprof_wxa8_4bpw.nsys-rep   # 时间线逐 kernel 查看
