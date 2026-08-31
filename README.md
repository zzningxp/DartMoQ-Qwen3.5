# DartMoQ-Qwen3.5

面向 Qwen3.5-35B-A3B MoE 模型的混合精度量化推理框架，及所有依赖 Qwen3.5 混合注意力机制模型，如 Qwen3.6/3.7/3.8 等。
基于 [DartMoQ](https://github.com/zzningxp/DartMoQ) 算法，适配 Qwen3.5 MoE 的 grouped_gemm 权重格式，
目前支持在 RTX 5090 上的 triton kernel wxa16 的量化推理方法，wxa16 代表支持 w1a16/w2a16/w4a16/w8a16 的混合精度推理。wxa8 和 wxa4 的工程开发正在进行中。

## 核心特性

- **混合 bit 权重量化（WxA16）**：1/2/4 bit 混合分组量化，权重按 expert 独立分配 bit
- **Triton 融合 kernel**：MoE 全路径 Triton 实现，反量化 + 矩阵乘 + epilogue 融合
- **Group-First 布局**：权重按 group 连续存储，提升 L2 缓存命中率
- **Rotation Hoisting**：Hadamard 旋转提升到加载期，kernel 内零开销
- **Kernel 自动调优**：支持 BlockN / BLOCK 尺寸 / warps / stages 等参数的网格搜索，自适应匹配硬件最优配置
- **Linear Attention 深度优化**：chunk delta rule 全路径 Triton 融合
  - pad-skip、WY 闭式解、cast 缩减、wy_prep 融合、kernel 自动调优


## 性能

> Qwen3.5-35B-A3B, WxA16, bpw2, RTX 5090 (400w 功率下）, sequential eval

| 数据集 | FP16 | WxA16 | 加速比 | ppl |
|---|---|---|---|---|
| wikitext2（145 samples） | 90.05 s | **57.52 s** | **35.6% 更快** | 7.7944（持平） |
| c4（256 samples） | 114.7 s | **88.36 s** | **23.0% 更快** | 11.268（持平） |



> 端到端时间含约 20s 的 PPL 计算开销。MoE 层与 Linear Attention 的相对加速比见各模块说明。

## 快速开始

### 0. 环境

```bash
conda create -n dart312 python=3.12
conda activate dart312
pip install torch transformers datasets triton
```

### 1. 量化模型产生并存储任意 bpw 的量化模型

```
# 或先量化保存，再单独评测
python run_qwen35.py $model_path wikitext2 \
        --nsamples 64 --slices 4 --quant-scheme global-a8s8m2bpw \
        --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-cpu \
        --inference-quant-mode wxa16 --save-quantized ./quant_ckpt 

python eval_qwen35.py --load-quantized ./quant_ckpt

```

注意，这里的量化模式为 MoE 参数量化为混合精度（0、1、2、4），此处 bpw 未考虑 0.127 左右的量化参数。Attention 部分参数统一量化为 8bit。其他参数，lm_head 和多模态 clip 模型暂时未量化，依然保持 fp16 格式。
量化结束后会执行一次 eval 评测。
量化方法支持 turboquant 和 gptq
--standby-layer-cpu 可以保证小显卡也能逐层载入 GPU 完成量化。

2bpw + 0.127 模型量化完成后约为 14GB 磁盘占用。

### 2. FP16 baseline 评测

```bash
python eval_qwen35.py --inference-quant-mode wxa16 /path/to/qwen3.5/model
```

### 3. WxA16 量化 + 评测

```bash
# 量化 + 评测（一步完成）
python run_qwen35.py $model_path wikitext2 \
    --inference-quant-mode wxa16 \
    --nsamples 64 \
    --slices 4 \
    --quant-scheme global-a8s8m2bpw \
    --rank-mode turboquant_innerproduct \
    --quantmode turboquant \
    --standby-layer-cpu

```

### 4. 推理量化模式切换

```bash
# WxA16（默认，FP16 激活 + FP16 计算）
python eval_qwen35.py --load-quantized ./quant_ckpt --inference-quant-mode wxa16

# WxA8（INT8 激活 + INT8 Tensor Core，MoE 部分；attention 保持 W8A16 直到 P3）
python eval_qwen35.py --load-quantized ./quant_ckpt --inference-quant-mode wxa8
```

WxA8 与 WxA16 共用同一份 packed checkpoint（码本转 INT8 是加载期算的），
`--load-quantized` 加载后 `qwen35_quant_io.convert_model_to_wxa8` 原地切换，零拷贝。

## 路线图

- **WxA16** ✅ 已完成（FP16 激活 + FP16 Tensor Core + 混合 bit 权重量化）
  - **MoE 全路径 Triton 融合**：反量化、矩阵乘、epilogue 全部融合进单个 kernel，消除中间访存
  - **Group-First 权重布局**：权重按量化分组连续存储，提升 L2 缓存命中率与多 expert 并行效率
  - **Rotation Hoisting**：Hadamard 旋转提前到权重加载阶段，kernel 内零开销完成
  - **autotune 自适应调优**：BlockN 可搜索，根据硬件自动挑选最优配置
  - **Linear Attention 深度优化**：chunk delta rule 全路径 Triton 化，包括 pad-skip 跳过、WY 闭式解、cast 缩减、wy_prep 融合、kernel 自动调优等
  - 详细优化记录详见 [roadmaps/wxa16-optimization-backlog-260824.md](roadmaps/wxa16-optimization-backlog-260824.md)

- **WxA8** 🚧 进行中（INT8 激活 + INT8 Tensor Core + INT32 累加）
  - MoE 全路径已落地：融合 kernel（kernel 级 1.52x）+ rotate-quantize 融合
    （18.75x）+ 加载接线（`--inference-quant-mode wxa8`）
  - attention 均匀码本改造 + WxA8Linear 待做（P3）
  - 详见 [roadmaps/wxa8-plan-260829.md](roadmaps/wxa8-plan-260829.md)
- **WxA4** 🔮 规划中（WGMMA int4，需 Machete 库或 CUTLASS）
