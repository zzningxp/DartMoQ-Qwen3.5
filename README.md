# DartMoQ-Qwen3.5

面向 Qwen3.5-35B-A3B MoE 模型的混合精度量化推理框架，及所有依赖 Qwen3.5 混合注意力机制模型，如 Qwen3.6/3.7/3.8 等。
基于 [DartMoQ](https://github.com/zzningxp/DartMoQ) 算法，适配 Qwen3.5 MoE 的 grouped_gemm 权重格式，
Triton kernel 量化推理支持三种精度模式（共用同一份 packed checkpoint）：

- **WxA16** ✅：FP16 激活张量 + FP16 Tensor Core，支持 w1a16/w2a16/w4a16/w8a16 混合精度权重
- **WxA8** ✅：INT8 激活张量 + INT8 Tensor Core（MoE + attention 全路径）
- **WxFP8** ✅：e4m3 激活 + FP8 Tensor Core（MoE 路径；attention 混合部署保 WxA8。
  定位是 WxFP4 的基础设施预备步骤）
- **WxFP4** ✅ 保守版落地（bit1/4 expert：e2m1 激活 + FP4 MX MMA；bit2 主力与
  attention 维持 WxA8。⚠ 需 dart312-t38 环境——triton ≥3.8 + ptxas 12.8，
  见路线图）。Triton 原生 `mma.kind::mxf4nvf4`，dense 492 TF（1.39× fp8）；
  另含面向普通 MoE 的专家级纯 MX GEMM 路径


## 核心特性

- **混合 bit 权重量化**：1/2/4 bit 混合分组量化，权重按 expert 独立分配 bit
- **Triton 融合 kernel**：MoE 全路径 Triton 实现，反量化 + 矩阵乘 + epilogue 融合
- **WxA8 INT8 激活推理**：per-token per-group 对称量化 + INT8 Tensor Core（IMMA、INT32 累加），
  码本转 INT8 在加载期完成，checkpoint 格式不变
- **WxFP8 e4m3 激活推理**：per-token per-group e4m3 量化（satfinite 饱和）+ FP8 Tensor Core
  原生 fp32 累加；码本 e4m3 LUT 与 cb_scale 网格搜索在加载期完成，checkpoint 格式不变
- **WxFP4 e2m1 激活推理**：bit1/4 expert 走 e2m1 + per-32 e8m0（MX 原生格式，
  `mma.kind::mxf4nvf4` 块缩放 MMA）；码本→nibble LUT 连续 scale 搜索
  （1-bit 精确落格）+ residual 折 norms，全部加载期完成，checkpoint 格式不变
- **Rotate+Quantize 融合**：分组旋转与激活量化融合为单个 Triton kernel，
  中间结果不落地（hoist 规模实测 18.75x vs 两段式）
- **Group-First 布局**：权重按 group 连续存储，提升 L2 缓存命中率
- **Rotation Hoisting**：分组 QR 旋转提升到 expert 循环外（全 token 预旋转一次，
  per-expert 零旋转开销），WxA8 下旋转与量化一并融合
- **Kernel 自动调优**：BlockN / BLOCK 尺寸 / warps / stages 网格搜索，自适应硬件最优配置
  （WxA16 与 WxA8 各有独立的调优表）
- **Linear Attention 深度优化**：chunk delta rule 全路径 Triton 融合
  - pad-skip、WY 闭式解、cast 缩减、wy_prep 融合、kernel 自动调优


## 性能

> Qwen3.5-35B-A3B, bpw2, RTX 5090, sequential eval, HEAD 6eabe23

| 数据集 | WxA16 | **WxA8** | 加速比 | ppl |
|---|---|---|---|---|
| wikitext2（256 samples） | 53.12 s | **45.08 s** | **-15.1%** | 7.7947（-0.0008，略好于 A16） |
| c4（145 samples） | 76.71 s | **61.84 s** | **-19.4%** | 11.2676（+0.0045） |



MoE 单独贡献 -7.0%/-6.7%，attention 路径贡献 -13.2%/-16.8%。
WxA8 全路径数字基于 260831-u8 checkpoint（attention 均匀码本，MoE 与 260824 同型）。

> WxFP8（混合部署：MoE fp8 + attention wxa8）同 checkpoint 另批实测（git 870e1f2+）：
> c4 62.93s（vs WxA8 +0.8%，速度中性）；wiki 55.43s 疑首轮 JIT 污染（待热缓存复跑确认）；
> ppl wiki 7.8064 / c4 11.2769（vs WxA16 +0.011/+0.014，代价远小于预期）。
> 详见 [roadmaps/wxfp8-plan-260924.md](roadmaps/wxfp8-plan-260924.md) §七 WF-5。

> **WxFP4 保守版**（bit1/4 expert fp4 + bit2/attention wxa8，t38 env，git 966e25a+）：
> ppl wiki 7.7954 / c4 11.2711——**与 WxA8 持平（+0.0007/+0.0035）且全面优于 WxFP8**；
> c4 59.37s（**比 WxA8 快 4.9%**）；wiki 60.1s 疑首轮 JIT 污染待复跑。
> 详见 [roadmaps/wxfp4-plan-260924.md](roadmaps/wxfp4-plan-260924.md) WF4-4。

> 端到端时间含约 20s 的 PPL 计算开销。MoE 层与 Linear Attention 的相对加速比见各模块说明。
> ⚠ 首次运行 WxA8 会触发 Triton JIT 编译（逐 expert 形状边跑边编译，端到端可虚增 10%+），
> 测速前先用任意一次 wxa8 eval 捂热磁盘缓存。

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
# --save-quantized 自动启用真量化路径（packed 落盘），无需再加 --wxa16
python run_qwen35.py $model_path wikitext2 \
        --nsamples 64 --slices 4 --quant-scheme global-a8s8m2bpw \
        --rank-mode turboquant_innerproduct --quantmode turboquant --standby-layer-cpu \
        --save-quantized ./quant_ckpt 

python eval_qwen35.py --load-quantized ./quant_ckpt

```

注意，这里的量化模式为 MoE 参数量化为混合精度（0、1、2、4），此处 bpw 未考虑 0.127 左右的量化参数。Attention 部分参数统一量化为 8bit。其他参数，lm_head 和多模态 clip 模型暂时未量化，依然保持 fp16 格式。
量化结束后会执行一次 eval 评测。
量化方法支持 turboquant 和 gptq
--standby-layer-cpu 可以保证小显卡也能逐层载入 GPU 完成量化。

2bpw + 0.127 模型量化完成后约为 14GB 磁盘占用。

### 2. FP16 baseline 评测

```bash
# 原始 fp16 模型直接评测（--inference-quant-mode 只对 --load-quantized 生效）
python eval_qwen35.py /path/to/qwen3.5-origin-fp16/model
```
如果显卡放不下可以测试：
```bash
python eval_qwen35.py $quant_modelname --sequential-eval --standby-cpu
```

### 3. WxA16 量化 + 评测

```bash
# 量化 + 评测（一步完成；不落盘时需显式 --wxa16 开真量化）
python run_qwen35.py $model_path wikitext2 \
    --wxa16 \
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

# WxFP8（e4m3 激活 + FP8 Tensor Core；MoE 走 fp8，attention 默认混合部署保 WxA8）
python eval_qwen35.py --load-quantized ./quant_ckpt --inference-quant-mode wxfp8

# WxFP4 保守版（bit1/4 expert e2m1+FP4 MX MMA；bit2 与 attention 维持 WxA8）
# ⚠ 必须用 dart312-t38 环境（triton 3.8 + ptxas 12.8；依赖版本已对齐主环境）
conda run -n dart312-t38 eval_qwen35.py --load-quantized ./quant_ckpt --inference-quant-mode wxfp4
```

WxA8 与 WxA16 共用同一份 packed checkpoint（码本转 INT8 是加载期算的），
`--load-quantized` 加载后 `qwen35_quant_io.convert_model_to_wxa8` 原地切换，零拷贝。
WxFP8 同理（`convert_model_to_wxfp8`，码本 e4m3 LUT + cb_scale 搜索加载期算），
WxFP4 同理（`convert_model_to_wxfp4`，nibble LUT + 常量 e8m0 + norms×residual 副本
加载期算），四种模式读的是同一份 checkpoint，切换只是加载标志。

## 路线图

- **WxA16** ✅ 已完成（FP16 激活 + FP16 Tensor Core + 混合 bit 权重量化）
  - **MoE 全路径 Triton 融合**：反量化、矩阵乘、epilogue 全部融合进单个 kernel，消除中间访存
  - **Group-First 权重布局**：权重按量化分组连续存储，提升 L2 缓存命中率与多 expert 并行效率
  - **Rotation Hoisting**：分组 QR 旋转提升到 expert 循环外（每 bit 全 token 预旋转一次，per-expert 零旋转开销）
  - **autotune 自适应调优**：BlockN 可搜索，根据硬件自动挑选最优配置
  - **Linear Attention 深度优化**：chunk delta rule 全路径 Triton 化，包括 pad-skip 跳过、WY 闭式解、cast 缩减、wy_prep 融合、kernel 自动调优等
  - 详细优化记录详见 [roadmaps/wxa16-optimization-backlog-260824.md](roadmaps/wxa16-optimization-backlog-260824.md)

- **WxA8** ✅ MoE + attention 全路径落地（INT8 激活 + INT8 Tensor Core + INT32 累加）
  - **INT8 融合 kernel**：kernel 级 1.52x（gate_up 1.61x / down 1.39x，真实 eval 形状），
    独立 INT8 tile 调优表（与 WxA16 最优点完全不同）
  - **Rotate+Quantize 融合**：分组旋转 + per-token per-group 量化单 kernel，hoist 规模 18.75x
  - **零格式变更**：MoE 码本转 INT8 加载期完成，与 WxA16 共用同一份 packed checkpoint
  - **attention 路径**：8-bit 均匀码本（indices 即 int8 权重，免查表），
    kernel 实测 2.63x/2.85x vs WxA16Linear；旧 checkpoint 的 Lloyd-Max 码本
    自动保持 W8A16（安全阀，静默降级防护）
  - **端到端实测**：wiki -15.1% / c4 -19.4%（vs WxA16），ppl 基本持平
    （wiki -0.0008 / c4 +0.0045）
  - 待做：per-expert 循环开销（占 MoE forward wall 约 80%，多 expert 合并 kernel 方向）
  - 详见 [roadmaps/wxa8-plan-260829.md](roadmaps/wxa8-plan-260829.md)
- **WxFP8** ✅ MoE 路径落地（e4m3 激活 + FP8 Tensor Core + 原生 fp32 累加；
  WxFP4 的基础设施预备步骤）
  - **码本 e4m3 化 + cb_scale 搜索**：浮点网格尺度不变性带来的 int8 没有的优化自由度，
    2-bit 码本 relerr 0.00021（优于 int8 的 0.00054）
  - **混合部署**：attention W8 走 e4m3 LUT 实测速度 0.58x + 精度 3.9x 双输，
    故 attention 保 WxA8、MoE 走 fp8（MoE 形态速度 0.89~1.00x 中性）
  - **端到端**：c4 与 WxA8 持平（+0.8%），ppl wiki +0.011 / c4 +0.014 vs WxA16
    （激活 relerr 2.57% 的 ppl 代价远小于预期——旋转的离群值抑制在 ppl 层面兑现）
  - 详见 [roadmaps/wxfp8-plan-260924.md](roadmaps/wxfp8-plan-260924.md)
- **WxFP4** ✅ 保守版全链路落地（e2m1 激活 + 块缩放 MMA + Triton 原生 mxf4nvf4）
  - **打通 sm_120 原生 FP4 MMA**：triton 3.8 + ptxas 12.8 组合实测 546 TFLOPS @4096³
    （1.55× fp8；四个坑与配方见 roadmap/memory：rhs scale 转置布局、scale 不走 TMA 等）
  - **专家级通用纯 MX GEMM**（面向普通 MoE）：纯预打包操作数 + TMA/persistent，
    dense 492 TF（1.39× fp8）；W 直量化 e2m1+per-32 e8m0 + 全局 scale 搜索
  - **保守版混合部署**：bit1/4 expert 走 fp4（码本→nibble LUT 连续 scale，
    **1-bit 精确落格**；residual 折 norms 副本，不动共享 buffer）、bit2 主力与
    attention 维持 WxA8——隔离验证（FP4_BITS 置空精确复现 wxa8）背书接线
  - **端到端**：ppl 与 WxA8 持平（wiki +0.0007 / c4 +0.0035）且全面优于 WxFP8；
    c4 比 WxA8 快 4.9%
  - 激进版（全 expert 迁移，3-4）已立项挂起，入口条件三条见 roadmap；
    普通结构 MoE 大 B 部署可基本绕开全部条件
  - 详见 [roadmaps/wxfp4-plan-260924.md](roadmaps/wxfp4-plan-260924.md)
