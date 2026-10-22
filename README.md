# DartMoQ-Qwen3.5

面向 Qwen3.5 MoE 模型的混合精度量化推理框架，及所有依赖 Qwen3.5 混合注意力机制模型，如 Qwen3.6/3.7/3.8 等。已支持 Qwen3.5-35B-A3B（子专家级切分搜索），Qwen3.5-122B-A10B（专家级 bit 搜索 + 流式量化）。
基于 [DartMoQ](https://github.com/zzningxp/DartMoQ) 算法，适配 Qwen3.5 MoE 的 grouped_gemm 权重格式，
Triton kernel 量化推理支持三种精度模式（共用同一份 packed checkpoint）：

- **WxA16** ✅：FP16 激活张量 + FP16 Tensor Core，支持 w1a16/w2a16/w4a16/w8a16 混合精度权重
- **WxA8** ✅：INT8 激活张量 + INT8 Tensor Core（MoE + attention 全路径）
- **WxFP8** ✅：e4m3 激活 + FP8 Tensor Core（MoE 路径；attention 混合部署保 WxA8。
  定位是 WxFP4 的基础设施预备步骤）
- **WxFP4** ✅ 保守版落地 + W 侧折叠直存（bit1/4 expert：e2m1 激活 + FP4 MX MMA，
  权重加载期折叠为直存 e2m1，kernel 零查表；bit2 主力与 attention 维持 WxA8。
  ⚠ 需 dart312-t38 环境——triton ≥3.8 + ptxas 12.8，见路线图）。
  Triton 原生 `mma.kind::mxf4nvf4`，dense 492 TF（1.39× fp8）；
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
  （1-bit 精确落格）+ residual 折 norms + **packed 索引折叠直存 e2m1**
  （kernel 零查表零转换，bit-exact），全部加载期完成，checkpoint 格式不变
- **Rotate+Quantize 融合**：分组旋转与激活量化融合为单个 Triton kernel，
  中间结果不落地（hoist 规模实测 18.75x vs 两段式）
- **Group-First 布局**：权重按 group 连续存储，提升 L2 缓存命中率
- **Rotation Hoisting**：分组 QR 旋转提升到 expert 循环外（全 token 预旋转一次，
  per-expert 零旋转开销），WxA8 下旋转与量化一并融合
- **Kernel 自动调优**：BlockN / BLOCK 尺寸 / warps / stages 网格搜索，自适应硬件最优配置
  （WxA16 与 WxA8 各有独立的调优表）
- **Linear Attention 深度优化**：chunk delta rule 全路径 Triton 融合
  - pad-skip、WY 闭式解、cast 缩减、wy_prep 融合、kernel 自动调优
- **122B 级模型流式量化**：meta 空结构 + 逐层流式反量化载入/落盘（`--standby-layer-meta`），
  bf16 全量载入需 ~244GB 的模型全程 CPU RAM 峰值 ~45GB；W8A8 int8 输入逐层读盘、
  GPU 反量化，产物分 shard 落盘，visual/mtp 原样存档
- **评测设备自适应**：按 checkpoint 大小自动选设备策略——双卡对半切分 batch eval
  （单边界 pre-hook 跨卡搬运、bs 自动收敛），单卡装不下自动回退 standby CPU +
  sequential，超出内存的基模型走流式评测（算一层读一层），无需任何设备 flag


## 性能

### Qwen3.5-35B-A3B（bpw2，RTX 5090，sequential eval，各模式分属不同 git 批次）

| 数据集 | fp16 基线 | WxA16 | **WxA8** | WxFP4 | WxFP8 |
|---|---|---|---|---|---|
| wikitext2（256 samples） | 87.19 s | 53.12 s | **45.08 s** | 60.10 s † | 55.43 s † |
| c4（145 samples） | 90.52 s | 76.71 s | 61.84 s | **59.37 s** | 62.93 s |
| **合计（vs 基线）** | 177.7 s | 129.8 s（-26.9%） | **106.9 s（-39.8%）** | 119.5 s †（-32.8%） | 118.4 s †（-33.4%） |
| ppl（wiki / c4） | 6.5852 / 9.6725 | 7.7955 / 11.2631 | 7.7947 / 11.2676 | 7.7954 / 11.2711 | 7.8064 / 11.2769 |

† wiki 的 WxFP4/WxFP8 疑首轮 Triton JIT 污染（未复跑确认），c4 数字干净；合计与
百分比受此影响仅供参考。

- WxA8 vs WxA16：wiki -15.1% / c4 -19.4%，ppl 持平；MoE 单独贡献 -7.0%/-6.7%，
  attention 路径贡献 -13.2%/-16.8%。WxA8 全路径数字基于 260831-u8 checkpoint
  （attention 均匀码本，MoE 与 260824 同型），与 WxA16（6eabe23）跨批次
- **W 侧折叠直存 e2m1 后（当前代码形态）**：同批次对照合计 **WxFP4 100.79s
  < WxA8 107.82s < WxFP8 112.24s**——WxFP4 在 35B 上为最快模式（vs WxA8 -6.5%），
  ppl 逐位一致（wiki 7.7972 / c4 11.2708）；gate_up kernel 2.0-2.4×
  （B64 40.2→16.6us；B2048 bit1 185→367 TF）。
  与上表跨批次数字并存，对比以同批次为准
- 35B 上 WxFP4 能登顶而 122B 不能：35B 的 bit1/4 专家占比更高，fp4 路径覆盖
  更多权重；122B 为 2-bit 主导（见下节）
- 端到端时间含约 20s 的 PPL 计算开销；首次运行 WxA8 触发 Triton JIT 编译
  （逐 expert 形状边跑边编译，端到端可虚增 10%+），测速前先捂热磁盘缓存
- 各批次明细：[roadmaps/wxfp8-plan-260924.md](roadmaps/wxfp8-plan-260924.md) §七 WF-5、
  [roadmaps/wxfp4-plan-260924.md](roadmaps/wxfp4-plan-260924.md) WF4-4、
  [roadmaps/wxfp4-fold-plan-260927.md](roadmaps/wxfp4-fold-plan-260927.md)

### Qwen3.5-122B-A10B（bpw2，RTX 5090，sequential eval，git da4bd33+）

输入为 W8A8 int8 checkpoint（磁盘 119GB）；fp16 基线 = int8 权重逐层反量化 bf16 后的
原生前向（W16A16）；量化 checkpoint 2bpw 磁盘 40GB。评测口径与基线完全对齐：
同逐层循环、同 hidden states 全缓存、同 batch（16/层内分批，delta rule 的 fp32
瞬时峰值随 bs 线性增长，是所有模式的共同上限）。

| 数据集 | fp16 基线 | WxA16 | **WxA8** | WxFP4 | WxFP8 |
|---|---|---|---|---|---|
| wikitext2 | 121.18 s | 137.03 s | **83.01 s** | 85.21 s | 90.14 s |
| c4 | 152.25 s | 193.67 s | **123.18 s** | 125.93 s | 130.86 s |
| **合计（vs 基线）** | 273.4 s | 330.7 s（+21%） | **206.2 s（-25%）** | 211.1 s（-23%） | 221.0 s（-19%） |
| ppl（wiki / c4） | 5.0766 / 8.9142 | 6.037 / 10.032 | 6.040 / 10.037 | 6.037 / 10.041 | 6.043 / 10.042 |

- **WxA8 最快（-25%）**，排名与 35B 相反：122B 的 bit 分布为 2-bit 主导
  （MoE packed 27.05GB 中 2-bit 占 26.15GB），WxFP4 的 fp4 路径只覆盖 bit1/4
  专家（~4% 权重），bit2 主力仍走 WxA8 kernel，故 WxFP4 ≈ WxA8 + LUT 开销
- 四种量化模式 ppl 几乎一致（wiki 6.037~6.043），相对基线 +0.96 / +1.12
  ——2bpw 的质量代价，与推理 kernel 无关
- WxA16 是唯一慢于基线的模式：packed MoE 每次 forward 约 210ms 的 per-expert
  固定开销（init 占 ~87%，实际 kernel 计算仅 ~29ms），per-expert 合并 kernel
  是明确优化方向

122B 推理配置对比（同一 2bpw checkpoint）：

| 配置 | 显存峰值 | CPU RAM | bs | 适用 |
|---|---|---|---|---|
| 单卡 sequential + standby（上表口径） | ~12 GB | ~45 GB | 16 | 公平评测/单卡部署 |
| 双卡自动切分 batch eval | ~24 GB/卡 | ~45 GB | 8 | 双卡常驻部署（免搬运，bs 受限） |
| fp16 基线流式评测 | ~23 GB | ~15 GB | 16 | int8 原模型直接评测（算一层读一层） |

### 吞吐（prefill / decode，RTX 5090，bench_throughput.py P0 口径）

batch=1、wiki 真实文本切片。单位：prefill/decode 均为 **tok/s（越高越快）**，
ttft 秒（越低越快）。prefill 只计时 transformer（不含 lm_head），decode 含
cache 路径；每条结果都经过 cache-vs-full logits 一致性守卫（check PASS 才入表）。

| 模型 / 模式 | pf16 | pf128 | pf2048 | pf4096 | dec（256/512/1024） | ttft | check |
|---|---|---|---|---|---|---|---|
| 官方 FP8（双卡） | 215.6 | 1657.9 | 10009.8 | 10082.3 | 15.77 / 15.82 / 15.84 | 0.142 | PASS |
| 官方 GPTQ-Int4（单卡） | 51.7 | 329.2 | 3298.6 | 4941.3 | 3.93 / 3.95 / 3.94 | 0.644 | PASS |
| 4bpw wxa16（全 4bit） | 43 | 163 | 1416–1428 | 2403–2411 | 11.00–11.13 | 0.884–0.888 | PASS |
| 4bpw wxa8 | 111.5 | 492.7 | **3619.4** | 4898.8 | 13.33–13.36 | **0.322** | PASS |
| 4bpw wxa8（逐 expert，旧基线） | 46.2–46.4 | 181 | 1604–1611 | 2692–2701 | 11.26–11.39 | 0.794–0.796 | PASS |
| 4bpw wxfp8 | 84.7–85.1 | 445–446.2 | 3605.4–3620 | **4932.2–4959.9** | **13.93–14.03** | 0.36–0.361 | PASS |
| 4bpw wxfp8（逐 expert，旧基线） | 46.4–46.5 | 181–184 | 1606–1609 | 2680–2686 | 11.30–11.43 | 0.791–0.797 | PASS |
| 4bpw wxfp4（逐 expert + LUT） | 45.4 | 178.6 | 1567.3 | 2641.0 | 11.12–11.14 | 0.806 | PASS |
| 2bpw wxa16（1/2/4 混合） | 43 | 160.3 | 1421.7 | 2421.6 | 10.91–10.97 | 0.890 | PASS |
| 2bpw wxa8 | 77.9 | 407.2 | **3648.8** | 5186.8 | 11.69 | 0.406 | PASS |
| 2bpw wxa8（逐 expert，旧基线） | 44.6 | 177.6 | 1611.2 | 2765.6 | 9.78–9.84 | 0.824 | PASS |
| 2bpw wxfp8 | 77.2 | 402.3 | 3590.9 | **5188.2** | **11.62–11.64** | **0.404** | PASS |
| 2bpw wxfp8（逐 expert，旧基线） | 45.2 | 180.6 | 1628.2 | 2785.0 | 9.92–9.93 | 0.819 | PASS |
| 2bpw wxfp4 | 44.9 | 176.8 | 1611.5 | 2760.6 | 9.95–9.96 | 0.824 | PASS |

- **P5-2 合并路径（2026-10-20 落地，wxa8 + wxfp8）**：按 bit 池合并多 expert 为单次
  grouped kernel launch，消灭逐 expert 的 Python/torch 调度链——4bpw 与 2bpw
  的 prefill 均 **~2.2×**（wxa8/wxfp8 两模式同幅）、decode **+17~23%**、MoE 段
  57.3→41.3ms（wxa8 4bpw）；prefill 反超官方 GPTQ-Int4，与官方 FP8 的差距
  收窄到 decode 1.12~1.18× / prefill 2.77×（per-GPU 口径：官方 FP8 双卡是
  device_map 串行切分无重叠，双卡只解决容量不折半差距）。`DARTMOQ_MERGED_MOE=0`
  可回退逐 expert 对照；wxa16/wxfp4 的合并同型为下一步
- **wxfp8 合并后 decode 反超 wxa8**（4bpw 13.93–14.03 vs 13.33–13.36，+5%）：
  逐 expert 时代两模式 decode 持平（11.3 vs 11.3），调度链消失后 FP8 Tensor
  Core 的算力优势才兑现（MoE 段 39.9 vs 41.3ms）；prefill 两模式持平
  （3620 vs 3619），小 prefill（pf16/pf128）wxa8 仍更快。2bpw 两模式全档
  基本持平（bit2 主导，kernel 形态相同）
- **4bpw vs 2bpw**（同合并路径）：decode 4bpw 快 14~21%（wxa8 13.34 vs 11.69；
  wxfp8 14.03 vs 11.63），2bpw 仅 pf4096 赢回 ~6%——低 bit 混合路径（1/2-bit
  解包 + per-bit 分池）的固定开销在开销主导的场景里净亏，带宽收益只在长
  prefill 兑现（合并后带宽收益可见度从 +3% 提高到 +6%）
- 4bpw wxfp4 为「逐 expert + WFF-5 预算跳过 fold 的 in-kernel LUT」双重旧路径
  叠加的数字（首个可跑通的 4bpw wxfp4，此前 OOM），待 fp4 合并接线后重测
- 全部差异的逐项分析（四个问题的根因拆解）见
  [roadmaps/throughput-analysis-261020.md](roadmaps/throughput-analysis-261020.md)
- 口径与明细：[roadmaps/throughput-bench-261019.md](roadmaps/throughput-bench-261019.md)；
  日志 1015.thrput.1（2bpw 逐 expert）/ 1018.thrput.1.w4（4bpw 逐 expert）/
  1018.thrput.2.official.fp8 / 1018.thrput.3.official.gptq4int /
  1018.thrput.4–5（wxa8 合并）/ 1018.thrput.7（wxfp8 合并）

## 下游任务评测（lm-eval 七任务，对齐 BitsMoE 口径）

2026-10 起用 lm-eval-harness 对 35B 2bpw checkpoint（260930-s1，三种 kernel 模式
共用同一份）跑 BitsMoE 同款七任务（five-shot）：HellaSwag / ARC-Easy / MathQA /
MMLU / OpenBookQA / WinoGrande / GSM8K。评测入口 `eval_tasks.py`（自定义
DartMoQLM 直接驱动量化模型，不经 HFLM；lm-eval 0.4.12 + transformers 5.13.0，
量化模式整模型驻留 RTX 5090），结果 JSON 落 `results/`，日志解析用
`task_logs_parser.py`，数据集拉取与各任务口径见 `benchmarks/roadmap.md`。
MathQA 因 hub 旧式 script 被 datasets 4.x 禁用，走本地 parquet
（官方 zip 转换，`benchmarks/mathqa_parquet/`，数据量与官方 metadata 一致）。

主指标口径：HellaSwag / ARC-Easy / OpenBookQA / WinoGrande 取 acc_norm，
MathQA / MMLU 取 acc，GSM8K 取 exact_match（strict-match，括号内 flexible-extract）。

| 模式 | HellaSwag | ARC-Easy | MathQA | MMLU | OpenBookQA | WinoGrande | GSM8K | 七任务均值 ² |
|---|---|---|---|---|---|---|---|---|
| fp16 基线 ¹ | 0.8199 | 0.8998 | 0.6647 | 0.8487 | 0.4880 | 0.7648 | 0.8939 † | 0.7685 |
| wxa8 | 0.7609 | 0.8948 | 0.5776 | 0.8094 | 0.4520 | 0.7553 | 0.8415（0.8453） | 0.7279 |
| wxfp8 | 0.7607 | 0.8948 | 0.5762 | 0.8093 | 0.4500 | 0.7545 | 0.8362（0.8423） | 0.7268 |
| wxfp4 | 0.7624 | 0.8969 | 0.5796 | 0.8096 | 0.4500 | 0.7435 | 0.8347（0.8431） | 0.7264 |

² 七任务均值中 GSM8K 取 flexible-extract（fp16 行 † 为外部参照，仅此口径可得；
量化行自测 flexible 见括号内）。

† fp16 行的 GSM8K 非本仓库评测：Qwen3.5-35B-A3B-FP8（官方 FP8 量化）
vLLM 评测 89.39%（flexible-extract，
[AMD 模型卡](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4)）——官方
README benchmark 表不含 GSM8K，fp16 无官方数字可引，此处取同模型官方量化档位
的数字作外部参照（口径为 vLLM 部署 + flexible-extract）。

¹ fp16 基线 70GB 无法整驻 GPU：loglikelihood 六任务走 sequential 层循环
（层驻 CPU 逐层搬 GPU，18.8 小时，`--fp16-mode sequential`）；GSM8K 需要
KV-cache 自回归（sequential 不支持、device_map offload 实测 268 小时不可行），
未跑，故取外部参照（见 †）。另一 vLLM 实测约 0.78-0.81
（[issue #37804](https://github.com/vllm-project/vllm/issues/37804)，修复精度
bug 后）。我们 2bpw 的 flexible-extract 0.842-0.845 相对官方 FP8 数字 -4.9 分，
与「2bpw 比 FP8 更激进」的预期方向一致。

- 三种 kernel 模式七任务均值互差 ≤0.0015，与 ppl 结论一致：kernel 实现不引入质量损失
- 2bpw 相对 fp16 基线七任务掉 4.1 分（0.7685 → 0.7279）；对照 BitsMoE 的
  2-bit 标量量化（71.61 → 64.29，-7.3 分；Qwen3-30B 不同代模型，仅作间接对照），
  旋转量化 2bpw 的掉分约其一半
- 掉分最大的是 MathQA（-8.7）与 HellaSwag（-5.9），GSM8K -4.9
  （flexible-extract，外部参照口径），OpenBookQA -3.6，其余 ≤1
- eval 墙钟：wxa8/wxfp8/wxfp4 七任务约 8.1-8.3 小时（热跑，Triton 磁盘缓存；
  HellaSwag + GSM8K 占 ~87%：前者 4 万次长 context 前向，后者 1319 条 batch=1
  自回归）。此墙钟是评测耗时，不代表 decode 吞吐；速度对照另见 P0 decode 通路

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

2bpw 压缩后 checkpoint 磁盘占用构成（global-a8s8m2bpw、slices=1 专家级 bit 搜索，
260930-s1 实测；MoE 为 packed 索引 + norms 的落盘大小，attention 为 8-bit packed）：

| 部分 | 35B-A3B | 122B-A10B |
|---|---|---|
| MoE packed（1/2/4-bit 混合） | 7.52 GB（1b 0.05 / 2b 7.36 / 4b 0.11） | 27.05 GB（1b 0.30 / 2b 26.15 / 4b 0.60） |
| Attention / shared expert（8-bit） | 1.31 GB | 4.27 GB |
| 全精度剩余（embed、lm_head、norms 等） | 2.43 GB | 4.67 GB |
| visual/mtp 存档（原样搬运） | — | 3.3 GB |
| 合计 | 11.24 GB | 35.94 GB（+存档 ≈ 39.5 GB） |

MoE 的 2-bit 主力专家占压缩主体的约 2/3，1/4-bit 专家为零头；attention 统一 8-bit；
embed/lm_head 保持 BF16。

### 1.1 122B 级模型（Qwen3.5-122B-A10B）

122B 暂不支持切分子专家，bit 搜索在专家级进行（`--slices 1`，scheme 的 bpw 为
全局目标值）。量化走流式路径（bf16 全量载入超出常规内存），输入为分 shard 的
W8A8 int8 checkpoint 时逐层读盘并在 GPU 反量化：

```bash
python run_qwen35.py /path/to/Qwen3.5-122B-A10B-W8A8 wikitext2 \
        --inference-quant-mode wxa8 --nsamples 64 --slices 1 \
        --quant-scheme global-a8s8m2bpw --rank-mode turboquant_innerproduct \
        --quantmode turboquant --standby-layer-meta \
        --save-quantized models/qwen3.5-122B-2bpw/
```

量化产物逐层落盘（分 shard + index.json，约 40GB）。评测无需任何设备 flag：
双卡自动对半切分 batch eval（bs 自动收敛），单卡自动回退 standby CPU +
sequential eval；性能对比请用 sequential 口径（与 fp16 基线同循环结构同 bs）：

```bash
python eval_qwen35.py models/qwen3.5-122B-2bpw/ --inference-quant-mode wxa16

# 性能对比口径（公平评测，见性能一节）：
python eval_qwen35.py models/qwen3.5-122B-2bpw/ --inference-quant-mode wxa8 \
    --sequential-eval --standby-cpu
```

### 1.2 量化的 standby 模式选择

量化主循环本身是逐层的，三种 standby 模式控制**权重如何在内存中流转**：

| 模式 | 行为 | CPU RAM 峰值 | 适用 |
|---|---|---|---|
| 默认（无 standby） | `device_map="auto"` 全模型上 GPU | 低 | 模型 + 中间量装得进显存 |
| `--standby-layer-cpu` | 整模型 CPU 常驻，逐层搬上 GPU 量化 | ≈ 全量 bf16 | RAM 装得下 bf16 全量的中型模型（35B 的经典跑法） |
| `--standby-layer-meta` | meta 空结构，逐层从盘流式反量化载入 | ~45GB | 全量 bf16（如 122B 的 ~244GB）超出 RAM 的模型 |

`--standby-layer-meta` 的特点：

- 输入为分 shard 的 checkpoint（`model.safetensors.index.json`）时，每层按需读盘、
  **int8 在 GPU 上反量化成 bf16**（不依赖 compressed-tensors 库）；BF16 权重直读
- 量化产物**逐层落盘**（按层边界切 shard + index.json），不等全部算完；
  量化过程中每层校验磁盘余量，不够会在第一层就报错
- visual/mtp 部分原样搬运到 `visual_mtp.safetensors` 存档，不进加载路径
- 与 `--standby-layer-cpu` 互斥；`--quant-layers` 部分量化时未量化层自动从
  checkpoint 反量化兜底归档

```bash
# 122B 级（W8A8 int8 输入）：
python run_qwen35.py $model_path wikitext2 \
        --nsamples 64 --slices 1 --quant-scheme global-a8s8m2bpw \
        --rank-mode turboquant_innerproduct --quantmode turboquant \
        --standby-layer-meta --save-quantized ./quant_ckpt
```

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
+ 索引折叠直存 e2m1 加载期算），四种模式读的是同一份 checkpoint，切换只是加载标志。

### 5. 吞吐基准（bench_throughput.py）与官方基线

能测什么：同一 harness 两条线——**我们的 checkpoint**（`--quant-dir`，
wxa16/wxa8/wxfp8/wxfp4 四种 kernel 模式共用同一份 checkpoint）与
**外部 HF 模型**（`--model-path`：官方 FP8 / GPTQ-Int4，或任意 fp16/w4a16
目录；ConditionalGeneration 结构自动用 ImageTextToText 加载，GPTQ 走
gptqmodel 原生加载 + TritonV2）。测量项：prefill 16/128/2048/4096
（tok/s，仅 transformer 不含 lm_head，每档 3 次取中位）、decode
256/512/1024（tok/s，batch=1 greedy 含 cache）、TTFT、MoE stage 逐项拆分
（我方模式独有，moe_ms 列）、cache-vs-full logits 一致性守卫（check 列，
不过守卫的数字不入表）。结果 JSON 落 `results/`，`logs_parser.py` 直接解表。

```bash
# 我们的 checkpoint（dart312-t38 环境）
sh run.throughput.sh

# 官方 FP8 / GPTQ-Int4 基线（用他们的 kernel；复用 dart312 = torch 2.9 栈，
# 官方 kernel 包最新版按 torch 2.11 编译，版本必须钉死；⚠ 不能装进 dart312-t38）
conda activate dart312
pip install "gptqmodel==7.3.6" "fbgemm-gpu==1.4.0" "torchao==0.17.0" \
            optimum "kernels==0.17.2"
sh run.throughput.official.sh   # FP8 35GB 需双卡，脚本内已设好

# 单条运行（自由组合）
python benchmarks/bench_throughput.py --quant-dir models/<ckpt>/ --mode wxa8 \
    --prefill-lens 16,128,2048,4096 --decode-tokens 256,512,1024
```

口径、结果与差异分析见性能一节「吞吐」小节、
[roadmaps/throughput-bench-261019.md](roadmaps/throughput-bench-261019.md) 与
[roadmaps/throughput-analysis-261020.md](roadmaps/throughput-analysis-261020.md)。

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
  - **W 侧 load 期折叠直存 e2m1**：索引→e2m1 映射挪到 ctx 构建期一次完成，
    kernel 直读零查表（bit-exact；bit4 零膨胀、bit1 全模型 +47MB）；
    gate_up kernel 2.0-2.4×、eval wall -9.8%，**WxFP4 成为最快模式**
  - **端到端**：ppl 与 WxA8 持平（wiki +0.0007 / c4 +0.0035）且全面优于 WxFP8；
    c4 比 WxA8 快 4.9%
  - 激进版（全 expert 迁移，3-4）已立项挂起，入口条件三条见 roadmap；
    普通结构 MoE 大 B 部署可基本绕开全部条件
  - 详见 [roadmaps/wxfp4-plan-260924.md](roadmaps/wxfp4-plan-260924.md)
    与 [roadmaps/wxfp4-fold-plan-260927.md](roadmaps/wxfp4-fold-plan-260927.md)

- **吞吐基准与外部基线**（2026-10-19 立项；2bpw/4bpw/官方 FP8+Int4 已出全量数字）
  - bench_throughput.py P0 口径（batch=1 prefill/decode + cache 一致性守卫），
    我方 checkpoint 四模式 + 官方 FP8（双卡）/GPTQ-Int4（gptqmodel TritonV2）
  - 核心发现：与官方 FP8 的差距主因是 per-expert 循环 vs grouped kernel
    （prefill per-GPU 6.2×；FP8 双卡系 device_map 串行切分、无算力红利）；
    低 bit 混合路径有固定开销惩罚（2bpw decode
    慢 4bpw 13%）；我们小 B 场景反超 gptqmodel Int4 decode 2.9×
  - 详见 [roadmaps/throughput-bench-261019.md](roadmaps/throughput-bench-261019.md)
    与 [roadmaps/throughput-analysis-261020.md](roadmaps/throughput-analysis-261020.md)
