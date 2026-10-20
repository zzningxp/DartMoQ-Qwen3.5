# 吞吐基准（prefill/decode）与外部基线（2026-10-19 立项）

bench_throughput.py 的结果记录与外部官方基线计划。P0 口径：batch=1、wiki 真实
文本切片，prefill 16/128/2048/4096（tok/s，仅 transformer 不含 lm_head），
decode 256/512/1024（**ms/token**，含 cache 路径），ttft、cache-vs-full logits
一致性守卫。硬件 RTX 5090（31.4GB 可用）。

## 一、35B 吞吐结果汇总（2026-10-19/20，1015/1018/1019 日志）

checkpoint：4bpw = `models/qwen3.5-35B-4bpw-261016-{gptq,turboquant}`
（`global-a4s4m4bpw`，无 bit 搜索空间即全 4bit；两份结果取范围），
2bpw = `qwen3.5-2bpw-260930-s1`。官方基线 = `~/models/` 的 FP8（双卡）与
GPTQ-Int4（单卡，gptqmodel 原生加载 TritonV2）。单位：prefill/decode 均为
**tok/s（越高越快）**，ttft 秒（越低越快）。

| 模型 / 模式 | pf16 | pf128 | pf2048 | pf4096 | dec（256/512/1024） | ttft | check |
|---|---|---|---|---|---|---|---|
| 官方 FP8（双卡） | 215.6 | 1657.9 | 10009.8 | 10082.3 | 15.77 / 15.82 / 15.84 | 0.142 | PASS |
| 官方 GPTQ-Int4（单卡） | 51.7 | 329.2 | 3298.6 | 4941.3 | 3.93 / 3.95 / 3.94 | 0.644 | PASS |
| 4bpw wxa16 | 43–43 | 162–163 | 1416–1428 | 2403–2411 | 11.00–11.13 | 0.884–0.888 | PASS |
| 4bpw wxa8 | 46.2–46.4 | 181–181 | 1604–1611 | 2692–2701 | 11.26–11.39 | 0.794–0.796 | PASS |
| 4bpw wxfp8 | 46.4–46.5 | 181–184 | 1606–1609 | 2680–2686 | 11.30–11.43 | 0.791–0.797 | PASS |
| 4bpw wxfp4 | — | — | — | — | OOM（WFF-5，见 §二） | — | — |
| 2bpw wxa16 | 43 | 160.3 | 1421.7 | 2421.6 | 10.91–10.97 | 0.890 | PASS |
| 2bpw wxa8 | 44.6 | 177.6 | 1611.2 | 2765.6 | 9.78–9.84 | 0.824 | PASS |
| 2bpw wxfp8 | 45.2 | 180.6 | 1628.2 | 2785.0 | 9.92–9.93 | 0.819 | PASS |
| 2bpw wxfp4 | 44.9 | 176.8 | 1611.5 | 2760.6 | 9.95–9.96 | 0.824 | PASS |

差异分析（四个问题：FP8 为何快 / 4bpw 为何慢 / 2bpw 为何慢于 4bpw /
Int4 为何 prefill 快 decode 慢）见
[throughput-analysis-261020.md](throughput-analysis-261020.md)。

⚠ 2026-10-20 勘误：本文件 §一 初版把 dec 列误读为 ms/tok，得出「2bpw 快
13%」的错误结论；实际 dec 列为 tok/s，**4bpw decode 比 2bpw 快 ~13%**，
上表与结论已按正确单位重写。

## 二、wxfp4 在 4bpw 上 OOM（已定位，处理中）

1018 日志两次 wxfp4 均在首层 `_get_bit_context` 的 `fold_indices_to_e2m1`
OOM（ Tried to allocate 4.00 GiB / free 3.94 GiB ）。两层原因：

1. **瞬时分配**（已修）：fold 整矩阵一次物化 `idx.long()`（8B/元素），全
   4bit 池（256 expert 全是 bit4）下单次 4GB。已改按 G 分块（~32M 元素/块，
   瞬时峰值 ~数百 MB），CPU 对拍 bit 1/2/4 与整矩阵路径 bit-exact。
2. **驻留副本**（已拍板先不动，2026-10-19）：WFF-1 的设计前提是「fp4 线内仅
   bit1/4 小池」（fold plan §二 +47MB 上限），4bpw 全 4bit 时
   `indices_e2m1_gf` 变成全量 MoE 索引的副本（40 层 ctx 缓存累计 ≈ +10GB），
   31.4GB 卡装不下（18GB 模型 + 10GB 副本 + 激活）。kernel 保留 in-kernel LUT
   分支（`w_e2m1_gf=None` 走 legacy）。候选方案留档：① 按池大小自适应跳过
   fold（小池折叠、大池走 kernel 内 LUT）——内存受限的 bit 级自适应而非整体
   回退；② fold plan 风险表的 kernel 内算术化（bit4 无现成算术化需另行论证）。
   当前状态：4bpw 上 wxfp4 不跑，2bpw 路径不受影响。

## 三、官方 FP8 / GPTQ-Int4 基线（待跑）

官方发布（`~/models/`，结构 `Qwen3_5MoeForConditionalGeneration`，含
visual/mtp，权重 key 带 `model.language_model.` 前缀）：

| 模型 | 磁盘 | 格式 | kernel 后端 |
|---|---|---|---|
| Qwen3.5-35B-A3B-FP8 | 35GB | fp8 block 128×128 + dynamic act（attn/conv/embed/lm_head/visual/mtp 不量化） | transformers fbgemm-fp8（`fbgemm-gpu`） |
| Qwen3.5-35B-A3B-GPTQ-Int4 | 23GB | gptq 4bit gs128 sym（attn/shared/visual/mtp 排除） | gptqmodel（marlin/exllama） |

实施要点：

- **bench 已支持**：`--model-path` 路径自动识别 ConditionalGeneration 结构用
  `AutoModelForImageTextToText` 加载；`_text_tower()` 解析
  `model.model.language_model` 做 prefill 计时与 MoE 聚合；decode/一致性守卫
  走顶层前向。脚本 `run.throughput.official.sh`。
- **复用 `dart312` 环境**（已不在主流程使用，torch 2.9.0+cu128 栈）。⚠ 官方
  kernel 包最新版全部按 **torch ≥ 2.11** 编译，pip 依赖解析不报（运行时 import
  才暴露），必须钉版本（1018.thrput.2.official 实测）：

  | 包 | 最新版（torch 2.11 栈） | torch 2.9 栈实测定版 | 失败模式 |
  |---|---|---|---|
  | gptqmodel | 7.6.0 | **7.3.6** | 7.4-7.6 的 w4a_nvfp4 import `torch.nn.functional.ScalingType`（2.11 API）→ transformers 加载任何量化模型都被连带挂 |
  | fbgemm-gpu | 1.9.0 | **1.4.0** | 1.5/1.9 的 .so 链接 torch 2.11 符号（c10_cuda_check_implementation 新签名）加载失败；1.3 另有 .so 挂 |
  | torchao | 0.18.0 | **0.17.0** | import 链经 transformers quantizer 注册表触发；cpp 扩展跳过警告无碍（两条基线不用 torchao 算子） |

  triton 3.5.0 / torch / numpy 均不动（t38 的 ptxas 软链目标在 dart312 的
  triton 里，冻结）。gptqmodel 7.3.6 的 cpp 扩展在 torch 2.9 下同样跳过，
  sm_120 无 marlin/exllama 也可用，走 **tritonv2 后端**（gptqmodel 的 triton
  kernel）。transformers 5.20 另有两个硬依赖（1018.thrput.2 第二轮实踩）：
  **optimum**（GPTQ 加载经 optimum 胶水层，内部再调 gptqmodel）与
  **kernels==0.17.2**（finegrained-fp8 kernel，FP8 块量化路径）。第三轮实踩：
  本机 HF hub 直连挂起 → 内核拉取必须 `HF_ENDPOINT=https://hf-mirror.com`；
  且 kernels 包的**离线快照校验不认部分缓存**（只缓存了选中 build 的 11 个
  文件时，HF_HUB_OFFLINE=1 仍报 no local snapshot）→ 官方基线脚本走镜像
  在线（datasets 仍离线）。第四轮实踩：gptqmodel 7.3.6 后端自动选择挑
  **MarlinLinear**，官方 Int4 checkpoint 含 N=1 形状（K=2048）触发
  `pack_factor=8` 不整除 NotImplementedError（marlin/exllama 在 sm_120 本就
  不可用）。第五轮实踩：强制 TritonV2 后仍然挂——transformers 5.20 的
  GPTQConfig 丢掉了官方配置里 `dynamic` 的 `-:` 排除规则（BF16 的
  shared_expert_gate 2048→1 被误包成量化层），且 Qwen3.5 融合 MoE 的
  per-expert qweight 在 get_layers 里没有对应模块（transformers 路径根本
  加载不了这种 checkpoint；官方 README 也只给 vLLM/SGLang
  `--quantization moe_wna16` 用法）→ bench 的 gptq 分支改走
  **GPTQModel.load 原生加载**（认识 dynamic 排除与融合 MoE，backend 强制
  gptq_triton）。**FP8 基线 2026-10-19 已跑通**（check PASS，max logits
  diff 2.06，argmax 8/8）。备选：vLLM（moe_wna16 kernel，官方钦定，
  需新环境 torch 2.11 栈）。GPU 真实加载与 forward 属首次验证范围
  （check PASS 为准）。
  备选（若 7.3.6 加载仍有坑）：新建 torch 2.11 环境装 gptqmodel 7.6 +
  fbgemm 1.9（不可在 dart312 升 torch——会把 triton 3.5 顶掉，破坏 t38 软链）。
- **FP8 需双卡**：35GB > 单卡 31.4GB，`CUDA_VISIBLE_DEVICES=0,1` +
  device_map=auto；吞吐数字含跨卡 hop，对比时注明。Int4 23GB 单卡可跑。
- 口径与我们的 checkpoint 完全一致（同 prefill 档/decode 档/wiki 数据切片/
  一致性守卫），`check PASS` 后数字才可入表。

```bash
# 环境一次性安装（复用 dart312 = torch 2.9 栈，版本钉死；勿装进 dart312-t38）
conda activate dart312
pip install "gptqmodel==7.3.6" "fbgemm-gpu==1.4.0" "torchao==0.17.0" \
            optimum "kernels==0.17.2"

# 跑基线（Int4 单卡 + FP8 双卡，脚本内已设好 CUDA_VISIBLE_DEVICES）
sh run.throughput.official.sh
```

## 变更记录

- 2026-10-19：立项；记录 4bpw vs 2bpw 结论；wxfp4 4bpw OOM 定位（分块 fold
  已修，驻留副本方案待拍板）；官方 FP8/Int4 基线接入完成待跑。
