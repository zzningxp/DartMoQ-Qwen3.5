# 35B 吞吐差异分析（2026-10-20）

数据与口径见 [throughput-bench-261019.md](throughput-bench-261019.md) §一
（git 99336b8+ / 8e0f61c+；日志 1015.thrput.1、1018.thrput.1.w4、
1018.thrput.2.official.fp8、1018.thrput.3.official.gptq4int）。单位均为
tok/s（越高越快）。硬件 RTX 5090 ×2（带宽 ~1.79TB/s/卡，显存 31.4GB）。

先摆几个换算好的基础量（decode batch=1，逐 token 一步）：

| 路径 | step 耗时 | 其中 MoE stage | 说明 |
|---|---|---|---|
| 官方 FP8（双卡） | 63.4 ms | — | 35GB 双卡 device_map |
| 我们 4bpw wxa8（单卡） | 88.7 ms | 57.3 ms（65%） | bench moe_ms 列 |
| 我们 2bpw wxa8（单卡） | ~102 ms | ~63 ms | moe_ms 0.0632 |
| 官方 GPTQ-Int4（单卡） | ~254 ms | — | step1 0.258s≈稳态，非 JIT |

权重带宽下限参照：4bpw checkpoint 18GB 单卡全量读一遍 ~10ms，逐 token
实际只需激活的 ~4GB（~2-3ms）；FP8 35GB 双卡激活 ~5GB——**所有路径的
decode 都远高于带宽下限（>20×），全部是开销主导而非带宽主导**。

**decode 开销的构成（B=1，2026-10-20 修正口径）**：每 token 每层激活 8 个
专家、40 层 = **320 次专家迭代**。纯 kernel launch 约 5µs × ~2000 次 ≈
10ms，占比不大；真正贵的是每次专家迭代外面裹的 **Python/torch 调度小算子
链**（切片、gather、silu、乘、scatter），每个算子 dispatch 10-30µs 且
B=1 时工作量趋近零、开销一分不少——320 次 × ~10 个算子 ≈ **40-90ms**，
与实测 MoE 段 57.3ms 对上。即：decode 贵在 launch 外面的调度链，不只是
launch 本身。（B=32 的 eval profile 结论「warm 循环 GPU-bound」与 B=1 的
调度链主导并存——小 B 时算子工作量→0 但 dispatch 不变，开销占比反超；
大 B 时 GPU 时间重新盖过调度链。）

## 1）官方 FP8 为什么这么快（pf2048 10009 vs 我们 1603，6.2×）

- **kernel 形态是根因**：transformers 5.20 的 finegrained-fp8 kernel 提供
  `matmul_2d / matmul_batched / matmul_grouped`——MoE 走 **grouped 融合
  kernel**（一次 launch 覆盖全部 active expert，fp8 权重 × 128×128 block
  scale × fp8 tensor core，源头是 DeepGEMM 系 kernel）。没有 per-expert
  Python 循环、没有逐 expert launch。
- **双卡红利**：35GB 装不进单卡，device_map=auto 两卡切分，prefill 两卡
  并行——6.2× 里约一半是卡数。单卡等价 ~5000 tok/s，per-GPU 差距 **3.1×**。
- **attention 保 BF16**：官方只量化 experts（modules_to_not_convert 列了
  attn/conv/embed/lm_head/visual/mtp），attention 走原生 BF16 GEMM。虽然
  本机没装 flash-linear-attention、delta rule 落到 torch 参考实现（慢），
  但 GEMM 类部分全速，且这不影响结论方向——他们的 MoE prefill 快到即使
  attention 拖后腿仍领先 6×。
- ttft 0.142s / pf16 215.6 tok/s：引擎路径固定开销极低。

## 2）我们的 4bpw 为什么比 FP8 慢不少（最关键）

prefill 6.2×（单卡等价 3.1×）、decode 1.4×。拆 88.7ms/step（MoE 57.3ms
占 65%）与 pf2048 的 32ms/层：

1. **MoE kernel 形态（主因）**：main 上逐 expert 循环是唯一路径，代价随 B
   两头爆：**小 B（decode）**：320 次专家迭代 × 每次 ~10 个 Python/torch
   调度小算子（切片/gather/silu/乘/scatter，每个 10-30µs、工作量→0）≈
   40-90ms，对上实测 57.3ms——launch 本身（~10ms）不是大头；**大 B
   （eval/prefill）**：小 kernel SM 填不满（gate_up 256 programs /
   170 SM = 1.51 波，尾波利用率 50.6%，backlog §4.3 实测）+ 串行链 ×256。
   fp8 引擎 grouped kernel 一次覆盖全部 active expert，launch 与调度链
   同时消失、SM 填满。
   **P5-2 注脚**：按 bit 合并多 expert 的 v2 kernel 已做过但在 `opt-p5-2`
   分支，真实 eval 变慢（269.4→288.6s）被否、未并入 main；否决发生在
   P5-3 tile 调优之前（当时还是 B=32 旧配置），SM 填充度证据说明合并动机
   仍成立，「新配置下重测 P5-2」挂在 backlog §4.3——官方 grouped kernel
   正是该方向的正确完成态，本次对比是其外部佐证；且合并 kernel 同时
   消灭 decode 的调度链与 prefill 的填充度问题，双场景受益。
2. **4bpw 的 attention 更重**：`global-a4s4m4bpw` 的 qscheme attn=[4]，
   attention 是 4-bit Lloyd-Max 码本（非均匀）→ 不能转 WxA8（量化日志
   "Converted to WxA8: 40 MoE layers, 0 attention/shared linears"），
   跑 W4A16 in-kernel LUT 反量化——比官方 BF16 attention 慢。2bpw 的
   attention 是 8-bit 均匀码本（identity，免查表），反而轻。
3. **激活量化链**：我们 wxa8 每层 per-token-per-group 旋转+量化（已 hoist
   优化）；官方 dynamic fp8 是 per-tensor 缩放直进 tensor core。
4. **decode 差距小（1.4×）而 prefill 差距大（3.1×/卡）的机制**：decode
   单 token 只激活 6-8 个专家——sort/scatter 数据量与循环长度小一个量级，
   且我们小 B tile 配置是 P5-3 专门调过的；fp8 侧 decode 每步还付跨卡
   同步。prefill 是 256 expert 全激活、逐 expert 串行被放大的场景。

方向：杠杆在 grouped/合并 kernel 的**正确形态**（P5-2 重测挂在 backlog
§4.3，需带 P5-3 新配置），以及 4bpw 配置的 attention 均匀码本（4bit
Lloyd-Max 无法转 wxa8 是该配置特有的坑）。

## 3）我们的 2bpw 为什么比 4bpw 慢（decode -13%，仅 pf4096 +3%）

- decode 9.78 vs 11.27 tok/s；pf16 -4%、pf128 -2%；唯一 2bpw 赢的档是
  pf4096（2765 vs 2692，+3%）。
- **固定开销解释**：2bpw 是混合 1/2/4-bit——per-bit 分池 dispatch、
  1/2-bit 解包每 byte 处理 8/4 个元素（bit4 只 2 个）、混合池的 mask/sort
  更复杂；4bpw 全层统一 bit4，kernel 配置全层复用、每 byte 工作最少。
  decode 是开销主导场景，所以 4bpw 反而快。
- **带宽只在长 prefill 兑现**：pf4096 时 MoE 字节减半（4bpw 的 MoE packed
  ≈ 2bpw 的 2 倍）开始起作用，但只换回 3%——prefill 也大半被固定开销吃掉。
- **同 checkpoint 内部的旁证**：2bpw decode 排序 wxa16 10.91 > wxfp4 9.95
  > wxfp8 9.92 > wxa8 9.78——激活量化链在小 B 时净亏（wxa16 最快）；
  4bpw 则 wxa8/wxfp8 > wxa16——权重大了 A8 才开始净赚。两条曲线交叉，
  说明开销项（激活量化、低 bit 解包）与带宽项的平衡点随权重规模移动。
- **勘误**：throughput-bench §一 初版把 dec 列误读为 ms/tok 得出「低 bit
  无惩罚」，方向相反；正确结论是**低 bit 路径存在真实固定开销惩罚**。

## 4）GPTQ-Int4 为什么 prefill 快（2.06×）decode 慢（我们快 2.9×）

- **prefill 3298 vs 1603**：gptqmodel 把融合 experts 模块整体替换
  （日志 "Experts (before replacement) … Qwen3_5MoeExperts"）后走
  TritonV2 GEMM kernel——大 B（2048 行）时 GEMM 效率高、dequant 摊薄，
  attention 还是 BF16。即使 delta rule 跑 torch 参考实现拖后腿，仍赢 2×。
- **decode 3.93 tok/s（254ms/step）**：TritonV2 是为大 B 设计的 GEMM
  kernel，batch=1 时 tile 占用率极低 + 替换实现里的逐 expert dispatch；
  step1 0.258s ≈ 稳态 254ms，排除 JIT 编译因素，是稳态路径就这么慢。
  我们的 kernel 恰恰为小 B 优化（WXA8_MIN_B、P5-3 tile 表），decode 反超。
- **口径警示**：这是 transformers 生态（gptqmodel 7.3.6 TritonV2）的 Int4
  数字，**不代表官方 vLLM moe_wna16 的 Int4 上限**——官方 README 只推荐
  serving 引擎跑这个 checkpoint，vLLM 的 decode 会显著更好。对比时标注
  kernel 来源；如需 vLLM 口径另立环境补测。

## 汇总一图

| 差距 | 量化 | 主因 |
|---|---|---|
| FP8 vs 我们 4bpw（prefill） | 6.2×（单卡 3.1×） | grouped kernel vs per-expert 循环 + 双卡 + BF16 attention |
| FP8 vs 我们 4bpw（decode） | 1.4× | 同上但小 B 场景我们撑住更多 |
| 我们 4bpw vs 2bpw（decode） | 4bpw 快 13% | 低 bit 混合路径固定开销惩罚 |
| Int4 vs 我们 4bpw（prefill） | Int4 快 2.06× | TritonV2 大 B GEMM |
| Int4 vs 我们 4bpw（decode） | 我们快 2.9× | TritonV2 小 B 崩 + 我们小 B 专优 |

## 变更记录

- 2026-10-20：立项；四问分析成文（基于 1015/1018 系列日志与 WxA8 路线
  既有 profile 结论）；勘误 dec 单位误读。
