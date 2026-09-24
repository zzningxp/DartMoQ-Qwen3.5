# WxFP8 路线规划（2026-09-24）

> **定位**：WxFP8 **不是为了在 8-bit 上赢过 WxA8**——`wxa8-plan-260829.md:186-191` 已有决策记录， e4m3 在 kernel 速度（403 vs 526 TOPS）、激活精度（2.57% vs 0.65%）、码本精度（2.65% vs 0.53%） 三个轴上全败给 uniform-codebook int8。出自 test/test_wxa8_attn_fp8_spike.py（WxA8 P3 阶段在本机跑的决策 spike）。
> 本路线的动机是：**为 WxFP4 建立浮点量化的基础设施**（scale 语义、浮点码本转换、浮点激活存储与 kernel 结构），fp4 阶段直接复用。SM120 上 int4 没有 tensor core（WxA4 规划中的 WGMMA int4 路线在 Triton 上不可行），**FP4 是激活侧唯一还能"存储减半 + MMA 吞吐翻倍"的方向**；而 fp4 的 scale/存储/量化框架与 fp8 同构，先把 fp8 走通是风险最低的预备步骤。

---

## 一、调研结论（本机实测，2026-09-24）

环境：RTX 5090（sm_120，driver 570.169），torch 2.9.0+cu128，triton 3.5.0，conda dart312。

### 1.1 硬件/软件能力实测表（探测脚本：`test/test_wxfp8_capability_probe.py`）

| 能力 | 状态 | 实测数据 |
|---|---|---|
| Triton `tl.dot(fp8e4m3, fp8e4m3)` | ✅ | 353 TFLOPS @4096³（int8 448 / bf16 190，即 fp8 ≈ 0.79×int8 ≈ 1.86×bf16） |
| Triton `tl.dot(fp8, bf16)` 混合 | ❌ | `Unsupported lhs dtype fp8e4nv`，不允许混合 dtype dot |
| Triton `tl.dot_scaled(e2m1,e2m1)`（mxfp4） | ⚠ 仅功能 | 112 TFLOPS @4096³，低于 bf16 → 确认走**软件模拟**（upcast bf16），非原生 MX MMA |
| cuBLASLt fp8（`torch._scaled_mm`）tensorwise | ✅ | 377 TFLOPS @4096³，rel_err 0.038 |
| cuBLASLt fp8 rowwise | ✅ | rel_err 0.037 |
| cuBLASLt fp8 **1x128 blockwise**（DeepSeek 风格） | ❌ | sm_120 校验不通过（报错枚举中无有效布局组合走通） |
| cuBLASLt **mxfp8**（1x32 + e8m0 scale） | ⚠ 存疑 | 配置被接受、能出结果，但 rel_err 0.68 异常偏大 → 疑 scale 被按 swizzle 块布局解释，eager 侧无配套 swizzle 工具 |
| cuBLASLt **nvfp4**（1x16 + e4m3 scale） | ⚠ 未走通 | dtype `float4_e2m1fn_x2` 已注册、`_scaled_mm` 错误信息枚举了该配置；但 eager `copy_` 未实现（无法从 float 转换）、4 种布局枚举均被形状校验拒绝 → 布局约定待考古（fp4 阶段任务） |
| cuBLASLt int8（`_scaled_mm`） | ❌ | 不支持（无冲突：当前 WxA8 int8 走自写 Triton kernel） |
| torch eager `.to(float8_e4m3fn)` | ⚠ 有坑 | 越界（>448）产生 **NaN 而非饱和**（500→nan） |
| Triton `.to(tl.float8e4nv)` | ✅ | **饱和到 ±448**（satfinite），kernel 内量化天然安全 |

关键结论：
1. **主路径仍是 Triton**——fp8 `tl.dot` 在 sm_120 + triton 3.5.0 原生可用，走真 fp8 MMA（1.86×bf16
   可证），且与 WxA8 同技术栈，kernel 结构可平移。
2. **fp8 在 Triton 里比 int8 慢 ~20%**（353 vs 448）——与 wxa8-plan 旧记录一致，接受（fp4 铺路定位）。
3. fp4 的三条候选路径全部不成熟：Triton `dot_scaled` naive 用法仅模拟（112 TFLOPS，无 TMA 所致，
   §1.2）、cuBLASLt nvfp4 eager 未走通、CUTLASS 需大工程 → **佐证"先 fp8 铺基础设施、fp4 路径
   边走边定"的路线**（fp4 四条候选路线详见 §1.2b，其中 Triton TMA+persistent 与 TE 两条可行性
   高于最初预期）。

### 1.2 生态调研（除 Triton 外的库；含 web 调研，来源见文末）

| 库 | sm_120 (5090) 状态 | 对本项目的可用性 |
|---|---|---|
| cuBLASLt（经 `torch._scaled_mm`） | ✅ fp8 tensorwise/rowwise 实测可用；CUDA 12.9+ 另有 16/32/128 元素 1D 块缩放与 outer-vector fp8 缩放 | perf 参照系（实测 377 TFLOPS > Triton 353）；注意 Cloudrift 报告 cuBLAS 在 sm_120 曾欠调优（~60% 峰值） |
| CUTLASS | ✅ 3.9.0（2025-04-24）起支持 SM120 kernel；SM12x block-scaled（MXFP8/NVFP4）kernel 存在，Colfax 有完整教程系列；⚠ SM120 smem 仅 ~100 KiB（SM100 tile 配置不可直接复用） | 后备；dense fp8 可用，grouped GEMM（MoE）在 sm_120 尚不成熟（vllm#43814 未合）；fp4 阶段的主要蓝图 |
| DeepGEMM（DeepSeek） | ❌ 官方仅 SM90/SM100（#236：维护者无 sm_120 设备、无计划）；社区 PR #447 在侧分支待合 | 不可用；其 1x128 激活 + 128x128 权重的 scale 方案仍是设计参考 |
| Marlin（vLLM） | fp8 Marlin = **W8A16 权重-only**，不是激活 fp8 路线；NVFP4 Marlin 可在 SM120 跑 | 非 wxfp8 候选 |
| TensorRT-LLM | ✅ v0.17.0 起官方支持 RTX 50 系；fp8 w8a8 与 fp4/nvfp4 均可在 sm_120 跑 | engine 级，非 kernel 库；证明 NVIDIA 自家 sm_120 fp8/fp4 kernel 存在且快 |
| vLLM | ✅ block fp8（per-128）SM120 kernel 已合（v0.10.1，#22131）；且正在调优 **Qwen3.5 专用** fp8 GEMM（#54182） | 佐证 per-128 块缩放 fp8 在 sm_120 的 CUTLASS 路线可行；int8 w8a8 被 vLLM 政策性关闭（非硬件限制） |
| TransformerEngine | ✅ v2.15（2026-05）起 sm_120 硬件 NVFP4 可用（实测 1.77× bf16，TE#2968）；fp8 recipe 包装 `_scaled_mm` | **fp4 阶段的现成库候选**（"consumer Blackwell 上唯一有硬件 fp4 加速的上游路径"，forgather#38） |
| FlashInfer | ✅ SM120/121（fp8/nvfp4 KV cache + GEMM） | attention 侧远期候选 |
| torchao | fp8 rowwise 同 `_scaled_mm`；mxfp4/nvfp4 路径 gated 到 SM100+ | 量化策略参考，不直接引入 |

**Triton 版本关键事实**（triton#7188）：fp8 `tl.dot` 在 **3.3.x 会静默降级 fp16 MMA**（PTX 里是
`mma...f16.f16`），**3.4.0 起才是原生 `mma.sync...e4m3.e4m3`**（m16n8k32 族，与 int8 同族）。
本机 triton 3.5.0 ✓（1.86×bf16 的实测比例亦印证原生路径）。⚠ 若环境里残留 `pytorch-triton`
遮蔽新版会重新触发降级——运行时检查 `triton.__version__`。

**Triton MX/fp4 路径的关键事实**（triton#8548, forgather#38）：`tl.dot_scaled` 在 sm_12x 上
**必须 TMA descriptor + persistent 调度才走原生块缩放 MMA，否则静默 bf16 模拟**——这正是
本机探测 112 TFLOPS（< bf16 190）的原因：naive kernel 无 TMA。vLLM #31089 的 SM120 MXFP4
Triton GEMM（TMA+persistent）是可参考的实现。另：Triton `dot_scaled` 只支持 MX（1x32 + e8m0），
**不支持 NVFP4**（1x16 + e4m3 scale）。

### 1.2b FP4 路线前瞻（WF-6 的输入，本阶段不决策）

5090 硬件有 FP4 tensor core（dense ≈ 2× fp8），但 sm_120 **没有 tcgen05/TMEM**（FA4、SM100
kernel 均不可用），块缩放 MMA 要求编译目标 **sm_120a**（PyTorch 自动 gencode 会丢 `a` 后缀，
pytorch#172807——经 stock PyTorch 走 fp4 很脆）。可用路线按工程量排序：

1. **Triton TMA + persistent MXFP4**（vllm#31089 参考）——与我们技术栈最连续；
2. **CUTLASS SM12x block-scaled**（Colfax 教程为蓝图，`mma.sync .kind::mxf8f6f4/.nvf4`）；
3. **TransformerEngine v2.15 NVFP4 recipe**——现成库，1.77× bf16 实测，但引入 TE 依赖；
4. cuBLASLt 块缩放 fp4（CUDA 12.9+；本机 torch eager 的 `_scaled_mm` fp4 仍未走通，§1.1）。

### 1.3 与 WxA8/WxA16 的关系（本次调研的核心问题）

**Q2：从 A16→A8 调过一次量化过程（码本），FP8 是否还要对应改一次？——是，且机制完全同构。**

A16→A8 改了什么（`triton_kernels_a8.py`）：
- `build_int8_codebook`（:526-550）：fp16 码本 → int8 网格，`cb_step = max|cb|/127` 折进激活
  scale（`extra_scale` 通道，`wxa8/bit_partitioned_moe.py:73-78`）
- attention W8 特殊处理：Lloyd-Max 256 级码本 → 均匀 int8 网格会塌成 ~195 级（W8→W6），
  解决方案是 **uniform 码本 + IDENTITY_CB**（`idx - 128` 免查表，`triton_kernels_a8.py:120-124`）

FP8 对应改造（WxFP8 要做的）：
- `build_fp8_codebook`：fp16 码本 → e4m3。**浮点网格尺度不变 → 单一 cb_scale 可搜索**，
  让少量质心尽量落在 e4m3 网格点上（int8 版无此自由度）。WF-1 已实测（§七）：
  1-bit 精确、2-bit 0.00021（优于 int8 的 0.00054）、4-bit 0.011（int8 0.0036，差 3 倍
  但绝对值小）、`cb_scale` 折进激活 scale（同 `cb_step` 机制，复用 `extra_scale` 通道）
- **W8 uniform 码本 → e4m3 已实测确认是精度风险点**：码本占用 256→81 Byte（e4m3 有限值总共仅
  254 个有限幅度，结构性塌级），码本 relerr 0.0231，attention 路径总 relerr 0.037
  vs wxa8 的 0.009（~3.9×）→ 支持"attention 保 wxa8、MoE 上 wxfp8"的混合部署备选，
  最终由 WF-5 ppl 决定
- 结论：**码本需要一次 fp8 化改造，位置与 `build_int8_codebook` 对称（load 时），checkpoint 不动**

**Q3：存储是否独立？——持久化存储不独立，运行时产物独立。**

- **权重 checkpoint（持久化）完全共享**：packed uint8 indices + fp16 码本 + fp16 norms，
  A16/A8/FP8(/FP4) 同一份文件（`quantization/wxa8/__init__.py:1-4` 已明确此原则）。
  fp8 化与 int8 化一样发生在 **load 时**，不落盘。
- **激活（transient，本就不持久化）各自独立**：A8 = int8 + fp16 per-group scale；
  FP8 = e4m3 + per-group scale；FP4 = packed e2m1 + 块 scale（32 或 16 元素/块）。
  这不是"独立一套存储系统"，只是 kernel 输入 tensor 的 dtype 不同。
- **kernel 文件独立**（惯例）：`triton_kernels_a8.py` → `triton_kernels_fp8.py`；
  `quantization/wxa8/` → `quantization/wxfp8/`（`linear.py` + `bit_partitioned_moe.py` 平移结构）。
- **FP4 阶段才出现真正的存储分化**：若走 Triton 模拟路径，激活 packed e2m1 只是 transient；
  若走 cuBLASLt nvfp4，则 scale 需要 swizzle 布局——那是 fp4 阶段的决策，fp8 阶段不预设。

### 1.4 待确认清单（web 调研返回后更新）

- [x] DeepGEMM sm_120 支持状态 → **不支持**（#236，维护者无计划；社区 PR #447 待合）
- [x] Triton 后续版本 sm_120 原生 MX 支持 → **有条件支持**：需 TMA + persistent（triton#8548）；
      `dot_scaled` 不支持 NVFP4（仅 MX 1x32 + e8m0）；本机 3.5.0 落后当前 3.8.0 一年，升级与否
      留作 fp4 阶段决策（遵守"不降级"原则，升级需独立验证）
- [ ] cuBLASLt nvfp4 的 eager/PTX 级布局约定（fp4 阶段任务；TE/CUTLASS 路线可能绕开）
- [ ] mxfp8 `_scaled_mm` 的 scale swizzle 要求（仅当 fp8 想用 1x32 块 scale 时才需要，非主路径）

---

## 二、技术方案（WxFP8）

### 数据流（一层 MoE 为例，★ = 相对 WxA8 的改动点）

```
上一层输出 (FP16)
    ↓
[分组旋转]  per-group QR 正交旋转（group_size=128）            ← 不变
    ↓
[激活量化]  per-token per-group 对称量化 → ★ E4M3 + scale
    scale = amax(group)/448（e4m3 max），Triton satfinite 饱和   ← 改输出 dtype 与 scale 语义
    ↓
[矩阵乘法]  ★ E4M3 激活 × E4M3 码本权重 → tl.dot 原生 FP32 累加
    权重 unpack → ★ 查 E4M3 码本（或 W8 uniform 的 e4m3 LUT）    ← 改码本路径
    ↓
[反量化]   ★ acc_fp32 × act_scale[b,g] × norms_gf[n,g] → FP16
    （含 cb_scale，同 extra_scale 机制；无 int32→fp32 组边界转换） ← 结构不变，去 int32 技巧
    ↓
[非线性/下一层]                                               ← 不变
```

### 核心要点

- **A-FP8 = 参与矩阵运算的激活输入是 e4m3**（e4m3 而非 e5m2：尾数多 1 bit、max 448 配合
  satfinite 更适合激活；e5m2 留作离线对照）
- **累加器从"int32 组内 + fp32 组间"简化为原生 fp32 dot 累加**——`tl.dot(fp8,fp8)` 默认
  fp32 acc，组边界 epilogue 仍乘 `xs_g * norm_g`，`_wxa8_fused_matmul_kernel_grouped_gf`
  的骨架（行/列切片、hoisted 旋转、per-expert 三量化点）原样平移
- **量化点不变**：gate_up hoisted（`_build_hoisted_rotations`）、down per-expert、attention
  linear（`WxA8Linear.forward` 结构）——`_rotate_quantize_kernel` 输出改 e4m3，
  `quantize_act_per_token_group` 参考实现同步改
- **scale 粒度 = group 128**（与旋转/norms/现有 epilogue 对齐）；1x32 mx 风格留作对照实验
  （cuBLASLt mxfp8 的 swizzle 问题未解，Triton 侧 dot_scaled 无加速，不作为主路径）
- **速度预期**：GEMM 约 0.79×int8（353/448）→ wxfp8 全链路预计略慢于 wxa8，约 -10~-20%；
  这是铺路成本，fp4 阶段回收（fp4 存储减半 + 若走通原生 MMA 再翻倍）

### 精度预期与风险

| 项 | WxA8 实测 | WxFP8 实测（WF-1） | 风险 |
|---|---|---|---|
| 激活 relerr | 0.65%（int8 per-group） | 2.57%（e4m3 per-group，复现 spike） | **主风险**，~4× 退化 |
| 2-bit 码本 relerr | 0.00054 | **0.00021**（scale 搜索有效，优于 int8） | 已消除 |
| 4-bit 码本 relerr | 0.0036 | 0.011（naive 0.025 的 2.3 折改善） | 低（绝对值小，且 4-bit 只覆盖少量 expert） |
| W8 uniform 码本 | 0%（int8 精确映射） | 0.0231（码本占用 256→81 Byte，结构性） | 中-高，attention 敏感 → 混合部署备选 |
| MoE 总 relerr（2-bit，Gaussian 模拟） | 0.0065 | 0.0257（激活主导，码本贡献 0.0003） | ppl 由 WF-5 定 |
| attention 总 relerr（8-bit uni，模拟） | 0.0095 | 0.0367（~3.9×） | 同上；不可接受则 attention 保 wxa8 |

---

## 三、实施步骤

| 阶段 | 内容 | 产出 | 验证方式 |
|---|---|---|---|
| WF-0 | 本调研 + 能力探测脚本固化 | 本文件 + `test/test_wxfp8_capability_probe.py` | 已跑通（本文表格即输出） |
| WF-1 | `build_fp8_codebook`（低 bit + W8 uniform LUT）+ 离线精度表 | `triton_kernels_fp8.py` 码本工具 | `test/test_wxfp8_codebook_prec.py`（对照 wxa8-plan:531-542 的 relerr 表格式） |
| WF-2 | `_rotate_quantize_kernel` fp8 版（e4m3 输出 + satfinite + scale=amax/448） | 同上文件 | `test/test_wxfp8_act_quant_prec.py`（relerr + 饱和边界用例） |
| WF-3 | wxfp8 GEMM kernels：先 attention 全矩阵，后 MoE gate_up/down 三量化点 | `triton_kernels_fp8.py` + tile 配置表 | `test/test_wxfp8_gemm_align.py`（与 fp16 参考逐 tile 对齐 + GPU 填充度/SM 占用分析，遵守 kernel 测试规范） |
| WF-4 | `quantization/wxfp8/`（linear.py + bit_partitioned_moe.py）+ `--inference-quant-mode wxfp8` 入口 | 包 + run/eval 入口接线 | 单层 forward 对齐测试 |
| WF-5 | 全模型 ppl 验证 | 结果记录进本文件 | **手动**：`eval_qwen35.py`（命令见 §四） |
| WF-6 | （展望）WxFP4：kernel 路径三选一（cuBLASLt nvfp4 布局考古 / CUTLASS / Triton 模拟仅存储收益） | 新规划文件 | — |

> 遵守项目规范：不自定版本号；每阶段 relerr/耗时数据记录在本文件，git 由本人操作。
> **执行日志规则**：每完成一个阶段（或阶段内值得记录的中间结果），必须在 §七 执行日志
> 对应小节追加一条（日期 + 做了什么 + 实测数据 + 结论/下一步），再给一段 git comment 的信息，当天完成当天记。

---

## 四、手动测试命令

> 小显存测试（能力探测、码本精度、后续 kernel 对齐等）由 Claude 自跑并在 §七 记录数据，
> 各测试脚本头部有 usage 行可随时复跑。此处只列**需要本人手动跑的大显存命令**。

```bash
# WF-5 全模型 ppl（占大量显存，本人手动跑；混合部署：MoE fp8 + attention wxa8）
conda run -n dart312 eval_qwen35.py models/qwen3.5-2bpw-260831-u8 --inference-quant-mode wxfp8

# 对照组（如需同批比较）
conda run -n dart312 eval_qwen35.py models/qwen3.5-2bpw-260831-u8 --inference-quant-mode wxa8
conda run -n dart312 eval_qwen35.py models/qwen3.5-2bpw-260831-u8 --inference-quant-mode wxa16
```

---

## 六、参考来源（web 调研）

- Triton fp8 sm_120 修复与版本线：triton#7188（3.3.x 静默 fp16 降级 → 3.4.0 原生 e4m3 MMA）
- Triton MX 需 TMA+persistent：triton#8548、forgather#38（GB10/sm_121，Triton 3.6 实测）
- vLLM SM120 MXFP4 Triton GEMM：vllm#31089；SM120 block fp8 per-128：vllm#22131（v0.10.1）；
  Qwen3.5 fp8 GEMM SM120 调优：vllm#54182；SM120 grouped GEMM 未合：vllm#43814/#43507
- DeepGEMM 不支持 sm_120：DeepGEMM#236；社区 sm_120a PR：DeepGEMM#447（侧分支）
- `_scaled_mm` 1x128 blockwise 为 sm90/sm100 CUTLASS 专属：pytorch#130359 背景；
  公共 API RFC：pytorch#157950；gencode 丢 `a` 后缀：pytorch#172807
- cuBLAS 12.9 块缩放：NVIDIA blog "Boosting Matrix Multiplication ... with cuBLAS 12.9"；
  sm_120 欠调优：Cloudrift 博客
- CUTLASS SM120 支持：CUTLASS changelog 3.9.0（2025-04-24）；NVFP4 SM12x 蓝图：
  Colfax "NVFP4 Blockscaled GEMM on RTX PRO Blackwell (SM12x)" 及其优化续篇（2026-08）
- TransformerEngine sm_120 NVFP4：TE#2956/#2968（v2.15，1.77× bf16）
- TensorRT-LLM RTX 50 系支持：v0.17.0 release notes、TRT-LLM#5018
- FP8/e4m3 数值与 PTX cvt.rn.satfinite：OCP FP8 spec（arXiv:2209.05433）、PTX ISA
- 5090 FP4 硬件与 sm_120 缺失 tcgen05/TMEM：NVIDIA RTX Blackwell 架构白皮书

## 五、变更记录

- 2026-09-24 WF-0：初版调研 + 本机能力实测 + web 生态调研（本文件）。
- 2026-09-24：§三 补执行日志规则，新增 §七 执行日志区（本人要求：每做一步更新日志）。
- 2026-09-24 WF-1：`build_fp8_codebook` 落地 + 码本/端到端精度实测（数据见 §七 WF-1）；
  §1.3 与精度表按实测修正（4-bit 预测过乐观、W8 uni 塌级确认）。
- 2026-09-24 WF-2：`_rotate_quantize_kernel_fp8` 落地 + 对拍/边界/填充度实测（§七 WF-2）。
- 2026-09-24 WF-3：融合 GEMM kernel + attention 路径落地，contract 对拍 2.1e-04，
  MoE 形态 fp8/int8 = 0.89~1.00x（速度中性），attention 0.58x（LUT gather 代价）
  → 混合部署证据链闭合（§七 WF-3）。
- 2026-09-24 WF-4：`quantization/wxfp8/` 包 + 三处入口接线 + convert_model_to_wxfp8
  （默认混合部署）；test_quant_io 追加对拍段全绿（MoE 0.05042 / attn 0.03695，
  均与预测一致）（§七 WF-4）。
- 2026-09-24 WF-5：全模型 ppl（本人手动跑）：wiki +0.011 / c4 +0.014 vs wxa16
  （代价远小于预期，ppl 门槛通过）；c4 速度与 wxa8 持平，wiki 疑首轮 JIT 污染
  待复跑确认（§七 WF-5）。**wxfp8 六步全部完成**。

---

## 七、执行日志

> 条目格式：`- 日期 事项：做了什么 → 实测数据 → 结论/下一步`。数据必须抄实测输出，不写主观估计。

### WF-0：调研 + 能力探测 ✅ 已完成

- 2026-09-24 能力探测：固化 `test/test_wxfp8_capability_probe.py` 并跑通 →
  §1.1 表格全项（tl.dot fp8 353 TFLOPS / int8 447 / mxfp4 dot_scaled 110=模拟 /
  _scaled_mm tensorwise+rowwise OK、1x128 ❌、mxfp8 存疑 0.68、nvfp4 未走通 /
  torch NaN vs Triton satfinite）→ 关键结论见 §1.1"关键结论"三条。
- 2026-09-24 web 生态调研：DeepGEMM ❌ sm_120、Triton MX 需 TMA+persistent、TE v2.15 NVFP4 可用 →
  §1.2/§1.2b/§六。
- 2026-09-24 姊妹文件：`wxfp4-plan-260924.md`（下游预研规划）。

### WF-1：码本 fp8 化 ✅ 已完成（2026-09-24）

- 2026-09-24 `build_fp8_codebook`：`turboquant_utils/triton_kernels_fp8.py` 落地。
  与 `build_int8_codebook` 同契约（`cb ≈ lut × cb_scale`，折 extra_scale），
  新增 **cb_scale 对数网格搜索**（coarse 512 点 ±1 octave + 两轮细化，load 时一次性，
  浮点网格尺度不变性带来的 int8 没有的优化自由度）。
- 2026-09-24 码本 relerr 实测（`test/test_wxfp8_codebook_prec.py`，口径=‖lut×s−cb‖/‖cb‖）：

  | 码本 | int8 | fp8 naive | fp8 opt | int8 占用(Byte) | fp8 占用(Byte) |
  |---|---|---|---|---|---|
  | 1-bit LM | 0 | 0 | **0** | 2 | 2 |
  | 2-bit LM | 0.00054 | 0.01347 | **0.00021** | 4 | 4 |
  | 4-bit LM | 0.00360 | 0.02523 | **0.01097** | 16 | 16 |
  | 8-bit LM | 0.00527 | 0.02631 | 0.02518 | 195 | 86 |
  | 8-bit uni | 0.00394 | 0.02554 | **0.02305** | 254 | 81 |

- 2026-09-24 端到端模拟（Gaussian B2048 K2048 N1024 group128，参考=理想量化计算；
  锚点全对上：激活 int8 0.00646≈0.0065、LM8→int8 0.00527≈0.0053、
  8bitLM 总 0.01277≈wxa8-plan 0.0128）：
  - 2-bit（MoE 主力）：wxa8 0.00650 / wxfp8 0.02573（激活主导，码本贡献仅 0.00028）
  - 4-bit：wxa8 0.00859 / wxfp8 0.03062
  - 8-bit uni（attention 现役）：wxa8 0.00948 / wxfp8 0.03669（**3.9×，码本占用 256→81 Byte**）
- 结论：①MoE 权重侧 fp8 化基本免费（2-bit 甚至优于 int8），2bpw checkpoint 可直接用；
  ②attention W8→e4m3 有实质退化，混合部署（attention 保 wxa8）列为 WF-5 后的备选决策；
  ③激活 e4m3 2.57% 是全局主误差源，ppl 影响待 WF-5。下一步 WF-2（激活量化 kernel）。

### WF-2：激活量化 fp8 kernel ✅ 已完成（2026-09-24）

- 2026-09-24 `_rotate_quantize_kernel_fp8` + `rotate_quantize_fused_fp8` +
  `quantize_act_per_token_group_fp8`：与 int8 版（`triton_kernels_a8.py:428`）逐行对应，
  只有量化尾不同（`round+clamp+int8` → `.to(tl.float8e4nv)`，satfinite 饱和天然安全，
  无需 clamp）。scale=amax/448，EXTRA_SCALE 折叠机制不变。
- 2026-09-24 实测（`test/test_wxfp8_act_quant_prec.py`）：
  - 锚点复现：int8 0.00646 / e4m3 0.02573
  - 融合 kernel 对拍：relerr 0.02573 == 参考实现；e4m3 字节差 37/4194304
    （旋转 fp32 求和顺序差导致的 RTNE 边界翻转）；scale 最大差 2.4e-7；无 NaN/Inf
  - 边界用例：全零行 dequant 精确为 0；1e4 离群值行 relerr 0.026（旋转摊平 →
    QuaRot 效应符合设计）；幅度 1e-3 行（scale 深入 fp16 次正规区）relerr 0.0253
    —— **fp16 次正规 scale 存储实测不是问题**（失真低于量化噪声底）
  - GPU 填充度：B=2048 驻留 50%（1 wave）1.00 TB/s；B=16384 驻留 100%（4 wave）
    1.38 TB/s（峰值的 77%，带宽受限符合预期，与 int8 版同结构同量级）
- 结论：激活量化 fp8 化闭环，可直接供 WF-3 的 GEMM kernel 消费。下一步 WF-3。

### WF-3：wxfp8 GEMM kernels ✅ kernel + attention 路径完成（2026-09-24）

- 2026-09-24 `_wxfp8_fused_matmul_kernel_grouped_gf` + `_launch_fp8` +
  `wxfp8_matmul_grouped_gf`（attention 全矩阵）：与 int8 版
  （`triton_kernels_a8.py:35`）逐行对应；累加器改 group 内原生 fp32 dot；
  无 IDENTITY_CB（e4m3 网格对 idx 非线性，W8 也必须查 LUT）。
  实现细节：fp8 指针的 masked load 不能给整型 `other`（int32→fp8e4nv
  不可转换）→ 不给 other（被 mask 行列不写出，安全）。
  tile 配置表先以 WxA8 表为种子（`_WXFP8_CONFIG_*`，待扫回填）。
- 2026-09-24 实测（`test/test_wxfp8_gemm_align.py`）：
  - contract 对拍（torch fp32 复算 kernel 数学契约）：
    bit=2/4/8×4 形态全部 **2.1e-04**（fp16 scale 存储 + 累加序差异）；vs-理想量化 0.00021；
    vs-fp32 与 WF-1 模拟一致（2bit 0.341 / 4bit 0.101 / attn 0.038）
  - tile 扫描：6 组配置全部对齐（relerr 恒 2.07e-04）；attn 最优 (128,128,128,8,3) 223 TFLOPS
  - **性能定位（关键发现）**：
    | 形态 | fp8 | int8 | 比值 | 原因 |
    |---|---|---|---|---|
    | attn B16384 N4096 K2048 bit8 | 223 TF | 385 TO | **0.58x** | int8 走 IDENTITY_CB 免查表，fp8 必须逐元素查 256 项 LUT |
    | MoE B2048 N1024 K2048 bit2 | 177 TF | 200 TO | **0.89x**（最优 cfg；1.00x@128,64,128） | 双方都查 LUT，公平 |
  - 填充度：attn 最优 cfg 4096 CTA / 170 SM = 24.1 CTA/SM，驻留 100%（4 wave）
- 结论：①kernel 数学正确性闭环；②**MoE 路径 fp8 速度中性（0.89~1.00x）**，
  达到"fp4 铺路"的设计预期；③attention W8 fp8 三输（速度 0.58x + 精度 3.9x +
  LUT 复杂度）——**混合部署（attention 保 wxa8、MoE 上 wxfp8）证据链闭合**，
  最终由 WF-5 ppl 确认。MoE 切片 wrapper（gate_up 行切片 / down in_features
  切片）随 WF-4 包接线一起做（kernel 本身已支持切片语义）。

### WF-4：`quantization/wxfp8/` 包 + 入口接线 ✅ 已完成（2026-09-24）

- 2026-09-24 落地清单：
  - `turboquant_utils/triton_kernels_fp8.py`：补 gate_up 行切片 / down in_features
    切片两个 wrapper（`wxfp8_matmul_grouped_slice_rows_gf` /
    `wxfp8_matmul_grouped_slice_in_features_gf`，与 wxa8 对应版本逐行同构）
  - `quantization/wxfp8/`（新包）：`WxFP8BitPartitionedGroupMoE`（覆盖
    `_get_bit_context`/`_build_hoisted_rotations`/两个 matmul，与 wxa8 包逐点
    对应）+ `WxFP8Linear`（attention；e4m3 LUT 对 Lloyd-Max 码本也开放——
    int8 路线的均匀性安全阀在 fp8 下不需要；cb_scale 在 _ensure_gf 缓存）
  - `qwen35_quant_io.py`：`convert_model_to_wxfp8(model, attn="wxa8")`——
    **默认混合部署**（MoE fp8 + attention 保 wxa8，依据 §七 WF-3 的速度 0.58x
    + 精度 3.9x 双输）；attn="wxfp8" 供 full-fp8 对比（CLI 未暴露，研究用改一行）；
    `load_quantized_model` 的 mode 分支接线
  - `run_qwen35.py` / `eval_qwen35.py`：`--inference-quant-mode` choices 增加
    `wxfp8`；`qwen35_simple_wrapper.py` 内存内 eval 路径同步
- 2026-09-24 实测（`test/test_quant_io.py` 追加 WxFP8 段，已有断言未动）：
  - WxFP8 MoE forward relerr vs A16 = **0.05042**（预测 ~0.05：激活 2.57% ×
    silu/topk 放大 ≈2x；对照 A8 0.65%→0.01336 的同比例关系）✓
  - full-fp8 attn linear relerr = **0.03695**（WF-1 模拟预测 0.0367）✓
  - 转换链路：WxA16→WxA8→WxFP8 连续 __class__ 切换（零拷贝）验证通过
- 结论：wxfp8 全链路（checkpoint→加载→转换→forward）闭环，等价于 wxa8 当年
  P2 完成时的状态。下一步 WF-5：全模型 ppl（本人手动跑，命令见 §四）。

### WF-5：全模型 ppl 验证 ✅ 已完成（2026-09-24，本人手动跑，git 870e1f2+）

| 模式 | wiki ppl | c4 ppl | wall(s) | t_wiki | t_c4 |
|---|---|---|---|---|---|
| wxa16 | 7.7955 | 11.2631 | 134.15 | 53.13 | 81.02 |
| wxa8 | 7.7947 | 11.2676 | 106.58 | 44.16 | 62.42 |
| **wxfp8（混合）** | **7.8064** | **11.2769** | 118.36 | 55.43 | 62.93 |

- **精度结论：fp8 激活的 ppl 代价很小**——vs wxa16：wiki +0.0109（+0.14%）/
  c4 +0.0138（+0.12%）；vs wxa8：wiki +0.0117 / c4 +0.0093。
  对照当初悲观预测（0.1~0.5）实际只有 ~0.01 量级，与激活 relerr 比
  （2.57% vs int8 0.65%，约 4x）的放大远小于预期——旋转对离群值的抑制
  在 ppl 层面兑现。**float 激活方向通过 ppl 门槛**。
- **速度结论：c4 与 wxa8 持平（62.93 vs 62.42，+0.8%）**，符合 kernel 级
  0.89~1.00x 预测；wiki 55.43 异常偏慢（比 wxa8 +25%，甚至慢于 wxa16）——
  **疑为 wxfp8 新 kernel 首轮 JIT 编译落在了 wiki 阶段**（wiki 先跑，
  wxa8/wxa16 的 kernel 已有 triton cache 而 wxfp8 全新），待热缓存复跑确认。
- 混合部署最终判定：**成立**（MoE fp8 + attention wxa8）。wxfp8 作为
  WxFP4 基础设施的任务完成，WF4-P1 前置探测可以启动。
- 待办：热缓存复跑一次 wxfp8 确认 wiki 时间（预期回落到 ~46s / 总 ~108s）。
