# WxFP4 路线规划（2026-09-24）

> **定位**：WxFP4 是 WxFP8（`wxfp8-plan-260924.md`）的下游阶段。动机：
> SM120 上 int4 没有 tensor core（README 里"WxA4 = WGMMA int4"的旧设想在 Triton 上不可行，
> 且 sm_120 无 tcgen05），**FP4 是激活侧唯一还能"存储减半 + MMA 吞吐再翻倍"的方向**
> （5090 FP4 dense ≈ 2× FP8 ≈ 4× FP16）。
> **依赖关系**：fp4 的 scale 语义、浮点码本转换、量化点接线全部复用 wxfp8 的基础设施
> （WF-1~WF-5），本文件在 wxfp8 落地前只做前置探测，不动主流程。
>
> ⚠ 本文件是**预研规划**，多处决策点（§三）显式留白，等 wxfp8 的实测数据回来再定。

---

## 一、能力事实（2026-09-24，实测 + web 调研，详见 wxfp8-plan §1.1/§1.2/§1.2b）

| 项 | 事实 | 来源 |
|---|---|---|
| 5090 FP4 硬件 | ✅ 5th-gen tensor core，FP4 dense = 2× FP8；无 tcgen05/TMEM（FA4、SM100 kernel 不可用）；smem 仅 ~100 KiB | RTX Blackwell 白皮书 |
| 块缩放 MMA 指令 | `mma.sync .kind::mxf8f6f4`（MXFP8/6/4，e8m0 scale）与 `.kind::nvf4`（NVFP4，e4m3 scale），**要求编译目标 sm_120a**（带 `a` 后缀） | PTX ISA；Colfax SM12x 系列 |
| PyTorch gencode 坑 | PyTorch 自动 gencode 丢 `a` 后缀 → stock PyTorch 走块缩放 fp4 很脆（`_scaled_mm` fp4 在 cu128/5090 eager 未走通，实测） | pytorch#172807；本机探测 |
| Triton `tl.dot_scaled` | ✅ 功能可用（实测 e2m1×e2m1 数值正确）；**naive 写法 110 TFLOPS = bf16 软件模拟**；需 **TMA descriptor + persistent 调度**才走原生块缩放 MMA；**只支持 MX（1x32+e8m0），不支持 NVFP4** | 本机探测；triton#8548；forgather#38 |
| Triton 参考实现 | vLLM #31089：SM120 MXFP4 Triton GEMM（TMA+persistent），可作为 kernel 蓝图 | vllm#31089 |
| CUTLASS | ✅ 3.9.0+ 支持 SM120，SM12x block-scaled（NVFP4）kernel + Colfax 完整教程 | CUTLASS changelog；Colfax |
| TransformerEngine | ✅ v2.15 起 sm_120 硬件 NVFP4 可用，实测 1.77× bf16（consumer Blackwell 唯一上游可用路径） | TE#2956/#2968；forgather#38 |
| cuBLASLt | CUDA 12.9+ 提供 FP4 1D 块缩放（16/32/128 元素）；eager 侧布局约定未考古 | NVIDIA blog；本机探测（未走通） |
| DeepGEMM | ❌ sm_120 官方不支持；社区 PR #447（侧分支）有 sm_120a dense+grouped BF16/FP8/FP4（**含 mixed FP8×FP4**） | DeepGEMM#236/#447 |

## 二、格式与数值

- **e2m1 网格**：±{0, 0.5, 1, 1.5, 2, 3, 4, 6}——只有 **7 个非零幅度**。逐元素误差远大于
  e4m3（e4m3 有 3-bit 尾数），**必须配块缩放**才能用：
  - **MXFP4**：32 元素/块，e8m0（2 的幂）scale —— Triton `dot_scaled` 唯一支持的格式
  - **NVFP4**：16 元素/块，e4m3 scale + 全局 fp32 scale —— 精度更好，TRT-LLM/TE 主推，
    但 Triton 不支持，需 CUTLASS/TE
- **本项目利好**：已有 per-group(128) QR 旋转把激活压近高斯、抑制离群值——这正是 fp4 激活
  量化最依赖的前置条件（QuaRot/SpinQuant 系的核心思想，我们已经在做）。128 的 group 内
  均匀再切 4×32 或 8×16 个块，结构与旋转/norms 天然对齐
- **精度预期**：fp4 激活 relerr 待实测（类比：int8 per-128 是 0.65%，e4m3 per-128 是 ~2.5%，
  e2m1+块缩放预计 8~20% 量级，**这是本路线最大风险**）；W 侧是否受损取决于 §三决策点 2

## 三、决策点（等 wxfp8 数据后定，现在只列选项）

**DP-1：格式 —— MXFP4 还是 NVFP4？**
- MXFP4：Triton 可达（TMA+persistent），栈连续性最好，工程量中
- NVFP4：精度上限高，但只能 CUTLASS（大工程）或 TE（引依赖）
- 倾向：**先 MXFP4 走通 Triton 路线**（与"科研加速探索"的项目定位和技术栈复用一致），
  精度不够再上 NVFP4/CUTLASS

**DP-2：W 侧操作数 —— e2m1 还是 e4m3（mixed）？**
- `kind::mxf8f6f4` 允许 A/B 混合类型（DeepGEMM #447 也有 FP8×FP4 mixed）
- **A=e2m1 × W=e4m3（mixed）**：W 侧直接复用 wxfp8 的码本→e4m3 转换（WF-1），
  checkpoint 不动；代价：MMA 吞吐大概率为 fp8 档（≈2× bf16），收益主要是激活存储减半
- **A=e2m1 × W=e2m1（全 fp4）**：吞吐 4× bf16，但 W 侧码本→e2m1 有损
  （Lloyd-Max 16 级 ≠ e2m1 的 7 幅度网格，与当年 W8→int8 塌级同类问题），
  低 bit expert（1/2/4 bpw 混合）到 MX 格式的映射需要重新设计，可能动 checkpoint
- 倾向：**先 mixed 走通（W 精度零损失、只赚激活带宽），再评估全 fp4**

**DP-3：块缩放粒度与旋转的复合**
- 旋转 group=128 内再切 MX 块（32 或 16）：scale 在旋转后计算（同 wxfp8 原则）
- 待实测：128 内 4×32 块的 e8m0 vs wxfp8 的整 128 fp16 scale，精度差多少

## 四、实施步骤（依赖 wxfp8 完成度，WF4-x 与 wxfp8 的 WF-x 区分）

| 阶段 | 内容 | 前置 |
|---|---|---|
| WF4-0 | 本规划文件 | — |
| WF4-P1 | 前置探测（可与 wxfp8 并行，均小显存）：① Triton TMA+persistent MXFP4 GEMM 原型（参考 vllm#31089 结构）在 5090 的实测吞吐——验证能否 >353（fp8 档）；② e2m1+32 块 e8m0 激活量化 relerr（离线，模拟旋转后分布）；③ TE v2.15 NVFP4 recipe 安装试用（可选） | 无 |
| WF4-1 | 决策点 DP-1/DP-2 复审（用 WF4-P1 数据 + wxfp8 WF-5 ppl） | wxfp8 WF-5 |
| WF4-2 | mixed（A=e2m1 × W=e4m3）kernel：`triton_kernels_fp4.py`，量化点接线平移 wxfp8 | WF4-1 |
| WF4-3 | 全 fp4（若 DP-2 选了）：W 侧 MX 量化离线工具 + checkpoint 格式决策（需本人确认，涉及 checkpoint 变更） | WF4-2 |
| WF4-4 | 集成 `quantization/wxfp4/` + eval ppl（手动） | WF4-2/3 |

> 遵守项目规范：WF4-3 若要动 checkpoint 格式，**必须先征得本人同意**；禁止降级回退；
> 不自定版本号。
> **执行日志规则**：同 wxfp8-plan §三——每完成一步在本文件"执行日志"区对应小节追加
> 一条（日期 + 事项 + 实测数据 + 结论/下一步）。

## 五、风险表

| 风险 | 等级 | 缓解 |
|---|---|---|
| fp4 激活精度崩（e2m1 仅 7 幅度） | **高** | 旋转已就位是最大底气；块缩放粒度加密（16 块）；敏感层保持 wxfp8/wxa8（混合部署）；实在不行 fp4 只用于 MoE down 等实测不敏感处 |
| Triton TMA+persistent 复杂度 | 中 | vllm#31089 有完整参考；wxfp8 阶段先积累 persistent kernel 经验 |
| sm_120a 编译链（Triton 内部处理，风险小；CUTLASS/PTX 路线风险大） | 中 | 主路径 Triton 自动处理 gencode；CUTLASS 路线再评估 |
| mixed fp4×fp8 无加速（只有带宽收益） | 中 | WF4-P1 实测先行，数据说话再定 DP-2 |
| cuBLASLt nvfp4 eager 布局 | 低 | 非主路径，暂不投入 |

## 六、参考来源

见 `wxfp8-plan-260924.md` §六（fp4 相关：triton#8548、vllm#31089、Colfax SM12x NVFP4 系列、
TE#2956/#2968、DeepGEMM#236/#447、pytorch#172807、PTX ISA mma kind::mxf8f6f4/.nvf4）。

## 变更记录

- 2026-09-24 WF4-0：初版预研规划（与 wxfp8-plan 同日，作为其下游阶段的独立文件）。
- 2026-09-24：新增执行日志区（本人要求：每做一步更新日志）。

---

## 执行日志

> 条目格式：`- 日期 事项：做了什么 → 实测数据 → 结论/下一步`。数据必须抄实测输出，不写主观估计。

### WF4-0：规划 ✅ 已完成

- 2026-09-24：本文件创建；能力事实汇总自 wxfp8-plan §1.1/§1.2 探测与本机 fp4 探测
  （dot_scaled 功能正确 c[0,0]=128.00@cfg 64x128x256；吞吐 110 TFLOPS=软件模拟；
  nvfp4 `_scaled_mm` 四种布局枚举均被拒，eager copy_ 未实现）。

### WF4-P1：前置探测（未开始，可与 wxfp8 WF-1..5 并行）

- （待记：① Triton TMA+persistent MXFP4 原型吞吐；② e2m1+32 块 e8m0 激活 relerr 离线；
  ③ TE v2.15 试用（可选））

### WF4-1：DP-1/DP-2 决策复审（未开始）

- （待记：用 WF4-P1 + wxfp8 WF-5 数据定格式与 W 侧操作数）

### WF4-2：mixed（A=e2m1 × W=e4m3）kernel（未开始）

- （待记）

### WF4-3：全 fp4 路线（未开始，若 DP-2 选中）

- （待记；动 checkpoint 前需本人确认）

### WF4-4：集成 + eval ppl（未开始）

- （待记，ppl 数据 + 与 wxfp8/wxa8 基线对比）
