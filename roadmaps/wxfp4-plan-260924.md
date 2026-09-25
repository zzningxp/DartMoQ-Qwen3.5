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
| Triton `tl.dot_scaled` | ✅ **原生可用（triton 3.8.0 + ptxas 12.8 组合，实测 546 TFLOPS @4096³ = 1.55× fp8）**；四个前置条件见执行日志 WF4-P1（3.5.0 有布局 bug、ptxas-blackwell 13.3 的 cubin 被 570 驱动拒载、rhs scale 转置布局、scale 不走 TMA）；**只支持 MX（1x32+e8m0），不支持 NVFP4** | 本机 WF4-P1 实测（2026-09-24/25） |
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
- 2026-09-24 WF4-P1：② e2m1 relerr 实测（格式地板 ~10%）；① dot_scaled 被
  triton 3.5.0 sm_120 布局 bug 阻断（正确 rhs=(K/2,N) 布局考古 + 全变体复现），
  早先"功能可用"结论作废并同步修正相关文档（执行日志）。
- 2026-09-25 WF4-P1：**原生 MXFP4 打通**——隔离 env `dart312-t38`（triton 3.8.0 +
  ptxas-blackwell 换 12.8）下 546 TFLOPS @4096³（1.55× fp8），无需升级驱动/CUDA；
  四个坑与完整配方见执行日志与 memory/sm120a-mxfp4-triton-recipe。
- 2026-09-25 WF4-P1 补测：指针加载也是原生 MMA（W 侧查表结构可对接）；mixed
  e2m1×e4m3 走模拟（57-67 TF）→ DP-2 改为全 e2m1 + kernel 内 nibble 映射主线。
- 2026-09-25 WF4-2 bring-up：fp4 GEMM kernel 契约对拍 0.00021 ✓，MoE 形态 177 TF
  （与 wxfp8 持平，546 上限的 32%），优化方向已列（执行日志）。
- 2026-09-25 WF4-2 迭代 1：swapped-operand 落地（修转置编址 bug，对拍 ✓）；
  真实 per-expert 形状重定向优化目标（down 已持平 fp8，gate_up 0.33x 待转换消解；
  dense 吞吐对 MoE wall 无意义——循环开销占 80%）。
- 2026-09-25 WF4-2 迭代 2：专家级通用纯 MX GEMM（普通 MoE 主路径）闭环——
  契约对拍 0.00021；down 反超 fp8（1.15×）、gate_up 差 4us（launch 量级）、
  dense 372 TF（TMA 参照 68%）；debug 记录：全局 scale 乘反（g² 偏差）曾被
  误判为 lhs scale bug，已澄清并留档。

---

## 执行日志

> 条目格式：`- 日期 事项：做了什么 → 实测数据 → 结论/下一步`。数据必须抄实测输出，不写主观估计。

### WF4-0：规划 ✅ 已完成

- 2026-09-24：本文件创建；能力事实汇总自 wxfp8-plan §1.1/§1.2 探测与本机 fp4 探测
  （dot_scaled 功能正确 c[0,0]=128.00@cfg 64x128x256；吞吐 110 TFLOPS=软件模拟；
  nvfp4 `_scaled_mm` 四种布局枚举均被拒，eager copy_ 未实现）。

### WF4-P1：前置探测（进行中，2026-09-24）

- 2026-09-24 ② e2m1 激活量化 relerr（`test/test_wxfp4_p1_probe.py` §1，高斯=旋转后代理）：
  | 格式 | relerr |
  |---|---|
  | int8 per-128（wxa8 基线） | 0.00646 |
  | e4m3 per-128（wxfp8 基线） | 0.02573 |
  | **e2m1 per-32 + e8m0（MXFP4）** | **0.11545** |
  | e2m1 per-16 + e4m3（NVFP4） | 0.09514 |
  | e2m1 per-128 + fp16（上界参考） | 0.10887 |

  结论：e2m1 格式地板 ~10%（7 个非零幅度的结构性代价），缩放粒度加密改善有限
  （MX→NVFP4 0.115→0.095）；fp4 激活误差 ≈ e4m3 的 4x、int8 的 16x。
  但 wxfp8 的先例是"4x 误差只换 +0.01 ppl"，fp4 的 ppl 代价要实测才知道
  （可能进入不同 regime，DP-3 的关键输入）。
- 2026-09-24 ① TMA+persistent MXFP4 GEMM（§2）：**被编译器阻断**。
  - 布局考古：`dot_scaled` 的 rhs 必须是 **(K/2, N)**（沿 K 打包、N 作列，
    semantic.py:1612 `K_RHS, N = rhs.shape`）；早先所有 naive 探测传的是 (N, K/2)——
    方形形状下侥幸过类型校验，数值检查也因均匀填充数据误判通过，**全部作废**
  - 正确布局后：最小用例（无循环无 TMA）、TMA+persistent、纯指针 × 全部 tile 配置
    均在 `TritonGPUAccelerateMatmul` pass 崩溃（`convert_layout` 形状断言，
    PassManager::run failed）→ **triton 3.5.0 在 sm_120 无法编译正确布局的 dot_scaled**
  - smem 约束顺带确认：(128,128,256,3s) 需 102464B > 101376B 上限（SM120 ~99KB），
    大 tile 需降 stages 或 BK
  - 能力探测脚本（test_wxfp8_capability_probe.py）与 wxfp8-plan §1.1/§1.2 的
    dot_scaled 相关结论已同步修正
- 2026-09-25 ① **突破：Triton 3.8 + ptxas 12.8 组合下原生 MXFP4 MMA 跑通，546 TFLOPS @4096³**
  （对照：fp8 353 / int8 447 / bf16 190 —— 即 **1.55× fp8、2.87× bf16**，数值精确）。
  破解过程（四层问题依次剥开，均有实测依据）：
  1. **布局**（3.5/3.8 通用约定，semantic.py）：rhs 的 value 是 **(K/2, N)**（沿 K 打包、
     N 作列），但 rhs 的 **scale 是 (N, K/32)**——scale 与 value 转置；
     lhs 为 value (M, K/2) + scale (M, K/32)。早先全部探测传反，作废
  2. **Triton 版本**：3.5.0 在正确布局下 AccelerateMatmul pass 崩溃（前述）；3.8.0 的
     PTX 正确产出 `.target sm_120a` + `mma.sync...kind::mxf4nvf4...ue8m0`（原生指令）
  3. **ptxas 版本才是 cubin 拒载真因**：triton 3.8 wheel 对 arch≥100 调
     `bin/ptxas-blackwell`（**CUDA 13.3**），其 cubin 被 570 驱动拒载（invalid image）；
     **换成 CUDA 12.8 的 ptxas 后编译/加载/计算全部正确——无需升级驱动或 CUDA**
     （sm_120a 正是 12.8 引入；做法：探测 env 里 `ptxas-blackwell` 软链到 dart312
     triton 3.5 自带的 12.8 ptxas）
  4. **scale 不能走 TMA**：box 最内维 BK/32=4~8 字节不满足 TMA 16B 对齐 →
     cuTensorMapEncodeTiled 报 invalid argument；数据走 TMA、scale 走普通指针即可
  - 环境备忘：探测 env `dart312-t38`（torch 2.9.0+cu128 + triton 3.8.0 + ptxas-blackwell
    软链 12.8）；`TRITON_PTAXAS_BLACKWELL_PATH` 环境变量实测不生效，直接换二进制才有效
  - 最优 cfg (64,128,256,4w,3s) 546 TF / (128,128,128,8w,4s) 537 TF；smem 约束：大 tile
    需 ≤2 stages（SM120 ~99KB，(128,256,128,3s) 需 113KB 超限）
  - **判定**：546/353 = 1.55×（未到理论 2×，scale 指针加载与配置未调优有空间）→
    **DP-1 倾向 MXFP4/Triton 路线成立**；DP-2（mixed vs 全 fp4）待精度数据与 ppl 决策
- 2026-09-25 补测（两个关键架构事实）：
  1. **纯指针加载（非 TMA）同样产出原生 mxf4nvf4**（k_min PTX 16 处）→ 权重侧
     "packed 索引 + kernel 内 LUT 查表"结构可直接对接 dot_scaled，TMA 仅为 A 侧
     性能优化项
  2. **mixed A=e2m1 × W=e4m3 走 bf16 模拟**（实测 57-67 TF，数值亦异常）→
     Triton 3.8 的原生 mxf4nvf4 只在 e2m1×e2m1 时触发，混合格式无硬件路径
- **DP-2 修正（基于上述数据）**：mixed 路线经 Triton 不可行；主线改为
  **全 e2m1×e2m1 + kernel 内码本索引→e2m1 nibble 映射**：
  - 1/2-bit expert：码本 2/4 级可精确落入 e2m1（含 per-32 e8m0 块缩放 +
    epilogue 的 fp16 norms 兜底，数学链待 WF4-2 展开）
  - 4-bit expert：16 级 → e2m1 7 幅度塌缩（少量 expert，误差量化后接受与否 ppl 定）
  - attention W8：维持 wxa8（混合部署结论不变）
  - W 侧离线展开为 e2m1 张量的方案否决（2-bit expert 显存 2× 膨胀，
    9.5GB→19GB 不可接受），必须 kernel 内转换

### WF4-1：DP-1/DP-2 决策复审（未开始）

- （待记：用 WF4-P1 + wxfp8 WF-5 数据定格式与 W 侧操作数）

### WF4-2：mixed（A=e2m1 × W=e4m3）kernel（进行中，2026-09-25 bring-up 完成）

> DP-2 已修正为全 e2m1 主线（见 WF4-P1 补测），本阶段实际实现为
> A=e2m1+per-32 e8m0 × W=码本索引→e2m1 nibble（kernel 内转换）。

- 2026-09-25 `turboquant_utils/triton_kernels_fp4.py` bring-up 版落地：
  - `build_e2m1_codebook`：码本→nibble LUT + **2 的幂约束** scale 搜索
    （分解链 `w ≈ nib[idx] × 2^E0 × norms_gf`，与 wxfp8 的 LUT+cb_scale 同构，
    W 侧 e8m0 通道全常量）
  - `quantize_act_e2m1_per32`：torch 参考（MX 原生格式）
  - `_wxfp4_fused_matmul_kernel_grouped_gf`：W 侧 unpack→LUT→`tl.split` 拼 byte
    →`tl.trans`→dot_scaled；A/scale 指针加载；per-group norms epilogue
  - ⚠ 只能在 dart312-t38（triton 3.8）下编译运行
- 2026-09-25 实测（`test/test_wxfp4_gemm_align.py`，t38 env）：
  - **契约对拍 0.00021 ✓**（整条分解链数学正确）
  - 端到端 vs fp32 = 0.35941（预测 ~0.36 ✓：W2 量化 0.34 ⊕ 激活 e2m1 0.115）；
    vs 理想量化 0.04155（码本→e2m1 增量，好于预期 0.05-0.09）
  - 码本→e2m1 转换 relerr（pow2 约束）：1-bit 0.060 / 2-bit 0.031 / 4-bit 0.104
    ⚠ 4-bit 偏高——连续 scale + 残差折进 norms（norms×residual 常数因子）
    是现成的改进路径，待做
  - **吞吐 @ MoE 形态（B2048 N1024 K2048 bit2）：最优 177 TF**（(128,128,128,8w,1s)）
    = wxfp8 同形状持平；离 546 上限还有 3.1× 空间——瓶颈在 W 侧 kernel 内转换
    （reshape/split/trans 的 smem 布局转换，ns≥2 即超 135KB smem）
- 下一步（WF4-2 优化迭代）：
  1. swapped-operand 设计：W 作 lhs（自然方向免 trans），A 在量化时预转置存储
  2. 拼包替代：reshape+split → 乘法归约（byte = Σ nib×[1,16]）
  3. 码本 LUT 连续 scale + norms 折残差（救 4-bit 的 0.104）
  4. A 侧 TMA + persistent（对齐 546 参照 kernel 的结构）
- 2026-09-25 优化迭代 1（swapped-operand + 乘法归约拼包）完成：
  - 修复转置布局编址 bug：A 预转置后 group 偏移必须乘行 stride B
    （`(g+k)//2` 是字节序号不是地址；group0 偏移 0 侥幸正确——诊断方法：
    逐 group 单独激活对拍，group1-only 复现）
  - swapped 对拍 0.00021 ✓；但 dense B=2048 吞吐不变（177 TF）——trans 不是瓶颈，
    W 转换的 smem 布局转换才是（ns≥2 恒超限 147-286KB）
  - **真实 per-expert 形状实测（修正优化方向的关键数据）**：
    | 形状 | fp4 | fp8 | 比值 |
    |---|---|---|---|
    | gate_up Be=32/64 (N1024 K2048) | 37-39us | 13-15us | **0.33-0.40x** |
    | down Be=32/64 (N2048 K512) | 12us | 12us | **0.96-1.00x（持平）** |
  - **结论修正**：dense 177→546 的优化对本项目当前 MoE 几乎无意义——真实 MoE 是
    per-expert 小 B（~16-64 行，单 B-tile，W 转换零重复），且 per-expert Python
    循环占 MoE wall ~80%（P4 profile 结论），GEMM 微秒差被循环开销淹没。
    fp4 在 MoE 的实际价值 = 激活存储减半（gather/scatter 字节减半）+
    未来大 B 场景；**down 已达 fp8 持平，gate_up 需转换开销消解（0.33x）**
- 2026-09-25 优化迭代 2（专家级通用纯 MX GEMM，普通 MoE 主路径）完成：
  - `quantize_weight_e2m1`（W 直量化 e2m1+per-32 e8m0 + 全局 scale 搜索；
    返回 epilogue 乘法因子 1/g）+ `_wxfp4_expert_matmul_kernel` /
    `wxfp4_expert_matmul`（纯预打包操作数、零 kernel 内转换、group-M swizzle）
  - **debug 记录**：契约对拍一度恒差 0.111，先误判为"lhs 真实 scale 被 Triton
    错误应用"（常数 scale 隔离实验 0.0 的假象）；后经"kernel 输出 = 0.3076×参考
    的常数比例 + A=精确 1.0 探针"定位真因——**全局 scale 乘反**（packed 代表
    w·g，epilogue 应乘 1/g 而非 g，输出偏 g²）。修正后两取向均正确，
    定稿 W-lhs（dense 373 vs 302 TF 更快）
  - 全局 scale 搜索对 MX 的增益实测可忽略（0.1155→0.1151）：e8m0 的 2 的幂
    浪费是逐块无偏的，全局因子无法吸收（与 NVFP4 的 e4m3+全局两级缩放本质
    不同）——保留参数但非必需
  - 实测（`test/test_wxfp4_expert_gemm.py`）：契约对拍 **0.00021** ✓；
    vs fp32 0.1625（=预测 0.163）；吞吐见下
    | 形态 | fp4 | 对照 |
    |---|---|---|
    | per-expert down Be64 | 10.4us | **fp8 12us 的 1.15×（反超）** |
    | per-expert gate_up Be64 | 17.4us | fp8 13us（差 4us，launch 量级） |
    | B=2048 N1024 K2048 | 319 TF | — |
    | dense 4096³ | **372 TF** | TMA 参照 546（68%） |
  - 结论：**专家级主路径（普通 MoE）数值与性能闭环**——小 B 与 fp8 互有胜负
    （±15-30%），大 B 场景 319-372 TF 显著超过 fp8 的 353 门槛一半以上；
    剩余优化空间 = A 侧 TMA + persistent（372→546）
- 下一步（WF4-2 迭代 3 或直接 WF4-4）：
  1. A 侧 TMA + persistent（可选：普通 MoE 大 B 场景的收益项）
  2. 激活 rotate+quantize fp4 融合 kernel（目前 A 量化是 torch 参考，主流程
     接线前必须 kernel 化——WF-2 的 fp4 变体）
  3. WF4-4 集成：quantization/wxfp4/ + 入口 + eval（含本项目 checkpoint 的
     适配器路径决策）

### WF4-3：全 fp4 路线（未开始，若 DP-2 选中）

- （待记；动 checkpoint 前需本人确认）

### WF4-4：集成 + eval ppl（未开始）

- （待记，ppl 数据 + 与 wxfp8/wxa8 基线对比）
