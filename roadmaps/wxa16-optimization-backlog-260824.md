# WxA16 当前可优化点梳理（2026-08-24）

> 本文档基于 `roadmaps/` 下全部 md 文件（ROADMAP.md、ROADMAP_ALTERNATIVE_QWENMULTILINEAR.md、wxa16-plan.md、new-wxa16-plan-260818.md、new-wax16-plan-260820-reduce-fp32.md、profile-wxa16-bottleneck.md），结合当前代码（`turboquant_utils/triton_kernels.py`、`turboquant_utils/rotation.py`、`wxa16_bit_partitioned_moe.py`、`turboquant_utils/cutile_kernels.py`）的实际落地状态，整理「当前仍然可以优化的点」与优先级。
>
> 本文档是阶段性梳理，**不是最终方案**。所有涉及主流程全模型 eval（`run_qwen35.py` / `eval_qwen35.py`）的验证由本人手动跑，不自动执行。

---

## 〇、当前已落地状态（基线，已核对代码）

以下优化**已在代码中确认落地**，作为后续优化的基线：

| 优化项 | 落地证据 | 收益（实测） |
|--------|----------|--------------|
| FP16 全链路（P2 Step1-3：output/codebook/norms/旋转 acc fp16） | `triton_kernels.py` 类型链 | Layer0 forward ~12.3s→8.44s（含 P4-4/2/7 叠加） |
| 8-bit constexpr 快速分支（跳过 unpack） | `triton_kernels.py:98/230/424` `if BIT_WIDTH == 8` | 小收益 |
| 多 group 融合单 kernel（P4-4） | `triton_kernels.py:542-545` fast path，三个 grouped 函数均接入 `batch_rotate_input` | **Layer0 2.04x**（8.44s→4.13s） |
| Block/warp 离线调优（P4-2） | `triton_kernels.py` 各 bit 最优配置 | **+9.5%** |
| 旋转 bmm 优化（P4-7） | `triton_kernels.py:39/550/640/729` 全部走 `batch_rotate_input` | **+7~8%**（MoE e2e） |
| 旋转缓存上限 128→2048（P3-a） | `rotation.py:35` `_MAX_CACHE_SIZE = 2048` | 消除每 forward ~284ms QR 重算 |
| 删 forward 末尾 gc/empty_cache（P3-b） | `wxa16_bit_partitioned_moe.py` forward 内已无 gc | Layer0 ~35% |
| 去 `.item()` D2H 同步（P3-c） | `wxa16_bit_partitioned_moe.py:187/277/310-314` `_expert_offsets_cpu` | 256exp ~1.6%（大 H 下稀释） |
| `index_add_` 替代 `scatter_reduce_`（P3-d） | `wxa16_bit_partitioned_moe.py:449` | 零风险、数值一致 |

**当前端到端状态（2026-08-26 更新，P5-3 落地后）**：量化 overhead 已被完全抹掉并反超 FP16。
全模型 eval（Qwen3.5-35B-A3B, `models/qwen3.5-2bpw-260824/`, sequential mode, RTX 5090，
见 `logs/0822.4.wxa16.p5-3.log`）：

| 数据集 | FP16 | W2A16（P5-3 后） | W2A16（P6 后，2026-08-26） | W2A16（P8 后，2026-08-29） | 差距 |
|---|---|---|---|---|---|
| wikitext2（145 samples） | 90.05 s | **76.32 s** | **68.96 s** | **57.52 s** | 比 FP16 快 **36.1%**，比 P6 再 -16.6% |
| c4（256 samples） | 114.7 s | **110.85 s** | **107.28 s** | **88.36 s** | 比 FP16 快 **23.0%**，比 P6 再 -17.6% |

ppl：wikitext2 7.7942 → **7.7944**；c4 11.2683 → **11.268**（均无损失）。

> **2026-08-27 勘误闭环**：0822.8 轮（69.64/108.74）因 `patch_delta_rule` 未接线而无效；
> 接线补上后（git 77dbfe4+）：c4 107.28 → 98.12s，与 GPU 口径估算几乎 1:1 兑现——
> 「eval 是 CPU bound」假说证伪，稳态下 wall 跟随 GPU。
> **P8-4 第一步（Triton chunk kernel，git 3265adc+）**：c4 98.12 → **95.43s（-2.69s，
> 与预估 -2.9s 一致）**；wiki 首次含 Triton 版实测 **60.43s**。
> **P8-4 第二步（cast 缩减，git 3265adc+）**：c4 95.43 → **93.64s（-1.79s）**，
> wiki 60.43 → **60.6s**。ppl 11.2674 / 7.7932 无损失。
> **P8-4 第三步（wy_prep 融合 kernel，git 746d1f8+wy_triton 修复）**：
> c4 93.64 → **89.74s（-3.9s）**，wiki 60.6 → **58.01s（-2.6s）**。
> ppl 11.2672 / 7.7946 无损失。
> **P8-6（kernel 自动调优，git 766a5d3+）**：c4 89.74 → **88.36s（-1.38s）**，
> wiki 58.01 → **57.52s（-0.49s）**。ppl 11.268 / 7.7944 无损失。

MoE 层 warm 稳态 forward：0.145 s → **~0.105 s（约 -28%）**（layer 10/20/30 实测）。
P6-0 轮实测拆分（wall-clock，仍有异步归因噪声，仅供参考）：
- compute 0.054 s → 0.042 s（triton 0.042 s → 0.036 s），init 0.080 s → 0.065 s
- layer0 冷启动（含 JIT）：2.98 s → 0.91 s（`warmup_kernels` 修复后首次真正生效）

**p6-0 轮的完整 eval 拆解**（`logs/0822.5.wxa16.p6-0.log`，c4）：总 107.28 s 里
attention 是最大头——linear_attn 层每层 ~0.65 s × 30 层（另 10 层 full attention 更快），MoE forward 只有 ~0.105 s。
**后续优化的最大杠杆已不在 MoE 内部**，而在 linear attention 与其他层间开销
（后续按 P6-0 的方法对 attention 也做 event 拆分可确认）。
> ⚠️ 该 0.145s 的**内部拆分不可信**：现有计时全是 `time.time()` 且无一次 `cuda.synchronize()`，
> 日志里 `init` 占 55% 是 allocator 阻塞吸收上一层异步队列的假象，不是真实工作量。
> 精确拆分需先做 **P6-0**（CUDA event 计时）。

**目标转变**：早期目标「追平 dense/FP16」已达成，后续优化的参照系不再是 FP16，
而是 A16 路线本身的天花板（受限于 fp16 tensor core 吞吐）——参见 P9-1（WxA8/WxA4）。

---

## 一、当前仍可优化的点（按 P 阶段排序）

> P5 最先做，P6/P7/P8 依次往后。同一 P 阶段内的子项主题相关、可并行或顺序灵活。
> （2026-08-27 更新：P7 阶段已整体清零，见下方 P7 节；P8 为 attention 优化，
> P9 为长期方向。）
> P2 / P3 / P4 为已完成阶段，见 `new-wax16-plan-260820-reduce-fp32.md`。

---

### P5 — 结构性主线优化（最高优先级，直接啃真实差距）

> 端到端几十倍差距的三层拆解（profile 文档 §十一）：纯 kernel ~6x（10%）、分组架构 ~3.5x（20%）、**per-expert × per-bit Python 循环 ~3~4x（70%）**。P5 针对后两层，外加零风险快速 win。

#### P5-1：加载期 group-first 连续布局重排（方案 A）✅ 高优先（低难度、不降级）
- **问题**：去 clone（col_start 寻址）在 micro 小形状省 44.5%，但真实 eval 形状（B≈9280，packed 张量 ~134MB 超 L2）下宽 stride 跳读惩罚随 B 增长，净回归（gate_up +113.7µs/call、down +229.8µs/call），eval +26%/+17%。
- **方案**：加载期把 packed 权重按 group 重排为 group-first 连续布局 `(num_groups, N, group_packed_bytes)`（一次 permute+contiguous，134MB 一次性 <0.1s）。推理时每组切片天然连续 → kernel 无 clone、无宽 stride、col_start 恒 0。同时优于旧 clone 路径与现 col_start 路径。
- **改动点**：`wxa16_bit_partitioned_moe.py` 的 `set_packed_data` 一线 + wrapper 切片方式。
- **预期收益**：修复真实 eval 净回归，并使 col_start 寻址回到 micro 测量收益。
- **难度**：低。**约束**：不降级退回 clone 旧路径（按规范需本人同意，故用重排方案 A 替代）。
- **验证**：`test/test_eval_shape_bench.py`（新路径应 ≤ clone 旧路径）+ 数值一致性 + 本人手动 `run.q.sh` 复测。
- **来源**：profile-wxa16-bottleneck.md §十 记录 4 方案 A。

#### P5-2：多 expert 合并（MoE grouped GEMM）：同一 bit 所有 expert 一次 kernel ⚠️ 已实现未采纳
- **问题**：`wxa16_bit_partitioned_moe.py:366/385` 仍是 `for expert_idx` × `for bit_str` 的逐 expert 循环，每 expert 单独 launch。expert 多但每 expert token 少时 GPU 利用率极低，且 Python 循环体开销被放大。
- **方案**：参考 false-grouped 的 gather/scatter + bad-triton 的 3D grid 思路，但用 grouped GEMM 方式——把同一 bit 的所有 expert 权重打包好（已是 bit-partitioned 布局），一个 kernel 内用 `expert_info_ptr` 索引各 expert 的 token_start/token_count/weight_range，每个 SM 动态分配给某个 expert 避免负载不均衡。
- **预期收益**：MoE 部分 launch 从 `O(experts × bits × groups)` 降到 `O(bits × groups)`，并显著降低 Python 循环占比（约 70% 的差距来源）。
- **难度**：中高。**注意**：3D grid 按 expert 并行的 bad-triton 方案已被证伪（负载不均衡 + 边界检查开销）；正确做法是 expert_info 指针 + 仅算需要的行（grouped GEMM 思路）。
- **分阶段推进**：
  1. 先做同一 bit 内所有 expert 的合并（gate_up 一个 kernel + down 一个 kernel），bit 间保留外层循环。改动小，先吃到大部分 launch 减少收益。
  2. 再优化负载均衡 + down 路径 in_feature-slice 适配。
- **来源**：new-wxa16-plan-260818.md §1.2。
**效果不好**：wxa16 load eval 速度下降从 269.4 到 288.6s。

> **状态补充（2026-08-26）**：实现存在于分支 `opt-p5-2`（commit `2564a87`，kernel
> `moe_gate_up_grouped_gf_v2` / `moe_down_grouped_gf_v2`），**未并入 main**；
> main 上该路径已完全移除，逐 expert 循环是唯一路径。
> 已决定**不重测**（见 §4.3），否决记录与 SM 填充度佐证保留备查。

#### P5-3：BLOCK_B / BLOCK_N / BLOCK_K / num_warps / num_stages 联合调优 ✅ 已完成
- **问题**：P4-2 单维离线调优（+9.5%）的结果是局部最优。五个 tile 参数之间强耦合，单维扫无法找到全局最优。
- **关键发现**：**per-expert token 数（B 值）对最优配置的影响远大于 bit-width**。B=64 时最优是 BLOCK_N=16/BLOCK_K=128，但真实 eval 中 per-expert 约 2048 tokens，最优反而是 BLOCK_N=128/BLOCK_K=32。B 规模差 30 倍，配置方向完全相反——这是初期踩的主要坑（单 kernel +30% 但全模型 eval -9%）。
- **最终方案**：
  - 离线全量网格搜索（324 组，过滤后 ~150 组有效）
  - gate_up / down 拆两套配置（K/N 互换导致 BLOCK_K 最优值不同）
  - B 自适应两档（阈值 256）：small 档针对 B<256 长尾 expert，large 档针对 B≥256 主力场景
  - large 档 2-bit 配置：gate_up=(64,128,32,4,4), down=(64,128,128,4,2)
- **实测收益（RTX 5090, Qwen3.5-35B-A3B, wikitext2 sequential eval）**：
  - MoE triton 部分：**~1.9x**
  - 全模型总时间：95.52s → 76.32s，**+25%**（时间减少 20%）
  - ppl 不变（7.7939 → 7.7925，纯 tile 调整不影响数学结果）
  - c4 提升更显著（MoE 占比更高的场景）
  - **里程碑**：2-bit bpw WxA16 推理速度首次超过 FP16（wiki 76.32s vs 90.05s，快 18%, c4 256 110.85s vs 114.7，快 3%），量化 overhead 被完全抹掉。比上一个版本快 （wiki 76.32s vs 92.39，快 18%, c4 256 110.85s vs 162.4s，快 32%）
- **难度**：低，**收益高（+20% 总时间）**，零风险。
- **遗留 / 可继续优化**：
  - 冷启动：Triton JIT 编译让第一层变慢（~3s 额外开销，占总时间 3-4%）。可通过预编译（加载时 dummy forward 触发编译）消除。
  - 1-bit / 4-bit / 8-bit 配置用了 2-bit 的模板，未精细搜索。混合 bit 模型需重搜。
  - 调优工具已沉淀为 `turboquant_utils/kernel_autotune.py`，后续换硬件/换 shape 可一键重跑。
- **学术价值**：MoE 量化 GEMM 的 tile size 性能特征分析、B 规模对最优配置的影响规律、形状自适应配置策略——公开文献中这类细粒度 empirical study 较少。
- **详细方案 + 完整数据**：`roadmaps/wxa16-p5-3-joint-kernel-autotuning.md`
- **来源**：new-wax16-plan-260820 P4-2。

---

### P6 — 旋转去冗余 & per-expert 开销消除（P5 之后，2026-08-26 重排）

> **重排说明（2026-08-26）**：P5 完成后，用 `models/qwen3.5-2bpw-260824/meta.json` 与
> `logs/0822.4.wxa16.p5-3.log` 复核了 P6 原四项的前提，发现**其中三项的前提不成立**。
> 原 P6-1~P6-4 的正文全部保留在下方（标题已标记「已废弃」，附裁决与证据），
> 新的执行项为 **P6-0 / P6-1 / P6-2 / P6-3**（P6-1~P6-4 编号复用给新内容，
> 旧的已标记「已废弃（旧）」加以区分）。

#### 复核依据（三条实测证据）

**证据 1：本模型每个 expert 只用一个 bit，bit 是在 expert 之间「划分」而非在 expert 内「混合」**

`meta.json` 全 40 层 `active_experts_per_bit` 统计：

| 层数 | bit-1 | bit-2 | bit-4 | 合计 |
|---|---|---|---|---|
| 17 | 4 | 250 | 2 | 256 |
| 13 | 2 | 253 | 1 | 256 |
| 7 | 6 | 247 | 3 | 256 |
| 1 | 8 | 244 | 4 | 256 |
| 1 | 0 | 256 | 0 | 256 |

三个 bit 的 active expert 数**正好加起来 = 256**。bit-2 覆盖 95.3%~100% 的 neuron。

> 注：日志里的 `active_bits: 768` 是**循环进入次数**，不是 kernel 发射次数——计数器
> `active_bits_count += 1` 在 `wxa16_bit_partitioned_moe.py:544`，而
> `if actual_inter_size == 0: continue` 在 `:553`。真实 (expert, bit) 活跃对 ≈ **257**。

**证据 2：gate_up 的旋转与 expert 无关，真正的冗余是跨 expert 而非跨 bit**

`triton_kernels.py:1184`：`x_rot = batch_rotate_input(x, group_size, seed)`——`seed` 是 packed 里的常量，
`row_start/row_end` 只切权重、不进旋转。256 个 expert 用的是**同一个旋转矩阵**，只是作用在不同 token 子集。
top_k=8 ⇒ 每个 token 被重复旋转 8 次：每层旋转 256×2048 = 524288 行，而 unique token 只有 65536 行。

**证据 3：seed 已烧进 checkpoint，「统一 seed」= 重新量化整个模型**

`quantize.py:338/435` 把 `seed` 写进 pack dict，加载时校验
（日志：`[OK] safetensors qmeta seeds match meta.json (547 keys)`）。

#### 原 P6-1~P6-4 裁决表

| 项 | 裁决 | 依据 |
|---|---|---|
| P6-1 跨 bit 旋转复用 + dual_matmul 接线 | ⛔ **理由作废，换方向复活为新的 P6-1（见下）** | 证据 1（无跨 bit 冗余）+ 证据 3（需重量化） |
| P6-2 混合 bit fusion | ❌ **删除** | 证据 1：expert 内没有多个 bit 段可合并 |
| P6-3 块对角 R + 大 K matmul | ❌ **删除** | 块对角 2048×2048 稠密乘比现有 16×(128×128) bmm 多做 **16 倍 FLOPs**，严格劣于现状；且 P5-3 实测 gate_up 最优 BLOCK_K=**32** 而非 128，「大 K 提升 Tensor Core 利用率」已被自身扫描否证 |
| P6-4 dual_matmul grouped 版 | ❌ **删除** | 依赖 P6-1 的 SAME_INPUT（已不成立）；且唯一调用点 `module.py:757` 从**本仓库不存在的 `turboquant_model` 包**导入，`_HAS_TRITON_DUAL` 恒为 False，该路径从未执行过 |

顺带清理 P5-3 遗留：
- **冷启动预编译**：`warmup_kernels` 的调用点已接线（`eval_qwen35.py:209`），但函数体里写的是
  `self.hidden_dim`（类上只有 `hidden_size`），**从 `b7ce480` 起就一直 AttributeError**，
  即该预编译**从未真正执行过**。已于 2026-08-26 修正为 `self.hidden_size`。
  修正前该项不能算完成——`logs/0822.4.wxa16.p5-3.log` 里 layer 0 的 2.98s（含 JIT）
  正是因为当时跑的 HEAD（`e794fb7+`）还没有 warmup_kernels，冷启动开销仍在。
  **修好后的真实收益需重新测量。**
- **1/4/8-bit 重搜配置**：价值低（仅覆盖 0.6%~3% neuron），四 bit 共用 2-bit 配置可接受。

---

#### P6-0：分阶段 CUDA event 计时（前置条件，必须先做）

- **问题**：当前 forward 内所有计时都是 `time.time()`，**全程没有一次 `torch.cuda.synchronize()`**
  （`wxa16_bit_partitioned_moe.py:449-672`）。warm 层日志 `init: 0.080s / compute: 0.054s (triton: 0.042s)`
  中，`init` 段实际只有一个 `reshape` + `zeros_like`，却占 55%——这是 CPU 首次在 allocator 上阻塞、
  把上一层排队的 GPU 工作全部吸收进来的假象。现有 `triton` 子项测的是 **CPU launch 时间**，不是 GPU 执行时间。
- **方案**：在 rotation / gate_up / silu / down / scatter 上打 CUDA event 对，forward 末尾统一 sync 取 elapsed。
  沉淀为可复用的 `turboquant_utils/cuda_profiler.py`，MoE 侧用独立开关接入，**不删除现有 wall-clock 计时路径**。
- **为什么是前置**：在测出旋转到底占 10% 还是 50% 之前，无法给 P6-1/P6-2 排序，也无法验证收益。
- **难度**：低。**风险**：零（仅新增开关，默认关闭时不改变执行路径）。
- **状态**：✅ 已落地。`turboquant_utils/cuda_profiler.py`（`CudaStageProfiler` +
  `sm_occupancy_report`），MoE 侧开关 `moe.set_cuda_profile(True)`，结果存 `moe.last_cuda_stats`。

**实测分阶段拆分**（`test/test_p6_rotation_hoist.py`，32 experts / T=8192 /
per-expert B=2048，等比缩小但保持真实 per-expert B）：

| stage | 提升前 ms | 占比 | 提升后 ms | 占比 |
|---|---|---|---|---|
| gate_up_kernel（含旋转） | 3.174 | 47.2% | 2.286 | 38.7% |
| down_kernel | 1.629 | 24.2% | 1.626 | 27.5% |
| **scatter (index_add_)** | 1.072 | **15.9%** | 1.035 | **17.5%** |
| silu_mul | 0.317 | 4.7% | 0.328 | 5.6% |
| gather | 0.269 | 4.0% | 0.235 | 4.0% |
| sort_prep | 0.201 | 3.0% | 0.212 | 3.6% |
| rotation_hoisted | — | — | 0.127 | 2.1% |
| router | 0.059 | 0.9% | 0.062 | 1.0% |
| 合计 | 6.719 | | 5.911 | |

**两条重要结论**：
1. **`init` 根本不是一个 GPU 阶段**——CUDA event 拆分里它压根不存在，
   证实了原 wall-clock 日志里「init 占 55%」是分配器阻塞吸收异步队列的假象。
2. **scatter（per-expert `index_add_`）已经是第二大开销**（~16%），
   仅次于 gate_up kernel。这是原 backlog 完全没有提到的项，值得单列后续优化。

#### P6-1：gate_up 旋转提到 expert 循环外（原「已废弃 P6-1」的正确形式）

- **依据**：证据 2。旋转按 K 维分组线性、逐行独立 ⇒ `rotate(x[idx]) ≡ rotate(x)[idx]`，
  **数学严格等价，不动量化侧，不改 ppl，不需重量化**。
- **收益**：每层 gate_up 旋转 FLOPs 与访存**均降 8 倍**（= top_k），bmm launch 从 256 次降到 1 次。
  注意 `batch_rotate_input` 每次调用做 **2 次全量 (B,K) 拷贝 + 1 次 bmm**（`rotation.py:203-209`），
  所以访存收益与 FLOPs 收益同量级：4.3GB → 537MB。
- **收益判据（重要）**：预旋转对某个 bit 只有在
  `该 bit 下所有 expert 的 token 行数之和 > T` 时才划算。bit-2：250×2048 = 512000 ≫ 65536 ⇒ 提升 8x；
  bit-1：4×2048 = 8192 < 65536 ⇒ **预旋转反而是 8 倍亏损**，必须跳过。故按 bit 逐一判定，不能无脑全提。
- **代价**：为被提升的 bit 保留一块 (65536, 2048) fp16 ≈ **256MB**；
  `batch_rotate_input` 内部瞬时峰值约 768MB（两次 contiguous 拷贝）。当前 layer30 显存 10.4GB / 32GB，有余量。
- **down 方向不可提**：`seed_base = seed + original_start` 含 expert 偏移（`triton_kernels.py:1272`），
  且其输入 `act_out` 本就是 per-expert 数据，不存在跨 expert 冗余。
- **后续可继续**：`batch_rotate_input` 末尾的 `transpose(0,1).contiguous()` 是为了还原 (B, K) 布局；
  若让 group-first kernel 直接吃 (G, B, gs) 布局的激活，可再省一次全量拷贝。本轮先做等价提升，便于隔离验证。
- **状态**：✅ 已落地。开关 `moe.enable_rotation_hoist`（默认 True）、
  判据阈值 `moe.rotation_hoist_threshold`（默认 1.5）。
  kernel 侧新增 `x_is_rotated` 参数（`triton_fused_matmul_grouped_slice_rows_gf`），
  非主路径传 True 会直接报错而不是静默算错。

**实测**（`test/test_p6_rotation_hoist.py`）：

| 项 | 结果 |
|---|---|
| 数值等价（3 种 bit 配置） | **max_abs = 0.000e+00，逐位相同** |
| 收益判据 | bit-2 (30 experts, rows/T=7.50x) → 提升；bit-1/bit-4 (1 expert, 0.25x) → 跳过 ✓ |
| 旋转本体（缩小规模 T=8192） | 0.856 ms → 0.112 ms，**7.64x** |
| 旋转本体（**真实规模** T=65536, 256 experts） | 6.819 ms → 1.209 ms，**5.64x** |
| MoE forward 整体（缩小规模） | 6.503 ms → 5.678 ms，**1.15x** |

> ⚠️ **注意 micro≠real**：缩小规模下旋转提升 7.64x，真实规模只有 5.64x
> ——单次大 bmm 在 T=65536 时已经转为带宽受限，优势会衰减。
> 按真实规模算，旋转从占 MoE GPU 时间约 13% 降到约 2%，**净省约 10% 的 MoE GPU 时间**；
> 换算到端到端 eval 的收益取决于该层有多大比例真的是 GPU-bound，需本人跑真实 eval 确认。
> 真实规模下提升路径峰值显存 **1042 MB**（`batch_rotate_input` 内两次 contiguous 拷贝所致）。

**真实 eval 结果（2026-08-26，`logs/0822.5.wxa16.p6-0.log` + wiki 补测）**：
- c4 总时间 110.85 → **107.28 s（-3.2%）**，ppl 11.2680 → **11.2683**。
- wikitext2 总时间 76.32 → **68.96 s（-9.6%）**，ppl 7.7925 → **7.7942**。
  两者 ppl 变化均在正常波动量级，bf16 提升路径的 ~1e-2 舍入噪声未伤及 ppl。
- 收益来自三部分：warmup 预编译首次生效（layer0 2.98s→0.91s）+ 旋转提升 +
  P6-2 的 Python 层清理。MoE warm forward 0.145 → ~0.105 s（约 -28%）。
- **关键发现**：端到端里 attention 才是大头（linear_attn 层每层 ~0.65s × 30 层），
  MoE forward 只剩 ~0.105s/层。**后续最大杠杆已不在 MoE 内部**。

#### P6-2：消除 per-expert 的重复常量计算与冗余分配

> **注意：原设想的「一次 gather + 一次 index_add_ 全向量化」不可行**——需要
> (T·top_k, H) = 524288×2048 fp16 ≈ **2.1GB** 缓冲区。当前逐 expert (2048, 2048) = 8MB 的分块反而是对的。
> 记录于此避免重复踩坑。可做的是下面四项：

1. **预计算 active (expert, bit) 对**：现在每层跑 256×3 = 768 次内层迭代，其中约 512 次在
   `:553` 命中 `continue` 空转。改为一次性算出 ~257 条活跃列表。
2. **`norms_scaled` 预乘**：`norms_slice / sqrt(group_size)` 在
   `triton_kernels.py:1189` / `:1282` 每次调用重算，每层约 512 次微 kernel。scale 是常量，
   应在 `_build_group_first` 时预乘进 `norms_gf`。
3. **codebook 设备缓存**：`codebook.to(x.device)`（`wxa16_bit_partitioned_moe.py:565/577/600/612`）
   每层调用约 512 次。
4. **去掉 per-expert `torch.zeros_like(expert_tokens)`**（`:539`）：每层 256 次 8MB 分配。
   因每个 expert 实际只有一个活跃 bit，改为首个 bit 直接接管、后续 bit 累加。

- **难度**：低。**风险**：低（2 需与 kernel 侧对齐，其余为纯 Python 层）。
- **状态**：✅ 四项均已落地。
  1. `_ensure_active_bits()` → `_active_bits_by_expert`（实测内层迭代 96 → 32，省 67% 空转）
  2. `_build_group_first` 里预乘 `norms_gf`，kernel 侧新增 `norms_prescaled` 参数
     （两个 gf wrapper 都支持；非主路径会还原缩放，保证各分支都正确）
  3. `_get_bit_context(device)` 缓存 codebook/seed/group_size/shape
     （顺带把 `seed.item()` 的潜在 D2H 同步从每层 ~512 次降到 1 次）
  4. `expert_out` 首个活跃 bit 直接接管，去掉 `torch.zeros_like`
- **实测**：norms 预乘数值 **逐位相同**（max_abs = 0.000e+00）。

#### P6-3（新增候选，由 P6-0 实测发现）：per-expert `index_add_` scatter

P6-0 的分阶段拆分显示 **scatter 已是第二大开销（~16%）**，每层 256 次
`final_hidden_states.index_add_(0, exp_token_idx, expert_out)`，每次是一整块
(B, hidden) 的带原子累加写回。原 backlog 未覆盖此项。

> **已排除的做法**：一次性 gather/scatter 全向量化需要 (T·top_k, hidden) 缓冲区
> = 524288×2048 fp16 ≈ **2.1GB**，不可行（见 P6-2 开头的注记）。
> 可探索方向：分块批量 scatter（每 K 个 expert 合并一次）、
> 或利用 sorted_tokens 已排序的性质改用 segment-reduce。**待评估，未实现。**

---

#### 已废弃 P6-1（旧）：旋转结果跨 bit 复用 + 接线 `triton_fused_dual_matmul`

> ⛔ **已裁决：理由作废**（见上方裁决表）。以下为原始记录，保留备查。
> 编号已复用于新 P6-1（gate_up 旋转提到 expert 循环外）。
- **问题**：
  - 旋转作用在输入 x 上，仅依赖 `g_dim/group_size/g_start`（跨 bit 相同），但当前每 bit 用**不同 seed**（`seed=42+bit`/`42+bit+1000`），导致旋转矩阵随 bit 不同而不同，无跨 bit 复用（profile §8.4 已验证）。
  - `triton_fused_dual_matmul` 已实现（`triton_kernels.py`），但**主 MoE 路径完全未调用**（唯一调用在 `turboquant_utils/module.py:757` 的单 Linear 路径），属死代码。
- **方案**：统一跨 bit seed → `x_rot_g` 跨 bit 相同，计算一次喂给多 bit kernel；`SAME_INPUT` 成立后可接线 `triton_fused_dual_matmul` 合并两次 matmul。
- **预期收益**：视情况，旋转占 grouped 时间 ~40%（cold）/~30%+（warm 稳态）；合并两次 matmul 减少一次 kernel。
- **难度**：中（耦合量化侧改动：统一 seed + 重新验证精度）。
- **前置依赖**：先做消融实验，验证统一 seed 后 perplexity 变化，确认精度可接受再推进。
- **来源**：profile §8.4、§十 遗留待办；new-wax16-plan-260820 P4-4 未做项。

#### 已废弃 P6-2（旧）：混合 bit-width 场景的 fusion

> ❌ **已裁决：删除**（见上方裁决表）。以下为原始记录，保留备查。
> 编号已复用于新 P6-2（per-expert 重复常量/冗余分配消除）。

- 2-bit 段 + 4-bit 段分别 fused 后，再加一次加法（目前逐 bit 各自 fused 再 Python 累加）。
- **难度**：低-中，**收益中等**。
- **与 P5-2 协同**：多 expert 合并后再做 bit 间 fusion，kernel 结构更清晰。
- **来源**：new-wax16-plan-260820 P4-4 未做项。

#### 已废弃 P6-3（旧）：方案 D：旋转矩阵 fusion + 大 K matmul（理论收益最大）

> ❌ **已裁决：删除**（见上方裁决表）。以下为原始记录，保留备查。
> 编号已复用于新 P6-3（per-expert `index_add_` scatter 优化，由 P6-0 实测发现）。

- **思路**：多个 group 的旋转矩阵合并成块对角 `R = diag(R0, R1, ..., Rn)`，则 `x_rot = x @ R^T` 整个 K 维度一次旋转完成，之后用完整 K 的 dequant + matmul，Tensor Core 利用率大幅提升。
- **预期收益**：理论 30-50%+（取决于利用率提升），但 packed indices 需确认 K 维连续性。
- **难度**：高（算法层面支持 + 内核改动）。
- **建议**：先做小范围 PoC 验证 Tensor Core 利用率提升上限，再决定是否全面推进。
- **来源**：new-wax16-plan-260820 P4-4 未做项 / 方案 D。

#### 已废弃 P6-4（旧）：`triton_fused_dual_matmul` 的 grouped 版本 fusion

> ❌ **已裁决：删除**（见上方裁决表）。以下为原始记录，保留备查。
> 编号未被复用。

- 主路径未使用 dual_matmul；若接线（依赖 P6-1 统一 seed），可进一步合并。
- **难度**：中，**优先级低**（主路径未使用）。
- **来源**：new-wax16-plan-260820 P4-4 未做项。

---
### P7

- MoE 这条线——量化 + kernel + 布局 + 调优 + 旋转 + Python 层——探索性空间已经吃干净了，P7 之前规划的方法已经没有优化空间。

---

### P8 - 整体优化 attention WxA16 加速方法

> **背景（2026-08-27）**：P6 后实测 linear_attn 层每层 ~0.65s × 30 层（另 10 层 full attention），而 MoE forward
> 只剩 ~0.105s/层（`logs/0822.5.wxa16.p6-0.log`），attention 是当前最大杠杆。
> 关键现状：**FLA / causal-conv1d 未安装**（`modeling_qwen3_5_moe.py:216-218`），
> `torch_chunk_gated_delta_rule`（:245）整体 fp32，内部有 Python 循环；
> causal conv1d 走 `F.silu(self.conv1d(...))` torch 原生 fallback（:497-498）。
> **路线决定（本人）**：不换 FlashAttention / 不依赖 FLA 安装，先按 P4/P5/P6 经验
> 优化 kernel 细节（fp16 化、循环消除、融合、调优），P8-0 暂缓。

#### P8-0：Flash Attention 风格在线反量化 Attention（KV cache 量化）⏸ 暂缓

> 本人倾向不换 FlashAttention，先做 kernel 细节优化；本项保留待将来重新评估。

- **问题**：Attention 的 Q/K/V 投影被量化了，但 attention 计算本身全精度；若 K/V cache 也做 TurboQuant 量化，在 Flash Attention kernel 内部实时反量化，可大幅减少 KV cache 带宽。
- **难度**：很高（深度修改 Flash Attention kernel）。
- **来源**：new-wxa16-plan-260818.md §9。

#### P8-1：attention 分阶段 CUDA event 测量（P6-0 方法移植）

- **为什么是第一步**：和 MoE 侧一样，`[Layer N] time` 是 wall-clock，无一次 synchronize，
  各子段真实 GPU 占比未知。P6-0 的经验证明不先测量就排序会踩错方向的坑。
- **方案**：复用 `turboquant_utils/cuda_profiler.py` 的 `CudaStageProfiler`，
  在 `Qwen3_5MoeGatedDeltaNet.forward`（:438）接 in_proj_qkv / in_proj_z / in_proj_b /
  in_proj_a / conv1d / delta_rule / norm / out_proj 八个 stage，
  `Qwen3_5MoeAttention.forward`（:675）接 qkv 投影 / RoPE / attn / o_proj。
  开关默认关闭，不影响主流程。
- **判据输出**：决定 P8-2 ~ P8-6 的优先级排序，以及 delta rule 内部
  （chunk 内 WY 递归 vs 32 chunk 串行循环 vs fp32 cast）各自占比。
- **难度**：低。**风险**：零（测量开关）。

**状态**：✅ 已落地。`turboquant_utils/attn_profile.py`（monkeypatch 包装，可逆、
关闭时零开销，`patch_attn_profiling` / `unpatch_attn_profiling`），
测试 `test/test_p81_attn_profile.py`（真实 config + batch=32/seq=2048 + bf16）。

**实测结果（RTX 5090，真实 eval 形状，GatedDeltaNet 模块级，GPU 时间）**：

| stage | ms | 占比 |
|---|---|---|
| **delta_rule_chunk** | **95.4** | **70.4%** |
| norm（RMSNormGated，fp32） | 11.6 | 8.6% |
| conv1d（torch fallback） | 9.6 | 7.1% |
| in_proj_qkv | 9.2 | 6.8% |
| out_proj | 4.7 | 3.4% |
| in_proj_z | 4.6 | 3.4% |
| in_proj_b / a | 0.4 | 0.3% |
| 合计 | **135.4** | 100% |

**delta rule 内部拆解**（逐行复制版插 stage，与原函数逐位一致 max_abs=0，
复制版总耗时 94.1ms ≈ 模块级 95.4ms，拆解可信）：

| 内部 stage | ms | 占比 |
|---|---|---|
| **d_in_wy_prep**（k_beta@k^T 等 3 个大 bmm + 63 次 WY 循环） | **48.1** | **51.1%** |
| **d_in_chunk_loop**（32 chunk 串行递推 + 小 matmul） | **23.8** | **25.3%** |
| d_in_pad（F.pad × 5，**pad_size=0 时仍做 5 次全量拷贝**） | 5.7 | 6.0% |
| d_in_cast（bf16→fp32 transpose+contiguous × 5） | 5.5 | 5.9% |
| d_in_l2norm | 3.5 | 3.8% |
| d_in_beta / d_in_decay_mask / d_out_finalize | 7.5 | 7.9% |

**由此修正对 P8 子项的排序**：

1. **P8-2（fp16 化）收益最大**：wy_prep 的三个 bmm（约 29ms）+ chunk loop 内 bmm
   （约 20ms）全部 fp32，转 fp16 直接吃 Tensor Core。原 P8-3/P8-4 都排在它后面。
2. **新零风险项（P8-2 前置顺手做）**：seq=2048 整除 chunk=64 时 `pad_size=0`，
   F.pad 是 5 次纯浪费的全量拷贝（5.7ms/层 ≈ 模块 4%）——加一行 `if pad_size:` 即可。
3. P8-3（WY 63 次循环向量化）≈ 19ms 上限；P8-4（chunk 循环融合 kernel）
   目标 23.8ms 的 launch 开销与访存。
4. **重要口径修正**：eval 日志的 `attn: ~0.65s/层` 是 wall-clock（CPU 侧排队/
   分配器膨胀），实测 GPU 时间只有 **~135ms/层**（P6-0 教训在 attention 侧复现）。
   折算 attention GPU 约占总 eval GPU 时间 ~32%（MoE 约 ~15%），仍是最大单项杠杆，
   但 0.65s 的量级被夸大了 ~4.6 倍，后续收益预期按 GPU 口径算。

**实施载体待定（需本人决定）**：主流程实际执行 transformers 内置版的
`torch_chunk_gated_delta_rule`（仓库根副本与内置版逐字节相同）。P8-2~P8-6 的
改动要么（a）在仓库内实现优化版并在模型加载时 monkeypatch 替换（与 MoE wrapper
同模式，不碰 site-packages）；要么（b）直接改 site-packages 内置文件。
建议 (a)，与项目现有架构一致。

#### P8-2：delta rule / conv1d 全链路 fp16 化（P2 方法）

- **问题**：`torch_chunk_gated_delta_rule` 把 q/k/v/beta/g 全部 `.to(torch.float32)`
  （`modeling_qwen3_5_moe.py:261-263`），整个 delta rule 在 fp32 下跑。MoE 侧 P2 已经
  证明「输出/码本/norms 全 fp16 + 关键处保留 fp32」路线可行，attention 侧还没有做。
- **精度敏感点（必须先验证再落）**：
  - `g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`（:519）——
    注释已警告 fp16 下 A 可能 -inf，g 的 cumsum/exp 链是数值重灾区；
  - l2norm、decay_mask 的 exp、状态递推的乘法链。
- **方案**：q/k/v 在 fp16 下做 bmm/einsum，g/decay/状态递推保留 fp32 或 fp32 计算后
  写回 fp16（分点验证 ppl）。
- **预期收益**：delta rule 的 matmul 部分 2x（fp32→fp16 Tensor Core），
  视 P8-1 测量占比折算。
- **难度**：中。**风险**：精度——以 ppl 为准，不行就局部回退 fp32（不整体回退）。

**状态（2026-08-27 实测，`test/test_p82_delta_rule_fp16.py`）**：
实现载体已落定 (a)——`turboquant_utils/delta_rule.py`（fast 版 + `patch_delta_rule`
monkeypatch，不碰 site-packages），拆成两个子项分别验证：

- **P8-2a pad 跳过 ✅ 已落地**：seq 整除 chunk 时 `pad_size=0`，跳过 5 次 F.pad 全量拷贝。
  与原函数**逐位一致**（含 seq 不整除时仍走 pad 的路径）。实测 delta_rule 94.3→90.1ms
  （函数级）、stage 95.5→91.3ms（模块级），≈ **-3% 模块 forward**。默认开启。
- **P8-2b fp16 bmm ❌ 实测无收益，重定向到 P8-4**：真实 g（恒负）下误差 3.9e-3 相对
  （可接受），但 torch 层 fp16 反而 **+9.5ms 回归**。原因：这些 bmm 的成本是
  **16384 个 64×64 小 GEMM 的 cublas 批量 launch 开销，不是 FLOPs**——fp16 只省
  FLOPs 不省 launch，还要付 cast。**fp16 的正解位置在 P8-4 融合 kernel 内部**
  （tl.dot fp16 无 cublas 批量开销）。`ENABLE_FP16_BMM` 保留作消融开关，默认关闭。
- P8-2c（scale 并入 beta）暂缓，收益过小。

**排序调整**：P8-1 实测后原「fp16 化收益最大」的判断修正——delta rule 的瓶颈
不是精度类型而是**批量小 GEMM 的 launch 开销 + Python 循环**（wy_prep 48ms +
chunk 循环 24ms 里 FLOPs 只占零头）。**P8-4（融合 kernel）与 P8-3（WY 向量化）
的优先级提升，成为 delta rule 的主攻方向**。

#### P8-3：消除 chunk 内 Python 循环（P6-2 / P5-2 方法）

- **问题**：`torch_chunk_gated_delta_rule:290-293` 的
  `for i in range(1, chunk_size)`（63 次迭代，每次 2 个 `.clone()` + 小张量乘加）
  以及 :306-316 的 32 chunk 串行循环——和 MoE 逐 expert 循环同一类病：
  Python 循环体开销 + 小形状 kernel launch 放大。
- **方案**：chunk 内 WY 递归用 cumsum / associatve-scan 向量化重写（消除 63 次迭代）；
  chunk 串行循环因递推依赖不能并，但可以搬进单个 Triton kernel 内做
  （见 P8-4），先做前者成本低。
- **难度**：中。**风险**：低（纯重写，逐位可对拍验证）。

**状态（2026-08-27）**：✅ 已落地。发现该递归有闭式解——三角递归
`L_new = L + L·L_new` ⟺ `L_new = (I−L)⁻¹L`（单位下三角批量 solve_triangular
一次完成，cuBLAS trsm）。实测（`test/test_p83_wy_solve.py`，B=32 seq=2048）：
- 数值：solve vs 63 次循环 max_rel **5.5e-5**（fp32 trsm 舍入，远低于已上线的
  1e-2 噪声量级）；模块级 forward 对拍 3.2e-3 PASS
- 收益：delta rule 94.1 → **56.9 ms（函数级 -33ms）**；模块级 stage 95.3 → **58.1 ms（1.64x）**

#### P8-4：chunk delta rule 融合 Triton kernel（P4-4 / P4-7 方法）

- **问题**：delta rule 内是大量 64×64×128 级的小 bmm/einsum
  （`attn @ v_beta`、`q_i @ k_i^T`、`k_i^T @ v_new`……），单个体量小、launch 多。
- **方案**：把 chunk 内计算（qk^T + decay + WY + @v）写成一个 Triton kernel，
  参考 P4-4 的「多 group 融合单 kernel」和 P4-7 的「小 matmul 合并 bmm」思路。
  状态递推（chunk 间串行）留在 kernel 外循环，每个 chunk 一次 launch（32 次/层）。
- **难度**：中高。**风险**：数值对齐以 P8-2 的 fp32 路径为参考逐步逼近。
- **前置**：P8-1 测量确认 delta rule 内部 matmul 占比够大。

**状态（2026-08-27）**：范围已按 P8-2b/P8-4a 的实测修正。两项 torch 层预演都证明
**小批量 GEMM 的成本在 cublas launch 开销而非 FLOPs**（fp16 化 +9.5ms、bmm 合并
+7.5ms 双双回归），正确载体是 Triton kernel 内部的 tl.dot。P8-4a 已实测
（数值逐位一致，性能回归默认关闭，概念并入本项）。

**P8-3 落地后的 fast 版内部 stage 实测（49.9ms，P8-4 的目标清单）**：

| stage | ms | 占比 | P8-4 可吃掉的量 |
|---|---|---|---|
| **chunk_loop（32 次串行 × 5 个小 bmm）** | **23.9** | **47.8%** | 大部分（launch + 访存） |
| cast（bf16→fp32 transpose × 5） | 9.0 | 18.1% | 部分（kernel 直接吃 bf16/fp16） |
| wy_bmm（k_beta@key^T） | 4.8 | 9.6% | 全部 |
| wy_solve（批量 trsm） | 4.7 | 9.5% | 全部（kernel 内 WY 递推） |
| decay / beta_reshape / finalize | 7.5 | 14.9% | 大部分 |
| pad | 0.0 | 0% | —（P8-2a 已消） |

**当前累计**：P8-2a + P8-3 已把 delta rule 从 95.4ms 压到 58.1ms（1.64x），
attention 模块 forward 从 135.4ms 降到 ~98ms（约 -28% GPU）。

**真实 eval 结果（2026-08-27，`logs/0822.8.wxa16.p8.log`，git 77dbfe4+，接线生效后）**：

| 数据集 | P6 基线 | P8 后 | 变化 |
|---|---|---|---|
| c4 | 107.28 s | **98.12 s** | **-8.5%（-9.16s）** |
| ppl | 11.2683 | **11.2679** | 无损失 ✓ |

**关键结论（勘误后的定案）**：GPU 侧 delta rule 1.64x 在端到端上**几乎 1:1 兑现**
（-9.16s ≈ GPU 口径估算 -9.5s）。「eval 是 CPU bound」假说**证伪**——之前的判断
全部建立在 0822.8 未接线的无效数据上。稳态下 wall 跟随 GPU，GPU 加速会直接转化为
eval 时间。

**由此恢复 P8-4 的吸引力**：既然 wall 跟随 GPU，fast 版内部剩余的
chunk_loop 23.9ms（47.8%）+ cast 9.0ms + wy_bmm/wy_solve 11.7ms 若再砍掉一半，
端到端还能再拿 ~4-6%。P8-4（融合 Triton kernel）**重新排队，待本人确认开工**。

**原方案（保留备查）**：

- **问题**：delta rule 内是大量 64×64×128 级的小 bmm/einsum
  （`attn @ v_beta`、`q_i @ k_i^T`、`k_i^T @ v_new`……），单个体量小、launch 多。
- **方案**：把 chunk 内计算（qk^T + decay + WY + @v）写成一个 Triton kernel，
  参考 P4-4 的「多 group 融合单 kernel」和 P4-7 的「小 matmul 合并 bmm」思路。
  状态递推（chunk 间串行）留在 kernel 外循环，每个 chunk 一次 launch（32 次/层）。
- **难度**：中高。**风险**：数值对齐以 P8-2 的 fp32 路径为参考逐步逼近。
- **前置**：P8-1 测量确认 delta rule 内部 matmul 占比够大。

**第一步 ✅ 已落地（2026-08-27，`turboquant_utils/delta_rule.py` 的
`_delta_chunk_kernel` + `_triton_chunk_loop`，`ENABLE_TRITON_CHUNK` 默认开）**：
- 每块 5 个小 bmm + elementwise 融合成一个 kernel（每 program 一个 (batch, head)，
  K×V 双层分块 32×32，状态在 kernel 外循环的缓冲间交换）。
- 数值：vs torch 版 max_rel **3.9e-3**（tl.dot 与 cublas fp32 累加顺序噪声，
  与 fast-torch 版对原函数的距离同量级）；vs transformers 原函数 3.9e-3 PASS；
  模块级 forward 对拍 6.4e-3 PASS。
- 收益（真实形状 B=32 seq=2048）：chunk_loop **23.9 → 11.6ms（2.06x）**，
  delta rule 函数级 56.9 → 45.7ms；模块级 stage **95.6 → 46.7ms（2.05x vs 原函数）**。
- SM 分析：grid=1024 / 170 SM = 6.02 波，整体利用率 86.1%。
- 测试：`test/test_p84_triton_chunk.py`（5 项全过）。

**fast 版当前内部拆分（38.6ms）**：chunk_loop 11.6（30.1%）、**cast 9.3（24.0%）**、
wy_solve 5.4、wy_bmm 4.9、其余 7.5。

**真实 eval 确认（2026-08-27，git 3265adc+，`logs/0822.8.wxa16.p8.log`）**：
c4 98.12 → **95.43s（-2.69s）**，与预估 -2.9s 一致；wiki **60.43s**（比 P6 -12.4%）；
ppl 7.7938 / 11.2671 无损失。**wall 跟随 GPU 再次验证**。

**第二步 ✅ 已落地（2026-08-27，cast 缩减）**：q/k 保持 bf16 直通 Triton kernel
（transpose 只搬一半字节、省掉 .to(fp32) 转换），scale 乘法移进 kernel 内以 fp32 完成
（bf16→fp32 转换精确，数值与旧路径一致）。实测：cast **9.3 → 7.1ms**、
函数级 45.7 → 42.9ms、模块级 stage **95.7 → 44.0ms（2.18x vs 原函数）**，
对拍不变（3.9e-3 舍入噪声）。

**fast 版当前内部拆分（24.7ms，P8-6 调优后）**：
cast 7.1（28.8%）、chunk_loop 6.7（27.3%）、wy_fused 3.6（14.6%）、其余 7.3。

**真实 eval（wy_fused 后，git 746d1f8+wy_triton 修复）**：
c4 93.64 → **89.74s（-3.9s）**，wiki 60.6 → **58.01s（-2.6s）**，
ppl 11.2672 / 7.7946 无损失。

**第三步 ✅ 已落地（wy_prep 融合 kernel）**：wy_bmm + WY 递归（63 步寄存器内）
+ 2 个后置 bmm 单 kernel 完成，attn 不出寄存器（省 536MB 写 + 两次读），
同时替代 solve_triangular。wy_bmm + wy_solve 从 11.4ms → **5.5ms**。
模块级 stage：原 95.6 ms → fast 版 **33.1 ms（2.89x）**。

**P8-6 ✅ kernel 自动调优（eval 验证，git 766a5d3+）**：chunk + wy 两个 Triton kernel
的 BLOCK/warps/stages 网格搜索（参照 P5-3 方法）。
chunk：BLOCK_V=32→128, warps 8→4, stages 1→2，6.3→2.9ms（+54%）；
wy：BLOCK_K=32→128, stages 1→2，2.5→1.8ms（+28%）。
函数级 31.3→**24.7ms（+21%）**，模块级 2.89x→**3.53x**。
端到端：c4 89.74→**88.36s（-1.38s，-1.5%）**，wiki 58.01→**57.52s（-0.49s，-0.8%）**，
ppl 11.268 / 7.7944 无损失。收益小于预期（delta rule 只是全模型一部分）。

**下一步候选**：
- P8-5 causal conv1d（需先测占比，当前非瓶颈）
- 更多 kernel 融合（cast/decay/beta_reshape 合并进 chunk 循环的第一个 kernel）

#### P8-5：causal conv1d torch fallback 优化（P4-7 方法）

- **问题**：`F.silu(self.conv1d(mixed_qkv))` 走 torch 原生 Conv1d（:498），
  groups=conv_dim 的深度卷积，torch 实现对小 kernel 未必最优。
- **方案**：Triton 分组深度卷积（或 shift+scale 实现）；若 P8-1 测出 conv1d 占比小
  则降优先级。
- **难度**：中。**风险**：低（数值可对拍）。

#### P8-6：新 kernel 的 tile 联合调优（P5-3 方法）

- **方案**：P8-2~P8-5 落地后，对每个新 Triton kernel 用
  `turboquant_utils/kernel_autotune.py` 做 BLOCK/warps/stages 联合搜索。
  教训照搬：**先确认真实 eval 形状再搜**（P5-3 的 B=2048 vs B=32 反方向教训），
  delta rule 的 chunk 形状 (64, 64, 128) 与 MoE 完全不同，旧配置不可复用。
- **难度**：低。**收益**：稳定 +5~20%（参照 P5-3）。

---

### P9 — 备选 / 独立技术路线（长期方向，非增量优化）

#### P9-1：WxA16 到 WxA4，WxA8 的整体进化。
- 改变现在 weights 反量化为 fp16 计算的方法，而是整体切换为输入量化后和 weights 继续 int 乘法的方法。

#### P9-2：ROADMAP 第四阶段：Machete / Marlin CUDA kernel（Blackwell wgmma）
- 针对 RTX 5090 (SM12.0) 的高性能推理替代路线：从 vLLM 提取 Machete kernel（支持 wgmma），或 Marlin MoE kernel，替代 Triton 全路线。
- **难度**：高（CUDA/C++ 扩展 + 集成）。
- **来源**：ROADMAP.md 第四阶段。

#### P9-3：cuTile 后端集成到 WxA16 MoE 推理路径 ✅ 探索
- **现状**：`turboquant_utils/cutile_kernels.py` 已实现 `cutile_fused_matmul` / `cutile_fused_matmul_autotuned` / `cutile_fused_dual_matmul`，但**未接入 WxA16 MoE forward**（MoE 文件无 `use_cutile`/`cutile` 引用）。CLI 有 `--disable-gpu-fused` 开关但面向 Metal/macOS。
- **方案**：给 `WxA16BitPartitionedGroupMoE` 加 cuTile 后端选项，实测对比 Triton vs cuTile（cuTile 的 MMA 可能比 Triton `tl.dot` 更高效，尤其 TF32/FP16 Tensor Core 利用率）。
- **难度**：中，**预期收益不确定（需实测）**。
- **建议**：先做单 kernel 对比（测试程序对比 cutile vs triton 在真实形状下性能），有明显收益再接入主流程。
- **来源**：new-wxa16-plan-260818.md §1。

#### P9-4：QwenMultiLinear 备选方案（保持 grouped_gemm 格式）
- 不转换为传统格式，直接在 Qwen3.5 grouped_gemm 格式上工作，打包所有专家指针实现一次 kernel launch，保存 `bit_to_indices` 元数据避免信息丢失。
- **现状**：纯讨论备选，未实现。
- **来源**：ROADMAP_ALTERNATIVE_QWENMULTILINEAR.md。

---

## 二、已评估 / 已否决 / 已降级（明确不做，避免重复踩坑）

| 项目 | 结论 | 原因 |
|------|------|------|
| 旋转融合进 Triton kernel（Hadamard） | ⛔ 降级 | 纯矩阵乘仅占 ~5%，收益极小 |
| P4-3 Gate-Up epilogue fusion（silu+mul 合进 kernel） | ⏸ 暂不做 | 收益 <5%（silu+mul 仅 ~2-3%），且拆成两次小 tl.dot 反而慢 |
| P4-1 codebook gather 深度优化（shared mem / tl.where 展开） | ⏸ 优先级下调 | 大 K 场景 gather 占 86.5% 但 Triton 编译器层面未找到有效方法；unpack 向量化 per-K 收益 <5% |
| P4-5 旋转跨 bit 复用（原判不可行） | 🔄 复活条件 | 原因 seed 不同；若统一跨 bit seed（Tier1-3）即可复用 |
| shared-mem-codebook | ❌ 不做 | 码本仅 2^bit 个 fp32（最大 1KB），L1 已缓存，手动管理反而增 barrier 开销 |
| Triton autotune 调参 | ❌ 不做 | 动态小 kernel，搜索编译开销 > 收益 |
| 退回 clone 旧路径 | ❌ 不作为选项 | 仅回 dfb8fc1 水平、无增益，且按规范须本人同意 |
| P7-2 Split-K（2026-08-26 移入） | ❌ 不做 | 前提「单 expert 几十 token」对 eval 不成立（实测 per-expert B=2048，B 维并行度充足）；仅在 decode（B≈1）场景才重新成立 |
| P7-4 权重预排序（2026-08-26 移入） | ❌ 不做 | 2-bit 码本仅 4 个 fp16，常驻寄存器/L1，不存在「相邻访问提升 L1 命中率」空间；与 shared-mem-codebook 同理 |
| P7-3 Bit-packing 32/128-bit 向量化加载（2026-08-26 移入） | ❌ 不做 | 判定测试实测持平（gate_up 95.4% / down 96.6%，变体相对基线）：packed 权重只占 kernel 访存一小部分，瓶颈在输入 fp16 tile 加载与 tl.dot，与 packed 加载宽度无关 |

> **P7-2 / P7-4 原文备份**（移入否决表前的原始记录，保留备查）：
> - **P7-2 Split-K 优化**（来源 new-wxa16-plan-260818.md §6）：当 M 很小（单 expert 仅几十 token）
>   但 K 很大时 GEMM 并行度不够；对 K 维 split-K，多 block 算同一输出 tile 不同 K 段，最后原子加。
>   适用：MoE token 数少的 expert 场景。
> - **P7-4 权重预排序**（来源 new-wxa16-plan-260818.md §10）：按激活数值排序权重行/列，使相邻位置
>   访问相近码本条目，提升 L1 命中率。需配合量化算法改。

> P7 — 已清零（2026-08-26 全部移入「已否决」表）
> 原 P7-1（cuTile）、P7-5（Flash Attention 在线反量化）已由本人重新归类至 P8-3 / P8-0。
> 原 P7-2 / P7-4 因前提失效移入「已否决」表（原文保留在表后附注）。
> 原 P7-3 经判定测试实测后同样移入「已否决」表，至此 **P7 阶段整体清零**。
> 该阶段遗留的可执行项为 0 —— MoE 内部的探索性优化空间已被 P5/P6 吃干净，
> 下一个优化方向是 attention（见 §〇「目标转变」）。

> P7-3 判定记录（2026-08-26，测试：`test/test_p73_bitpack_judge.py` + `test_p73_ref_check.py`）

判定前提：「kernel 是访存 bound 且 packed 权重加载路径没吃满」。
在真实 eval 形状（per-expert B=2048，2-bit）下对比 uint8 逐字节 vs uint32 向量化加载：

| 方向 | uint8 基线 | uint32 变体 | 相对 | 结论 |
|---|---|---|---|---|
| gate_up (B=2048, N=1024, K=2048) | 0.064 ms | 0.061 ms | 95.4% | 基本持平 |
| down (B=2048, N=2048, K=512) | 0.034 ms | 0.032 ms | 96.6% | 基本持平 |

- 数值：两版对 fp32 参考的误差比 **1.000x**（变体数学正确；A/B 间差异是 tl.dot
  累加顺序的 fp16 噪声）。
- 结论：packed 权重只占 kernel 访存的一小部分（2-bit 下每 group 仅 32 字节），
  瓶颈在输入 fp16 tile 加载与 tl.dot 计算，与 packed 加载宽度无关。**P7-3 判死刑。**

---

## 三、方法论教训（critical，写进文档防复发）

1. **micro 测试 ≠ 真实 eval（三次教训）**：
   - 旋转缓存 cold 假象：micro 每轮 `clear_rotation_cache()` 打冷得 "90% 节省"，真实 eval 缓存本就热，端到端无收益。
   - COL_START 编译风暴：constexpr 参与编译 key，真实流程上千 col_start 取值触发 ~2000 次重编译（首 mini_batch 240s）；micro 仅 1~2 取值测不出。
   - 去 clone 净回归：micro 小形状省 44.5%，真实 B≈9280 宽 stride 惩罚反超 +113.7/+229.8µs 每 call。
2. **constexpr 坑**：Triton 把 constexpr 取值写进编译缓存 key。**凡是每个 expert/每个 group/每个 bit 都会变的量，一律不得做 constexpr**，必须改运行时参数。复用 `test/test_colstart_recompile.py` 以 `~/.triton/cache` 目录增量计数排查。
3. **"省掉一次小操作"类优化**必须在真实 B、真实张量尺寸（尤其是否超 L2）下复测 kernel 本体；寻址 stride 代价随 B 与张量尺寸非线性放大，micro 形状完全不可见。
4. **测试程序区分 cold/warm 路径**：warm 才是真实场景；全模型 eval 由本人手动跑，不自动执行。

---

## 四、建议的迭代顺序

### 4.1 原顺序（2026-08-24，已执行完 P5，保留备查）

1. ~~**P5-1（group-first 重排）**~~ ✅ 已完成
2. ~~**P5-3（BLOCK 联合调优）**~~ ✅ 已完成（+25% 总时间）
3. ~~**P5-2（多 expert grouped GEMM）**~~ ⚠️ 已实现但 eval 变慢（269.4→288.6s），未并入 main；已决定不重测
4. ~~**P6-1（跨 bit 旋转复用 + dual_matmul 接线）**~~ ⛔ 前提作废（对应「已废弃 P6-1（旧）」）
5. ~~**P6-2 + P6-4**~~ ❌ 删除（对应「已废弃 P6-2/P6-4（旧）」）
6. ~~**P6-3（方案 D 大 K 融合）**~~ ❌ 删除（对应「已废弃 P6-3（旧）」）
7. ~~**P7**~~ ⛔ 已整体清零（2026-08-27，全部移入「已否决」表）
8. **P8**：见下方 4.2 的调整

### 4.2 当前顺序（2026-08-27 更新）

1. **P6-0（CUDA event 分阶段计时）** — ✅ 已落地。前置条件，不做这个后面无法排序也无法验证收益。
2. **P6-1（gate_up 旋转提到 expert 循环外）** — ✅ 已落地，数学严格等价，旋转 FLOPs 与访存均降 8 倍。
3. **P6-2（per-expert 重复常量 / 冗余分配消除）** — ✅ 已落地，低风险 Python 层清理。
4. **P6-3（per-expert `index_add_` scatter）** — P6-0 实测发现的第二大开销（~16%），待做。
   注：attention 已成更大杠杆（见 §〇），本项相对优先级下调。
5. ~~**P8-1（attention 分阶段 CUDA event 测量）**~~ ✅ 已落地（`turboquant_utils/attn_profile.py`
   + `test/test_p81_attn_profile.py`，实测见 P8 节）。
6. **P8-2（pad 跳过 + fp16 化）** — ✅ P8-2a 已落地（逐位一致，-3% 模块 forward）；
   P8-2b 实测无收益已重定向到 P8-4。
7. **P8-3（WY 循环向量化）** — ✅ 已落地（1.64x GPU），但真实 eval 无兑现
   （见 P8 节结论）。**P8-4 ~ P8-6。**
8. **新方向（进行中）：定位 eval 的 CPU 侧开销**。⚠️ 注意：本项最初的动机
   （P8「零兑现」）已勘误撤销——0822.8 轮 P8 未接线，需重测后才能知道端到端
   到底有没有 CPU 差额。本地探针（`test/test_cpu_bound_probe.py`，
   真实形状 attn+MoE 背靠背 40 层）先排除三个嫌疑：
   - **层内无 CPU 瓶颈**：wall 163.0ms ≈ GPU 164.3ms（差额 -1.3ms）——
     模块级 forward 的 Python/launch/同步都不构成瓶颈；
   - D2H 同步（bincount().cpu()）代价随 GPU 队列长度线性增长
     （0.07→1.72ms @ 16 个排队 GEMM），但层内量级 <2ms，非主嫌；
   - 本地 JIT 缓存全命中（+0 编译），eval 首 batch 的 ~2s/层 属 JIT/冷启动类
     一次性成本，本地无法复现。
   结论：差额（若有）在 **eval 侧机制**（sequential 层搬运 ~0.04s/层、
   per-batch kwargs 构造、分配器）。已给 `eval_qwen35.py` 加
   `--cpu-profile` 插桩（逐层 wall/gpu/move/kwargs/forward/attn/moe 拆分，
   测量模式才 sync）。**待本人跑一轮 `--cpu-profile` 后按数据定位。**
7. **P9-1（WxA8 / WxA4）** — 战略优先级已上调。A16 的天花板是 fp16 tensor core 吞吐，
   而 W2A16 现在已经比 FP16 快 18%，P6 之后的增量空间有限；输入侧也量化后走整数乘法才是下一个数量级。
8. **P9-2（Machete/Marlin） / P9-3（cuTile） / P9-4（QwenMultiLinear）** — 独立技术路线，长期备选。

### 4.3 待定 / 需本人决策

- **P5-2**：已决定**不重测**（2026-08-26）。否决记录（eval 269.4→288.6s）与
  SM 填充度佐证数据保留在 P5-2 条目内备查，后续如硬件/形状变化可再评估。
- **P7**：阶段已整体清零（2026-08-27），全部移入「已否决」表。

> 每一步改动都需配套：测试程序先对齐主流程逻辑（含 GPU 填充度 / 并行 SM 分析）+ 本人在真实 eval 形状下手动复测，避免再次落入 micro≠real 陷阱。
