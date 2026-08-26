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

**当前端到端状态**：WxA16 量化层 forward 已从早期 ~19.7s 量级压到 ~3.5s 量级（Layer0，约 **2.4~5.6x** 累计），但相对 dense 层（~1.9s）仍有明显差距，真实 10 倍差距已被啃掉大半但**未完全消除**。

---

## 一、当前仍可优化的点（按 P 阶段排序）

> P5 最先做，P6/P7/P8 依次往后。同一 P 阶段内的子项主题相关、可并行或顺序灵活。
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

#### P5-2：多 expert 合并（MoE grouped GEMM）：同一 bit 所有 expert 一次 kernel ✅ 高优先
- **问题**：`wxa16_bit_partitioned_moe.py:366/385` 仍是 `for expert_idx` × `for bit_str` 的逐 expert 循环，每 expert 单独 launch。expert 多但每 expert token 少时 GPU 利用率极低，且 Python 循环体开销被放大。
- **方案**：参考 false-grouped 的 gather/scatter + bad-triton 的 3D grid 思路，但用 grouped GEMM 方式——把同一 bit 的所有 expert 权重打包好（已是 bit-partitioned 布局），一个 kernel 内用 `expert_info_ptr` 索引各 expert 的 token_start/token_count/weight_range，每个 SM 动态分配给某个 expert 避免负载不均衡。
- **预期收益**：MoE 部分 launch 从 `O(experts × bits × groups)` 降到 `O(bits × groups)`，并显著降低 Python 循环占比（约 70% 的差距来源）。
- **难度**：中高。**注意**：3D grid 按 expert 并行的 bad-triton 方案已被证伪（负载不均衡 + 边界检查开销）；正确做法是 expert_info 指针 + 仅算需要的行（grouped GEMM 思路）。
- **分阶段推进**：
  1. 先做同一 bit 内所有 expert 的合并（gate_up 一个 kernel + down 一个 kernel），bit 间保留外层循环。改动小，先吃到大部分 launch 减少收益。
  2. 再优化负载均衡 + down 路径 in_feature-slice 适配。
- **来源**：new-wxa16-plan-260818.md §1.2。
**效果不好**：wxa16 load eval 速度下降从 269.4 到 288.6s。

#### P5-3：BLOCK_B / BLOCK_N / BLOCK_K / num_warps / num_stages 联合调优
- **问题**：P4-2 单维离线调优（+9.5%）的结果是局部最优。五个 tile 参数（BLOCK_B / BLOCK_N / BLOCK_K / num_warps / num_stages）之间强耦合——例如 BLOCK_B 变大后需要更多 warps 藏延迟，BLOCK_K 变大后需要更深的软件流水线——单维扫无法找到全局最优。
- **方案**：离线全量网格搜索，在真实 eval 形状下对每 bit-width 找最优配置并硬编码到 `_FUSED_GROUPED_CONFIG`。
  - 搜索空间：BLOCK_B∈{16,32,64}, BLOCK_N∈{16,32,64,128}, BLOCK_K∈{32,64,128}, num_warps∈{2,4,8}, num_stages∈{2,3,4}（过滤后约 150-200 组有效配置）。
  - **gate_up / down 拆两套配置**：两者 K/N 维度互换，计算模式不同，最优 tile 大概率不同。
  - **多 B 值鲁棒性测试**：在 B∈{8,16,32,64,128} 上分别搜索，验证最优配置的形状鲁棒性；若差异大则做 B 自适应配置表（launch 前查表，开销可忽略）。
  - 单 kernel micro-bench 做搜索，e2e bench (`test_triton_mp_moe_e2e_bench.py`) 做最终验证。
- **难度**：低，**收益中等（5-15%）**，零风险（纯配置参数，不改计算逻辑）。
- **学术价值**：MoE 小批量量化 GEMM 的 tile size 性能特征分析、bit-width 与最优配置的关系规律、跨形状鲁棒性研究——公开文献中这类细粒度 empirical study 较少。
- **注意**：
  - 不用 runtime autotune（首次编译开销太大），离线扫后硬编码。
  - 用真实 eval 形状扫，避免 micro≠real 陷阱。
  - 每组配置多次测量取中位数，减少 GPU 功耗/温度波动干扰。
- **详细方案**：`roadmaps/wxa16-p5-3-joint-kernel-autotuning.md`
- **来源**：new-wax16-plan-260820 P4-2。

---

### P6 — 跨 bit 融合 & Kernel 深化（中优先级，P5 之后或并行推进）

#### P6-1：旋转结果跨 bit 复用 + 接线 `triton_fused_dual_matmul`
- **问题**：
  - 旋转作用在输入 x 上，仅依赖 `g_dim/group_size/g_start`（跨 bit 相同），但当前每 bit 用**不同 seed**（`seed=42+bit`/`42+bit+1000`），导致旋转矩阵随 bit 不同而不同，无跨 bit 复用（profile §8.4 已验证）。
  - `triton_fused_dual_matmul` 已实现（`triton_kernels.py`），但**主 MoE 路径完全未调用**（唯一调用在 `turboquant_utils/module.py:757` 的单 Linear 路径），属死代码。
- **方案**：统一跨 bit seed → `x_rot_g` 跨 bit 相同，计算一次喂给多 bit kernel；`SAME_INPUT` 成立后可接线 `triton_fused_dual_matmul` 合并两次 matmul。
- **预期收益**：视情况，旋转占 grouped 时间 ~40%（cold）/~30%+（warm 稳态）；合并两次 matmul 减少一次 kernel。
- **难度**：中（耦合量化侧改动：统一 seed + 重新验证精度）。
- **前置依赖**：先做消融实验，验证统一 seed 后 perplexity 变化，确认精度可接受再推进。
- **来源**：profile §8.4、§十 遗留待办；new-wax16-plan-260820 P4-4 未做项。

#### P6-2：混合 bit-width 场景的 fusion
- 2-bit 段 + 4-bit 段分别 fused 后，再加一次加法（目前逐 bit 各自 fused 再 Python 累加）。
- **难度**：低-中，**收益中等**。
- **与 P5-2 协同**：多 expert 合并后再做 bit 间 fusion，kernel 结构更清晰。
- **来源**：new-wax16-plan-260820 P4-4 未做项。

#### P6-3：方案 D：旋转矩阵 fusion + 大 K matmul（理论收益最大）
- **思路**：多个 group 的旋转矩阵合并成块对角 `R = diag(R0, R1, ..., Rn)`，则 `x_rot = x @ R^T` 整个 K 维度一次旋转完成，之后用完整 K 的 dequant + matmul，Tensor Core 利用率大幅提升。
- **预期收益**：理论 30-50%+（取决于利用率提升），但 packed indices 需确认 K 维连续性。
- **难度**：高（算法层面支持 + 内核改动）。
- **建议**：先做小范围 PoC 验证 Tensor Core 利用率提升上限，再决定是否全面推进。
- **来源**：new-wax16-plan-260820 P4-4 未做项 / 方案 D。

#### P6-4：`triton_fused_dual_matmul` 的 grouped 版本 fusion
- 主路径未使用 dual_matmul；若接线（依赖 P6-1 统一 seed），可进一步合并。
- **难度**：中，**优先级低**（主路径未使用）。
- **来源**：new-wax16-plan-260820 P4-4 未做项。

---

### P7 — 探索性优化（高风险高回报，逐个验证）

#### P7-1：cuTile 后端集成到 WxA16 MoE 推理路径 ✅ 探索
- **现状**：`turboquant_utils/cutile_kernels.py` 已实现 `cutile_fused_matmul` / `cutile_fused_matmul_autotuned` / `cutile_fused_dual_matmul`，但**未接入 WxA16 MoE forward**（MoE 文件无 `use_cutile`/`cutile` 引用）。CLI 有 `--disable-gpu-fused` 开关但面向 Metal/macOS。
- **方案**：给 `WxA16BitPartitionedGroupMoE` 加 cuTile 后端选项，实测对比 Triton vs cuTile（cuTile 的 MMA 可能比 Triton `tl.dot` 更高效，尤其 TF32/FP16 Tensor Core 利用率）。
- **难度**：中，**预期收益不确定（需实测）**。
- **建议**：先做单 kernel 对比（测试程序对比 cutile vs triton 在真实形状下性能），有明显收益再接入主流程。
- **来源**：new-wxa16-plan-260818.md §1。

#### P7-2：Split-K 优化
- 当 M 很小（单 expert 仅几十 token）但 K 很大时 GEMM 并行度不够；对 K 维 split-K，多 block 算同一输出 tile 不同 K 段，最后原子加。
- **适用**：MoE token 数少的 expert 场景。
- **来源**：new-wxa16-plan-260818.md §6。

#### P7-3：Bit-packing 布局优化：按 32/128-bit 对齐
- 当前 uint8 逐字节加载，对 memory coalescing 非最优；改成 32-bit/128-bit vector 存储+加载。
- **来源**：new-wxa16-plan-260818.md §8。

#### P7-4：权重预排序（codebook lookup 友好）
- 按激活数值排序权重行/列，使相邻位置访问相近码本条目，提升 L1 命中率。需配合量化算法改。
- **来源**：new-wxa16-plan-260818.md §10。

#### P7-5：Flash Attention 风格在线反量化 Attention（KV cache 量化）
- **问题**：Attention 的 Q/K/V 投影被量化了，但 attention 计算本身全精度；若 K/V cache 也做 TurboQuant 量化，在 Flash Attention kernel 内部实时反量化，可大幅减少 KV cache 带宽。
- **难度**：很高（深度修改 Flash Attention kernel）。
- **来源**：new-wxa16-plan-260818.md §9。

---

### P8 — 备选 / 独立技术路线（长期方向，非增量优化）

#### P8-1：ROADMAP 第四阶段：Machete / Marlin CUDA kernel（Blackwell wgmma）
- 针对 RTX 5090 (SM12.0) 的高性能推理替代路线：从 vLLM 提取 Machete kernel（支持 wgmma），或 Marlin MoE kernel，替代 Triton 全路线。
- **难度**：高（CUDA/C++ 扩展 + 集成）。
- **来源**：ROADMAP.md 第四阶段。

#### P8-2：QwenMultiLinear 备选方案（保持 grouped_gemm 格式）
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

## 四、建议的迭代顺序（基于以上梳理）

1. **P5-1（group-first 重排）**：低难度、不降级、修复真实 eval 净回归，立即见效。
2. **P5-3（BLOCK 联合调优）**：可与 P5-1 并行，零风险稳定收益。
3. **P5-2（多 expert grouped GEMM）**：啃掉 ~70% 差距来源的逐 expert Python 循环，P5 阶段最大的结构性 win。分两阶段推进。
4. **P6-1（跨 bit 旋转复用 + dual_matmul 接线）**：需先做 seed 统一的精度消融实验，确认精度可接受再推进。
5. **P6-2（混合 bit fusion） + P6-4（dual_matmul grouped 版）**：与 P5-2 / P6-1 有协同，放其后。
6. **P6-3（方案 D 大 K 融合）**：PoC 先行，验证 Tensor Core 利用率提升上限。
7. **P7（cuTile / Split-K / Bit-packing / 权重预排序 / Flash Attention）**：探索性验证，逐个试、有收益再上主流程。
8. **P8（Machete/Marlin / QwenMultiLinear）**：独立技术路线，作为 Triton 路线的备选或长期替代。

> 每一步改动都需配套：测试程序先对齐主流程逻辑（含 GPU 填充度 / 并行 SM 分析）+ 本人在真实 eval 形状下手动复测，避免再次落入 micro≠real 陷阱。
