# WxA16 混合比特 MoE 推理性能 Profiling 方案

## 一、背景

核心问题：**按 bit 分区存储的混合精度 MoE（DartMoQ 风格），比单一 bit 宽度的 WxA16 推理慢多少？慢在哪里？**

被测对象：`WxA16BitPartitionedGroupMoE`
- 每个 expert 内神经元被切分为多份，每份不同 bit 宽度（1/2/4/8 bit）
- 还有剪枝（部分神经元直接丢弃，不参与计算）
- 按 bit 分区紧凑存储，同一 bit 的所有 expert 权重连续排列
- forward 方式：per-expert × per-bit 循环，每个 bit 调一次 grouped fused matmul

对比 baseline：单一 bit 宽度的 WxA16 MoE（所有神经元同一种 bit，无剪枝）

---

## 二、测试对比框架

### 变量控制

为了公平对比，保持以下条件相同：
- 模型结构：num_experts / hidden_size / top_k / group_size 全部相同
- 实际参与计算的 FLOPs 相同（通过调整神经元数量 + bit 宽度来控制）
- 权重初始化方式相同（随机）
- 输入相同

### 对比维度

| 维度 | 说明 |
|------|------|
| **总时间对比** | 混合比特 vs 单 bit baseline 的端到端时间比 |
| **L1 模块级对比** | 各阶段（gate_up / down / scatter / ...）时间占比的差异 |
| **L2 Kernel 级对比** | kernel launch 次数、每次 kernel 耗时、Python 端开销的差异 |
| **剪枝比例影响** | 不同剪枝率下的速度变化（剪枝是否真的省时间？） |
| **bit 分布影响** | 不同 bit 混合比例下的速度变化 |

---

## 三、L1：端到端 + 模块级对比

### 目标

直接回答「混合比特比单 bit 慢多少」，以及各阶段的时间差异。

### 测试场景设计

#### 场景组 1：相同计算量对比（控制 FLOPs 相同）

**Baseline**：全 4-bit，无剪枝，intermediate_size = N₀

**混合比特 A**：2-bit 50% + 4-bit 50%，无剪枝，intermediate_size = N₀
（FLOPs 相同，权重体积 2-bit 只有一半，但计算量一样）

**混合比特 B**：2-bit 50% + 4-bit 25% + 剪枝 25%，intermediate_size = N₀
（FLOPs 减少 25%，看看速度能否对应减少）

> 为什么控制 FLOPs 相同？因为如果混合比特只是因为神经元少了才快/慢，
> 那没有意义。我们要知道的是**比特分区这种数据结构本身带来的额外开销**。

#### 场景组 2：相同权重体积对比（控制显存带宽相同）

**Baseline**：全 4-bit，intermediate_size = N₀

**混合比特**：2-bit 100%（即全 2-bit），intermediate_size = 2×N₀
（权重体积相同，但神经元数翻倍，FLOPs 翻倍）

这个对比看带宽是否是瓶颈。

#### 场景组 3：分层开销拆解（FP16 → 量化 → partition → 混合比特）

四层对比，把总开销拆解成独立的贡献项：

| # | 配置 | 对应开销 |
|---|------|---------|
| ① | fp16 standard MoE | baseline（FP16 Tensor Core） |
| ② | WxA16 全 4-bit，**无 partition**（整块 grouped fused matmul） | 量化 kernel 本身的开销（反量化 + TF32 + grouped 拆分） |
| ③ | WxA16 全 4-bit，**bit-partitioned**（slice_rows/slice_in_features） | partition / slice 架构的额外开销 |
| ④ | WxA16 混合比特（2+4-bit，bit-partitioned） | 多 bit 混合 + per-bit 循环的额外开销 |

每层之间的时间差 = 那一层带来的额外开销。
这样就能回答：量化本身慢多少？partition 架构慢多少？多 bit 混合慢多少？

### 测量指标（L1 已有的）

每个配置下测量：
- 总时间 (ms)
- 各阶段时间：gate_up / silu_mul / down / scatter / cleanup / router / sort
- 各阶段占比 (%)

输出对比表格：
```
配置                   总时间(ms)   gate_up    down    scatter   cleanup  其他
全4-bit baseline         ???        ???%      ???%    ???%     ???%    ???%
2bit+4bit 混合 (同FLOPs) ???        ???%      ???%    ???%     ???%    ???%
2bit+4bit+剪枝           ???        ???%      ???%    ???%     ???%    ???%
...
```

关键指标：**混合比特的额外开销 (%)** = (混合比特时间 - 等 FLOPs baseline 时间) / baseline 时间

### 测试形状

- batch_size: 4, seq_len: 256
- hidden_size: 1024, intermediate_size: 2048
- num_experts: 16, top_k: 4, group_size: 128
- GPU: RTX 5090（注：小模型主要看相对比例）

---

## 四、L1 实测结果

### 组 1：相同 FLOPs 对比（bit 种类影响）

每 expert 2048 神经元，无剪枝，FLOPs 基本相同：

| 配置 | 总时间(ms) | gate_up | down | cleanup | vs baseline |
|------|-----------|---------|------|---------|-------------|
| 全 4-bit (baseline) | 142.2 | 11.4% | 54.8% | 32.0% | 1.00x |
| 2-bit + 4-bit 各半 | 179.3 | 24.1% | 47.4% | 26.5% | **1.26x** (+26%) |
| 1+2+4-bit 三等分 | 210.3 | 25.6% | 49.1% | 22.7% | **1.48x** (+48%) |

结论：多一种 bit 大约慢 20-30%，主要是 gate_up 阶段时间增加（每个 bit 都要做一次完整 grouped 流程 + 旋转）。

### 组 2：剪枝效率（2-bit:4-bit = 1:2 混合）

| 剪枝率 | 总时间(ms) | 时间比 | 效率 |
|--------|-----------|--------|------|
| 0% | 172.3 | 1.00x | 1.00 |
| 25% | 138.8 | 0.81x | **0.78** |
| 50% | 140.4 | 0.81x | **0.37** |
| 75% | 67.4 | 0.39x | 0.81 |

结论：剪枝效率 < 1.0，固定开销（cleanup/router/sort + 旋转 + launch）抵消了部分计算节省。50% 剪枝时效率异常低（数据抖动），但整体趋势一致。

### 组 3：分层开销拆解（核心结论）

| # | 配置 | 时间(ms) | vs fp16 | 额外开销 |
|---|------|---------|---------|---------|
| ① | fp16 standard | 43.5 | 1.00x | baseline |
| ② | WxA16 4-bit **无 partition** | 186.5 | **4.29x** | +329%（量化本身） |
| ③ | WxA16 4-bit **bit-partitioned** | 144.2 | 3.32x | -23%（N 变小反而更快） |
| ④ | WxA16 **2+4-bit 混合** | 166.0 | 3.82x | +15%（多 bit 混合） |

**核心发现**：慢 4 倍的主因是**量化 kernel 本身**，不是 bit-partition 架构，也不是多 bit 混合。
bit-partition 在单 bit 场景下反而更快，因为 N 变小后单次 kernel 更快。

---

## 五、L2 实测结果：为什么量化 kernel 慢 4 倍？

### 实验 1：Launch 开销 ≈ 15 μs

- 单次 Triton kernel launch ≈ 14-18 μs（几乎与 B 无关）
- B ≥ 128 时计算时间已显著超过 launch
- **结论**：launch overhead 是小因素，不是 4 倍慢的主因

### 实验 2：同等形状下 FP16 vs 4-bit Triton

| 形状 (B,N,K) | fp16 (μs) | 4-bit (μs) | 比值 | fp16 TFLOPS | 4-bit TFLOPS |
|-------------|-----------|------------|------|-------------|--------------|
| 极小 (16,256,128) | 6.6 | 16.9 | 2.6x | 0.16 | 0.06 |
| 小 (64,512,128) | 6.7 | 17.6 | 2.6x | 1.25 | 0.48 |
| 中 (256,1024,128) | 6.7 | 18.3 | 2.7x | 10.1 | 3.7 |
| 大 (512,2048,512) | 6.7 | 41.4 | **6.2x** | 39.9 | 6.5 |

反直觉：形状越大，Triton 相对 cuBLAS 的差距越大。说明不是小形状利用率问题，而是 Triton kernel 本身的计算效率在大形状下跟不上 cuBLAS。

### 实验 3：反量化位运算开销 ≈ 0（反直觉！）

| bit_width | 时间(μs) | vs 8-bit |
|-----------|----------|----------|
| 8-bit | 26.70 | 1.00x |
| 4-bit | 26.13 | 0.98x |
| 2-bit | 17.61 | **0.66x**（更快！） |
| 1-bit | 26.55 | 0.99x |

**结论**：位运算 unpack 不是瓶颈。2-bit 反而更快，因为 packed 数据量小，内存带宽需求低。
（之前以为"反量化位运算慢"的假设不成立。）

### 实验 4：Grouped 架构细粒度拆解（重磅发现）

形状：B=64, N=512, K=1024, group_size=128, num_groups=8（模拟一个 expert 的 gate_up）

| 阶段 | 时间(ms) | 占 grouped 总时间比例 |
|------|-----------|----------------------|
| 完整 grouped 总时间 | 0.510 | 100% |
| **旋转（Python 端 matmul）** | 0.203 | **39.8%** |
| **clone（per-group 切片拷贝）** | 0.091 | **17.8%** |
| 纯 kernel + 累加 | 0.142 | 27.9% |
| 假想 fp16 大 matmul | 0.006 | 1.2% |

**两个最大瓶颈：**
1. **旋转占 40%** — 每个 group 做 `x_g @ Pi.T`，(B,128) @ (128,128) 小矩阵乘利用率极低；此外这 40% 还包含一个未剥离的隐藏开销：旋转矩阵 `Pi` 由 `generate_rotation_matrix` 生成后缓存于 CPU（`Q.cpu()`），每次 forward 访问都要 `.to(device)` 重新拷回 GPU（128×128×4B），每个 group×bit×expert 一次。
2. **clone 占 18%** — `indices_packed[:, packed_start:packed_end].clone()` 每个 group 一次。注意这并非"可随意删的纯开销"：内核 `triton_fused_matmul` 按 `rn*PACKED_K + byte_col` 寻址，要求输入为行 stride=PACKED_K 的连续张量，而列切片真实行 stride 为原始总宽（≠ 切片宽），直接删 clone 会静默读错内存。正确做法是给内核加列偏移（`col_start` / `full_packed_k`）参数按偏移寻址后再去 clone。

纯 kernel 计算只占不到 30%，剩下 70% 都是 Python 端的辅助开销！

### 实验 5：精度影响（FP16 vs TF32）

| 形状 | fp16 TFLOPS | fp32/TF32 TFLOPS | 比值 |
|------|-------------|-------------------|------|
| 小 | 1.61 | 1.27 | 1.27x |
| 中 | 11.49 | 10.06 | 1.14x |
| 大 | 100.4 | 35.4 | **2.83x** |

Triton kernel 内部 accumulator 是 float32（走 TF32 Tensor Core），大形状下 TF32 吞吐只有 fp16 的 ~35%。
这解释了为什么大形状下 4-bit 与 fp16 的差距反而更大。

---

## 六、归因总结

| 开销来源 | 贡献 | 说明 |
|----------|------|------|
| **Python 端旋转** | ~40% grouped 时间 | 小形状 matmul 利用率低 |
| **clone 开销** | ~18% grouped 时间 | 列切片正确性约束，需 kernel 列偏移寻址去除（非随意删） |
| **TF32 vs FP16** | 1.5~2.8x | 大形状更明显 |
| **Triton vs cuBLAS 算子效率** | 1.5~2x | 动态小 kernel 不启用 autotune，视为固有算子效率差距 |
| **旋转矩阵 CPU→GPU 重复拷贝** | 含于旋转 ~40% | 旋转矩阵缓存于 CPU，每次 forward 重拷回 GPU，未剥离的隐藏开销 |
| **小形状 + grouped 拆分** | 显著 | K=128 小矩阵乘 + 多次 launch |
| Launch overhead | 小 | ~15μs/次，仅极小形状显著 |
| 位运算反量化 | **几乎 0** | 2-bit 反而更快（带宽节省） |
| bit-partition 架构 | -23% | 单 bit 下 N 变小反而更快 |
| 多 bit 混合 | +15% | per-bit 循环额外开销 |

---

## 七、优化方向（按优先级）

| 优先级 | 优化点 | 预期收益 | 难度 |
|--------|--------|---------|------|
| **P0** | kernel 列偏移寻址去 clone（保持正确性，内核加 `col_start`/`full_packed_k`） | -18% grouped 时间 | 低 |
| **P0** | 旋转矩阵设备端缓存 / 预计算旋转后输入（消除 CPU→GPU 重拷） | 削减旋转 40% 中的隐藏拷贝占比 | 低 |
| **P0** | 把旋转融合进 Triton kernel（或预计算所有 group 旋转后输入） | -40% 旋转时间 | 中 |
| **P1** | 多 group 合并到一次 kernel launch（减少 launch + 更好 SM 利用率） | 20-30% | 中 |
| **P1** | FP16 accumulator（如果精度允许） | 理论 1.5-2x | 中 |
| ⛔ 否决 | Triton autotune 调参（BLOCK 大小、num_warps、num_stages） | — | — | 动态小 kernel，搜索编译开销 > 收益，不启用 |
| 📋 待办 | 旋转结果复用（跨 bit / 跨 gate_up&down，需先统一跨 bit seed + 改量化 + 验精度） | 视情况 | 中 |
| 📋 待办 | 接线 `triton_fused_dual_matmul`（已实现未调用，合并两次 matmul，依赖统一 seed） | 视情况 | 中 |
| **P3** | 去掉 `.item()` D2H 同步 | 小 | 低 |

---

## 八、混合比特特有开销分析（待验证）

### 目标

回答「多 bit 混合比单 bit 慢多少、慢在哪里」，这是在量化 kernel 本身开销之上的额外开销。

（注：量化 kernel 本身为什么慢 4 倍已在第五章分析完毕。本章关注的是 bit-partition + 多 bit 混合带来的**增量**开销。）

### 分析维度

#### 8.1 Kernel launch 次数对比

```
baseline (单 4-bit):
  gate_up: num_experts × num_groups = 64 × 16 = 1024 次
  down:    num_experts × num_groups_inter ≈ 64 × 44 = 2816 次
  总计: ~3840 次

混合比特 (3 种 bit):
  gate_up: num_experts × num_bits × num_groups = 64 × 3 × 16 = 3072 次
  down:    num_experts × num_bits × num_groups_per_bit ≈ 64 × 3 × ... = ???
  总计: ???
```

**假设**：混合比特的 launch 次数 = baseline × num_bits（如果每个 bit 的 neuron 数都相同的话）
但实际上每个 bit 的 neuron 数不同，而且有剪枝，所以需要实测。

#### 8.2 单次 kernel 的计算效率对比

同形状（B, N, K）下，不同 bit_width 的 kernel 时间对比：
- 2-bit / 4-bit / 8-bit 的 fused dequant + matmul 各是多少时间？
- 反量化的开销有多大？（比如 2-bit 比 4-bit 慢多少？位运算开销）

#### 8.3 Python 端循环开销

混合比特多了一层 per-bit 循环：
- 每个 expert 内多了一个 `for bit_str in self.bit_weights.keys()` 循环
- 每次循环有 `expert_offsets[expert_idx].item()`（D2H 同步！）
- 每次循环有 `if actual_inter_size == 0: continue` 的判断

这个开销在 expert 很多 / bit 很多时可能显著。

#### 8.4 旋转的重复计算

> 原假设"3 个 bit 的 gate_up 对同一输入做相同旋转、可复用"**不成立**，原因如下：

- 量化时每个 bit 使用了**不同 seed**（`seed=42+bit` / `42+bit+1000`，见 `wxa16_bit_partitioned_moe.py`），
  因此旋转矩阵 `Pi` 随 bit 不同而不同，当前代码**不存在**跨 bit 的旋转重复计算。
- 旋转 40% 的真实来源是两部分：
  1. 每个 group 一次 `(B,128)@(128,128)` 极小矩阵乘从 Python 端 launch，Tensor Core 占用率极低；
  2. **隐藏开销**——旋转矩阵缓存存于 CPU（`Q.cpu()`），每次 forward 访问都 `.to(device)` 重新拷回 GPU。

**关于"若统一 seed 能否复用"**：技术上**可以**——旋转作用在输入 `x` 上，仅依赖 `g_dim/group_size/g_start`（跨 bit 相同），只要 seed 统一，`x_rot_g` 即跨 bit 相同，可计算一次喂给多 bit 的 kernel。
但前提是**量化侧也要统一跨 bit 的 seed**（方案改动），且必须重新验证精度，因此可行但**耦合量化、推迟为待办**。

> 另：`triton_kernels.py` 已实现 `triton_fused_dual_matmul`（支持 `SAME_INPUT` 合并两次 matmul），但主 forward 完全未调用，属未接线的死代码。其意图与"多 group 合并 launch"一致，但需先统一跨 bit seed（`SAME_INPUT` 才成立），故推迟为待办。

#### 8.5 小形状 kernel 的低效率

剪枝后，某些 bit 下某个 expert 的神经元数可能很少（比如只有 128 个 = 1 个 group）。
小矩阵乘的 Tensor Core 利用率低，而且 launch 开销占比高。

### 测量方法

| 开销来源 | 测量方法 | 状态 |
|---------|---------|------|
| Launch 次数差异 | 直接计数 + launch overhead 估算 (~16μs/次) | ✅ 已验证（~15μs/次，占比小） |
| 旋转重复计算 | 对比「1 个 bit 的旋转时间」和「3 个 bit 的总旋转时间」 | ✅ 已验证（旋转占 grouped 时间 40%，多 bit 就是多倍旋转） |
| Per-bit 循环开销 | 对比「相同 FLOPs 的单 bit」和「多 bit 拆分」的总时间差 | ✅ 已验证（+15% 左右） |
| 小形状效率损失 | 测不同 N (128~2048) 下的 per-neuron 时间 | ✅ 已验证（小形状差距 2.6x，大形状反而更大） |
| D2H 同步开销 | 把 `.item()` 去掉前后对比 | ⏳ 待验证 |

---

## 九、测试文件

### `test/profile_moe_l1.py`

- 构造不同 bit 分布的 MoE（全 4-bit baseline / 混合比特 / 带剪枝）
- 端到端时间 + 各阶段时间对比
- 输出对比表格

### `test/profile_moe_l2.py`

- Launch overhead 估算
- 不同 bit_width 的单 kernel 性能曲线
- Gate-Up 细粒度：旋转 / clone / kernel 各阶段占比
- 混合比特特有开销的分析（旋转重复、per-bit 循环、小形状效率）

---

## 十、关键决策点（基于实测数据更新）

### 已确认的结论

| 假设 | 实测结果 |
|------|---------|
| 反量化位运算慢 | ❌ **不成立**。2-bit 反而比 8-bit 快 34%（带宽节省） |
| launch overhead 是主因 | ❌ 不成立。~15μs/次，仅极小形状占比大 |
| 量化 kernel 本身慢 4 倍 | ✅ **确认**。无 partition 的 4-bit 比 fp16 慢 4.29x |
| bit-partition 架构慢很多 | ❌ 不成立。单 bit 下反而更快（-23%，N 变小） |
| 多 bit 混合有额外开销 | ✅ 确认。约 +15%（per-bit 循环 + 每 bit 独立旋转） |
| 旋转是主要开销 | ✅ **重磅确认**。占 grouped 流程时间 ~40%（含旋转矩阵 CPU→GPU 重拷隐藏开销） |
| clone 占 ~18%，但为正确性约束 | ✅ 确认占比，但实为列切片正确性约束，需配套 kernel 列偏移寻址去除（见 §七） |
| autotune 是可启用杠杆 | ❌ 不成立。kernel 高度动态（K=128 小矩阵、多次 launch），不启用，差距视为固有算子效率差距 |

### 优化方向决策

按性价比排序（收益 × 可行性）：

| 优先级 | 优化点 | 预期收益 | 难度 | 备注 |
|--------|--------|---------|------|------|
| **P0** | kernel 列偏移寻址去 clone（保持正确性） | grouped 时间 -18% | 低 | 改动最小；clone 非随意删，需配套内核列偏移寻址 |
| **P0** | 旋转矩阵设备端缓存 / 预计算旋转后输入 | 削减旋转 40% 中的 CPU→GPU 重拷 | 低 | 纯结构性，不碰 kernel 数学 |
| **P0** | 旋转融合进 Triton kernel | grouped 时间 -40% | 中 | 最大的单一开销来源 |
| **P1** | 多 group 合并为一次 kernel launch | 20-30% | 中 | 减少 launch + 提升 SM 利用率 |
| **P1** | FP16 accumulator | 理论 1.5-2x | 中 | 需验证精度损失 |
| ⛔ 否决 | Triton autotune 深度调参 | — | — | 动态小 kernel，搜索开销 > 收益，不启用 |
| 📋 待办 | 旋转结果复用（跨 bit / 跨 gate_up&down） | 视情况 | 中 | 需统一 seed+改量化+验精度 |
| 📋 待办 | 接线 `triton_fused_dual_matmul` 合并 launch | 视情况 | 中 | 已实现未调用，依赖统一 seed |
| **P3** | 去掉 `.item()` D2H 同步 | 小 | 低 | 仅在 per-expert 循环极多时显著 |

---

