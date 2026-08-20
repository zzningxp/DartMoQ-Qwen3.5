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
2. **clone 占 18%** — `indices_packed[:, packed_start:packed_end].clone()` 每个 group 一次。注意这并非"可随意删的纯开销"：内核 `triton_fused_matmul` 按 `rn*PACKED_K + byte_col` 寻址，要求输入为行 stride=PACKED_K 的连续张量，而列切片真实行 stride 为原始总宽（≠ 切片宽），直接删 clone 会静默读错内存。**已落地**：给内核加 `COL_START` 偏移（`byte_off = rn*PACKED_K + (COL_START + byte_col)`），传整张 `indices_packed` + `col_start=packed_start`，去掉 clone。热路径实测省 44.5%（见 §十 代码开发记录）。

纯 kernel 计算只占不到 30%，剩下 70% 都是 Python 端的辅助开销！

> 注：本实验测的是** warm 路径**（旋转矩阵缓存已热，仅含 `x_g @ Pi.T` 矩阵乘 + 每次访问的 CPU→GPU 重拷）。实验 6 进一步拆分表明：warm 路径里矩阵乘与重拷各占约一半；而每个 (bit,group) **首次**出现时还有一次 QR 重算（约 1.9ms，占 cold 路径 90%），该 QR 因 seed 确定、可在模型加载时预计算一次消除。详见 §五 实验 6。

### 实验 5：精度影响（FP16 vs TF32）

| 形状 | fp16 TFLOPS | fp32/TF32 TFLOPS | 比值 |
|------|-------------|-------------------|------|
| 小 | 1.61 | 1.27 | 1.27x |
| 中 | 11.49 | 10.06 | 1.14x |
| 大 | 100.4 | 35.4 | **2.83x** |

Triton kernel 内部 accumulator 是 float32（走 TF32 Tensor Core），大形状下 TF32 吞吐只有 fp16 的 ~35%。
这解释了为什么大形状下 4-bit 与 fp16 的差距反而更大。

### 实验 6：旋转开销拆分（matmul vs CPU→GPU 拷贝 vs QR 重算）

（完整测量见 `test/profile_moe_l2.py` 的 `experiment6_rotation_split`，形状同实验 4：B=64, K=1024, group_size=128, num_groups=8）

| 变体 | 时间(ms) | 占 cold 比 |
|------|-----------|-----------|
| A 纯矩阵乘（GPU 钉住，无拷贝） | 0.103 | 4.9% |
| B warm（含 CPU→GPU 重拷） | 0.199 | 9.5% |
| C cold（QR 重算 + 重拷） | 2.102 | 100% |

拆分结论：
- 纯矩阵乘仅占 4.9%；**CPU→GPU 重拷占 4.6%**（在会重复的 warm 路径里 ≈ 旋转开销的一半）；
- **QR 重算占 90.5%**——每个 (bit,group) 首次出现时算一次（seed 不含 expert 维度，同一 bit 内 8 个 Pi 跨 16 expert 复用），每 forward 仍重算 3×8=24 次；
- seed 确定 → 可在**模型加载时预计算一次钉在 GPU**，永久消除 QR 重算与逐次重拷。

**决策门原结论（已被 §补充复盘修正预期）**：当时基于实验 6 数据优先「旋转矩阵设备端缓存 / 加载时预计算」（同时干掉重拷与 QR 重算），**而非**「旋转融合进 kernel」（纯矩阵乘仅占 ~5%，收益极小）。落地后实测证明前者真实收益≈0（cold 假象），见下方补充复盘。

### 实验 6 补充：决策门落地后的实测复盘（重要纠正）

旋转矩阵设备端缓存（变体 1）**已实现**（`rotation.py` 新增 `_ROTATION_CACHE_DEV`，`triton_kernels.py` 调用时传 `device=x.device` 命中）。但接进真实 eval 后**端到端无任何提速**，复盘如下：

- 真实 run 里**没有任何 `clear_rotation_cache()` 调用**，旋转矩阵按 `(d, seed+g_start)` 缓存，首个 batch 冷一次、之后全热；
- 那个 CPU→GPU 重拷每 group 才 64KB（(d,g_dim) = (128,128)×4B），本就是微秒级；
- 实验 6 的 "warm 路径重拷占 ~50% / cold QR 占 90%" 是**基于人为打冷（每 forward `clear_rotation_cache()`）的 micro 假象**——真实 eval 缓存不冷，重拷只在首次出现，对 19.71s 的 WxA16 层贡献 ≈ 0；
- 佐证：`eval_qwen35.py` 跑 Layer 0（WxA16 量化层）forward 19.62s 对比 Layer 1（dense）1.92s，慢 10 倍，旋转缓存改前后完全一致——说明这 10 倍差距**与旋转无关**。

**结论**：旋转缓存改得对（正确性没问题、显存增量≈4MB 可忽略），但不该被当成"主优化方向"。它消除的是**冷启动**开销，而真实推理是热路径。真正吞掉 19.71s 的是**结构性**开销（见 §六/§十）。

**关于 micro 测试方法的纠正**：`test_triton_mixed_precision.py` 当时为"展示改进"每轮都 `clear_rotation_cache()` 强制打冷，得到 "90% 节省" 是误导。后续测试必须区分 cold/warm 路径，warm 路径才是真实场景。

---

## 六、归因总结

| 开销来源 | 贡献 | 说明 |
|----------|------|------|
| **Python 端旋转** | warm 路径 ~40% grouped 时间 | 小形状 matmul 利用率低；**仅首个 batch（cold）显著，warm 后该开销已在基线内**，详见实验 6 补充复盘 |
| **clone 开销** | ~18% grouped 时间 | 列切片正确性约束；**已落地去除**（内核 `COL_START` 偏移，真实省 44.5%） |
| **TF32 vs FP16** | 1.5~2.8x | 大形状更明显 |
| **Triton vs cuBLAS 算子效率** | 1.5~2x | 动态小 kernel 不启用 autotune，视为固有算子效率差距 |
| **旋转矩阵 CPU→GPU 重复拷贝** | cold 路径显著，warm ≈ 0 | 旋转矩阵缓存于 CPU，每次 forward 重拷回 GPU；设备端缓存已落地，但真实 eval 缓存本就热，**对端到端无贡献**（详见实验 6 补充复盘） |
| **小形状 + grouped 拆分** | 显著 | K=128 小矩阵乘 + 多次 launch |
| Launch overhead | 小 | ~15μs/次，仅极小形状显著 |
| 位运算反量化 | **几乎 0** | 2-bit 反而更快（带宽节省） |
| bit-partition 架构 | -23% | 单 bit 下 N 变小反而更快 |
| 多 bit 混合 | +15% | per-bit 循环额外开销 |

---

## 七、优化方向（按优先级）

| 优先级 | 优化点 | 预期收益 | 难度 | 状态 |
|--------|--------|---------|------|------|
| **P0** | kernel 列偏移寻址去 clone（保持正确性，内核加 `col_start`） | 热路径省 44.5%（grouped kernel 端开销） | 低 | ✅ **已落地**（实测 0.314→0.174ms/次，全 bit 宽度 allclose 通过） |
| **P0** | 旋转矩阵设备端缓存 / 加载时预计算 | 消除 cold 路径重拷+QR 重算 | 低 | ✅ **已落地，但真实收益≈0**（见实验 6 补充复盘：warm 路径本就热，对端到端无贡献；micro 的 90% 是打冷假象） |
| **P1** | 多 group 合并到一次 kernel launch（减少 launch + 更好 SM 利用率） | 20-30% | 中 | ⏳ 待做（真正能啃 10 倍差距的大头之一） |
| **P1** | FP16 accumulator（如果精度允许） | 理论 1.5-2x | 中 | ⏳ 待做（真正能啃 10 倍差距的大头之一，需验精度） |
| **P2** | 旋转融合进 Triton kernel | 视情况 | 中 | ⛔ 降级（纯矩阵乘仅占 ~5%，收益有限） |
| ⛔ 否决 | Triton autotune 调参 | — | — | 动态小 kernel，搜索编译开销 > 收益，不启用 |
| 📋 待办 | 旋转结果复用（跨 bit / 跨 gate_up&down，需先统一跨 bit seed + 改量化 + 验精度） | 视情况 | 中 | ⏳ 待做 |
| 📋 待办 | 接线 `triton_fused_dual_matmul`（已实现未调用，合并两次 matmul，依赖统一 seed） | 视情况 | 中 | ⏳ 待做 |
| **P3** | 去掉 `.item()` D2H 同步 | 小 | 低 | ⏳ 待做 |

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

### `turboquant_utils/test_clone_removal.py`（本次新增，验证 clone 去除正确性）

- 对比「clone 旧路径」与「`col_start` 新路径」在同一列切片上的输出逐元素一致性；
- 覆盖 bit=1/2/4/8 全宽度，`max_diff=0.000e+00`，确认内核列偏移寻址无静默读错内存；
- 运行：`PYTHONPATH=$PWD conda run -n dart312 python turboquant_utils/test_clone_removal.py`。

### `test/test_colstart_recompile.py`（本次新增，复现/验证 col_start 编译风暴）

- 64 个不同 col_start 各调一次 `triton_fused_matmul`，测耗时 + 统计 `~/.triton/cache` 新增目录数（= 新编译 kernel 变体数）；阶段 2 重复同批值验证缓存命中，阶段 3 固定单值模拟 micro 测试模式，另附 clone vs col_start 数值抽查；
- 修复前：62 个新编译变体 / 12.71s（平均 ~205ms/个新 col_start）；修复后：2 个变体 / 0.39s，数值 max_diff=0；
- 运行：`python test/test_colstart_recompile.py`（或 `PYTHONPATH=$PWD conda run -n dart312 python test/test_colstart_recompile.py`）。

### `test/test_eval_shape_bench.py`（本次新增，真实 eval 形状基准）

- micro 测试形状（B=64）与真实形状（eval per-expert B≈9280 / mini_batch B≈1024）差距大，本脚本在真实形状下对比「clone 旧路径」vs「col_start 新路径」的 kernel 本体时间，并单测 clone 本身开销与旋转访问流成本；
- 关键实测（2-bit，B=9280）：gate_up clone 旧路径 252.7µs（clone 本身仅 5.0µs）vs col_start 新路径 366.4µs（**新路径反而慢 +113.7µs**）；down 486.4µs vs 716.3µs（**+229.8µs**）；B=1024 时差值仅 +9.6µs / +31.8µs；
- 结论：clone 去除省的 ~5µs/call 被大 stride 寻址惩罚抵消且随 B 增大反超——mini_batch 尺度净收益≈0，eval 尺度净回归。

### `test/test_rotation_thrash_count.py`（本次新增，旋转缓存抖动精确计数）

- monkey-patch `torch.linalg.qr` 计数，模拟一次真实 forward 的旋转访问流（254 expert × 20 组 = 5080 次访问，1032 个不同 (d,seed) key，down 的 seed 含 expert 偏移故每 expert 不同）；
- 实测：上限 128 时**每个 forward QR 重算 1160 次 ≈ 284~308ms**（forward#2 与 #1 相同 → 抖动每 forward 重复支付）；上限提到 4096 后 cold 一次 1032 次，稳态 forward **0 次 / 0.9ms** → 抖动确实存在且可通过调大上限消除；
- **踩坑记录（双模块陷阱）**：`turboquant_utils/__init__.py` 的 `sys.modules.setdefault("turboquant_model", ...)` 别名 + vendored 模块用 `from turboquant_model.rotation import ...`，导致 rotation.py 被执行两次、进程内存在两个 rotation 模块对象，包属性 `rotation` 被重绑到副本；`import turboquant_utils.rotation as m` 可能拿到副本，monkey-patch 打空（本测试首版因此误判"抖动不存在"）。测试里必须用 `sys.modules[generate_rotation_matrix.__module__]` 反查函数真实所属模块；
- 运行：`python test/test_rotation_thrash_count.py`。

### `turboquant_utils/test_triton_mixed_precision.py`（已更新，用于对齐验证）

- 三个场景（单独 Linear / MoE up_gate slice_rows / MoE down slice_in_features）复用主流程同款 kernel；
- 场景汇总里直接织入 `Triton(改后 hot)` vs `Triton(改前 cold)` 对照 + 数值一致性；
- **注意**：该测试的 cold 路径是每轮 `clear_rotation_cache()` 人为打冷所得，仅用于展示 cold/warm 差异，不代表真实 eval 收益（详见实验 6 补充复盘）。

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
| 旋转是主要开销（仅 cold 路径） | ✅ 确认 warm 路径占 ~40%，但**仅首个 batch（cold）显著**；真实 eval 缓存热后该开销已在基线内（详见实验 6 补充复盘），不再是主优化方向 |
| clone 占 ~18%，但为正确性约束 | ✅ 确认占比，但实为列切片正确性约束，需配套 kernel 列偏移寻址去除（见 §七） |
| autotune 是可启用杠杆 | ❌ 不成立。kernel 高度动态（K=128 小矩阵、多次 launch），不启用，差距视为固有算子效率差距 |
| col_start 做成 constexpr 只多一次整数加法 | ❌ **不成立（真实流程回归事故）**。constexpr 参与 Triton 编译 key，每个不同 col_start 触发一次完整重编译（实测 ~205ms/个）；真实 MoE 里 col_start 随 expert×bit 有上千取值 → 编译风暴（首 mini_batch 240s）。micro 测试 col_start 恒为 1~2 个值所以测不出（详见 §十 记录 3） |
| clone 去除每次省 ~5µs，收益与形状无关 | ❌ **不成立（真实形状下为净回归）**。去 clone 后 kernel 直接按整张宽度（512B 行距）跳读 32B 切片，宽 stride 惩罚随 B 增长：B=9280 时 gate_up +113.7µs/call、down +229.8µs/call，远超省下的 ~5µs clone。mini_batch（B≈1024）净收益≈0 → 对应"与最早版本无区别"；eval（B≈9280）净回归 → 对应 eval +26%/+17%（详见 §十 记录 4） |
| 旋转缓存上限 128 足够（"缓存首个 batch 冷一次后全热"） | ❌ **不成立**。down 方向 seed = base + expert 偏移 + group 偏移，每 forward 工作集 1032 个 key > 上限 128 → 每 forward QR 重算 1160 次 ≈ 284ms（forward#2 不衰减）；上限提到 4096 后稳态 0 次 / 0.9ms。该开销在 dfb8fc1 与当前代码两代共有（详见 test/test_rotation_thrash_count.py） |

### 已落地代码记录（本次迭代）

#### 1. clone 去除（内核列偏移寻址）—— P0，真实可见改进 ✅

**背景**：md 实验 4 已确认 clone 占 grouped 时间 ~18%，且是列切片正确性约束（直接删会静默读错显存）。

**实现（改 `turboquant_utils/triton_kernels.py`）**：
- 内核 `_turboquant_fused_matmul_kernel_nbit` 新增 `COL_START` constexpr，寻址改为 `byte_off = rn*PACKED_K + (COL_START + byte_col)`；
- `triton_fused_matmul` 新增 `col_start` 参数（默认 0，向后兼容 `module.py` 等其他调用方）；
- 三个 wrapper（grouped / slice_rows / slice_in_features）去掉 `indices_packed[:, packed_start:packed_end].clone()`，改传整张张量 + `col_start=packed_start`。

**正确性验证**：`turboquant_utils/test_clone_removal.py` 对比 clone 旧路径 vs `col_start` 新路径，bit=1/2/4/8 全 `max_diff=0.000e+00`，确认无静默读错内存；`test_triton_mixed_precision.py` 三个场景「反量化+GEMM vs Triton」误差与改动前一致（max≈0.15~0.25）。

**热路径实测**（warm、不清缓存，模拟真实 eval）：clone 旧 0.314ms → col_start 新 0.174ms，**省 44.5%** 每次 kernel 调用。这是当前真实可测、且不影响数值的改进。

**后续纠正（真实 eval 形状实测，见记录 4）**：上述 44.5% 是 micro 小形状（B 小、张量 L2 驻留）下的结论。真实 eval 形状（B≈9280）下宽 stride 寻址惩罚随 B 增长，新路径反而比 clone 旧路径慢（+113.7µs/+229.8µs 每次调用），clone 去除在 eval 尺度为净回归、mini_batch 尺度约打平。修复方向不是退回 clone，而是加载期把权重按 group 重排成连续布局（见记录 4 方案 A）。

#### 2. 旋转矩阵设备端缓存（变体 1）—— P0，已落地但真实收益≈0 ✅（需纠正预期）

**实现（改 `turboquant_utils/rotation.py`）**：保留 CPU 基缓存 `_ROTATION_CACHE`，新增 device-keyed `_ROTATION_CACHE_DEV`（key=(d,seed,device)），首次派生 `.to(device)` 后常驻。显存增量≈4MB 可忽略。

**实测复盘（关键纠正）**：接进真实 eval（`eval_qwen35.py` 跑 Layer 0 moe WxA16 19.62s vs Layer 1 moe fp16 1.92s）**端到端无任何提速**。根因：真实 run 不调 `clear_rotation_cache()`，缓存首个 batch 冷一次后全热；CPU→GPU 重拷每 group 仅 64KB，本就微秒级。**此前 micro 测试显示的 90% 节省是每轮 `clear_rotation_cache()` 人为打冷的假象**，不代表真实收益（详见 §五 实验 6 补充复盘）。

**结论**：该改正确、零风险，但不应作为主优化方向；真实 10 倍差距来自结构性开销，须靠 P1（多 group 合并 launch / FP16 accumulator）啃。

#### 3. COL_START 编译风暴修复（constexpr → 运行时参数）—— 真实流程回归的定位与修复 ✅

**现象**：clone 去除落地后，真实 run.q.sh 流程 WxA16 MoE forward 从 ~1.2s/mini_batch 恶化为首个 mini_batch **240s**、后续 33/12/16s，PPL eval Layer0 forward 68s（此前 ~19.7s）；而 micro 测试（test_clone_removal / test_triton_mixed_precision）全部正常。

**根因**：内核 `COL_START` 当时声明为 `tl.constexpr`。Triton 把 constexpr 取值写进编译缓存 key，**每个不同的 col_start 值触发一次完整重编译**（实测 ~205ms/个）。真实 MoE 里 col_start = expert 在 packed 维偏移 + group 偏移，随 expert×bit 变化有上千个取值，叠加 B/N/K 运行时参数的整除特化类，后续 mini_batch 与 eval 仍持续出现新组合 → 编译贯穿全程。证据：run 期间 `~/.triton/cache` 新增 ~2000 个编译变体，全部是 `_turboquant_fused_matmul_kernel_nbit`；编译时间线（13:15–13:24）与各慢速阶段完全重叠。

**为何 micro 测试测不出**：micro 场景 col_start 只有 1~2 个取值，编译缓存恒热。这是又一次「micro 测试未反映真实流程」教训（同旋转 cold 路径假象）。

**修复**：`turboquant_utils/triton_kernels.py` 中 `COL_START` 由 `tl.constexpr` 改为运行时参数（仅这一处，调用点本来就是按位置传参）。寻址逻辑与数值不变；col_start 不再参与编译 key，仅剩整数整除特化（≤2 类）。**clone 去除优化保留，未降级**。

**验证**：
- `test/test_colstart_recompile.py`（新增）：64 个不同 col_start，修复前新增 62 个编译变体 / 12.71s → 修复后 2 个变体 / 0.39s，数值 max_diff=0；
- `turboquant_utils/test_clone_removal.py`：bit=1/2/4/8 全宽度 clone vs col_start max_diff=0.000e+00；
- `turboquant_utils/test_triton_mixed_precision.py`：三场景数值误差与改前一致（max≈0.15~0.25）。

**测试方法补充**：给 Triton kernel 增加 constexpr 参数前，必须确认该参数在真实流程里是否「动态且多取值」——凡是每个 expert/每个 group/每个 bit 都会变的量，一律不得做 constexpr。排查同类问题可复用 `test/test_colstart_recompile.py` 的做法：以 `~/.triton/cache` 目录增量计数编译次数。

#### 4. 与最早版本（dfb8fc1）端到端效率无差异的归因 —— clone 去除在真实形状下是净回归 ⚠️

**现象**：编译风暴修复后代码能跑通，但 mini_batch 与最早（dfb8fc1 时代，logs/0707.6）基本无差（稳态 1.1672/1.0683/1.0193s vs 1.0470/1.0513/1.0195s），PPL eval 反而回归：pass1 10.54→13.24s（+26%）、pass2 18.13→21.27s（+17%）。dense 层（L1/L2/L4）两代耗时完全一致，排除机器状态差异。

**代码差异范围**：dfb8fc1..当前，与 MoE 热路径相关的改动只有两处——clone 去除（col_start 寻址）与旋转设备端缓存（§十 记录 1/2）。设备端缓存只会更快不会更慢；嫌疑集中在 clone 去除。

**根因（test/test_eval_shape_bench.py 实测）**：去 clone 后 kernel 在整张宽（行距 512B）上跳读 32B 有用切片，宽 stride 惩罚随 B 增长，而 clone 本身只有 ~5µs：

| 形状 | clone 旧路径 | col_start 新路径 | 净差（新−旧） |
|------|-------------|-----------------|--------------|
| gate_up B=1024 | ~持平 | ~持平 | +9.6µs/call |
| gate_up B=9280 | 252.7µs | 366.4µs | **+113.7µs/call** |
| down B=9280 | 486.4µs | 716.3µs | **+229.8µs/call** |

**定量对上观测**：layer 0 一次 forward ≈ 4064 次 gate_up + 1016 次 down 调用。
- mini_batch（B≈1024）：净增 ≈ 4064×9.6µs + 1016×31.8µs ≈ **+70ms/forward**，占 ~1s 的 7% → 观感"无区别"（首个稳态 mini_batch 实测 +120ms，含一次性冷启动，量级吻合）；
- eval（B≈9280，仅 layer 0 量化、其余 39 层 dense）：bench 尺度外推 ≈ +0.7s，实测 +2.7s/+3.15s。缺口来自 bench 张量小（0.5MB，L2 驻留）而真实 gate_up packed 张量 ≈134MB 超出 L2，宽 stride 读叠加 DRAM/TLB 放大（约 3~4 倍），方向一致。
- 历史上 c2631d8（7 月 19 日 "rollback-and-remove-clone"）就是同款去 clone 尝试的回滚——同一坑第二次踩到，这次有了定量解释。

**附带发现（两代共有的可修开销）**：旋转缓存上限 128 < 工作集 1032 key（down 的 seed 含 expert 偏移），每 forward QR 重算 1160 次 ≈ 284ms，mini_batch 里占 ~25%；上限提到 ≥2048 后稳态 0 次（test/test_rotation_thrash_count.py）。这是独立于回归的现成收益。

**候选修复方向（未实施，待决策）**：
- **方案 A（推荐，不降级）**：加载期把 packed 权重按 group 重排为 group-first 连续布局（如 `(num_groups, N, group_packed_bytes)`，一次 permute+contiguous，134MB 一次性 <0.1s），推理时每组切片天然连续 → kernel 无 clone、无宽 stride、col_start 恒 0。同时优于旧 clone 路径（省掉每 call 的 clone）与现路径。改动点：packed 数据生成/装载处（`wxa16_bit_partitioned_moe.py` 的 `set_packed_data` 一线）+ wrapper 的切片方式。
- **方案 B（一行修复，独立收益）**：`rotation.py` 的 `_MAX_CACHE_SIZE` 128 → ≥2048，消掉每 forward ~284ms QR 重算（CPU 66MB + GPU 66MB 常驻，可接受；`clear_rotation_cache()` 仍可释放）。
- **方案 C（不推荐）**：退回 clone 旧路径——只回到 dfb8fc1 水平、无增益，且按规范须本人同意，不作为选项实施。
- 结构性方向不变：P1 多 group 合并 launch / FP16 accumulator（§七）仍是关 10 倍差距的主线。

**教训（第三次 micro≠真实）**：凡"省掉一次小操作"类优化，必须在真实 B、真实张量尺寸（尤其是否超 L2）下复测 kernel 本体；寻址 stride 的代价随 B 与张量尺寸非线性放大，micro 形状完全不可见。

#### 遗留待办 / 未解决

- **真实 10 倍差距未消除**：WxA16 量化层比 dense 慢 10 倍，主因是 64 expert × 多 bit × 多个小 Triton kernel（TF32 Tensor Core，比 FP16 慢 1.5~2.8x）+ 逐 expert Python 循环 + 多次 launch（md 实验 2/5 定性 Triton 4-bit 比 fp16 慢 4.29x）。clone 去除真实一步，但关不掉这 10 倍。
- **COL_START 编译风暴回归已修复**（见已落地记录 3）：240s 首 mini_batch 为 constexpr 重编译所致，与 clone 去除本身无关；修复后预期回到 baseline 水平且保留 clone 去除收益（每次 kernel 调用 -44.5%）。真实回归幅度以本人手动 run.q.sh 复测为准。**〔纠正〕**复测结果：编译风暴消除，但未快于 dfb8fc1，eval 反而 +26%/+17%——clone 去除本身在真实形状下是净回归（详见记录 4），"-44.5%"仅 micro 形状成立。
- **去 clone 净回归待修**（记录 4 方案 A）：加载期 group-first 连续布局重排，待决策后实施；实施后需跑 `test/test_eval_shape_bench.py`（新路径应 ≤ clone 旧路径）+ 数值一致性 + 本人手动 run.q.sh 复测。
- **旋转缓存上限待调**（记录 4 方案 B）：`_MAX_CACHE_SIZE` 128 → ≥2048，消每 forward ~284ms QR 重算（两代共有开销），一行改动，独立于方案 A。
- **P1 待做**：多 group 合并一次 launch、FP16 accumulator（见 §七）。
- **待办**：旋转结果跨 bit 复用（需统一 seed+改量化+验精度）、接线 `triton_fused_dual_matmul`、去 `.item()` D2H 同步。
- **测试方法**：micro 测试须区分 cold/warm 路径，warm 才是真实场景；全模型 eval 由本人手动跑（`run.q.sh` / `eval_qwen35.py`），不自动执行。

---

## 十一、mini-MoE e2e bench 补充发现（2026-08-20）

### 背景

micro test 和真实 run.q.sh 结果出入大（旋转缓存收益假象、col_start 编译风暴、clone 去除净回归），
根因是单 kernel micro test 的形状、调用次数、参数多样性都和真实流程差太远。

新增 `turboquant_utils/test_triton_mp_moe_e2e_bench.py`：构造和真实模型同形状的
`WxA16BitPartitionedGroupMoE`（权重随机初始化），完整走一遍 forward 流程，作为
单 kernel test 和全模型 run 之间的桥梁。

配套：`WxA16BitPartitionedGroupMoE` 新增 `enable_timing` 开关 + `last_timings` 属性，
可程序化获取各阶段时间（router / sort / triton / compute / cleanup 等）。

### 核心发现：端到端差距是几十倍，不是 4 倍

profile 文档实验 2 结论是"Triton 4-bit 比 fp16 慢 4.29x"，那是**单 kernel 对比**。
完整 MoE forward 端到端的差距要大得多：

| 配置（单 4-bit, 16 experts, B≈64/expert） | 时间 | 比值 |
|-------------------------------------------|------|------|
| FP16 cuBLAS MoE（预反量化权重） | 1.6 ms | 1.0x |
| Triton 混合比特 MoE | 103 ms | **63x** |

| 配置（单 4-bit, 8 experts, B≈128/expert） | 时间 | 比值 |
|-------------------------------------------|------|------|
| FP16 cuBLAS MoE | 3.6 ms | 1.0x |
| Triton 混合比特 MoE | 272 ms | **75x** |

### 差距分层拆解

端到端几十倍的差距由三层叠加而来：

| 层面 | 慢多少 | 占总差距比例 | 说明 |
|------|--------|------------|------|
| 纯 kernel（Triton vs cuBLAS） | ~6x | ~10% | TF32 vs FP16 + Triton 算子效率（同实验 2/5） |
| 分组架构开销（旋转 + clone + 多次 launch） | ~3.5x | ~20% | 小形状利用率低，纯 kernel 仅占 compute 时间 ~28%（同实验 4） |
| per-expert × per-bit Python 循环 | ~3~4x | ~70% | `.item()` D2H 同步 + 循环体开销 + 额外 launch 累积 |

**关键结论**：
- 纯 kernel 内的优化（如 clone 去除、反量化位运算）端到端收益有限——kernel 只占总时间的一小部分
- 真正能大幅提速的方向是减少 Python 循环和 launch 次数（多 group 合并、多 expert 合并）
- 这也解释了"micro test 有收益、run.q.sh 没效果"：micro 测的是 kernel 内优化，放到端到端被稀释了一个数量级

### 精度验证

端到端数值正确性已验证：
- Triton MoE vs FP16 反量化 reference（完整 router → sort → per-expert×per-bit → scatter 全路径）
- max_diff ≈ 0.002，相对误差 ≈ 0.3%，与单 kernel 测试误差量级一致

### 扩展性质疑（待解释）

B（per-expert token 数）从 32 增加到 256（8 倍），Triton MoE 总时间几乎不变（~100ms）。
FP16 的时间也增长远慢于线性。原因可能是：
- 小形状下 launch overhead 和固定开销主导，计算时间占比低
- per-expert Python 循环开销是固定的（expert 数不变）

需要更大 B（如 B=1024+）才能看到计算密集区的线性扩展。

### 与真实 run.q.sh 的量级对比（待补）

目前缺真实模型一层 MoE forward 的时间数据，无法确认 bench 和 run.q.sh 的量级是否一致。
需本人手动跑一次 `run.q.sh --quant-layers 0`，读取
`[WxA16BitPartitionedGroupMoE] forward total: X.XXXs` 进行对比。



---

