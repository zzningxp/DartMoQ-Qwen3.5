# P5-3：WxA16 MoE Kernel Tile Size 联合自动调优

> 定位：P4-2 单维调优的进阶版。对 BLOCK_B / BLOCK_N / BLOCK_K / num_warps / num_stages 做联合网格搜索，在真实 eval 形状下离线扫出每 bit-width 的最优配置并硬编码。
> 难度：低 | 预期收益：5–15% | 风险：零（纯配置参数调整，不改变计算逻辑）

---

## 一、背景与问题

### 1.1 当前状态

`turboquant_utils/triton_kernels.py` 中 `_FUSED_GROUPED_CONFIG` 表存放各 bit-width 的 kernel tile 配置：

```python
_FUSED_GROUPED_CONFIG = {
    1: (16, 32, 128, 2, 3),   # (BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages)
    2: (32, 32, 128, 2, 2),
    4: (32, 32, 128, 8, 3),
    8: (16, 32, 128, 4, 3),
}
```

这组配置来自 P4-2 的**单维扫**（每次只动一个参数，其余固定），针对 RTX 5090，测试形状约为 group_size=128, B≈32。单维扫的收益是 +9.5%（forward 4.13s → 3.77s）。

### 1.2 为什么单维调优不够

五个 tile 参数之间存在**强耦合**：

| 参数 | 作用 | 与其他参数的耦合 |
|---|---|---|
| BLOCK_B | 每个 program 处理的 token 行数 | 与 BLOCK_N 共同决定 tile 大小 → 决定 warps 需求 |
| BLOCK_N | 每个 program 处理的 neuron 列数 | 与 BLOCK_K 共同决定每次 tl.dot 计算密度 |
| BLOCK_K | K 方向每次切分大小（内循环步长） | 越大计算越密，但寄存器/共享内存压力越大 |
| num_warps | 每个 program 的 warp 数 | 必须匹配 tile 总工作量，少了喂不饱，多了寄存器溢出 |
| num_stages | 软件流水线深度 | 与 BLOCK_K 相乘就是预取数据量，受共享内存限制 |

典型的耦合案例：
- BLOCK_B 从 32 升到 64，如果 warps 还是 2，每个 warp 分到的活变多但并行度不够，性能可能不升反降。
- BLOCK_K 从 64 升到 128，计算量翻倍，如果 stages 不跟上，访存延迟就藏不住。

单维搜索容易停在**局部最优**。联合搜索有机会再挤出 5–15% 的收益。

### 1.3 为什么不做 Runtime Autotune

Triton 自带 `@triton.autotune`，但：
1. 首次编译开销太大：每个配置都要编译一次，MoE 场景 bit_width × direction × shape 组合多，冷启动代价不可接受。
2. 我们的形状相对固定（eval 时 per-expert token 数在一定范围内），离线调一次硬编码就够了。
3. 符合项目策略："Triton autotune 调参 ❌ 不做 | 动态小 kernel，搜索编译开销 > 收益"（backlog §P4-2 备注）。

---

## 二、参数含义与算法对应

### 2.1 Kernel 结构速览

以 group-first 布局的 `_turboquant_fused_matmul_kernel_grouped_gf` 为例：

```
一个 kernel program 负责一个 (BLOCK_B × BLOCK_N) 的输出块
│
├─ 外循环 for g in range(NUM_GROUPS)        // 遍历 16 个 group
│   ├─ 加载当前 group 的 norms [BLOCK_N]
│   └─ 内循环 for k in range(0, GROUP_SIZE, BLOCK_K)   // 每 group 内按 BLOCK_K 切块
│        ├─ 加载 inp_tile  [BLOCK_B × BLOCK_K]        // activation
│        ├─ 解包 + 码本查 w_quant [BLOCK_N × BLOCK_K]  // 权重反量化
│        └─ tl.dot(inp_tile, w_quantᵀ) → acc_g       // FP16 Tensor Core
│
└─ 写回 total_acc [BLOCK_B × BLOCK_N]
```

### 2.2 五参数详解

| 参数 | 算法含义 | 硬件含义 | 调优直觉 |
|---|---|---|---|
| **BLOCK_B** | 每个 program 处理的 token 行数（B 维切块） | grid 第 0 维的步长 | B 小（MoE 场景）时不能太大也不能太小，太小 launch 太多 program 开销大 |
| **BLOCK_N** | 每个 program 处理的 neuron 列数（N 维切块） | grid 第 1 维的步长 | 太小则 Tensor Core 利用率低，太大则权重寄存器压力大 + padding 浪费 |
| **BLOCK_K** | 每个 group 内 K 方向的切块大小（内循环步长） | 每次 tl.dot 的 mma K 维度 | 越大计算密度越高，但受 group_size(128) 上限和寄存器限制 |
| **num_warps** | 每个 program 用多少 warp（32 线程/warp） | CUDA block 内并行粒度 | tile 大、访存重（低 bit）需要更多 warps 藏延迟；但太多会寄存器溢出 |
| **num_stages** | 软件流水线深度（预取多少迭代的数据） | 计算-访存重叠程度 | 增加可隐藏访存延迟，但占更多共享内存/寄存器 |

---

## 三、搜索方案

### 3.1 搜索空间

**全量网格搜索，不分粗搜/精搜。**

搜索空间本身不大（324 组全组合，过滤后约 150-200 组有效），离线搜索完全可以承受。全量搜索比分阶段搜索结果更可靠，也更简单。

| 参数 | 候选值 | 备注 |
|---|---|---|
| BLOCK_B | 16, 32, 64 | 覆盖小/中/大 token 数场景 |
| BLOCK_N | 16, 32, 64, 128 | P4-2 最优是 32，往两边各扩两档 |
| BLOCK_K | 32, 64, 128 | 必须是 16 的倍数（mma 形状约束），且 ≤ group_size=128 |
| num_warps | 2, 4, 8 | 2 的幂，Triton 常规选项 |
| num_stages | 2, 3, 4 | 从 2 到 4，覆盖浅/中/深流水线 |

总量 3 × 4 × 3 × 3 × 3 = **324 组**。

**过滤规则**（明显不合理的组合直接跳过，节省编译+运行时间）：
1. `BLOCK_B * BLOCK_N < num_warps * 32 * 4`：每个 warp 分到的元素太少（少于 4 个），并行效率低
2. `BLOCK_K > GROUP_SIZE`：超过 group 大小无意义（内循环只跑一次就行）
3. Triton 编译失败的组合自动跳过（比如寄存器不够、共享内存溢出）
4. BLOCK_B > B 或 BLOCK_N > N 的跳过（实际不会用到的配置）

估计过滤后约 **150–200 组**有效配置。

**测量方法**：每组配置 warmup 3 次 + 正式测量 10 次，取**中位数**（比均值更抗 GPU 功耗/温度波动干扰）。

### 3.2 测试形状

以**真实 eval 形状**为准，避免 micro-bench 大矩阵得出的最优配置在真实小 B 场景无效（"micro≠real"陷阱）。

#### gate_up 方向

| 参数 | 值 | 来源 |
|---|---|---|
| B (token/expert) | 8, 16, 32, 64, 128 | 覆盖从极端不均匀到均匀满载的全范围 |
| N (neurons) | 2816 | inter_size × 2 (gate + up) |
| K (in_features) | 2048 | hidden_size |
| group_size | 128 | 项目默认 |
| num_groups | 16 | K / group_size |

#### down 方向

| 参数 | 值 | 来源 |
|---|---|---|
| B (token/expert) | 8, 16, 32, 64, 128 | 同上 |
| N (neurons) | 2048 | hidden_size |
| K (in_features) | 2816 | inter_size |
| group_size | 128 | 项目默认 |
| num_groups | ~22 | K / group_size |

**全 bit 覆盖**：1 / 2 / 4 / 8 bit 都跑，虽然 1-bit 和 8-bit 占比小，但搜索是零边际成本的。

### 3.3 B 自适应配置策略

真实 MoE 中 token 分布不均匀（有的 expert 分到上百 token，有的只有几个），**单一配置不可能在所有 B 下都最优**。因此搜索后需要决定配置策略：

**策略 A：单一鲁棒配置**
- 如果不同 B 值下的最优配置相同或很接近（性能差 < 3%），选加权平均最优的一套配置
- 最简单，零额外开销

**策略 B：B 自适应配置表**
- 如果差异大（> 5%），做一张 `bit_width × B_range` 的二维配置表
- kernel launch 前根据 B 的大小查表选配置（开销：一次比较 + 参数传值，可忽略）
- B 区间划分：B ≤ 16 / 16 < B ≤ 32 / 32 < B ≤ 64 / B > 64

**决策方式**：先跑 B=32（中位数场景）的全量搜索，拿到最优配置后，再在 B=16/64 上跑 top-10 配置验证鲁棒性。如果前三名在各 B 值下重合度高，用策略 A；否则上策略 B。

### 3.4 gate_up / down 拆两套配置

**本次就拆，不共用。**

原因：
- gate_up 和 down 的 K/N 维度互换（gate_up: K=2048, N=2816；down: K=2816, N=2048），计算模式不同
- 权重访存量差异大（gate_up 是 2×inter 的权重，down 是 hidden×inter 的权重）
- 最优 tile 配置大概率不一样，不拆会损失几个点
- 代码改动量很小：配置表从一张扩成两张，4 个调用点分别取各自配置

配置表命名：
- `_FUSED_GROUPED_CONFIG_GATE_UP` — gate_up 方向（slice_rows 路径）
- `_FUSED_GROUPED_CONFIG_DOWN` — down 方向（slice_in_features 路径）

如果后续验证差异不大，再合并也不迟。

### 3.5 测试方法

#### 单 kernel micro-bench（搜索用）

脚本：`test/test_p53_tune.py`（新建）

- 直接调用 `_triton_fused_matmul_grouped_gf`（绕过 rotation 准备，直接给拼接好的 x_rot_concat）
- 输入：B, N, K_total, group_size, bit_width, direction（direction 决定搜哪套配置表的空间）
- 流程：对每组配置 → 编译（命中缓存则跳过）→ warmup 3 次 → 计时 10 次取中位数
- 输出：按性能排序的配置表（含 TFLOPS、占峰值比）+ 最佳配置
- 用 `torch.cuda.Event` 计时（精度高于 `time.time()`）
- 结果存为 JSON 方便后续分析

#### e2e 验证（最终收益确认）

用现有 `turboquant_utils/test_triton_mp_moe_e2e_bench.py`：
- 把最优配置硬编码回配置表
- 跑 batch=1, seq_len=2048（真实 eval 尺度）
- 对比调优前后的 warm 稳态时间、MoE 阶段时间

---

## 四、执行步骤

### 步骤 1：写调优脚本

创建 `test/test_p53_tune.py`，核心功能：
- 参数化输入形状（B / N / K / group_size / bit_width）
- 生成全量网格 + 自动过滤不合理配置
- 编译缓存（Triton 自动磁盘缓存，首次慢后续快）
- 每组配置 warmup 3 次 + 计时 10 次取中位数
- 按性能排序输出，结果存 JSON

### 步骤 2：gate_up 方向基线搜索（B=32, 2-bit & 4-bit）

先跑占比最大的 2-bit 和 4-bit，B=32（中位数场景），看看联合搜索相对当前配置有多大收益空间。如果 top1 比当前快不了 5%，后面的可以简化。

```bash
# 2-bit gate_up, B=32
conda run -n dart312 python test/test_p53_tune.py \
    --bit 2 --B 32 --N 2816 --K 2048 --group-size 128 --direction gate_up

# 4-bit gate_up, B=32
conda run -n dart312 python test/test_p53_tune.py \
    --bit 4 --B 32 --N 2816 --K 2048 --group-size 128 --direction gate_up
```

### 步骤 3：B 鲁棒性验证（B=16, B=64）

对步骤 2 的 top-10 配置，在 B=16 和 B=64 上复测，看最优配置的形状鲁棒性。

- 如果前三名重合度高 → 单一配置策略，取加权平均最优
- 如果差异大 → 上 B 自适应配置表

### 步骤 4：down 方向搜索

对 2-bit / 4-bit 的 down 方向重复步骤 2-3。

```bash
# 2-bit down, B=32
conda run -n dart312 python test/test_p53_tune.py \
    --bit 2 --B 32 --N 2048 --K 2816 --group-size 128 --direction down

# 4-bit down, B=32
conda run -n dart312 python test/test_p53_tune.py \
    --bit 4 --B 32 --N 2048 --K 2816 --group-size 128 --direction down
```

### 步骤 5：1-bit / 8-bit 补全

1-bit 和 8-bit 占比小，但搜索是零成本的，顺手跑完。

### 步骤 6：更新配置表 + 代码接入

- 新增 `_FUSED_GROUPED_CONFIG_GATE_UP` 和 `_FUSED_GROUPED_CONFIG_DOWN` 两张配置表
- 新增 `_get_fused_grouped_config(bit_width, direction)` 函数
- 修改 4 个 kernel 调用点（gate_up 的 gf/非gf，down 的 gf/非gf），传 direction 参数
- 如果用 B 自适应，再加一个根据 B 选配置的逻辑

### 步骤 7：e2e 验证

```bash
# 真实 eval 尺度
conda run -n dart312 python turboquant_utils/test_triton_mp_moe_e2e_bench.py \
    --batch-size 1 --seq-len 2048
```

对比调优前后的 warm 稳态时间、MoE 阶段时间、有效 TFLOPS。

---

## 五、学术价值与创新点

### 5.1 问题的一般性

量化 GEMM kernel 的 tile size 调优是一个**普遍存在但研究不足**的问题：

1. **形状异构性**：MoE 场景下每个 expert 的 token 数差异很大（从几个到几百个），传统 GEMM 调优针对的是大方阵，在小 B 场景下结论完全不适用。
2. **计算-访存耦合**：WxA16 kernel 不是纯计算 bound——权重需要实时反量化（bit unpack + codebook lookup），访存模式和计算模式强耦合，tile size 同时影响两者。
3. **多维度联合空间**：5 个参数 × 多个 bit-width × 多种形状，搜索空间虽不大但耦合强，单维扫容易陷局部最优。

### 5.2 方法论价值

本探索的学术贡献可以从以下角度提炼：

**(1) 小批量量化 GEMM 的性能特征分析**

- 在 MoE 典型小 B（16–64）场景下，系统分析 BLOCK_B / BLOCK_N / BLOCK_K / num_warps / num_stages 对性能的联合影响
- 揭示"单维调优的局部最优偏差"有多大，量化联合搜索的收益上限
- 不同 bit-width 下的最优配置模式差异（1-bit 计算轻，4-bit 访存重，最优 warps 和 stages 模式不同）

**(2) 形状自适应配置选择的启发式规则**

- 基于搜索结果，提炼可推广的经验公式/启发式规则（而不只是硬编码一张表），例如：
  - `num_warps = f(B, N, bit_width)` 的经验关系
  - 小 B 场景下 BLOCK_B 的最优取值规律
  - bit-width 如何影响最优 warp 数（位宽越低，codebook lookup 越重，warps 需求如何变化）

**(3) 跨硬件平台的配置迁移性**

- 如果后续在多卡上测试，可以研究 RTX 5090 上找到的最优配置在其他 GPU（A100 / H100 / 消费级卡）上的迁移程度
- 哪些参数与硬件强相关（num_warps、num_stages），哪些相对鲁棒（BLOCK 比例）

**(4) 与 Auto-Tuning 方法的联系**

- 虽然我们用离线网格搜索，但搜索结果可以作为更高级 autotuning 方法的 baseline
- 可以讨论：对于 MoE 这种形状动态变化的场景，什么样的 autotuning 策略（离线预计算 vs 在线微调 vs 启发式）最有效
- 对比随机搜索、贝叶斯优化、遗传算法在这个空间的搜索效率（324 组的网格其实可以当 ground truth）

### 5.3 工程价值

- **直接收益**：零代码侵入的 5–15% 加速，无精度风险
- **方法复用**：调优脚本和方法论可以复用到项目其他 kernel（attention projection、P5-2 merged kernel、未来的 W8A8 kernel 等）
- **配置表沉淀**：形成 per-bit-width 的最优配置基准，后续 kernel 改动（如优化反量化路径）可以快速重调

---

## 六、风险与注意事项

### 6.1 风险评估

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| 联合搜索收益 < 5% | 中 | 低（白忙活但不损失） | 先跑 2-bit 和 4-bit 的粗搜，看 top1 vs 当前配置差距有多大，小就停 |
| micro-bench 有收益但 e2e 没收益 | 中 | 低 | 用真实形状测，且必须 e2e 验证才落地 |
| 不同 B 数下最优配置差异大 | 中 | 中 | 多测几个 B 值（16 / 32 / 64），取 Pareto 最优或加权平均 |
| 编译缓存导致测量不准 | 低 | 低 | 每组配置单独 warmup，用 cuda.Event 精确计时，多次平均 |

### 6.2 注意事项

1. **必须区分 cold/warm**：cold 包含 Triton JIT 编译时间，不算性能结论。warm 稳态才是。
2. **注意 GPU 功耗/温度波动**：长时间搜索时最好开固定功耗，或者每组配置随机顺序测试减少系统误差。
3. **配置必须能在所有形状下正确运行**：找到的最优配置要在 B=1 到 B=几百的范围内都能正确跑（边界检查 mask 是通用的，应该没问题，但要验证）。
4. **P5-2 合并 kernel 的配置另说**：本次只调单 expert gf kernel。如果 P5-2 后续救活，merged kernel 的 B_total 大很多，最优 tile 肯定不一样，需要单独调。

---

## 七、相关文档

- P4-2 单维调优结论：`new-wax16-plan-260820-reduce-fp32.md` §P4-2
- 优化总览 backlog：`wxa16-optimization-backlog-260824.md`
- Kernel 定义：`turboquant_utils/triton_kernels.py` `_turboquant_fused_matmul_kernel_grouped_gf`
- 当前配置表：`turboquant_utils/triton_kernels.py` `_FUSED_GROUPED_CONFIG` (L459-464)
- e2e 基准测试：`turboquant_utils/test_triton_mp_moe_e2e_bench.py`
