# FP32→FP16 类型转换优化方案（WxA16 Kernel 内）

## 背景

mini-MoE e2e bench 实测：同等形状下，Triton 混合比特 MoE 比 FP16 cuBLAS MoE 慢 **60~75 倍**。
其中纯 kernel 计算只占一小部分，大部分开销来自分组架构 + Python 循环。

但即便是纯 kernel 层面，也还有很大优化空间。当前 kernel 的类型转换链存在冗余：

```
输入 x (fp16)
    │
    ├→ tl.load → inp_tile (fp16)
    │
权重 packed (uint8)
    ├→ unpack → idx (int32)
    ├→ codebook[idx] → w_quant (fp32)  ← codebook 是 fp32
    │
    └→ tl.dot(fp16 × fp32, allow_tf32=True)
         │
         └→ acc (fp32)  ← TF32 Tensor Core，fp32 累加器

norms (fp32) → acc * norms → output (fp32)
```

**问题**：
1. codebook 和 norms 都是 fp32，导致 w_quant 是 fp32，tl.dot 两边类型不一致
2. 混合精度下 allow_tf32=True 可能无法充分利用 Tensor Core（或者需要先提升到 fp32）
3. fp32 累加器寄存器压力大，限制了并行度
4. output 也是 fp32，但下游激活（silu、scatter）都是 fp16，需要额外类型转换

**核心思路**：把整条链路从 fp32 降到 fp16，走 FP16 Tensor Core，理论吞吐 2~3 倍于 TF32。

---

## P2：FP16 链路改造（核心优化）

### 目标

kernel 内整个计算链路改为 fp16：
- codebook: fp32 → fp16
- norms: fp32 → fp16
- accumulator: fp32 → fp16
- output: fp32 → fp16
- tl.dot: fp16 × fp16 → FP16 Tensor Core（而非 TF32）

### 预期收益

**理论上限 2~3 倍**（FP16 Tensor Core 吞吐 vs TF32）。
实际收益受限于：
- 内存带宽是否饱和（如果瓶颈在访存，计算再快也没用）
- 寄存器释放后能否提升并行度
- 小形状下 launch overhead 占比高

纯 kernel 层面预计 1.5~2 倍加速，端到端预计 +10~30%（因为 kernel 只占总时间的一部分）。

### 改动范围

#### 1. `quantize.py`：codebook 和 norms 的 dtype

- `get_codebook()` 返回 fp16 版本（或调用方 cast）
- `turboquant_quantize_packed_full()` 中 norms 存 fp16
- packed_data dict 里增加 dtype 信息
- **注意**：需要保证量化精度不明显下降。codebook 从 fp32 降到 fp16 的误差应该很小
  （码本值本身就是标量，fp16 精度足够）

#### 2. `triton_kernels.py`：kernel 内类型

- `_turboquant_fused_matmul_kernel_nbit`：
  - codebook_ptr 元素类型从 fp32 → fp16
  - norms_ptr 元素类型从 fp32 → fp16
  - acc dtype 从 tl.float32 → tl.float16
  - tl.dot 去掉 allow_tf32（fp16 × fp16 直接走 FP16 Tensor Core）
  - output 从 fp32 → fp16
- 所有 wrapper 函数（`triton_fused_matmul` / grouped / slice_rows / slice_in_features）：
  - output tensor 从 float32 → float16
  - 输入输出 dtype 校验

#### 3. `wxa16_bit_partitioned_moe.py`：forward 内类型

- gate_up_out / down_out 当前是 fp32（来自 kernel 输出），改成 fp16
- silu + mul 在 fp16 下进行（本来就是，因为输入 x 是 fp16）
- `expert_out` 的初始零张量 dtype 可能需要调整

### 精度验证

FP16 accumulator 的精度影响需要验证：
1. **单 kernel 精度**：对比 fp32 acc vs fp16 acc 的输出误差
2. **端到端精度**：e2e bench 里 Triton vs FP16 reference 的误差是否增大
3. **PPL 精度**：最终需要 run.q.sh 验证 ppl 影响

阈值参考：当前 fp32 accumulator 下相对误差 ~0.3%，如果 fp16 accumulator 后
误差在 1% 以内应该可以接受。

### 实施步骤（三步走，风险递进，每步都有可停的中间态）

#### Step 1：只改 output dtype（fp32→fp16），accumulator 保持 fp32 ✅ 代码完成

- **改动**：
  - kernel 内 `acc` 还是 fp32，`tl.dot` 还是 TF32，store 时 `acc.to(tl.float16)`
  - 四个 wrapper（`triton_fused_matmul` / grouped / slice_rows / slice_in_features）
    的 output tensor 从 `torch.float32` → `torch.float16`
  - Python 端 `output = torch.zeros(..., dtype=torch.float32)` 全部改 fp16
  - forward 中 `expert_out = torch.zeros_like(expert_tokens)` 自动 fp16（因为输入是 fp16）
- **预期收益**：~5-10%（省输出带宽 50% + 下游 silu/mul/scatter 从 fp32 降回 fp16）
- **风险**：极低。accumulator 还是 fp32，数值精度不变（只是最后 trunc 到 fp16）
- **目的**：打通 fp16 输出链路，确认 Python 端所有下游操作在 fp16 下正常工作

#### Step 2：codebook + norms + 旋转输入 改为 fp16，accumulator 保持 fp32 🔧 开发中

- **改动**：
  - `quantize.py`：`turboquant_quantize_packed_full` 中 codebook 和 norms 存 fp16
    （`.half()` 后存入 result dict）
  - `triton_kernels.py` grouped 系列函数：旋转计算从 fp32 降到 fp16
    - `generate_rotation_matrix(...).half()` 旋转矩阵转 fp16
    - `x[:, g_start:g_end] @ Pi.T` 去掉 `.float()`，直接 fp16 × fp16
    - 目的：使 tl.dot 两边都是 fp16，走 FP16 Tensor Core
  - kernel 内：`codebook_ptr` / `norms_ptr` 元素类型自动为 fp16，
    `w_quant` 自动 fp16，`tl.dot` 两边 fp16 → FP16 Tensor Core
  - 测试程序：`quantize_weight_simple` 返回 fp16 codebook/norms，
    `dequantize_weight_simple` 转回 fp32 做高精度 baseline
- **关键约束**：Triton `tl.dot` 要求两边 operand dtype 必须相同，
  所以 codebook 改 fp16 必须同时把输入 x_rot 也改成 fp16
- **预期收益**：~10-20%（codebook/norms 带宽减半 + FP16 Tensor Core + 旋转计算降精度）
- **风险**：低。codebook fp32→fp16 误差极小，旋转 fp16 误差也很小，
  accumulator 仍为 fp32，不影响累加精度

#### Step 3：accumulator 改为 fp16（完整 FP16 链路）🔧 开发中

- **改动**：
  - kernel 内 `acc = tl.zeros(dtype=tl.float16)`
  - `tl.dot` 加 `out_dtype=tl.float16`（Triton tl.dot 默认返回 fp32，需显式指定）
  - 去掉 `allow_tf32=True`（fp16 输入下无效）
  - 两个 kernel（单 matmul + dual matmul）同步修改
- **关键坑**：Triton `tl.dot(fp16, fp16)` 默认返回 fp32，需要 `out_dtype=tl.float16`
  才能让累加器保持 fp16，否则会报 loop-carried variable type mismatch 错误
- **预期收益**：较小。因为 Step 2 已经让 tl.dot 走 FP16 Tensor Core 了，
  Step 3 只是累加器从 fp32 变 fp16，主要收益是寄存器压力降低 → 可能提高并行度，
  但当前 BLOCK_B=16, BLOCK_N=64 配置下提升不明显
- **实测**：e2e 性能几乎不变（~125ms → ~126ms，在测量误差内）
- **风险**：中。需验证 FP16 accumulator 对 PPL 的影响
  - 溢出风险低：K=128，权重归一化，累加值 ≈128，远小于 fp16 max=65504
  - 精度影响：单 kernel 约 0.05% mean_diff/std（额外于 Step 2 的 0.08%）
- **回退**：精度不达标就退回到 Step 2（fp16 输入 + fp32 累加）

### 风险与回退

- **精度下降**：如果 PPL 影响不可接受，回退到 fp16 输入 + fp32 累加（Step 2）
- **kernel 编译问题**：Triton 对 fp16 accumulator 的支持可能需要特殊处理
  （如某些架构下 fp16 dot 需要特定配置）
- **溢出风险**：fp16 范围有限（max 65504），如果累加值过大可能上溢。
  但 TurboQuant 是归一化量化，权重 norm 后幅值 ≈ 1，累加 K=128 个乘积，
  每个乘积 ≈ 1，总和 ≈ 128，远小于 fp16 上限。溢出风险低。

### P2 进度追踪

| Step | 单 kernel 精度 | e2e 精度 | e2e 性能 | run.q.sh 性能 | run.q.sh PPL | 状态 |
|------|--------------|----------|----------|---------------|--------------|------|
| Step 1: output fp16 | ✅ | ✅ | ✅ | ⏳ | ⏳ | 已完成 |
| Step 2: codebook+norms+rot fp16 | ✅ | ✅ | ✅ | ⏳ | ⏳ | 已完成 |
| Step 3: acc fp16 | ✅ | ✅ | ✅ | 🔶 | 🔶 | 单层层测通过，待全层验证 |

> **Layer 0 实测（run.q.sh, Qwen3.5-35B-A3B）**：
> - 性能：forward 从 ~12.3s → 8.44s，**加速 ~31%**（含 Step 2+3 叠加）
> - PPL：wiki 6.5644→6.5641（-0.0003），c4 9.6789→9.6784（-0.0005），**精度几乎不变**
> - 注：单层层测结果，全模型效果待完整 run.q.sh 验证

---

## P4：FP16 链路后的内核深度优化

P2 FP16 改造完成后，kernel 计算从 TF32 切换到 FP16 Tensor Core，性能格局发生变化：
- 2-bit kernel 加速明显（~1.3x+），计算仍是瓶颈，FP16 吃得满
- 4-bit kernel 加速有限（~1.0x），瓶颈从计算转向 unpack + codebook lookup
- 固定 block size 配置可能不再是 fp16 下的最优解

### 优先级排序（更新于 P4-1 实验后）

| # | 优化项 | 预期收益 | 难度 | 状态 |
|---|--------|---------|------|------|
| 1 | **多 group 融合到单个 kernel launch**（P4-4） | **高（e2e 2.04x，per-kernel 2-11x）** | 中高 | ✅ 已完成 |
| 2 | Block size / num_warps / num_stages 离线调优（P4-2） | 中（e2e ~10%, per-kernel 1.4-2x） | 低 | ✅ 已完成 |
| 3 | Gate-Up epilogue 融合（silu+mul 合进 kernel）（P4-3） | 中（gate_up 省 15-25%） | 低 | ⏳ 待做 |
| 4 | 4-bit codebook gather 深度优化（P4-1） | 中低（大 K 场景收益大，per-group 小） | 中 | ⏳ 待做 |
| 5 | 旋转跨 bit 复用 + dual matmul 接入（P4-5） | 中 | 中 | ⏳ 待做 |
| 6 | Grouped GEMM 化（同 bit 多 expert 一次 kernel）（P4-6） | 高 | 高 | ⏳ 待做 |

### P4-1：4-bit unpack / codebook lookup 优化（优先级下调）

**实验结论（已验证）**：
- **per-group 场景 (K=128)**：4-bit 只比 FP16 慢 27%（12.0 vs 15.2 TFLOPs），
  dequant 开销占比不大，unpack 向量化收益有限（<5%）
- **大 K 场景 (K=2048)**：4-bit 比 FP16 慢 6.5x，其中 **codebook gather 占 86.5%**
  的开销，unpack 本身几乎不耗时
- tl.where 完全展开（16 路选择）**反而更慢**（0.84x），选择指令数多于 gather 开销
- num_warps=2 比 num_warps=4 快 ~50%（大 K 场景），说明 warp 间存在 L1 竞争

**优化方向（重新定位）**：
1. **codebook gather 优化**：需要找比 `tl.load(ptr + idx)` 更高效的小表查找方式
   - shared memory / L1 路径调优
   - Triton 编译器层面的优化（暂未找到有效方法）
2. **unpack 向量化**：per-K 收益有限，搁置
3. **优先级**：等 P4-4（多 group fusion）做完后，再评估是否值得做
   - 如果 fusion 后 K 维度变大，codebook gather 优化的价值会上升

### P4-4：多 group 融合到单个 kernel launch（最高优先级）

**背景与问题**：
- 当前 grouped 函数逐 group 调用 `triton_fused_matmul`（每个 group K=128），
  然后 Python 侧 `output += out_g` 累加
- **核心瓶颈**：每个 group 只做 K=128 的小矩阵乘，Tensor Core 利用率极低
  - per-group (K=128): ~12 TFLOPs（4-bit），~15 TFLOPs（FP16）
  - 整矩阵 (K=2048): ~10 TFLOPs（4-bit grouped 模拟），~65 TFLOPs（FP16）
  - 4-bit grouped e2e 比 FP16 整矩阵慢 **5~7 倍**
- kernel launch overhead 很低（<1%），不是主要问题
- **真正的问题**：小 K 导致 Tensor Core 利用率低 + 每个 K-tile 的 dequant 开销固定

**优化思路（从易到难，逐步推进）**：

#### 方案 A：Python 侧循环消除 + 单 kernel 内多 group 累加
- 在一个 kernel 内处理多个 group（同 bit-width），输出直接累加
- 每个 group 仍用独立的 packed indices 偏移和独立的旋转输入
- 收益：省去 Python 循环的 `output += out_g` 额外开销（D2D 写入+读取）
- 难度：低
- 预期收益：小（5-10%），主要是省中间输出带宽

#### 方案 B：K 维度拼接 + 单 kernel 大矩阵乘
- 把多个 group 的 K 维度在 kernel 内拼接起来，做一次大的 tl.dot
- 关键点：
  - 输入 x_rot：每个 group 旋转不同，必须分别计算（CPU 侧已预计算）
  - 权重 packed indices：连续存储，可以按偏移访问
  - codebook：同 bit-width 共享同一 codebook
  - norms：每个 group 独立
- 实现方式：
  - kernel 循环 K 维度，但每 BLOCK_K 个元素可能跨越 group 边界
  - 或：kernel 内按 group 循环，每次处理一个完整 group 的 K=128，累加到 acc
- **实际上就是方案 A，但在 kernel 内做循环累加，省去多次 launch + 中间结果**
- 预期收益：中（10-20%），主要来自省去多次 launch 的固定开销 + 中间张量读写

#### 方案 C：同 bit-width 多 group 的 K 维融合（增大有效 K）
- 把多个同 bit-width group 当作一个更大的 K 来处理
- 输入 x_rot：每个 group 不同，无法简单拼接 K 维度
- 但可以**按 N 维度并行、逐个 group 累加** — 本质上和方案 B 一样
- 更大的收益点：**如果有办法让 tl.dot 做更大 K 的计算**，Tensor Core 利用率会提升
- 但由于 x_rot 每组独立，实际上 K 维不能跨 group 拼接
- 结论：方案 C ≈ 方案 B，收益上限受限于「单个 tl.dot 仍是 K=128」

#### 方案 D：旋转矩阵 fusion（需要算法层面支持）
- 核心想法：多个 group 的旋转矩阵可以合并成一个大旋转矩阵
  R = diag(R0, R1, ..., Rn)（块对角矩阵）
- 则 x_rot = x @ R^T，整个 K 维度一次旋转完成
- 然后可以用完整 K 的 dequant + matmul，Tensor Core 利用率大幅提升
- 但 packed indices 是按 group 存储的，需要确认 K 维连续性
- 这是**理论上收益最大**的方向，但实现复杂度也最高
- 难度：高
- 预期收益：高（30-50%+），取决于 Tensor Core 利用率提升幅度

**实施计划**：
1. 先做 **方案 A/B**（kernel 内多 group 累加）—— 低风险、立即可做
2. 评估收益后，再决定是否推进方案 D（旋转 fusion + 大 K matmul）

**改动范围**：
- `triton_kernels.py`：新增 `_turboquant_fused_matmul_kernel_grouped` kernel
- `triton_kernels.py`：新增 `_triton_fused_matmul_grouped_fused` 辅助函数
- `triton_fused_matmul_grouped` / `_slice_rows` / `_slice_in_features` 各加 fast path

---

#### P4-4 实验结果（已完成）

**实现方案**：方案 A/B 混合 — 单 kernel 内循环 NUM_GROUPS 个 group，每个 group 独立
dequant + matmul，乘 norms 后累加到 total_acc。x_rot 在 Python 侧预计算并拼接为
`(B, K_total)`，packed indices 直接按偏移访问。kernel 支持 `packed_col_start` /
`norms_group_start` 偏移参数，兼容 slice_in_features 场景。

**支持范围**：
- bit-width：1/2/4/8（BIT_WIDTH 为 constexpr）
- 接口：`triton_fused_matmul_grouped`、`_slice_rows`、`_slice_in_features`
- 接口完全不变，内部自动选择 fast path（条件：`in_features % group_size == 0 and num_groups >= 2`）

**性能结果**（MoE 场景，group_size=128）：

| 场景 | Baseline | Fused | Speedup |
|------|----------|-------|---------|
| down_proj 4-bit, B=128 (22 groups) | 0.53 ms (2.8 TFLOPs) | 0.16 ms (9.1 TFLOPs) | **3.26x** |
| down_proj 4-bit, B≤64 (22 groups) | 0.53 ms | 0.096 ms | **5.5x** |
| gate_up 2-bit 段, B=128 (8 groups) | 0.19 ms (3.9 TFLOPs) | 0.054 ms (13.8 TFLOPs) | **3.59x** |
| gate_up 2-bit 段, B≤32 (8 groups) | 0.19 ms | 0.025 ms | **7.7x** |
| gate_up 4-bit 段, B=128 (8 groups) | 0.20 ms (3.8 TFLOPs) | 0.089 ms (8.3 TFLOPs) | **2.21x** |
| gate_up 4-bit 段, B≤32 (8 groups) | 0.20 ms | 0.038 ms | **5.2x** |
| 1-bit, B=64 (16 groups) | 0.38 ms (1.4 TFLOPs) | 0.033 ms (16.1 TFLOPs) | **11.5x** |

**精度**：
- mean/std ≈ 0.03-0.05%（FP16 累加顺序差异，完全正常）
- max_diff 在 FP16 精度范围内
- slice_rows：max_diff = 0（完全一致）
- slice_in_features：mean/std ≈ 0.03%
- 全模型 PPL：wikitext2 6.5581 vs baseline 6.5613，c4 9.6765 vs 9.677 — 基本吻合

**全模型验证（Qwen3.5-35B-A3B Layer 0, 256 experts, 256 tokens）**：
- Baseline forward: **8.44s**
- P4-4 fused forward: **4.13s**
- **加速比：2.04x**

**关键洞察**：
1. **收益远超预期** — 最初预期 10-20%，实际 2-11x
2. 原因：baseline 中每个 group K=128 的小 matmul Tensor Core 利用率极低
   （2-4 TFLOPs vs 峰值 80 TFLOPs），fused 后有效工作负载变大，利用率大幅提升
3. **小 B 时收益更大** — MoE 场景 per-expert batch size 通常很小（top-k 分散），
   这正是收益最大的情况
4. **1-bit 收益最大（11.5x）** — 因为单 group 计算量最小，填充率最低

**未做项 / 后续可做**：
- [ ] `triton_fused_dual_matmul` 的 grouped 版本 fusion（主路径未使用，优先级低）
- [ ] 方案 D：旋转矩阵 fusion + 大 K matmul（理论收益更高，但复杂度高）
- [ ] 混合 bit-width 场景的 fusion（2-bit 段 + 4-bit 段可分别 fused，再加一次加法）
- [ ] BLOCK_B / BLOCK_N / BLOCK_K / num_warps 的联合调优（P4-2）

---

### P4-2：Block size / num_warps 离线调优 ✅ 已完成

**背景**：
- P4-4 fused kernel 初始配置：BLOCK_B=16, BLOCK_N=64, BLOCK_K=64，num_warps=4
- FP16 + fused 后寄存器压力变化大，默认配置远非最优
- 不用 runtime autotune（首次编译开销太大），离线扫参数硬编码最优值

**调优参数**：
- BLOCK_B: 16, 32
- BLOCK_N: 32, 64, 128, 256
- BLOCK_K: 32, 64, 128
- num_warps: 2, 4, 8
- num_stages: 2, 3, 4

**调优脚本**：`test_p4_tune.py`

**各 bit-width 最优配置（RTX 5090, group_size=128, B≈32）**：

| bit-width | BLOCK_B | BLOCK_N | BLOCK_K | num_warps | num_stages | per-kernel 加速 |
|-----------|---------|---------|---------|-----------|------------|----------------|
| 1-bit | 16 | 32 | 128 | 2 | 3 | 1.50x |
| 2-bit | 32 | 32 | 128 | 2 | 2 | 1.37x |
| 4-bit | 32 | 32 | 128 | 8 | 3 | 1.88x |
| 8-bit | 16 | 32 | 128 | 4 | 3 | （未精细调） |

**关键发现**：
1. **BLOCK_K=128 全场景最优**：大 K-tile 减少内层循环次数，更好利用 Tensor Core
2. **4-bit 需要更多 warps**（访存瓶颈，多 warps 藏延迟）
3. **1/2-bit warps=2 最优**（计算轻，warps 多了反而浪费寄存器）
4. **BLOCK_N=32 普遍优于 64**：fused 后 N 维度利用率已高，小 block 更灵活

**全模型验证（Qwen3.5-35B-A3B Layer 0）**：
- forward: 4.13s → 3.77s（**+9.5%**，c4, 256 samples）
- wikitext2: 2.51s → 2.31s（**+8.7%**）
- PPL: wiki 6.5581→6.5605，c4 9.6765→9.6777（基本不变 ✅）
- commit: abfd809+（P4-2）vs 575adf8+（P4-4 baseline）

---

## P3 快速验证项（低风险，先逐个试）

在做大改动之前，先验证几个低开销的小优化，确认各自的收益量级。
全部在 `test_triton_mp_moe_e2e_bench.py` 上测，有收益的再上 run.q.sh。

### P3-a：旋转缓存上限 128 → 2048 ✅ 已落地

- **背景**：down 方向 seed 含 expert 偏移，每 forward 工作集 ~1032 个 key > 上限 128，
  导致每 forward 都发生缓存抖动，QR 重算 ~1160 次 ≈ 284ms（详见 profile 文档 §十 记录 4）
- **改动**：`rotation.py` 的 `_MAX_CACHE_SIZE` 从 128 改到 2048
- **显存代价**：CPU ~66MB + GPU ~66MB（1032 个 128×128 矩阵 × 4B），可接受
- **风险**：极低，一行改动
- **实测结果**：
  - 双模块陷阱排查：`turboquant_utils.rotation` 和 `turboquant_model.rotation` 是同一模块对象
    （`sys.modules.setdefault` 别名机制），`_MAX_CACHE_SIZE = 2048` 已全局生效，无需重复修改
  - mini-MoE e2e bench（16 experts）：几乎无收益（expert 少，工作集未超 128，本来就不抖）
  - 真实 run.q.sh（256 experts）：收益应已包含在 gc 删除之前的 baseline 中，无法单独剥离
- **结论**：改正确、零风险，已经生效；真实场景有收益但无法单独量化，留着即可。

### P3-b：去掉 forward 末尾的 gc.collect() / empty_cache() ✅ 已落地

- **背景**：`WxA16BitPartitionedGroupMoE.forward` 末尾每轮都调 `gc.collect()` 和
  `torch.cuda.empty_cache()`。前者是 CPU 同步+扫描，后者触发 GPU 同步+释放，
  在每 forward 都调的情况下贡献显著开销
- **改动**：直接删除（不加开关）
- **风险**：显存碎片可能增加，但推理阶段分配模式稳定，实际未发现问题
- **实测结果**：
  - mini-MoE e2e bench：~24% 加速（forward 末尾同步开销被移除后，triton 占比从 ~70% → 97.8%）
  - 真实 run.q.sh：Layer 0 forward 从 ~19-20s → **12.51s**，约 **35-38%** 加速
    （比 bench 幅度更大，可能因为 attention + MoE 叠加效应）
- **结论**：本轮最大的"白捡"优化，已落地且效果显著。

### P3-c：去掉 expert_offsets[expert_idx].item() 的 D2H 同步 ✅ 已落地

- **背景**：per-expert × per-bit 循环里 `expert_offsets[expert_idx].item()` 每次触发
  D2H 同步（CPU 等 GPU 返回一个整数）。256 experts × 2 bits = 1024 次/forward
- **改动**：`expert_offsets` 加载时生成 CPU 常驻 Python list（`self._expert_offsets_cpu`），
  循环里直接读 list 索引。加懒初始化兼容所有构造路径
- **风险**：极低，不改变数值
- **实测结果**（e2e bench，含完整 GPU 同步）：

  | 配置 | 节省比例 | 绝对节省 |
  |------|---------|---------|
  | 16 experts, H=1024 | -2.3% | 0.59 ms |
  | 64 experts, H=1024 | -3.0% | 2.07 ms |
  | 256 experts, H=512 | **-5.9%** | 10.15 ms |

  真实 run.q.sh（256 exp, H=2048）：Layer 0 从 12.51s → **12.31s**，约 **-1.6%**。
  原因：大 H 下 kernel 计算占绝对主导，Python 同步开销被稀释。

- **结论**：确定有效、零风险、改动极小，已落地。大 H 场景收益有限但白捡。

### P3-d：scatter_reduce_ → index_add_ ✅ 已落地

- **背景**：`final_hidden_states.scatter_reduce_(0, exp_token_idx.view(-1,1).repeat(1, H), ...)`
  需要先把 1D index repeat 成 (M, H) 的大张量，每 expert 一次额外分配 + 拷贝
- **改动**：改用 `final_hidden_states.index_add_(0, exp_token_idx, expert_out)`，
  index 保持 1D，语义完全等价（dim=0 按行累加）
- **风险**：极低，e2e 精度验证 max_diff = 0.0（完全一致）
- **实测结果**（e2e bench）：

  | 配置 | 节省比例 | 绝对节省 |
  |------|---------|---------|
  | 16 experts, H=1024 | -1.2% | 0.31 ms |
  | 64 experts, H=1024 | -2.2% | 1.51 ms |
  | 256 experts, H=512 | **-4.5%** | 7.76 ms |

- **结论**：代码更简洁、零风险、数值完全一致，已落地。小 expert/大 B 场景收益小，
  expert 多且每 expert token 少的场景收益更明显。

### P3 小结

| 优化 | e2e bench (256 exp) | run.q.sh (Layer 0) | 风险 | 状态 |
|------|---------------------|---------------------|------|------|
| P3-a 旋转缓存 2048 | 无法单独测（已生效） | 已包含在 baseline | 极低 | ✅ 已落地 |
| P3-b 删 gc | ~24% | ~35%（19→12.5s） | 低 | ✅ 已落地 |
| P3-c 去 .item() | ~5.9% | ~1.6%（12.51→12.31s） | 极低 | ✅ 已落地 |
| P3-d index_add_ | ~4.5% | <1%（包含在上条内） | 极低 | ✅ 已落地 |

P3 全部为"纯 overhead 消除、不碰计算逻辑"类优化，零风险或极低风险，累计 run.q.sh 上
Layer 0 forward 从 ~19-20s → 12.31s，**约 36-38% 加速**。大头来自 P3-b（删 gc），
其余几项是"白捡"的叠加收益。

---

### P4 实验笔记（P4-1 瓶颈分析，已完成）

**实验脚本**：`test_p4_4bit_unpack.py`, `test_p4_codebook_opt.py`, `test_p4_e2e_sim.py`

#### 单 kernel 层面

| 场景 | FP16 | 4-bit | 2-bit | 4/2 比 |
|------|------|-------|-------|--------|
| per-group (K=128) | 15.2 TFLOPs | 12.0 TFLOPs | - | - |
| large K (K=2048) | 65 TFLOPs | 10 TFLOPs | ~10 TFLOPs | ~1.0x |

**关键结论**：
1. **4-bit 的瓶颈不在 unpack**，而在 **codebook gather（间接内存访问）**
   - 开销分解：tl.dot 占 ~15%，unpack 占 ~0%，gather 占 ~85%（大 K 场景）
   - codebook 只有 16 个 fp16 值（32 字节），完全在 L1 cache，但间接寻址指令开销大
2. **4-bit ≈ 2-bit**（1.04x），bit-width 几乎不影响性能
   - 因为 gather 开销相同，且都远大于 tl.dot 计算
3. **tl.where 展开反而更慢**（0.84x）：15 个选择指令 > 1 次 gather
4. **num_warps=2 比 4 快 ~50%**：warp 间 L1 cache 竞争

#### e2e grouped 调用层面

| 场景 | FP16 整矩阵 | 4-bit grouped | 差距 |
|------|------------|--------------|------|
| down (22 groups, K=2816) | 0.023 ms (65 TFLOPs) | 0.164 ms (9 TFLOPs) | 7.25x |
| gate_up (16 groups, K=2048) | 0.021 ms (72 TFLOPs) | 0.119 ms (12 TFLOPs) | 5.78x |

**关键结论**：
1. **kernel launch overhead 几乎为 0**（<1%），不是问题
2. **真正的问题是小 K 导致 Tensor Core 利用率低**
   - 每个 group K=128 的 matmul 效率只有整矩阵的 ~1/5
   - 22 个小 kernel 的总时间 = 22 × 单个小 kernel 时间
   - 而 FP16 整矩阵 1 个大 kernel 时间很短
3. **分组调用方式本身是最大瓶颈**，不是单个 kernel 的计算效率

#### 优先级调整

P4-1（unpack 向量化）收益有限，下调优先级。
**P4-4（多 group fusion）上调为最高优先级** — 通过减少分组次数提高 Tensor Core 利用率。

#### P4-4 实验结论（已完成 ✅）

- **收益**：
  - per-kernel: 2.2x - 11.5x（视 bit-width 和 B 大小而定）
  - **全模型 e2e (Layer 0): 2.04x**（8.44s → 4.13s, Qwen3.5-35B-A3B）
- **覆盖范围**：`triton_fused_matmul_grouped` / `_slice_rows` / `_slice_in_features`
- **支持 bit-width**：1/2/4/8
- **接口不变**，内部根据 `in_features % group_size == 0 and num_groups >= 2` 自动选择 fast path
- **精度**：
  - 数值：FP16 累加顺序差异 ≈ 0.03-0.05%，完全正常
  - 全模型 PPL：wikitext2 6.5581 vs 6.5613，c4 9.6765 vs 9.677（baseline），基本吻合
- **验证日志**：`logs/0818.5.wxa16.opt.p4-4.log`
- **commit 对比**：575adf8+ (P4-4) vs dfb8fc1 (baseline)

---

## 测试与验证

### 测试层次

1. **单 kernel 正确性**：`test_triton_mixed_precision.py` 增加 fp16 accumulator 对比
2. **端到端正确性**：`test_triton_mp_moe_e2e_bench.py` 精度验证
3. **性能基准**：`test_triton_mp_moe_e2e_bench.py` warm 路径时间对比
4. **全模型验证**：手动 run.q.sh（PPL + 速度）

### 必须对比的 baseline

- 当前主分支（fp32 accumulator + TF32）
- 优化后（fp16 accumulator + FP16 Tensor Core）
- 数值误差对比（max_diff, mean_diff）

---

## 相关历史经验（来自 new-wxa16-plan-260818.md）

- 第 7 条 "FP16 输出 + FP16 激活" 曾被列为中优先级，但没有实际落地
- 本次方案更彻底：不只是输出 fp16，而是整条链路 fp16
- autotune 去掉是对的（首次调用开销大），但 fp16 改造后可能需要重新调 block size
