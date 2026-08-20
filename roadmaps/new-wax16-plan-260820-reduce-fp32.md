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

### 实施步骤

1. **第一步**：只改 output dtype（fp32→fp16），accumulator 保持 fp32
   - 最小改动，验证输出类型转换是否正确
   - 收益：减少输出带宽，可能有限

2. **第二步**：codebook + norms 改为 fp16，accumulator 保持 fp32
   - 验证混合精度（fp16 输入，fp32 累加）下的数值正确性
   - 收益：w_quant 从 fp32 变 fp16，tl.dot 可能更高效

3. **第三步**：accumulator 改为 fp16（完整 FP16 链路）
   - 最大改动，也是最大收益
   - 需要仔细验证精度

### 风险与回退

- **精度下降**：如果 PPL 影响不可接受，回退到 fp16 输入 + fp32 累加（第二步）
- **kernel 编译问题**：Triton 对 fp16 accumulator 的支持可能需要特殊处理
  （如某些架构下 fp16 dot 需要特定配置）
- **溢出风险**：fp16 范围有限（max 65504），如果累加值过大可能上溢。
  但 TurboQuant 是归一化量化，权重 norm 后幅值 ≈ 1，累加 K=128 个乘积，
  每个乘积 ≈ 1，总和 ≈ 128，远小于 fp16 上限。溢出风险低。

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
