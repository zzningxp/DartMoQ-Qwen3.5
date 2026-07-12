# DartMoQ-Qwen3.5: 长期路线图

## 概述

本项目将 DartMoQ 混合精度量化框架适配到 Qwen3.5 MoE 架构上。
Qwen3.5 MoE 有一个完全不同的权重存储设计，可以实现高效的分组矩阵乘法。

---

## 架构背景

### Qwen3.5 MoE 的 grouped_gemm 原始格式

**Qwen3.5 MoE 原生格式：**
```
所有专家合并为两个大张量:
  - mlp.experts.gate_up_proj: (num_experts, 2 * intermediate_size, hidden_size)
  - mlp.experts.down_proj: (num_experts, hidden_size, intermediate_size)
```

---

## Qwen3.5 MoE 适配差异总结

### 1. 模型加载架构差异

- **原有模型**: 纯语言模型，直接加载 `AutoModelForCausalLM`
- **Qwen3.5 MoE**: 外层是多模态模型 `Qwen3_5MoeForConditionalGeneration`，需要提取 `model.language_model` 获取纯文本部分

### 2. 专家存储结构差异

- **原有模型**: 每个专家是独立的 Module
- **Qwen3.5 MoE**: 所有专家权重合并存储在大张量中

### 3. 注意力机制差异

- **原有模型**: 每层都使用相同的自注意力机制
- **Qwen3.5 MoE**: 交替使用两种注意力：`linear_attn` 和 `self_attn`

### 4. MoE 层结构差异

- **原有模型**: 通常只有路由专家
- **Qwen3.5 MoE**: 同时包含 `shared_expert` 和路由专家

### 5. 门控输出格式差异

- **原有模型**: 通常只返回 logits，需要手动计算 topk
- **Qwen3.5 MoE**: 返回三值 tuple `(logits, topk_weights, topk_indices)`

---

## 量化流程总览

我们的量化为混合精度模型分为四个阶段，通过三次格式转换完成：

```
  原始 grouped_gemm (Qwen3.5 原生)
       ↓ [Stage 1] TraditionalMoEWrapper 转换
  传统 MoE 格式 (每个专家独立)
       ↓ [Stage 2] DartMoQ 量化 + 神经元排序分组
  MoEBuildBlock + DartMoQHybridWrapper (按 bit 分子专家)
       ↓ [Stage 3] 重组为紧凑格式
  BitPartitionedGroupMoE (最终格式，按 bit 分区存储)
```

---

## 第一阶段: FP16 基线评估 + 格式转换

**目标:** 运行 Qwen3.5 MoE 原始的 FP16 困惑度评估，并将 grouped_gemm 格式转换为传统 MoE 格式供 DartMoQ 量化。

### 关键转换: TraditionalMoEWrapper

**作用**: 将 Qwen3.5 原生的 grouped_gemm 大张量拆分为独立专家

```python
# 原始格式
gate_up_proj: (num_experts, 2*inter_size, hidden_size)
down_proj: (num_experts, hidden_size, inter_size)

# 转换后
experts[i].gate_proj.weight: (inter_size, hidden_size)
experts[i].up_proj.weight: (inter_size, hidden_size)
experts[i].down_proj.weight: (hidden_size, inter_size)
```

**内存优化**: 转换后立即删除原始权重，释放内存

### 任务:
- [x] 创建文件夹结构并复制必要工具
- [x] 为 Qwen3.5 MoE 适配 `load_model`
- [x] 为 Qwen3.5 架构适配顺序评估（CPU 待机）
- [x] 实现 TraditionalMoEWrapper 格式转换
- [x] 探索并记录 Qwen3.5 MoE 的层结构

### 输入/输出:
- **输入:** 原始 Qwen3.5 MoE 模型 (FP16)
- **输出:** 在 wikitext2 和 C4 上的困惑度分数

### 关键文件:
- `qwen35_utils.py` - Qwen3.5 专用工具
- `eval_qwen35.py` - 评估脚本
- `explore_qwen35.py` - 架构探索脚本
- `grouped_gemm_moe_adapter.py` - 格式转换适配层

---

## 第二阶段: DartMoQ 混合 MoE 量化

**目标:** 将 DartMoQ 的混合精度量化方法应用到转换后的传统格式 MoE。

### 关键流程

1. **敏感度分析**: 根据权重分布确定每个神经元的最优 bit 宽度
2. **神经元排序**: 将同 bit 宽度的神经元排在一起
3. **分组**: 将神经元按 bit 分组，创建 sub_experts
4. **动态规划位宽分配**: 全局优化位宽分配

### 输出结构: MoEBuildBlock + DartMoQHybridWrapper

```
MoEBuildBlock
├── gate (router)
├── shared_expert (可选)
└── experts (nn.ModuleList)
    ├── expert[0]
    │   └── sub_experts (nn.ModuleList)
    │       ├── ExpertMLP (bit=2)
    │       ├── ExpertMLP (bit=3)
    │       └── ExpertMLP (bit=4)
    ├── expert[1]
    └── ...
```

### 当前状态 (已完成)
- [x] 为 Qwen3.5 的合并权重格式适配专家敏感度分析
- [x] 在 Qwen3.5 结构中实现神经元排序和分组（通过转换为传统格式）
- [x] 创建适配 Qwen3.5 架构的混合 MoE 包装器
- [x] 适配动态规划位宽分配

---

## 第三阶段: BitPartitionedGroupMoE 紧凑格式

**目标:** 将量化后的结构重组为高效的紧凑格式，按 bit 分区存储所有权重。

### BitPartitionedGroupMoE 架构（当前采用）

**设计思路**:
- 只保留按 bit 分开的权重，内存最优
- 优化前向计算：按 expert 批量计算，减少循环
- 无 padding，紧凑存储

**存储结构（无冗余）**:
```
BitPartitionedGroupMoE
├── gate (router)
├── shared_expert (可选)
└── bit_weights
    ├── gate_up: ParameterDict[str, Tensor]
    │   ├── '2': (total_neurons_2x, H)  # 所有专家 bit=2 的 gate+up 拼接
    │   ├── '3': (total_neurons_2x, H)
    │   └── '4': (total_neurons_2x, H)
    ├── down: ParameterDict[str, Tensor]
    │   ├── '2': (H, total_neurons)  # 所有专家 bit=2 的 down 拼接
    │   ├── '3': (H, total_neurons)
    │   └── '4': (H, total_neurons)
    └── expert_offsets: Dict[str, LongTensor]
        ├── '2': (E+1,)  # 每个专家在 bit=2 中的 start/end idx
        ├── '3': (E+1,)
        └── '4': (E+1,)
```

**紧凑存储详解**:
```
gate_up['2']: [gate_e0, up_e0, gate_e1, up_e1, gate_e2, up_e2, ...]
              ↑ 每个专家的 bit=2 神经元的 gate 和 up 拼接在一起

down['2']: [down_e0, down_e1, down_e2, ...]
           ↑ 每个专家的 bit=2 神经元的 down 拼接在一起

expert_offsets['2']: [0, n0, n0+n1, n0+n1+n2, ...]
                     ↑ 记录每个专家的起始位置
```

**前向计算优化**:
1. **先按 expert 组织 token**: 对每个 token，先找出选择了哪些 expert
2. **批量计算**: 对同一个 expert 的所有 token，一次性计算所有 bit
3. **scatter_reduce 累加**: 高效地将结果累加回对应位置

**优势**:
- ✅ 内存最优：无 padding，无冗余存储
- ✅ 速度快：批量计算，减少循环开销
- ✅ 内存管理完善：转换后立即清理中间结构

### 快速调试

- [x] **按需单层 convert**: 只在要量化某一层时才 convert 那一层，而不是一开始全 convert
- [x] **部分层量化调试模式**: 使用 `--quant_layers 0-5` 只量化前几层，后面直接 break，节省时间

```bash
python run_qwen35.py --quant_layers 0-5
```

---

## 测试结果

| git      | model            | sli  | q_scheme          | rank                     | q_mode       | q_layers          | wiki    | c4       | status  | time    | t_quant  | t_ppl   | t_wiki | t_c4   |
|----------|------------------|------|-------------------|--------------------------|--------------|-------------------|---------|----------|---------|---------|----------|---------|--------|--------|
| d6c5026  | Qwen3.5-35B-A3B  |      |                   |                          | fp16         |                   | 6.5807  | 9.6724   | ok      | 201.6   |          | 201.6   | 87.11  | 114.49 |
| d6c5026  | Qwen3.5-35B-A3B  | 4    | global-a8s8m2bpw  | turboquant_innerproduct  | turboquant   | [0]               | 6.5743  | 9.6766   | ok      | 430.98  | 221.44   | 209.54  | 89.02  | 120.52 |
| d6c5026  | Qwen3.5-35B-A3B  | 4    | global-a8s8m2bpw  | turboquant_innerproduct  | turboquant   | [0, 1, 2, 3, 4]  | 6.6445  | 9.7605   | ok      | 1284.25 | 1093.08  | 191.17  | 83.79  | 107.38 |
| d6c5026  | Qwen3.5-35B-A3B  | 4    | global-a8s8m2bpw  | turboquant_innerproduct  | turboquant   | all               | 7.6874  | 11.2625  | ok      | 8089.54 | 7921.86  | 167.68  | 68.95  | 98.73  |

---

## 第四阶段 (W4A16 真实量化存储 + 纯整数推理)

**目标：** 实现 W4A16 量化权重的紧凑存储 + 纯整数推理，不反量化回 FP16 做矩阵乘法。

**新的量化推理流程：**
```
量化阶段：w_fp16 → quantize → w_int4 (存下来)
推理阶段：w_int4 → 直接整数推理 → (x_fp16 @ w_dequant) 等价于 (x_fp16 @ w_int4 + scale)
```

---

### 目标架构一：快速验证版（Python + PyTorch 原生实现）

**目标：** 快速验证 W4A16 纯整数推理的精度和性能可行性，使用 PyTorch 原生操作。

**文件夹结构：**
```
dartmoq_qwen3.5/
├── kernels/
│   ├── __init__.py
│   └── w4a16_kernels.py     # W4A16 纯整数推理 kernel (PyTorch 实现)
├── quantization/
│   ├── __init__.py
│   ├── w4a16_quantizer.py   # 量化器，生成 W4A16 权重
│   └── weight_packing.py    # 权重打包，适配 kernel 格式
└── moe/
    ├── __init__.py
    └── w4a16_moe.py         # W4A16 版本的 MoE 推理模块
```

**核心设计：**
1. **量化器**：支持 per-group 量化，可配置 group_size
2. **权重打包**：将 INT4 打包为 INT32/INT64 存储
3. **MoE 推理**：按 bit 分组，批量处理每个 expert

**关键任务：**
- [ ] 实现 W4A16Quantizer (per-group 量化)
- [ ] 实现权重打包/解包函数
- [ ] 实现 PyTorch 原生 W4A16 推理 kernel
- [ ] 验证 ppl 精度 (对比现有反量化方案)
- [ ] 初步性能测试

---

### 目标架构二：高性能实现版（基于 Machete/Marlin）

**目标：** 针对 RTX 5090 (Blackwell SM12.0) 架构优化的高性能推理。

**技术选型分析：**
| 方案 | 优势 | 劣势 | 推荐度 |
|------|------|------|--------|
| **Machete** | 最新架构优化，支持 wgmma，来自 Neural Magic (Marlin 团队) | 仍在迭代中 | ⭐⭐⭐⭐⭐ |
| **Marlin MoE** | 成熟稳定，已有 MoE 专用优化 | 使用旧版 mma 指令 | ⭐⭐⭐⭐ |
| **exllamav3 EXL3** | 专门为 MoE 优化，有完整工具链 | 格式封闭 | ⭐⭐⭐ |

**文件夹结构：**
```
kernels/csrc/
├── machete_moe/
│   ├── machete_moe_kernel.cu   # Machete MoE kernel
│   └── machete_moe_wrapper.cpp # 绑定代码
├── marlin_moe/
│   └── marlin_moe_wna16_kernel.cu  # Marlin MoE kernel (备选)
└── utils/
    ├── weight_packing.cu       # 权重打包工具
    └── token_sorting.cu        # Token 排序工具
```

**核心设计：**
1. **Machete Kernel**：基于 CUTLASS 3.5+，支持 Blackwell wgmma 指令
2. **Marlin MoE Kernel**：备选方案，成熟稳定
3. **Token 排序优化**：按选择的 expert 排序 token，提高 GPU 利用率
4. **批量处理**：同一 expert 的所有 token 一起处理

**关键优化点：**
- **WGMMA 指令**：利用 Blackwell 新的 Tensor Core 指令
- **Shared Memory 流水线**：更大的 shared memory (128KB per SM)
- **Token 排序**：减少 warp 内分支，提高局部性
- **动态批处理**：根据当前 batch 的 expert 分布调整

**关键任务：**
- [ ] 从 vLLM 提取 Machete kernel
- [ ] 适配 MoE 场景
- [ ] 实现 C++/CUDA 扩展
- [ ] 集成到现有 BitPartitionedGroupMoE 架构
- [ ] 性能 profiling 和调优

---

### 量化后权重存储格式 (兼容两种架构)

```
BitPartitionedW4A16:
├── gate: Router 权重 (保持 FP16)
├── shared_expert: (可选，FP16 或 W8A16)
└── w4a16_weights:
    ├── gate_up:
    │   ├── '2': (packed INT4, scales, zeros, g_idx)
    │   ├── '3': (packed INT4, scales, zeros, g_idx)
    │   └── '4': (packed INT4, scales, zeros, g_idx)
    └── down:
        ├── '2': (packed INT4, scales, zeros, g_idx)
        ├── '3': (packed INT4, scales, zeros, g_idx)
        └── '4': (packed INT4, scales, zeros, g_idx)
    └── expert_offsets: (记录每个 expert 在每种 bit 中的起始位置)
```

---

### 验证策略

**精度验证：**
1. PPL 对比：wikitext2/c4 数据集上与现有方案对比
2. Expert 输出对比：验证每个 expert 的输出差异
3. 端到端生成测试：验证生成质量

**性能验证：**
1. Kernel microbenchmark：测量 kernel 的纯计算性能
2. 端到端推理：测量完整 MoE 层的推理时间
3. Memory 带宽测试：验证是否达到内存带宽瓶颈

---

## 关键挑战

### 1. 分组矩阵乘法支持
- PyTorch 有没有高效的分组矩阵乘法操作？
- 我们需要自定义 CUDA 内核吗？

### 2. 分组格式内的量化
- 如何在保持分组结构的同时应用分组量化？
- 如何在同一专家内处理不同位宽？

### 3. 内存布局
- Qwen3.5 的权重布局可能针对特定硬件优化
- 需要理解如何最好地适配我们的量化方法

---
