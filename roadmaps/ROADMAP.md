# DartMoQ-Qwen3.5: 长期路线图

## 概述

本项目将 DartMoQ 混合精度量化框架适配到 Qwen3.5 MoE 架构上。
Qwen3.5 MoE 有一个完全不同的权重存储设计，可以实现高效的分组矩阵乘法。

---

## 架构背景

### 传统 MoE vs Qwen3.5 MoE

**传统 MoE (DeepSeek, OLMoE 等):**
```
每个专家是独立的 Module:
  - expert[i].gate_proj: Linear(in_dim, inter_dim)
  - expert[i].up_proj: Linear(in_dim, inter_dim)
  - expert[i].down_proj: Linear(inter_dim, out_dim)
```

**Qwen3.5 MoE:**
```
所有专家合并为两个大张量:
  - mlp.experts.gate_up_proj: (num_experts, 2 * intermediate_size, hidden_size)
  - mlp.experts.down_proj: (num_experts, hidden_size, intermediate_size)

这可以实现高效的分组矩阵乘法！
```

### Qwen3.5 的 grouped_gemm 格式详解

**grouped_gemm 的本质：只是 gate 和 up 合并，专家还是独立的**

```
假设：
  num_experts = 4
  hidden_size = 10
  intermediate_size = 3

Qwen3.5 的权重格式：
  gate_up_proj: (4, 2*3, 10)  ← (num_experts, 2*inter_size, hidden_size)
  down_proj: (4, 10, 3)       ← (num_experts, hidden_size, inter_size)

让我们看看 gate_up_proj 的实际内容：

gate_up_proj[0]:  ← 专家 0
  [:, :10] = gate_proj of expert 0  ← 前一半是 gate
  [:, 10:] = up_proj of expert 0    ← 后一半是 up

gate_up_proj[1]:  ← 专家 1
  [:, :10] = gate_proj of expert 1
  [:, 10:] = up_proj of expert 1
```

**图解：Qwen3.5 的 grouped_gemm 格式**

```
gate_up_proj: (num_experts, 2*inter_size, hidden_size)

  ┌─────────────────────────────────────────────────────────────┐
  │  专家 0                      专家 1                          │
  │ ┌─────────────┬─────────────┐ ┌─────────────┬─────────────┐ │
  │ │   gate[0]   │    up[0]    │ │   gate[1]   │    up[1]    │ │
  │ ├─────────────┼─────────────┤ ├─────────────┼─────────────┤ │
  │ │ (3, 10)     │  (3, 10)    │ │  (3, 10)    │  (3, 10)    │ │
  │ └─────────────┴─────────────┘ └─────────────┴─────────────┘ │
  │ ┌─────────────┬─────────────┐ ┌─────────────┬─────────────┐ │
  │ │   gate[2]   │    up[2]    │ │   gate[3]   │    up[3]    │ │
  │ ├─────────────┼─────────────┤ ├─────────────┼─────────────┤ │
  │ │ (3, 10)     │  (3, 10)    │ │  (3, 10)    │  (3, 10)    │ │
  │ └─────────────┴─────────────┘ └─────────────┴─────────────┘ │
  │  专家 2                      专家 3                          │
  └─────────────────────────────────────────────────────────────┘
         ↑                              ↑
    gate 和 up 合并             但专家之间还是独立的！


down_proj: (num_experts, hidden_size, inter_size)

  ┌─────────────────────────────────────────────────────────────┐
  │  专家 0         专家 1         专家 2         专家 3        │
  │ ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
  │ │ down[0]  │   │ down[1]  │   │ down[2]  │   │ down[3]  │ │
  │ ├──────────┤   ├──────────┤   ├──────────┤   ├──────────┤ │
  │ │(10, 3)  │   │ (10, 3)  │   │ (10, 3)  │   │ (10, 3)  │ │
  │ └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
  └─────────────────────────────────────────────────────────────┘
         ↑
    每个专家的 down_proj 也是独立的！
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

## 第一阶段: FP16 基线评估

**目标:** 运行 Qwen3.5 MoE 原始的 FP16 困惑度评估，支持 CPU 待机模式。

### 任务:
- [x] 创建文件夹结构并复制必要工具
- [x] 为 Qwen3.5 MoE 适配 `load_model`
- [x] 为 Qwen3.5 架构适配顺序评估（CPU 待机）
- [ ] 验证困惑度符合预期结果
- [x] 探索并记录 Qwen3.5 MoE 的层结构

### 输入/输出:
- **输入:** 原始 Qwen3.5 MoE 模型 (FP16)
- **输出:** 在 wikitext2 和 C4 上的困惑度分数

### 关键文件:
- `qwen35_utils.py` - Qwen3.5 专用工具
- `eval_qwen35.py` - 评估脚本
- `explore_qwen35.py` - 架构探索脚本

---

## 第二阶段: DartMoQ 混合 MoE + FP16 反量化 (已完成)

**目标:** 将 DartMoQ 的混合精度量化方法应用到 Qwen3.5 MoE，仍然在 FP16 中评估（通过反量化）。

### 当前状态 (已完成)
- [x] 为 Qwen3.5 的合并权重格式适配专家敏感度分析
- [x] 在 Qwen3.5 结构中实现神经元排序和分组（通过转换为传统格式）
- [x] 创建适配 Qwen3.5 架构的混合 MoE 包装器
- [x] 适配动态规划位宽分配

---

## 第三阶段 (实现中 - 回归 grouped_gemm 格式，当前目标)

**目标架构:**
```
                    ┌─────────────────────────────────────────┐
                    │         Router (原始 gate 复用)          │
                    │         (保持路由一致性)                  │
                    └─────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────────────────────────┐
                    │      Grouped GEMM (按精度分组)           │
                    │  ┌──────────────┐                       │
                    │  │bit2 experts  │ gate_up_proj:        │
                    │  │              │ (E, 2*I2, H)         │
                    │  └──────────────┘                       │
                    │  ┌──────────────┐                       │
                    │  │bit3 experts  │ gate_up_proj:        │
                    │  │              │ (E, 2*I3, H)         │
                    │  └──────────────┘                       │
                    │         ...                             │
                    └─────────────────────────────────────────┘
```

**默认启用：** `args.true_quant` 默认为 True，直接使用第三阶段的 grouped_gemm 格式

### ⚠️ 关键问题分析：传统格式量化后能否重组回 grouped_gemm？

**问题：** 当前量化流程存在信息丢失！

**当前流程：**
```
原始 grouped_gemm
  ↓ [convert_grouped_gemm_to_traditional]
传统格式
  ↓ [quantize] (神经元排序 + 分组)
量化后的传统格式 (SimpleMoEBlock + DartMoQHybridWrapper)
  ↓ ??? [丢失信息]
grouped_gemm 格式？
```

**信息丢失分析：**

在 `qwen35_layer_reconstruct.py:273-277`，我们构建了关键映射：
```python
bit_to_indices = {}
for bit, group_indices in zip(orig_bit_config, expert_groups):
    if bit not in bit_to_indices:
        bit_to_indices[bit] = []
    bit_to_indices[bit].extend(group_indices)
```

这个 `bit_to_indices` 告诉我们：
- ✅ 专家 e 的 bit 2 包含神经元 [0, 7, 13, ...]
- ✅ 专家 e 的 bit 3 包含神经元 [1, 2, 4, ...]

**但是！** 我们只用它来收集权重，然后就丢弃了！

**量化后保留的信息：**
- ✅ 子专家的权重值
- ✅ 子专家的 bit 宽度（`_quant_bit` 属性）
- ❌ **丢失了：** 这些权重对应原始神经元的哪些索引位置

**解决方案：** 在量化过程中保存元数据

**方案 A（推荐）：在 `DartMoQHybridWrapper` 中保存元数据**
```python
class DartMoQHybridWrapper(nn.Module):
    def __init__(self, sub_experts, bit_to_indices=None, expert_bit_indices=None):
        super().__init__()
        self.sub_experts = nn.ModuleList(sub_experts)
        self.bit_to_indices = bit_to_indices  # 保存原始索引映射（按 bit）
        self.expert_bit_indices = expert_bit_indices  # 保存每个子专家对应的神经元索引
```

### 详细设计文档

详见 `PHASE3_DESIGN.md` 获取完整的第三阶段设计。

---

## 第四阶段 (以后考虑 - 真实量化推理)

**目标：** 在 RTX 5090 上使用 Qwen3.5 的分组格式实现真实量化推理（int4/int3/int2/int1）。当前暂不实现。

### 核心架构：DartMoQ-Accel

Qwen3.5 MoE 的权重存储设计与混合精度量化完美契合：
- **传统**: 每个专家是独立的 Module
- **Qwen3.5**: 所有专家合并为两个大张量

这使得高效的**分组矩阵乘法**成为可能，对混合精度是巨大优势！

### 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        DartMoQ-Accel Layer                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Router     │  │ Shared Expert│  │Bit Groups    │          │
│  └──────┬───────┘  └──────────────┘  └──────┬───────┘          │
│         │                                    │                  │
│         ▼                                    ▼                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Bit-Grouped Expert Storage             │       │
│  ├─────────────────────────────────────────────────────┤       │
│  │  ┌─────────────────────────────────────────────┐  │       │
│  │  │ Bit2 Group (merged weights)                 │  │       │
│  │  │  - gate_up_proj: (E2, 2*I2, H)              │  │       │
│  │  │  - down_proj: (E2, H, I2)                   │  │       │
│  │  │  - scales/zeros: quantization params        │  │       │
│  │  └─────────────────────────────────────────────┘  │       │
│  │  ┌─────────────────────────────────────────────┐  │       │
│  │  │ Bit3 Group (merged weights)                 │  │       │
│  │  │  - gate_up_proj: (E3, 2*I3, H)              │  │       │
│  │  │  - down_proj: (E3, H, I3)                   │  │       │
│  │  │  - scales/zeros: quantization params        │  │       │
│  │  └─────────────────────────────────────────────┘  │       │
│  │  ┌─────────────────────────────────────────────┐  │       │
│  │  │ Bit4 Group (merged weights)                 │  │       │
│  │  │  - gate_up_proj: (E4, 2*I4, H)              │  │       │
│  │  │  - down_proj: (E4, H, I4)                   │  │       │
│  │  │  - scales/zeros: quantization params        │  │       │
│  │  └─────────────────────────────────────────────┘  │       │
│  └─────────────────────────────────────────────────────┘       │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │               Group GEMM Execution Engine                │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │  Bit2 Group GEMM  ────┐                                 │  │
│  │  Bit3 Group GEMM  ────┼─── Parallel Execution            │  │
│  │  Bit4 Group GEMM  ────┘                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 任务:
- [ ] 研究 PyTorch 对分组矩阵乘法的支持
- [ ] 实现量化权重存储格式 (int2/3/4 + scales/zeros)
- [ ] 添加 W4A16/W3A16/W2A16 分组 GEMM 支持
  - 选项 1: 使用现有内核 (AutoAWQ, AutoGPTQ, Marlin)
  - 选项 2: 为分组 GEMM 实现自定义 CUDA 内核
- [ ] 在 RTX 5090 上进行性能基准测试

### 关键研究问题:
- PyTorch 是否原生支持我们用例的分组矩阵乘法？
- 我们能否为 Qwen3.5 的格式适配现有的 MoE 量化内核？
- 相比反量化 FP16，潜在加速比是多少？

---

## 关键设计决策

### 选项 A (长期目标): 保持 Qwen3.5 的分组格式，但按位宽拆分为不同组
```
mlp.experts_bit2.gate_up_proj: (num_experts, 2*inter_size_bit2, hidden_size)
mlp.experts_bit3.gate_up_proj: (num_experts, 2*inter_size_bit3, hidden_size)
...
```

### 选项 B (已完成): 转换为传统格式进行量化，然后再转换回来
(第二阶段更简单，但效率较低)

---

## 关键文件

- `qwen35_layer_reconstruct.py` - Qwen3.5 层重构（已扩展支持元数据）
- `qwen35_hybrid_moe.py` - 第三阶段按 bit 分组的 grouped_gemm 实现
- `dartmoq_hybridmoe.py` - 第二阶段传统格式的混合 MoE 包装器（已扩展）
- `run_qwen35.py` - 主量化脚本（支持 --true_quant）
- `grouped_gemm_moe_adapter.py` - 格式转换适配层
- `qwen35_utils.py` - Qwen3.5 专用工具

### 第四阶段文件（待实现）
- `qwen35_quant_kernels.py` - 量化内核
- `qwen35_inference.py` - 量化推理

---

## 文件夹结构

```
dartmoq-qwen3.5/
├── README.md
├── roadmaps/
│   ├── ROADMAP.md (本文件)
│   ├── PHASE3_DESIGN.md (第三阶段详细设计)
│   ├── ROADMAP_ALTERNATIVE_QWENMULTILINEAR.md (备选方案参考)
│   └── GROUPED_GEMM_VS_BLOCKSPARSE.md (格式对比参考)
├── explore_qwen35.py          # 第一阶段: 架构探索
├── eval_qwen35.py             # 第一阶段: FP16 评估
├── qwen35_utils.py            # 第一阶段+: Qwen3.5 工具
│
├── qwen35_layer_reconstruct.py  # 第二阶段/第三阶段: 层重构
├── qwen35_hybrid_moe.py         # 第三阶段: 按 bit 分组的 grouped_gemm 实现
├── dartmoq_hybridmoe.py         # 第二阶段: 传统格式的混合 MoE 包装器
├── run_qwen35.py                # 第二阶段/第三阶段: 主量化脚本
├── grouped_gemm_moe_adapter.py  # 格式转换适配层
│
└── test/                      # 测试脚本
    ├── test_phase1.py
    ├── test_phase2.py
    └── test_phase3.py
```

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

## 数学等价性保证 (Router Gate 设计)

**设计思路：**
原始路由：
```
gate_logits = gate(x)  # (batch, num_experts)
topk_weights, topk_indices = topk(gate_logits)
```

目标：复用原始 gate（不重新训练），保证路由一致性

**等价变换：**
对于每个 bit b：
1. 收集所有专家中，属于 bit b 的神经元索引
2. 对于 gate 来说，路由决策只关心 "选哪些专家"，不关心 "专家内部怎么分组"
3. 因此，可以完全复用原始 gate，只是在前向时对每个 bit 分别做 grouped_gemm

**前向传播等价性：**
```
原始：
  out = sum_{e in topk} weight_e * expert_e(x)

等价分解：
  out = sum_{e in topk} weight_e * [ sum_b expert_e^b(x) ]
      = sum_b [ sum_{e in topk} weight_e * expert_e^b(x) ]

其中 expert_e^b(x) 是 expert_e 中 bit b 的神经元贡献
```

因此，我们可以：
- 保持原始 gate 不变（保证路由一致性）
- 按 bit 分组专家的神经元
- 前向时，先用原始 gate 路由，再分别对每个 bit 做 grouped_gemm

---

## 依赖

与主 DartMoQ 项目相同：
- PyTorch
- Transformers
- (可选) 用于第四阶段自定义内核的 CUDA Toolkit
- (可选) 用于零样本评估的 lm_eval
