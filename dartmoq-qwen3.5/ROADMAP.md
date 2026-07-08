# DartMoQ-Qwen3.5: 长期路线图

## 概述

本项目将 DartMoQ 混合精度量化框架适配到 Qwen3.5 MoE 架构上。
Qwen3.5 MoE 有一个完全不同的权重存储设计，可以实现高效的分组矩阵乘法。

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


## 第二阶段: DartMoQ 混合 MoE + FP16 反量化

**目标:** 将 DartMoQ 的混合精度量化方法应用到 Qwen3.5 MoE，仍然在 FP16 中评估（通过反量化）。

### 设计原则
- ✅ 不训练，保持路由一致性，数学推导的等价结构
- ✅ 每一步验证正确性，用 git 节点保证可追溯
- ⏳ 第三阶段先不急

### 当前状态 (已完成 - 选项 B)
- [x] 为 Qwen3.5 的合并权重格式适配专家敏感度分析
- [x] 在 Qwen3.5 结构中实现神经元排序和分组（通过转换为传统格式）
- [x] 创建适配 Qwen3.5 架构的混合 MoE 包装器
- [x] 适配动态规划位宽分配
- [ ] 通过 FP16 反量化评估验证数值正确性 (**下一步 3.1**)

选项 A 是不转成传统格式，直接用 Qwen3.5 格式本身。但是之前尝试时，跑不通，错误信息没有保存。

### 第三阶段 (待实现 - 回归 grouped_gemm 格式)

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
- ❌ **丢失了**：这些权重对应原始神经元的哪些索引位置

**为什么这很关键？**
如果要重组回 grouped_gemm 格式，我们需要知道：
- 对于专家 e，bit 2 的神经元在原始 `intermediate_size` 中是哪些位置？
- 没有位置信息，就无法正确拼接到 `(E, 2*I, H)` 的张量中

**解决方案：** 在量化过程中保存元数据

**方案 A（推荐）：在 `DartMoQHybridWrapper` 中保存元数据**
```python
class DartMoQHybridWrapper(nn.Module):
    def __init__(self, sub_experts, bit_to_indices=None):
        super().__init__()
        self.sub_experts = nn.ModuleList(sub_experts)
        self.bit_to_indices = bit_to_indices  # 保存原始索引映射
```

**方案 B：创建独立的元数据结构**
```python
class QuantizationMetadata:
    def __init__(self):
        self.layer_metadata = []  # 每层的元数据
    
    def add_layer(self, layer_idx, expert_bit_indices):
        # expert_bit_indices[e][b] = 专家 e 中 bit b 的神经元索引列表
        self.layer_metadata.append(expert_bit_indices)
```

---

**任务分解:**
- [ ] **3.1 验证当前量化正确性** (先做)
  - 通过 FP16 反量化评估
  - 记录 PPL 基线

- [ ] **3.1.5 修改代码保存元数据**
  - 在 `DartMoQHybridWrapper` 中保存 `bit_to_indices`
  - 或创建独立的 `QuantizationMetadata` 结构
  - 确保可以追踪每个神经元的原始位置
  - 提交 git 节点

- [ ] **3.2 设计权重重组函数**
  - 输入: 量化后的传统格式 MoE (SimpleMoEBlock + DartMoQHybridWrapper)
  - 输出: 按精度分组的 grouped_gemm 格式
  - 关键逻辑: 收集每个精度的所有神经元，重新组织权重
  - 验证数值等价性

- [ ] **3.3 实现 Qwen35HybridMoE 类**
  - 完善 `qwen35_hybrid_moe.py` 中的 `Qwen35HybridExperts`
  - 实现按精度分组的前向传播
  - 复用原始 gate (不重新训练)
  - 验证数值等价性

- [ ] **3.4 端到端验证**
  - 完整流程验证 PPL
  - 与基线对比

### 关键设计决策

#### 选项 A (长期目标): 保持 Qwen3.5 的分组格式，但按位宽拆分为不同组
```
mlp.experts_bit2.gate_up_proj: (num_experts, 2*inter_size_bit2, H)
mlp.experts_bit3.gate_up_proj: (num_experts, 2*inter_size_bit3, H)
...
```

#### 选项 B (已完成): 转换为传统格式进行量化，然后再转换回来
(第二阶段更简单，但效率较低)

### 关键文件
- `qwen35_layer_reconstruct.py` - Qwen3.5 层重构
- `qwen35_hybrid_moe.py` - Qwen3.5 混合 MoE 包装器
- `run_qwen35.py` - 主量化脚本
- `grouped_gemm_moe_adapter.py` - 格式转换适配层

### 数学等价性保证 (Router Gate 设计)

**设计思路:**
原始路由:
```
gate_logits = gate(x)  # (batch, num_experts)
topk_weights, topk_indices = topk(gate_logits)
```

目标: 复用原始 gate (不重新训练)，保证路由一致性

**等价变换:**
对于每个 bit b:
1. 收集所有专家中，属于 bit b 的神经元索引
2. 对于 gate 来说，路由决策只关心 "选哪些专家"，不关心 "专家内部怎么分组"
3. 因此，可以完全复用原始 gate，只是在前向时对每个 bit 分别做 grouped_gemm

**前向传播等价性:**
```
原始:
  out = sum_{e in topk} weight_e * expert_e(x)

等价分解:
  out = sum_{e in topk} weight_e * [ sum_b expert_e^b(x) ]
      = sum_b [ sum_{e in topk} weight_e * expert_e^b(x) ]

其中 expert_e^b(x) 是 expert_e 中 bit b 的神经元贡献
```

因此，我们可以:
- 保持原始 gate 不变 (保证路由一致性)
- 按 bit 分组专家的神经元
- 前向时，先用原始 gate 路由，再分别对每个 bit 做 grouped_gemm


## 第四阶段: 真实量化推理 (RTL 内核)

**目标:** 在 RTX 5090 上使用 Qwen3.5 的分组格式实现真实量化推理。

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

### 关键文件:
- `qwen35_quant_kernels.py` - 量化内核包装器
- `qwen35_inference.py` - 量化推理引擎


## 文件夹结构

```
dartmoq-qwen3.5/
├── README.md
├── ROADMAP.md (本文件)
├── explore_qwen35.py          # 第一阶段: 架构探索
├── eval_qwen35.py             # 第一阶段: FP16 评估
├── qwen35_utils.py            # 第一阶段+: Qwen3.5 工具
│
├── qwen35_layer_reconstruct.py  # 第二阶段: 层重构
├── qwen35_hybrid_moe.py         # 第二阶段: 混合 MoE 包装器
├── run_qwen35.py                # 第二阶段: 主量化脚本
│
├── qwen35_quant_kernels.py    # 第三阶段: 量化内核
├── qwen35_inference.py        # 第三阶段: 量化推理
│
└── test/                      # 测试脚本
    ├── test_phase1.py
    ├── test_phase2.py
    └── test_phase3.py
```


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


## 依赖

与主 DartMoQ 项目相同:
- PyTorch
- Transformers
- (可选) 用于第三阶段自定义内核的 CUDA Toolkit
- (可选) 用于零样本评估的 lm_eval
