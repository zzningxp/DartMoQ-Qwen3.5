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

## 第三阶段 (路线调整 - SimpleMoEBlock vs Qwen35HybridMLP)

### 重要发现：SimpleMoEBlock 更优！(2026-07-09 更新)

在实际调试中，我们对比了两种架构：

| 维度 | SimpleMoEBlock (nn.ModuleList) | Qwen35HybridMLP (大张量) |
|------|-------------------------------|-------------------------|
| **速度** | ✅ 更快 | ❌ 更慢 |
| **内存** | ✅ 低，天然分散 | ❌ 高，易 OOM |
| **W4A16 存储** | ✅ 适合，无 padding | ⚠️ 有 padding，浪费空间 |
| **代码复杂度** | ✅ 简单直接 | ⚠️ 较复杂 |

**结论：直接使用 SimpleMoEBlock，不转回 grouped_gemm 格式！**

### SimpleMoEBlock 架构（当前采用）

**结构：**
```
SimpleMoEBlock
├── gate (router)
└── experts (nn.ModuleList)
    ├── DartMoQHybridWrapper[0]
    │   └── sub_experts (nn.ModuleList)
    │       ├── ExpertMLP (bit=2)  ← 独立 nn.Linear
    │       └── ExpertMLP (bit=4)  ← 独立 nn.Linear
    ├── DartMoQHybridWrapper[1]
    └── ...
```

**前向方式：** 逐个专家 mask 选择 + 独立 Linear forward

**优势：**
- ✅ 不会 OOM，内存天然分散
- ✅ 每个 Linear 是独立的，标准量化流程直接用
- ✅ 可以用现成的 LLM.int8, AWQ, GPTQ 等方案
- ✅ 实际测试速度反而更快！
- ✅ 无 padding，存储空间利用充分

**W4A16 存储方案：**
```python
# 每个专家独立存储，清晰简单
checkpoint = {
    'layer_0.mlp.experts.0.sub_experts.0.gate_proj.weight': pack_int4(w_int4),
    'layer_0.mlp.experts.0.sub_experts.0.gate_proj.scale': scale,
    'layer_0.mlp.experts.0.sub_experts.0.gate_proj.bit': 2,
    ...
}
```

### Qwen35HybridMLP 架构（已放弃）

**结构：**
```
Qwen35HybridMLP
├── gate (router)
└── experts (Qwen35HybridExperts)
    ├── bit_list: [1, 2, 4]
    ├── gate_up_proj_by_bit (nn.ParameterDict)
    │   ├── '1': Parameter (256, 2*max_n_1, 2048)  ← 大张量！
    │   ├── '2': Parameter (256, 2*max_n_2, 2048)  ← 所有专家的 bit=2 权重拼一起
    │   └── '4': Parameter (256, 2*max_n_4, 2048)
    ├── down_proj_by_bit (nn.ParameterDict)
    │   ├── '1': Parameter (256, 2048, max_n_1)
    │   ├── '2': Parameter (256, 2048, max_n_2)
    │   └── '4': Parameter (256, 2048, max_n_4)
    └── inter_size_by_bit: {1: n1, 2: n2, 4: n4}
```

**问题：**
- ❌ 易 OOM：需要一次性 gather 大张量
- ❌ 有 padding：为了对齐 max_n，空间浪费
- ❌ 速度慢：gather 开销大
- ❌ 不适合 W4A16 存储：需要额外元数据描述 padding

**当前状态：** 代码还在 (`qwen35_hybrid_moe.py`)，但默认不使用

### 调试优化 (已完成)

**问题：** 一开始就全 convert 所有层，调试时浪费时间。

**解决方案：**
- [x] **按需单层 convert**：只在要量化某一层时才 convert 那一层，而不是一开始全 convert
- [x] **部分层量化调试模式**：使用 `--quant_layers 0-5` 只量化前几层，后面直接 break，节省时间

**使用示例：**
```bash
python run_qwen35.py --quant_layers 0-5
```

### 测试结果：

| 模型类型 | 模型名称 | sli | q_scheme | rank | q_mode | q_layers | wiki | c4 | status | time | t_quant | t_ppl | err |
|---------|---------|-----|----------|------|--------|----------|------|----|--------|------|---------|-------|-----|
| Origin | Qwen3.5-35B-A3B | | | | fp16 | | 6.5807 | 9.6724 | ok | 320.21 | | 320.21 | |
| SimpleMoEBlock | Qwen3.5-35B-A3B | 4 | global-a8s8m2bpw | turboquant_innerproduct | turboquant | [0, 1, 2, 3, 4] | 6.6454 | 9.7606 | ok | 1526.86 | 1156.54 | 370.32 | |
| SimpleMoEBlock | Qwen3.5-35B-A3B | 4 | global-a8s8m2bpw | turboquant_innerproduct | turboquant | all | 7.6882 | 11.2645 | ok | 9533.79 | 9019.48 | 514.31 | |
| SimpleMoEBlock | Qwen3.5-35B-A3B | 4 | global-a8s8m2bpw | turboquant_innerproduct | turboquant | [0, 1, 2, 3, 4] | 6.6454 | 9.7606 | ok | 1485.75 | 1155 | 330.75 | |
| SimpleMoEBlock | Qwen3.5-35B-A3B | 4 | global-a8s8m2bpw | turboquant_innerproduct | turboquant | [0, 1, 2, 3, 4] | 6.6472 | 9.7624 | ok | 1496.25 | 1154.85 | 341.4 | |
| BitPartitionedGroupMoE | Qwen3.5-35B-A3B | 4 | global-a8s8m2bpw | turboquant_innerproduct | turboquant | | 7.6864 | 11.2656 | ok | 8654.59 | 8253.99 | 400.6 | |

TODO：当前目标应该是先将 all layer SimpleMoEBlock 的 t_ppl 时间优化到和 origin 相近。

### 当前流程

```
原始 grouped_gemm
  ↓ [convert_grouped_gemm_to_traditional]
传统格式
  ↓ [quantize] (神经元排序 + 分组)
SimpleMoEBlock + DartMoQHybridWrapper  ← 最终用这个！
  ↓ [直接验证] (不转回去)
PPL 评估
```

**不需要：** 重组回 Qwen35HybridMLP / grouped_gemm 格式

---

## 第四阶段 (W4A16 真实量化存储)

**目标：** 实现 W4A16 量化权重的紧凑存储，不反量化回 FP16，直接存 int4/int2。

**量化流程：**
```
量化阶段：w_fp16 → quantize → w_int4 (存下来)
推理阶段：w_int4 → dequantize → w_fp16 → x_fp16 @ w_fp16
```

### 方案：SimpleMoEBlock 独立存储（推荐）

**存储格式：**
```python
# 每个 Linear 独立存储，清晰简单
checkpoint = {
    # 量化权重（packed int4）
    'layer_0.mlp.experts.0.sub_experts.0.gate_proj.weight': pack_int4(w_int4),
    'layer_0.mlp.experts.0.sub_experts.0.up_proj.weight': pack_int4(w_int4),
    'layer_0.mlp.experts.0.sub_experts.0.down_proj.weight': pack_int4(w_int4),
    
    # 量化参数
    'layer_0.mlp.experts.0.sub_experts.0.gate_proj.scale': scale,
    'layer_0.mlp.experts.0.sub_experts.0.gate_proj.zero_point': zero_point,
    'layer_0.mlp.experts.0.sub_experts.0.gate_proj.bit': 2,
    
    # shared expert（8bit）
    'layer_0.mlp.shared_expert.gate_proj.weight': pack_int8(w_int8),
    'layer_0.mlp.shared_expert.gate_proj.scale': scale,
    ...
}
```

### 优势

| 特性 | SimpleMoEBlock | Qwen35HybridMLP |
|------|----------------|-----------------|
| **无 padding** | ✅ 每个专家刚好是自己的神经元数 | ❌ 为对齐 max_n 有 padding，浪费空间 |
| **存储简单** | ✅ 每个 Linear 独立存，标准格式 | ⚠️ 需要存额外元数据描述 padding |
| **兼容性** | ✅ 与现有量化 checkpoint 格式兼容 | ❌ 自定义格式，兼容性差 |
| **推理解包** | ✅ 简单直接 | ⚠️ 需要先切片再解包 |

### 任务清单
- [ ] 实现 int4/int2 打包/解包函数
- [ ] 在 `DartMoQHybridWrapper` 中保存量化参数
- [ ] 实现 W4A16 checkpoint 存储/加载
- [ ] 验证 ppl 不变（存储前 vs 加载后）
- [ ] 测试存储体积压缩比

### 关键文件
- `qwen35_layer_reconstruct.py` - 量化后权重已在这里！
- `dartmoq_hybridmoe.py` - DartMoQHybridWrapper 保存量化参数
- `simple_moe_block.py` - SimpleMoEBlock（已有）

---

## 关键设计决策

### 最终决定 (2026-07-09): SimpleMoEBlock 路线

**不转回 grouped_gemm 格式，直接使用 SimpleMoEBlock！**

原因：
1. ✅ SimpleMoEBlock 速度更快
2. ✅ 不会 OOM，内存天然分散
3. ✅ 更适合 W4A16 存储（无 padding）
4. ✅ 代码简单直接

### 选项对比总结

| 选项 | 架构 | 状态 |
|------|------|------|
| 选项 A | Qwen35HybridMLP (按 bit 分组大张量) | ❌ 已放弃 |
| 选项 B | SimpleMoEBlock (nn.ModuleList) | ✅ **当前采用** |

### 架构对比详情

**SimpleMoEBlock:**
```
组织方式：按专家组织 (ModuleList)
最小单元：单个专家的 nn.Linear
权重形状：(inter_size, hidden_size) 小矩阵
前向方式：逐个专家 mask 选择
```

**Qwen35HybridMLP:**
```
组织方式：按 bit 组织 (大张量)
最小单元：所有专家某 bit 的权重拼在一起
权重形状：(256, 2*inter_size, hidden_size) 大张量
前向方式：按 bit gather + batch matmul
```

### 为什么 SimpleMoEBlock 反而更快？

1. 避免了大张量 gather 开销
2. 内存局部性更好
3. PyTorch 对 Linear layer 优化非常好
4. 用 mask 只选择需要的专家，不做冗余计算

---

## 关键文件

- `qwen35_layer_reconstruct.py` - Qwen3.5 层重构（构建 SimpleMoEBlock）
- `qwen35_simple_wrapper.py` - 主量化流程（已跳过重组为 grouped_gemm）
- `dartmoq_hybridmoe.py` - 混合 MoE 包装器
- `grouped_gemm_moe_adapter.py` - 格式转换适配层（SimpleMoEBlock 定义在这里）
- `qwen35_utils.py` - Qwen3.5 专用工具

### 已放弃的文件
- `qwen35_hybrid_moe.py` - 按 bit 分组的 grouped_gemm 实现（不再使用）

### 第四阶段文件（待实现）
- `qwen35_quant_storage.py` - W4A16 量化存储
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
