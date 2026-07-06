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

### 任务:
- [ ] 为 Qwen3.5 的合并权重格式适配专家敏感度分析
- [ ] 在 Qwen3.5 结构中实现神经元排序和分组
- [ ] 创建适配 Qwen3.5 架构的混合 MoE 包装器
- [ ] 适配动态规划位宽分配
- [ ] 通过 FP16 反量化评估验证数值正确性

### 关键设计决策:
- **选项 A:** 保持 Qwen3.5 的分组格式，但按位宽拆分为不同组
  ```
  mlp.experts_bit2.gate_up_proj: (num_experts_bit2, 2 * inter_size_bit2, H)
  mlp.experts_bit3.gate_up_proj: (num_experts_bit3, 2 * inter_size_bit3, H)
  ...
  ```

- **选项 B:** 转换为传统格式进行量化，然后再转换回来
  (第二阶段更简单，但效率较低)

### 关键文件:
- `qwen35_layer_reconstruct.py` - Qwen3.5 层重构
- `qwen35_hybrid_moe.py` - Qwen3.5 混合 MoE 包装器
- `run_qwen35.py` - 主量化脚本


## 第三阶段: 真实量化推理 (RTL 内核)

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


## 成功指标

### 第一阶段
- [x] 可以加载 Qwen3.5 MoE 模型
- [x] 可以运行顺序（CPU 待机）评估
- [ ] PPL 匹配 HuggingFace 基线

### 第二阶段
- [ ] 可以应用 DartMoQ 量化
- [ ] PPL 在基线可接受范围内
- [ ] 保持混合 MoE 结构

### 第三阶段
- [ ] 可以在 RTX 5090 上运行量化推理
- [ ] 相比第二阶段有显著加速
- [ ] PPL 保持可接受


## 依赖

与主 DartMoQ 项目相同:
- PyTorch
- Transformers
- (可选) 用于第三阶段自定义内核的 CUDA Toolkit
- (可选) 用于零样本评估的 lm_eval
