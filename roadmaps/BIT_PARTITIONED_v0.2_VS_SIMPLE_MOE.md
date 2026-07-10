
# BitPartitionedGroupMoE vs SimpleMoEBlock 对比总结

## 1. 架构概览

| 维度 | SimpleMoEBlock | BitPartitionedGroupMoE |
|------|----------------|------------------------|
| **设计目标** | 临时结构，用于量化过程 | 最终结构，恢复原始 grouped_gemm 格式 |
| **存储格式** | 分散的 sub-experts | 统一的大张量 |

---

## 2. 详细结构对比

### SimpleMoEBlock 结构
```
SimpleMoEBlock
├── gate (router)
├── shared_expert (可选)
└── experts: ModuleList[DartMoQHybridWrapper]
    └── DartMoQHybridWrapper (256个)
        └── sub_experts: ModuleList[nn.Linear] (1-4个，不同bit)
            ├── gate_proj (nn.Linear, I_b × H)
            ├── up_proj (nn.Linear, I_b × H)
            └── down_proj (nn.Linear, H × I_b)
```

**特点：**
- 256个专家，每个专家是一个 `DartMoQHybridWrapper`
- 每个 wrapper 内部有 1-4 个 sub-experts（不同 bit 宽度）
- 每个 sub-expert 是标准的 `nn.Linear` 层
- 权重按 bit 分区存储在不同的 Linear 对象中

---

### BitPartitionedGroupMoE 结构
```
BitPartitionedGroupMoE
├── gate (router, 复用)
├── shared_expert (可选)
└── experts: Experts (内部类)
    ├── gate_up_proj: Parameter (E × 2I × H)  [完整张量]
    └── down_proj: Parameter (E × H × I)       [完整张量]
```

**特点：**
- 所有专家的所有权重拼接成两个大张量
- 完全恢复原始 Qwen3_5MoeSparseMoeBlock 的 grouped_gemm 格式
- 按原始神经元位置重新组织权重

---

## 3. Forward 计算流程对比

### SimpleMoEBlock forward
```python
for k in top_k:
    for e_idx in 256 experts:
        mask = expert_indices_k == e_idx
        if mask.any():
            token_x = x[mask]
            # 调用 DartMoQHybridWrapper.forward
            expert_out = self.experts[e_idx](token_x)
            # 内部再循环 sub-experts:
            #   for sub_expert in sub_experts:
            #       out += sub_expert(token_x)
            final_hidden_states[mask] += weight[mask] * expert_out
```

**层次：**
- L1: top_k × 256 循环
- L2: 每个 expert 内部再循环 sub-experts (1-4次)
- L3: 每个 sub-expert 内部调用 `nn.Linear.forward()` (优化的 matmul)

---

### BitPartitionedGroupMoE forward
```python
for k in top_k:
    for e_idx in 256 experts:
        mask = expert_indices_k == e_idx
        if mask.any():
            token_x = x[mask]
            # 直接用大张量做 matmul
            gate_up_out = token_x @ gate_up[e_idx].t()
            gate_out = gate_up_out[:, :I]
            up_out = gate_up_out[:, I:]
            act_out = F.silu(gate_out) * up_out
            down_out = act_out @ down[e_idx].t()
            final_hidden_states[mask] += down_out * token_w
```

**层次：**
- L1: top_k × 256 循环
- L2: 直接两次大矩阵乘法，无内部循环

---

## 4. 权重重组过程 (BitPartitionedGroupMoE from_simple_moe)

### 关键步骤
1. **获取元数据**：从 `layer_metadata` 获取 `expert_bit_indices`（每个 bit 对应的原始神经元索引）
2. **初始化大张量**：创建 `full_gate_up (E×2I×H)` 和 `full_down (E×H×I)`
3. **按位置回填**：
   ```python
   for expert_idx in 256:
       for bit in bit_list:
           indices = expert_bit_indices[expert_idx][bit]
           # gate 权重填到前 I 个位置
           full_gate_up[expert_idx, indices, :] = sub_expert.gate_proj.weight
           # up 权重填到后 I 个位置
           full_gate_up[expert_idx, intermediate_size + indices, :] = sub_expert.up_proj.weight
           # down 权重填到对应位置
           full_down[expert_idx, :, indices] = sub_expert.down_proj.weight
   ```

---

## 5. 关键区别总结

| 方面 | SimpleMoEBlock | BitPartitionedGroupMoE |
|------|----------------|------------------------|
| **存储方式** | 分散的 Linear 对象 | 统一的大张量 |
| **权重格式** | 按 bit 分区存储 | 按原始神经元位置重组 |
| **Forward 循环** | 两层循环 (expert + sub-expert) | 一层循环 (expert) |
| **计算方式** | 调用 nn.Linear.forward() | 直接 matmul |
| **架构演进** | 临时量化中间结构 | 最终目标结构 |

---

## 6. BitPartitionedGroupMoE 的优势

1. **架构清晰**：完全恢复原始 grouped_gemm 格式，为后续优化打下基础
2. **无额外开销**：避免了 sub-experts 层的循环
3. **可优化空间大**：可以用更接近原始的 grouped gemm 实现进一步加速
4. **便于理解**：和原始 Qwen3.5 结构一致，便于后续分析

---

## 7. 当前状态与下一步

**当前状态**：
- BitPartitionedGroupMoE v0.2 和 SimpleMoEBlock 性能持平
- PPL 几乎一致 (6.6454 vs 6.6472)，正确性验证通过
- 但比原始 Qwen3_5MoeSparseMoeBlock 还是慢约 1s

**下一步方向**：
- 继续优化 BitPartitionedGroupMoE 的 forward 实现
- 探索更接近原始 grouped gemm 的计算方式
- 目标：向原始 ~3s 的单层性能靠拢

