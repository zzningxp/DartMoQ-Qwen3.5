# 第三阶段详细设计：回归 grouped_gemm 格式

## 概述

本阶段的目标是把第二阶段量化后的传统格式模型，重新组织回 Qwen3.5 原始的 grouped_gemm 格式，但这次是按 bit 精度分组存储。

**触发条件：** 当 `args.true_quant=True` 时启用本阶段流程。

**关键原则：**
- 数学等价性：保证与第二阶段（反量化到 FP16）的数值结果一致
- 路由一致性：完全复用原始 gate，不重新训练
- 按 bit 分组：相同 bit 宽度的神经元放在一起，便于后续优化

---

## 流程设计

### 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│  0. 原始模型 (grouped_gemm 格式)                                │
│     mlp.experts.gate_up_proj: (E, 2*I, H)                       │
│     mlp.experts.down_proj: (E, H, I)                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  1. 转换为传统格式 (第二阶段已完成)                              │
│     convert_grouped_gemm_to_traditional()                        │
│     → TraditionalMoEWrapper (experts: ModuleList)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. 量化 + 神经元排序 + 分组 (第二阶段已完成)                    │
│     reconstruct_moe_from_existing()                              │
│     → SimpleMoEBlock + DartMoQHybridWrapper                     │
│     (保存元数据：bit_to_indices, expert_bit_indices)             │ ← 新增！
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. 重组回 grouped_gemm 格式 (第三阶段新增)                      │
│     restructure_to_grouped_gemm()                                │
│     → Qwen35HybridMLP                                            │
│       ├ gate (复用原始)                                          │
│       ├ experts (Qwen35HybridExperts)                           │
│       │  ├ gate_up_proj_by_bit: {2: (E,2*I2,H), 3: ...}         │
│       │  └ down_proj_by_bit: {2: (E,H,I2), 3: ...}               │
│       ├ shared_expert (可选)                                     │
│       └ shared_expert_gate (可选)                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. 前向传播 (第三阶段新增)                                      │
│     Qwen35HybridMLP.forward()                                    │
│     → 对每个 bit 分别计算，然后累加                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 数据结构设计

### 1. 元数据结构

在量化过程中保存的关键信息，用于后续重组回 grouped_gemm 格式。

```python
@dataclass
class LayerMetadata:
    """每层的量化元数据"""
    layer_idx: int
    
    # expert_bit_indices[e][b] = 专家 e 中 bit b 的神经元索引列表
    expert_bit_indices: List[Dict[int, List[int]]]
    
    # expert_groups[e] = 专家 e 的分组信息（从原始 expert_groups 来）
    expert_groups: List[List[List[int]]]
    
    # orig_bit_config[e] = 专家 e 的原始位宽配置
    orig_bit_config: List[List[int]]
    
    # 维度信息
    num_experts: int
    hidden_size: int
    intermediate_size: int
    
    # 所有用到的 bit 列表，例如 [2, 3, 4]
    bit_list: List[int]
```

### 2. Qwen35HybridExperts - 按 bit 分组的专家权重

保持 Qwen3.5 原始的 grouped_gemm 格式，但按 bit 精度分别存储。

```python
class Qwen35HybridExperts(nn.Module):
    """
    按 bit 分组的专家权重，保持 grouped_gemm 格式
    
    结构：
      - gate_up_proj_by_bit[bit]: (num_experts, 2*inter_size_bit, hidden_size)
      - down_proj_by_bit[bit]: (num_experts, hidden_size, inter_size_bit)
    """
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        bit_list: List[int],
        gate_up_proj_by_bit: Optional[Dict[str, torch.Tensor]] = None,
        down_proj_by_bit: Optional[Dict[str, torch.Tensor]] = None,
        inter_size_by_bit: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.bit_list = bit_list
        self.inter_size_by_bit = inter_size_by_bit or {}
        
        # 按 bit 存储权重，保持 grouped_gemm 格式
        self.gate_up_proj_by_bit = nn.ParameterDict()
        self.down_proj_by_bit = nn.ParameterDict()
        
        if gate_up_proj_by_bit is not None:
            for bit_str, weight in gate_up_proj_by_bit.items():
                self.gate_up_proj_by_bit[bit_str] = nn.Parameter(weight, requires_grad=False)
        
        if down_proj_by_bit is not None:
            for bit_str, weight in down_proj_by_bit.items():
                self.down_proj_by_bit[bit_str] = nn.Parameter(weight, requires_grad=False)
    
    def get_gate(self, expert_idx: int, bit: int) -> Optional[torch.Tensor]:
        """获取某个专家某个 bit 宽度的 gate_proj 权重"""
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj_by_bit:
            return None
        inter_size = self.inter_size_by_bit.get(bit, 0)
        if inter_size == 0:
            return None
        return self.gate_up_proj_by_bit[bit_str][expert_idx, :inter_size, :]
    
    def get_up(self, expert_idx: int, bit: int) -> Optional[torch.Tensor]:
        """获取某个专家某个 bit 宽度的 up_proj 权重"""
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj_by_bit:
            return None
        inter_size = self.inter_size_by_bit.get(bit, 0)
        if inter_size == 0:
            return None
        return self.gate_up_proj_by_bit[bit_str][expert_idx, inter_size:, :]
    
    def get_down(self, expert_idx: int, bit: int) -> Optional[torch.Tensor]:
        """获取某个专家某个 bit 宽度的 down_proj 权重"""
        bit_str = str(bit)
        if bit_str not in self.down_proj_by_bit:
            return None
        return self.down_proj_by_bit[bit_str][expert_idx, :, :]
```

### 3. Qwen35HybridMLP - 完整的 MoE 层

整合所有组件，实现前向传播。

```python
class Qwen35HybridMLP(nn.Module):
    """
    按 bit 分组的 Qwen3.5 MoE 层
    
    关键特性：
    - 复用原始 gate（保持路由一致性）
    - 按 bit 分组存储权重（grouped_gemm 格式）
    - 前向时对每个 bit 分别计算，然后累加
    """
    def __init__(
        self,
        gate: nn.Module,
        experts: Qwen35HybridExperts,
        shared_expert: Optional[nn.Module] = None,
        shared_expert_gate: Optional[nn.Module] = None,
        top_k: int = 6,
    ):
        super().__init__()
        
        self.gate = gate
        self.experts = experts
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.top_k = top_k
        
        # 复制一些方便访问的属性
        self.num_experts = experts.num_experts
        self.hidden_size = experts.hidden_size
        self.bit_list = experts.bit_list
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        数学等价性保证：
          out = sum_{e in topk} weight_e * expert_e(x)
              = sum_{e in topk} weight_e * [ sum_b expert_e^b(x) ]
              = sum_b [ sum_{e in topk} weight_e * expert_e^b(x) ]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)
        
        final_hidden_states = torch.zeros_like(x)
        
        # Shared expert 路径
        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_out = self.shared_expert(x)
            shared_gate_val = torch.sigmoid(self.shared_expert_gate(x))
            final_hidden_states = final_hidden_states + shared_out * shared_gate_val
        
        # Router
        gate_output = self.gate(x)
        if isinstance(gate_output, tuple):
            # Qwen3.5 风格: (logits, topk_weights, topk_indices)
            _, topk_weights, topk_indices = gate_output
        else:
            # 传统风格: logits
            router_logits = gate_output.softmax(dim=-1)
            topk_weights, topk_indices = router_logits.topk(self.top_k, dim=-1)
        
        # 对每个 bit 分别计算
        for bit in self.bit_list:
            bit_out = self._forward_bit(x, bit, topk_indices, topk_weights)
            final_hidden_states = final_hidden_states + bit_out
        
        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
    
    def _forward_bit(
        self,
        x: torch.Tensor,
        bit: int,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """单个 bit 宽度的前向计算"""
        bit_str = str(bit)
        if bit_str not in self.experts.gate_up_proj_by_bit:
            return torch.zeros_like(x)
        
        batch_size = x.shape[0]
        out = torch.zeros_like(x)
        
        # 对每个 top-k 专家
        for i in range(self.top_k):
            expert_idx_flat = topk_indices[:, i]
            weight_flat = topk_weights[:, i].unsqueeze(-1)
            
            # 对每个 unique 的专家索引，批量计算
            unique_experts, inverse_indices = expert_idx_flat.unique(return_inverse=True)
            
            for e_idx in unique_experts:
                mask = expert_idx_flat == e_idx
                if not mask.any():
                    continue
                
                x_e = x[mask]
                
                # 获取该专家该 bit 的权重
                gate_w = self.experts.get_gate(e_idx, bit)
                up_w = self.experts.get_up(e_idx, bit)
                down_w = self.experts.get_down(e_idx, bit)
                
                if gate_w is None or up_w is None or down_w is None:
                    continue
                
                # 计算：silu(x @ gate_w.T) * (x @ up_w.T) @ down_w.T
                gate_out = F.linear(x_e, gate_w)
                up_out = F.linear(x_e, up_w)
                act_out = F.silu(gate_out) * up_out
                down_out = F.linear(act_out, down_w)
                
                out[mask] = out[mask] + weight_flat[mask] * down_out
        
        return out
```

---

## 核心函数设计

### 1. reconstruct_moe_from_existing (修改版)

**文件：** `qwen35_layer_reconstruct.py`

**修改内容：**
- 保存 `bit_to_indices` 和 `expert_bit_indices`
- 返回 `(moe, layer_metadata)` 而不只是 `moe`

**关键代码片段：**
```python
# 在构建完每个专家后，保存元数据
layer_expert_bit_indices = []  # 每个专家的 bit 到神经元索引的映射

for expert_idx, expert in enumerate(layer.mlp.experts):
    # ... (原有代码)
    
    # 保存 bit 到神经元索引的映射
    bit_to_indices = {}
    for bit, group_indices in zip(orig_bit_config[expert_idx], expert_groups[expert_idx]):
        if bit not in bit_to_indices:
            bit_to_indices[bit] = []
        bit_to_indices[bit].extend(group_indices)
    
    layer_expert_bit_indices.append(bit_to_indices)
    
    # ... (原有代码)

# 构建返回的元数据
layer_metadata = LayerMetadata(
    layer_idx=layer_idx,
    expert_bit_indices=layer_expert_bit_indices,
    expert_groups=all_expert_groups,
    orig_bit_config=qscheme['slice_expert'],
    num_experts=ori_expert_num,
    hidden_size=layer.mlp.hidden_size if hasattr(layer.mlp, 'hidden_size') else model.config.hidden_size,
    intermediate_size=layer.mlp.intermediate_size if hasattr(layer.mlp, 'intermediate_size') else model.config.intermediate_size,
    bit_list=sorted(list(set([b for e_bit in layer_expert_bit_indices for b in e_bit.keys()]))),
)

return moe, layer_metadata
```

### 2. DartMoQHybridWrapper (扩展版)

**文件：** `dartmoq_hybridmoe.py`

**修改内容：**
- 添加 `bit_to_indices` 和 `expert_bit_indices` 属性
- 初始化时保存这些元数据

**关键代码片段：**
```python
class DartMoQHybridWrapper(nn.Module):
    def __init__(
        self,
        sub_experts,
        bit_to_indices=None,
        expert_bit_indices=None,
    ):
        super().__init__()
        self.sub_experts = nn.ModuleList(sub_experts)
        self.bit_to_indices = bit_to_indices  # 按 bit 的神经元索引映射
        self.expert_bit_indices = expert_bit_indices  # 每个子专家对应的神经元索引
```

### 3. restructure_to_grouped_gemm (新增)

**文件：** `qwen35_hybrid_moe.py`

**功能：** 从量化后的传统格式重组回 grouped_gemm 格式

**关键逻辑：**
1. 从 `SimpleMoEBlock` + `DartMoQHybridWrapper` 中提取权重
2. 利用 `bit_to_indices` 了解每个 bit 对应哪些神经元
3. 按 bit 分组，重新组织成 `(E, 2*I_b, H)` 和 `(E, H, I_b)` 格式

**关键代码片段：**
```python
def restructure_to_grouped_gemm(
    moe: SimpleMoEBlock,
    layer_metadata: LayerMetadata,
    device: Optional[torch.device] = None,
) -> Qwen35HybridMLP:
    """
    从量化后的传统格式重组回 grouped_gemm 格式
    
    参数：
        moe: 量化后的 SimpleMoEBlock
        layer_metadata: 量化过程中保存的元数据
        device: 目标设备
    
    返回：
        Qwen35HybridMLP: 按 bit 分组的 grouped_gemm 格式 MoE 层
    """
    if device is None:
        device = next(moe.parameters()).device
    
    num_experts = layer_metadata.num_experts
    hidden_size = layer_metadata.hidden_size
    bit_list = layer_metadata.bit_list
    
    # 收集每个 bit 的权重
    gate_up_proj_by_bit = {}
    down_proj_by_bit = {}
    inter_size_by_bit = {}
    
    for bit in bit_list:
        # 计算这个 bit 总共的神经元数
        total_neurons = 0
        for e_idx in range(num_experts):
            bit_indices = layer_metadata.expert_bit_indices[e_idx].get(bit, [])
            total_neurons = max(total_neurons, len(bit_indices))
        
        if total_neurons == 0:
            continue
        
        # 初始化张量
        gate_up = torch.zeros(
            num_experts, 2 * total_neurons, hidden_size,
            dtype=torch.bfloat16, device=device
        )
        down = torch.zeros(
            num_experts, hidden_size, total_neurons,
            dtype=torch.bfloat16, device=device
        )
        
        # 从每个专家的子专家中提取权重
        for e_idx in range(num_experts):
            wrapper = moe.experts[e_idx]  # DartMoQHybridWrapper
            
            # 找到这个专家该 bit 对应的子专家
            for sub_idx, sub_expert in enumerate(wrapper.sub_experts):
                if sub_expert._quant_bit != bit:
                    continue
                
                # 提取权重
                inter_size_b = sub_expert.gate_proj.weight.shape[0]
                
                gate_up[e_idx, :inter_size_b, :] = sub_expert.gate_proj.weight.data
                gate_up[e_idx, inter_size_b:2*inter_size_b, :] = sub_expert.up_proj.weight.data
                down[e_idx, :, :inter_size_b] = sub_expert.down_proj.weight.data
        
        gate_up_proj_by_bit[str(bit)] = gate_up
        down_proj_by_bit[str(bit)] = down
        inter_size_by_bit[bit] = total_neurons
    
    # 构建 Qwen35HybridExperts
    experts = Qwen35HybridExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        bit_list=bit_list,
        gate_up_proj_by_bit=gate_up_proj_by_bit,
        down_proj_by_bit=down_proj_by_bit,
        inter_size_by_bit=inter_size_by_bit,
    )
    
    # 构建 Qwen35HybridMLP
    hybrid_mlp = Qwen35HybridMLP(
        gate=moe.gate,
        experts=experts,
        shared_expert=getattr(moe, 'shared_expert', None),
        shared_expert_gate=getattr(moe, 'shared_expert_gate', None),
        top_k=getattr(moe, 'top_k', 6),
    )
    
    return hybrid_mlp
```

---

## 集成到主流程

### 修改 qwen35_simple_wrapper.py

在 `dartmoq_quant_grouped_gemm_moe` 函数中，当 `args.true_quant=True` 时：

```python
for layer_idx, layer in enumerate(layers):
    # ... (原有代码)
    
    if should_quantize:
        # 量化，返回 moe 和元数据
        moe, layer_metadata = reconstruct_moe_from_existing(...)
        layer.mlp = moe
        
        # 如果启用 true_quant，重组回 grouped_gemm 格式
        if args.true_quant:
            print(f"  Restructuring to grouped_gemm format (layer {layer_idx})")
            layer.mlp = restructure_to_grouped_gemm(moe, layer_metadata)
    else:
        # 不量化，直接使用原始层
        pass
    
    # ... (原有代码)
```

---

## 验证计划

### 数值正确性验证

1. **单 layer 验证：**
   - 对同一个 layer，分别用第二阶段和第三阶段计算前向
   - 比较输出的最大差值，应在 1e-5 以内

2. **端到端 PPL 验证：**
   - 在 wikitext2 和 c4 上分别计算 PPL
   - 第二阶段和第三阶段的 PPL 差值应在 1% 以内

### 内存/性能验证

1. **内存占用：**
   - 比较第二阶段和第三阶段的内存占用
   - 第三阶段应更优（权重格式更紧凑）

2. **推理速度：**
   - 比较推理速度（初期可能相近，后续优化后应更快）

---

## 文件清单

### 修改的文件
1. `qwen35_layer_reconstruct.py` - 扩展支持元数据保存
2. `dartmoq_hybridmoe.py` - 扩展 `DartMoQHybridWrapper`
3. `qwen35_simple_wrapper.py` - 集成第三阶段流程
4. `run_qwen35.py` - `--true_quant` 参数已存在，无需修改

### 新增的文件
1. `qwen35_hybrid_moe.py` - `Qwen35HybridExperts`、`Qwen35HybridMLP`、`restructure_to_grouped_gemm`
