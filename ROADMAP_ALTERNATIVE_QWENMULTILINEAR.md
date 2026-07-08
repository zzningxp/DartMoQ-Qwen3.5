
# 备选方案：QwenMultiLinear 设计

## 概述
这是一个基于 ExLlamaV3 的 MultiLinear 思想的备选设计方案，作为当前 roadmap 的补充。
**当前 roadmap 保持不变，此方案仅作为讨论备选。**

## 核心思想
- **不转换为传统格式**，直接在 Qwen3.5 的 grouped_gemm 格式上工作
- **打包所有专家的指针**，实现一次 kernel launch（理想情况下，初期可以先用 PyTorch 实现）
- **保存元数据**（bit_to_indices），避免信息丢失

## 设计细节

### 1. 核心数据结构：QwenMultiLinear

```python
class QwenMultiLinear(nn.Module):
    """
    类似 ExLlamaV3 的 MultiLinear，但专门为 Qwen3.5 的 grouped_gemm 格式设计。
    
    把所有专家的 gate/up/down 打包在一起，便于后续实现 fused kernel。
    初期可以先用 PyTorch 的方式实现，后期再优化。
    """
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        bits_list: List[int],  # 使用的 bit 宽度列表，如 [2, 3, 4]
        gate_up_proj_by_bit: Optional[Dict[str, torch.Tensor]] = None,
        down_proj_by_bit: Optional[Dict[str, torch.Tensor]] = None,
        bit_to_indices: Optional[Dict[int, List[int]]] = None,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bits_list = bits_list
        
        # 按 bit 分组存储权重（保持 Qwen3.5 的 grouped_gemm 格式）
        # 格式：{bit: (num_experts, 2 * inter_size_by_bit, hidden_size)}
        self.gate_up_proj_by_bit = nn.ParameterDict()
        # 格式：{bit: (num_experts, hidden_size, inter_size_by_bit)}
        self.down_proj_by_bit = nn.ParameterDict()
        
        if gate_up_proj_by_bit is not None:
            for bit, weight in gate_up_proj_by_bit.items():
                self.gate_up_proj_by_bit[bit] = nn.Parameter(weight, requires_grad=False)
        
        if down_proj_by_bit is not None:
            for bit, weight in down_proj_by_bit.items():
                self.down_proj_by_bit[bit] = nn.Parameter(weight, requires_grad=False)
        
        # 关键：保存 bit 到原始神经元索引的映射
        # 这样量化后还能还原回去
        self.bit_to_indices = bit_to_indices or {}
    
    def get_gate(self, expert_idx: int, bit: int):
        """获取某个专家某个 bit 宽度的 gate_proj"""
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj_by_bit:
            return None
        # 从 fused 权重中切分
        inter_size = self.gate_up_proj_by_bit[bit_str].shape[1] // 2
        return self.gate_up_proj_by_bit[bit_str][expert_idx, :inter_size, :]
    
    def get_up(self, expert_idx: int, bit: int):
        """获取某个专家某个 bit 宽度的 up_proj"""
        bit_str = str(bit)
        if bit_str not in self.gate_up_proj_by_bit:
            return None
        # 从 fused 权重中切分
        inter_size = self.gate_up_proj_by_bit[bit_str].shape[1] // 2
        return self.gate_up_proj_by_bit[bit_str][expert_idx, inter_size:, :]
    
    def get_down(self, expert_idx: int, bit: int):
        """获取某个专家某个 bit 宽度的 down_proj"""
        bit_str = str(bit)
        if bit_str not in self.down_proj_by_bit:
            return None
        return self.down_proj_by_bit[bit_str][expert_idx, :, :]
    
    def forward_single_expert(self, x: torch.Tensor, expert_idx: int, bit: int):
        """
        单专家前向（初期用 PyTorch 实现，后期可以用 CUDA 核替换）
        """
        gate = self.get_gate(expert_idx, bit)
        up = self.get_up(expert_idx, bit)
        down = self.get_down(expert_idx, bit)
        
        if gate is None or up is None or down is None:
            return torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        
        # 标准 MoE 计算
        g = F.linear(x, gate)
        u = F.linear(x, up)
        a = F.silu(g) * u
        d = F.linear(a, down)
        return d
```

### 2. 改进版 HybridWrapper：QwenHybridMoE

```python
class QwenHybridMoE(nn.Module):
    """
    保持 Qwen3.5 grouped_gemm 格式的混合精度 MoE。
    
    关键改进：
    1. 不转换为传统格式，保持原始的 grouped_gemm 结构
    2. 保存 bit_to_indices 元数据，避免信息丢失
    3. 使用 QwenMultiLinear 打包专家权重
    """
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        bits_list: List[int],
        gate_up_proj_by_bit: Dict[str, torch.Tensor],
        down_proj_by_bit: Dict[str, torch.Tensor],
        bit_to_indices: Dict[int, Dict[int, List[int]]],  # layer_idx -&gt; {bit: [neuron_indices]}
        top_k: int = 6,
        shared_expert: Optional[nn.Module] = None,
        shared_expert_gate: Optional[nn.Module] = None,
        router: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.bits_list = bits_list
        self.top_k = top_k
        
        # 打包所有专家
        self.multi_gate_up = QwenMultiLinear(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_sizes_by_bit={bit: w.shape[1]//2 for bit, w in gate_up_proj_by_bit.items()},
            bits_list=bits_list,
            gate_up_proj_by_bit=gate_up_proj_by_bit,
            down_proj_by_bit=None,
            bit_to_indices=bit_to_indices.get(0, {}),  # 这里简化，实际每层需要
        )
        
        self.multi_down = QwenMultiLinear(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_sizes_by_bit={bit: w.shape[2] for bit, w in down_proj_by_bit.items()},
            bits_list=bits_list,
            gate_up_proj_by_bit=None,
            down_proj_by_bit=down_proj_by_bit,
            bit_to_indices=bit_to_indices.get(0, {}),
        )
        
        # Router（保持原始 gate，不重新训练）
        self.gate = router
        
        # Shared expert
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.shape
        hidden_states = x.reshape(-1, hidden_dim)
        
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # Shared expert 路径
        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_out = self.shared_expert(hidden_states)
            shared_out = shared_out * torch.sigmoid(self.shared_expert_gate(hidden_states))
            final_hidden_states += shared_out
        
        # Router
        router_logits = self.gate(hidden_states).softmax(dim=-1)
        topk_weights, topk_indices = router_logits.topk(self.top_k, dim=-1)
        
        # 路由到 experts（初期用 PyTorch 方式，后期优化）
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            weight = topk_weights[:, i].unsqueeze(-1)
            
            for e_idx in range(self.num_experts):
                mask = expert_idx == e_idx
                if mask.any():
                    expert_input = hidden_states[mask]
                    
                    # 对每个 bit 分别计算，然后累加
                    expert_out = 0
                    for bit in self.bits_list:
                        bit_out = self._forward_single_bit(expert_input, e_idx, bit)
                        expert_out += bit_out
                    
                    final_hidden_states[mask] += weight[mask] * expert_out
        
        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
    
    def _forward_single_bit(self, x: torch.Tensor, expert_idx: int, bit: int):
        """单个 bit 宽度的计算"""
        gate = self.multi_gate_up.get_gate(expert_idx, bit)
        up = self.multi_gate_up.get_up(expert_idx, bit)
        down = self.multi_down.get_down(expert_idx, bit)
        
        if gate is None or up is None or down is None:
            return torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        
        g = F.linear(x, gate)
        u = F.linear(x, up)
        a = F.silu(g) * u
        d = F.linear(a, down)
        return d
```

### 3. 元数据结构：QuantizationMetadata

```python
@dataclass
class QuantizationMetadata:
    """
    量化过程中保存的元数据，确保可以还原回 grouped_gemm 格式。
    """
    # 每层的 bit 到神经元索引的映射
    # layer_to_bit_indices[layer_idx][bit] = [neuron_indices]
    layer_to_bit_indices: Dict[int, Dict[int, List[int]]]
    
    # 每层各 bit 对应的 intermediate_size
    layer_to_inter_size_by_bit: Dict[int, Dict[int, int]]
    
    def save(self, path: str):
        """保存到文件"""
        torch.save({
            'layer_to_bit_indices': self.layer_to_bit_indices,
            'layer_to_inter_size_by_bit': self.layer_to_inter_size_by_bit,
        }, path)
    
    @classmethod
    def load(cls, path: str) -&gt; 'QuantizationMetadata':
        """从文件加载"""
        data = torch.load(path)
        return cls(
            layer_to_bit_indices=data['layer_to_bit_indices'],
            layer_to_inter_size_by_bit=data['layer_to_inter_size_by_bit'],
        )
```

### 4. 量化流程（修改版，不转换为传统格式）

```python
def quantize_moe_in_grouped_format(
    model,
    layer,
    layer_idx: int,
    inps: torch.Tensor,
    bits_list: List[int],
    quantization_metadata: QuantizationMetadata,
):
    """
    在 grouped_gemm 格式上直接量化（不转换为传统格式）
    
    当前流程的问题：
      grouped_gemm -&gt; 传统格式 -&gt; 量化 -&gt; 丢失信息 -&gt; ???
    
    新流程：
      grouped_gemm -&gt; 直接在 grouped_gemm 上做量化 -&gt; 按 bit 分组 -&gt; 保存元数据
    """
    
    # 1. 获取原始权重
    gate_up_proj = layer.mlp.experts.gate_up_proj
    down_proj = layer.mlp.experts.down_proj
    num_experts = gate_up_proj.shape[0]
    intermediate_size = gate_up_proj.shape[1] // 2
    hidden_size = gate_up_proj.shape[2]
    
    # 2. 分析每个专家的敏感度，决定每个神经元的 bit 宽度
    # （这里复用 DartMoQ 的分析逻辑，但保持权重格式不变）
    bit_to_indices_for_layer = analyze_and_assign_bits(
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
        inps=inps,
        bits_list=bits_list,
    )
    
    # 3. 保存元数据！（关键）
    quantization_metadata.layer_to_bit_indices[layer_idx] = bit_to_indices_for_layer
    
    # 4. 按 bit 分组，切分权重
    gate_up_proj_by_bit = {}
    down_proj_by_bit = {}
    
    for bit in bits_list:
        # 获取这个 bit 对应的所有神经元索引
        neuron_indices = bit_to_indices_for_layer.get(bit, [])
        if len(neuron_indices) == 0:
            continue
        
        # 从原始 fused 权重中切分出这个 bit 对应的部分
        # 保持 grouped_gemm 格式，只是按 bit 分开存储
        gate_up_proj_bit = torch.zeros(
            num_experts, 2 * len(neuron_indices), hidden_size,
            dtype=gate_up_proj.dtype,
            device=gate_up_proj.device
        )
        down_proj_bit = torch.zeros(
            num_experts, hidden_size, len(neuron_indices),
            dtype=down_proj.dtype,
            device=down_proj.device
        )
        
        for e_idx in range(num_experts):
            # gate: 取对应神经元
            gate_up_proj_bit[e_idx, :len(neuron_indices), :] = \
                gate_up_proj[e_idx, neuron_indices, :]
            # up: 取对应神经元（在 fused 权重的后半部分）
            gate_up_proj_bit[e_idx, len(neuron_indices):, :] = \
                gate_up_proj[e_idx, [i + intermediate_size for i in neuron_indices], :]
            # down: 取对应神经元
            down_proj_bit[e_idx, :, :] = down_proj[e_idx, :, neuron_indices]
        
        gate_up_proj_by_bit[str(bit)] = gate_up_proj_bit
        down_proj_by_bit[str(bit)] = down_proj_bit
        
        # 记录 intermediate_size
        if layer_idx not in quantization_metadata.layer_to_inter_size_by_bit:
            quantization_metadata.layer_to_inter_size_by_bit[layer_idx] = {}
        quantization_metadata.layer_to_inter_size_by_bit[layer_idx][bit] = len(neuron_indices)
    
    # 5. 实际量化（这里可以插入量化逻辑）
    # gate_up_proj_by_bit = quantize_weights(gate_up_proj_by_bit, ...)
    # down_proj_by_bit = quantize_weights(down_proj_by_bit, ...)
    
    # 6. 构建 QwenHybridMoE
    new_mlp = QwenHybridMoE(
        num_experts=num_experts,
        hidden_size=hidden_size,
        bits_list=bits_list,
        gate_up_proj_by_bit=gate_up_proj_by_bit,
        down_proj_by_bit=down_proj_by_bit,
        bit_to_indices=quantization_metadata.layer_to_bit_indices,
        top_k=getattr(layer.mlp, 'top_k', 6),
        shared_expert=getattr(layer.mlp, 'shared_expert', None),
        shared_expert_gate=getattr(layer.mlp, 'shared_expert_gate', None),
        router=layer.mlp.gate,
    )
    
    return new_mlp
```

## 与当前方案的对比

| 特性 | 当前方案（已完成选项 B） | 备选方案（QwenMultiLinear） |
|------|------------------------|---------------------------|
| 权重格式 | 转换为传统格式 | 保持 Qwen3.5 grouped_gemm 格式 |
| 元数据保存 | ❌ 丢失 bit_to_indices | ✅ 保存完整元数据 |
| 专家打包 | ❌ 独立专家 | ✅ QwenMultiLinear 打包 |
| 量化后还原 | ❌ 无法还原 | ✅ 可以还原（有元数据） |
| 侵入性 | 改动较小 | 改动较大（需要重写量化流程） |

## 实施路径（如果选择此方案）

### Phase 3.1: 验证当前量化正确性（先完成！）
**（与当前 roadmap 相同）**
- 通过 FP16 反量化评估
- 记录 PPL 基线

### Phase 3.2: 实现元数据保存（可选，不改动现有流程）
- 在 `DartMoQHybridWrapper` 中添加 `bit_to_indices` 参数
- 保存 `QuantizationMetadata` 到文件
- **这一步可以在当前方案基础上叠加，为未来切换到备选方案做准备**

### Phase 3.3: 实现 QwenMultiLinear（可选，新模块）
- 实现 `QwenMultiLinear` 类
- 实现 `QwenHybridMoE` 类
- 在独立文件中测试，不影响现有流程

### Phase 3.4: 端到端测试（可选）
- 完整流程测试 PPL
- 与基线对比

## 总结

**此方案的优势：**
1. 不丢失元数据，量化后理论上可以还原回 grouped_gemm 格式
2. 为未来的 fused kernel 优化铺路
3. 保持与 Qwen3.5 原始格式的兼容性

**此方案的风险：**
1. 需要重写部分量化流程
2. 初期可能比当前方案慢（没有 fused kernel）
3. 工作量较大

**建议：**
- 先完成当前 roadmap 的 Phase 3.1
- 同时可以并行实现 Phase 3.2（元数据保存）
- 等有时间再深入研究 Phase 3.3-3.4
