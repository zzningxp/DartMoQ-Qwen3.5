# Qwen3.5 MoE 适配差异文档

本文档整理了将 Qwen3.5 MoE 模型适配到 DartMoQ 量化框架时，相比原有模型（DeepSeek、Mixtral 等）需要处理的主要差异点。

---

## 1. 模型加载架构差异

### 1.1 多模态包装
- **原有模型**：纯语言模型，直接加载 `AutoModelForCausalLM`
- **Qwen3.5 MoE**：外层是多模态模型 `Qwen3_5MoeForConditionalGeneration`，需要提取 `model.language_model` 获取纯文本部分

### 1.2 模型访问路径
```python
# 原有模型
model = AutoModelForCausalLM.from_pretrained(...)

# Qwen3.5 MoE
model = AutoModelForConditionalGeneration.from_pretrained(...)
if hasattr(model, 'language_model'):
    model = model.language_model
```

---

## 2. 专家存储结构差异

### 2.1 权重组织方式
- **原有模型**：每个专家是独立的 Module
  ```python
  layer.mlp.experts = nn.ModuleList([
      ExpertMLP(),  # 每个有自己的 gate_proj, up_proj, down_proj
      ExpertMLP(),
      ...
  ])
  ```

- **Qwen3.5 MoE**：所有专家权重合并存储在大张量中
  ```python
  class Qwen3_5MoeExperts(nn.Module):
      def __init__(self, config):
          self.gate_up_proj = nn.Parameter(  # (num_experts, 2 * intermediate_size, hidden_size)
              torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size)
          )
          self.down_proj = nn.Parameter(  # (num_experts, hidden_size, intermediate_size)
              torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
          )
  ```

### 2.2 专家访问方式
- **原有模型**：直接索引 `layer.mlp.experts[expert_idx]`
- **Qwen3.5 MoE**：需要从合并张量中切片，或使用封装对象

---

## 3. 注意力机制差异

### 3.1 混合注意力类型
- **原有模型**：每层都使用相同的自注意力机制
- **Qwen3.5 MoE**：交替使用两种注意力
  - `linear_attention`：基于线性注意力的 `Qwen3_5MoeGatedDeltaNet`
  - `full_attention`：标准的多头自注意力

### 3.2 配置驱动的层类型
```python
# 通过 config.layer_types 列表控制
config.layer_types = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",  # 每4层一个full attention
    ...
]
```

### 3.3 代码中需要处理的判断
```python
if hasattr(layer, 'self_attn'):
    # 使用 self_attn 前向
elif hasattr(layer, 'linear_attn'):
    # 使用 linear_attn 前向
```

---

## 4. MoE 层结构差异

### 4.1 共享专家设计
- **原有模型**：通常只有路由专家
- **Qwen3.5 MoE**：同时包含
  ```python
  class Qwen3_5MoeSparseMoeBlock(nn.Module):
      def __init__(self, config):
          self.gate = Qwen3_5MoeTopKRouter(config)
          self.experts = Qwen3_5MoeExperts(config)
          self.shared_expert = Qwen3_5MoeMLP(...)  # 共享专家
          self.shared_expert_gate = torch.nn.Linear(...)  # 共享专家门控
  ```

### 4.2 前向传播流程
```python
def forward(self, hidden_states):
    shared_output = self.shared_expert(hidden_states)
    shared_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output

    _, routing_weights, selected_experts = self.gate(hidden_states)
    expert_output = self.experts(hidden_states, selected_experts, routing_weights)

    return expert_output + shared_output  # 两部分相加
```

---

## 5. 门控输出格式差异

### 5.1 Router 返回值
- **原有模型**：通常只返回 logits，需要手动计算 topk
- **Qwen3.5 MoE**：返回三值 tuple
  ```python
  def forward(self, hidden_states):
      # ...
      return router_logits, router_top_value, router_indices
  ```

### 5.2 激活率分析时的判断
```python
def analyze_experts_activation(layer, ...):
    gate_output = layer.mlp.gate(inps)

    if isinstance(gate_output, tuple):
        # Qwen3.5 风格：gate返回tuple
        top_indices = gate_output[0]
    else:
        # 传统风格：需要自己计算topk
        router_logits = F.softmax(gate_output, dim=-1)
        _, top_indices = torch.topk(router_logits, K, dim=-1)
```

---

## 6. 位置编码差异

### 6.1 3D 位置编码
- **原有模型**：2D 位置编码 `(batch_size, sequence_length)`
- **Qwen3.5 MoE**：3D 位置编码 `(3, batch_size, sequence_length)`，包含
  - text 维度
  - height 维度
  - width 维度

### 6.2 Rotary Embedding 特殊处理
```python
# Qwen3.5MoeTextRotaryEmbedding 接受3D的position_ids
def forward(self, x, position_ids):
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    # ...
```

---

## 7. 专家数量获取差异

### 7.1 配置获取优先级
```python
# 原有模型
if hasattr(model.config, 'n_routed_experts'):
    ori_expert_num = model.config.n_routed_experts

# Qwen3.5 MoE
if hasattr(model.config, 'num_experts'):
    ori_expert_num = model.config.num_experts
```

### 7.2 备选方案
当配置中无法获取时：
```python
# 尝试 len(experts)
# 尝试从 named_children 计数
# 检查 experts.gate_up_proj.shape[0]
```

---

## 8. 量化时的模块匹配差异

### 8.1 路径匹配
- **原有模型**：`mlp.experts.0.up_proj`
- **Qwen3.5 MoE**：可能有不同的命名模式
  - 需要支持多种模式匹配
  - 通过正则或索引识别

### 8.2 共享专家处理
- 需要识别 `shared_expert` 的路径
- 共享专家通常使用与 attention 相同的量化位宽

---

## 9. 重构 MoE 时的处理差异

### 9.1 专家权重提取
```python
# 对于 Qwen3.5 MoE，需要从合并张量中提取单个专家
expert_gate_up = experts.gate_up_proj[expert_idx]
intermediate_size = expert_gate_up.shape[0] // 2

gate_proj_weight = expert_gate_up[:intermediate_size]
up_proj_weight = expert_gate_up[intermediate_size:]
down_proj_weight = experts.down_proj[expert_idx]
```

### 9.2 专家封装
可以创建包装类提供统一接口：
```python
class Qwen35ExpertWrapper(nn.Module):
    def __init__(self, gate_up_proj, down_proj, expert_idx, ...):
        self.gate_proj = nn.Linear(...)
        self.up_proj = nn.Linear(...)
        self.down_proj = nn.Linear(...)
        # 从合并张量中复制权重
```

---

## 10. 总结

| 方面 | 原有模型 | Qwen3.5 MoE |
|------|---------|------------|
| 模型包装 | 纯语言模型 | 多模态外层，需提取 language_model |
| 专家存储 | 独立ModuleList | 合并张量存储 |
| 注意力 | 统一self_attn | 混合linear_attn和full_attn |
| 共享专家 | 无 | 有，带门控 |
| Router输出 | logits | (logits, top_value, top_indices) |
| 位置编码 | 2D | 3D |
| 专家访问 | 直接索引 | 需切片或封装 |

---

## 11. 主要适配策略

1. **统一的专家访问接口**：封装不同的专家获取方式
2. **类型判断**：通过检查属性存在性来选择代码路径
3. **配置优先**：优先从 config 获取信息，而不是依赖模块结构
4. **容错设计**：多种方式尝试，提高健壮性
5. **注意力分支处理**：分别处理 self_attn 和 linear_attn 两种情况
