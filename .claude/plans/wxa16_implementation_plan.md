# WxA16 架构实现计划

## 概述
WxA16 是一种真实量化存储架构，区别于当前的 fake quant 模式。核心改动是：
1. 量化后不立即反量化，而是存储 packed 权重、scale、codebook 等
2. 删除原始 fp16 权重
3. 推理时再现场反量化

## 当前流程 vs 新流程

### 当前流程
```
每一层:
  1. Attention + Shared Expert 前向
  2. Router Expert 计算 loss -> 排序 -> Scheme
  3. Fake Quant: 量化 -> 立即反量化 -> 存 fp16
所有层完成后:
  4. PPL Test 推理（直接用 fp16）
```

### 新流程 (WxA16)
```
每一层:
  1. Attention + Shared Expert 量化为 int8 (W8A16)
  2. Router Expert: 计算 loss -> 排序 -> Scheme
  3. Real Quant: 量化 -> 存 packed 格式 -> 删除 fp16
  4. 立即基于量化结果前向推理
所有层完成后:
  5. PPL Test 推理（现场反量化）
```

## 文件改动清单

### 1. `turboquant_utils/quantize.py` - 增强 packed 量化
**新增功能**：
- `pack_nbit()`: 通用 n-bit 打包函数 (1/2/4/8 bit)
- `unpack_nbit()`: 通用 n-bit 解包函数
- `turboquant_quantize_packed_full()`: 完整的 packed 量化，支持 1/2/4/8 bit

### 2. `wxa16_linear.py` - 新模块
**新增文件**，包含：
- `WxA16Linear`: 替换 nn.Linear，存储 packed 权重
  - `__init__()`: 初始化，接受 packed 数据
  - `forward()`: 反量化 + 推理
  - `from_linear()`: 从 nn.Linear 量化转换

### 3. `wxa16_bit_partitioned_moe.py` - 新模块
**新增文件**，包含：
- `WxA16BitPartitionedGroupMoE`: 替换 `BitPartitionedGroupMoE`
  - 存储格式: 每个 bit 有 packed gate_up/down 权重
  - 高效前向推理

### 4. `wxa16_dartmoq_backend.py` - 新后端
**新增文件**，包含：
- `wxa16_quantize_linear()`: 替换 `turbo_fake_quant_linear`
  - 返回 `WxA16Linear` 而非修改原位

### 5. `dartmoq_utils.py` - 修改量化函数
**修改** `quant_layer_mix_precision()`:
- 不再调用 fake quant，而是调用 wxa16 真实量化
- 把 nn.Linear 替换为 WxA16Linear

### 6. `qwen35_simple_wrapper.py` - 修改主流程
**修改** `construct_moe()`:
- 量化 Attention 为 W8A16
- Shared Expert 为 W8A16 (或其他 bit)
- MoE 量化后转为 `WxA16BitPartitionedGroupMoE`
- 量化后立即前向推理

### 7. `wxa16_eval.py` - 新增评估辅助
**新增文件**：
- `wxa16_ppl_eval_sequential()`: WxA16 专用的顺序评估

## 核心数据结构设计

### WxA16Linear 存储结构
```python
class WxA16Linear(nn.Module):
    # 量化参数
    bit_width: int
    group_size: int
    
    # TurboQuant 专用
    codebook: torch.Tensor  # (2^bit,)
    rotation_seed: int
    
    # 打包的权重索引
    packed_indices: torch.Tensor  # (out_features, packed_in_features)
    
    # 缩放和归一化参数
    norms: torch.Tensor  # (out_features, n_groups)
    scales: torch.Tensor  # (out_features, n_groups) - 可选
    
    # 原始维度信息
    in_features: int
    out_features: int
    orig_dtype: torch.dtype
```

### WxA16BitPartitionedGroupMoE 存储结构
```python
class WxA16BitPartitionedGroupMoE(nn.Module):
    # Router
    gate: nn.Module
    
    # 按 bit 分组的量化权重
    bit_weights: nn.ModuleDict  # "8" -> WxA16Weights, "4" -> WxA16Weights, ...
    
    # 每个 expert 的神经元在不同 bit 中的位置
    expert_offsets: Dict[str, torch.LongTensor]
    
    # Shared expert (量化版)
    shared_expert: Optional[WxA16Linear]
    shared_expert_gate: Optional[nn.Module]
```

## 实现阶段

### Phase 1: 基础打包工具
1. 实现 `pack_nbit()` 和 `unpack_nbit()`
2. 实现 `turboquant_quantize_packed_full()`
3. 单元测试

### Phase 2: WxA16Linear 模块
1. 实现 `WxA16Linear` 类
2. 实现 `forward()` 反量化推理
3. 实现 `from_linear()` 转换函数
4. 单元测试

### Phase 3: WxA16BitPartitionedGroupMoE
1. 实现 `WxA16BitPartitionedGroupMoE` 类
2. 实现高效前向
3. 单元测试

### Phase 4: 集成到主流程
1. 修改 `dartmoq_utils.py`
2. 修改 `qwen35_simple_wrapper.py`
3. 实现 `WxA16` 量化流程

### Phase 5: 测试验证
1. PPL 正确性测试
2. 内存占用对比
3. 推理速度测试

## 关键设计考虑

### 1. 反量化 + GEMM 融合
当前计划：先反量化再 GEMM（简单但可能慢）
后续优化：将反量化融合到 GEMM 中（需要自定义 CUDA kernel）

### 2. 内存管理
- 量化后立即删除原始 fp16 权重
- 每一层完成后及时 gc.collect()
- 支持 CPU offload 用于大模型

### 3. 兼容性
- 保持与现有 qscheme 格式兼容
- 支持渐进式启用 WxA16

## 测试命令示例
```bash
# 现有命令
python run_qwen35.py --quant-scheme a8s4m1.5bpw --slices 4 --rank-mode turboquant_innerproduct

# WxA16 命令 (新增 --wxa16 flag)
python run_qwen35.py --wxa16 --quant-scheme a8s4m1.5bpw --slices 4 --rank-mode turboquant_innerproduct
```
