# DartMoQ-Accel: Mixed-Precision MoE Inference Acceleration Architecture

## Core Insight

Qwen3.5 MoE's weight storage design is a perfect match for mixed-precision quantization:
- **Traditional**: Each expert is independent Module (gate_proj, up_proj, down_proj)
- **Qwen3.5**: All experts merged into two large tensors:
  - `experts.gate_up_proj`: (num_experts, 2 * intermediate_size, hidden_size)
  - `experts.down_proj`: (num_experts, hidden_size, intermediate_size)

This enables efficient **group matrix multiplication** and is a huge win for mixed-precision!

## Architecture Overview

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

## Key Components

### 1. BitGroupedMoELayer

Main wrapper that:
- Takes the reconstructed hybrid MoE
- Groups sub-experts by their bit width
- Merges weights into Qwen3.5-style format
- Manages group GEMM dispatch

### 2. BitExpertGroup

Stores merged weights for a single bit width:
```python
class BitExpertGroup(nn.Module):
    bit_width: int
    # Merged weights (Qwen3.5 style)
    gate_up_proj: torch.Tensor  # (num_experts, 2 * inter_size, hidden_size)
    down_proj: torch.Tensor     # (num_experts, hidden_size, inter_size)
    # Quantization parameters
    q_scales_gate_up: torch.Tensor  # per-group scales
    q_zeros_gate_up: torch.Tensor   # per-group zeros
    q_scales_down: torch.Tensor     # per-group scales
    q_zeros_down: torch.Tensor      # per-group zeros
    # Mapping from original expert index to internal index
    expert_map: Dict[int, int]
    # Neuron indices for each expert (for reconstruction)
    expert_neurons: List[torch.Tensor]
```

### 3. GroupGEMMKernel

Custom CUDA kernel for group matrix multiplication:
- Supports W2A16, W3A16, W4A16
- Takes batch of inputs and batch of expert weights
- Computes all experts in parallel

## Data Flow

### Forward Pass

```
Input hidden_states: (batch_size, seq_len, hidden_size)
    │
    ▼
┌─────────────────────────────────┐
│ Router: Get top-k experts       │
│ Output: expert_indices (B, T, k)│
│         expert_weights (B, T, k)│
└─────────────┬───────────────────┘
              │
              ▼
┌───────────────────────────────────────────┐
│ Group experts by bit width                │
│ For each bit width:                       │
│   - Collect experts of this bit           │
│   - Collect corresponding hidden states   │
└─────────────┬─────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────┐
│ Parallel Group GEMM Execution             │
│ For each bit width in parallel:           │
│   1. Dequantize weights (or use WxA16)   │
│   2. Group GEMM: y = x @ w^T              │
│   3. Apply activation                     │
└─────────────┬─────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────┐
│ Aggregate Results                         │
│ Weighted sum by router weights            │
└─────────────┬─────────────────────────────┘
              │
              ▼
        Output: (B, T, H)
```

## Quantization Storage Format

### Per-Group Quantization (groupsize=128)

```
For gate_up_proj (num_experts, 2*I, H):
    - qweight: int4/3/2 packed, (num_experts, 2*I, H // pack_factor)
    - scales: (num_experts, 2*I, H // groupsize)
    - zeros: (num_experts, 2*I, H // groupsize)

For down_proj (num_experts, H, I):
    - qweight: int4/3/2 packed, (num_experts, H, I // pack_factor)
    - scales: (num_experts, H, I // groupsize)
    - zeros: (num_experts, H, I // groupsize)
```

## Implementation Roadmap

### Phase 1: Weight Merging (Simulation)
- [ ] Group experts by bit width
- [ ] Merge weights into Qwen3.5 format
- [ ] Verify numerical correctness (still using FP16)

### Phase 2: True Quantization Storage
- [ ] Store quantized weights (int2/3/4)
- [ ] Store scales/zeros
- [ ] Packing/unpacking utilities

### Phase 3: Custom Kernels
- [ ] W4A16 group GEMM kernel
- [ ] W3A16 group GEMM kernel
- [ ] W2A16 group GEMM kernel

### Phase 4: End-to-End Integration
- [ ] Full model conversion
- [ ] Performance benchmarking
- [ ] Accuracy verification

## Expected Benefits

1. **Memory Efficiency**: Reduced memory fragmentation
2. **Kernel Fusion**: Fewer kernel launches
3. **Better Locality**: Coalesced memory access
4. **True Quantization**: No dequantization before matmul
5. **Parallelism**: Multiple bit groups execute in parallel

## Compatibility

- Still supports hybrid MoE wrapper interface
- Backward compatible with original model outputs
- Works with existing evaluation pipeline
