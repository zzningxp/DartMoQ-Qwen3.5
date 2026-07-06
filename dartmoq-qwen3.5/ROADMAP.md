# DartMoQ-Qwen3.5: Long-term Roadmap

## Overview

This project adapts the DartMoQ mixed-precision quantization framework for Qwen3.5 MoE architecture.
Qwen3.5 MoE has a fundamentally different weight storage design that enables efficient group matrix multiplication.

## Architecture Background

### Traditional MoE vs Qwen3.5 MoE

**Traditional MoE (DeepSeek, OLMoE, etc):**
```
Each expert is an independent Module:
  - expert[i].gate_proj: Linear(in_dim, inter_dim)
  - expert[i].up_proj: Linear(in_dim, inter_dim)
  - expert[i].down_proj: Linear(inter_dim, out_dim)
```

**Qwen3.5 MoE:**
```
All experts merged into two large tensors:
  - mlp.experts.gate_up_proj: (num_experts, 2 * intermediate_size, hidden_size)
  - mlp.experts.down_proj: (num_experts, hidden_size, intermediate_size)

This enables efficient group matrix multiplication!
```

## Phase 1: Baseline FP16 Evaluation

**Goal:** Run Qwen3.5 MoE's original FP16 perplexity evaluation, with CPU standby mode support.

### Tasks:
- [ ] Create folder structure and copy essential utilities
- [ ] Adapt `load_model` for Qwen3.5 MoE
- [ ] Adapt sequential evaluation (CPU standby) for Qwen3.5's architecture
- [ ] Verify perplexity matches expected results
- [ ] Explore and document Qwen3.5 MoE's layer structure

### Inputs/Outputs:
- **Input:** Original Qwen3.5 MoE model (FP16)
- **Output:** Perplexity scores on wikitext2 and C4

### Key Files:
- `qwen35_utils.py` - Qwen3.5-specific utilities
- `eval_qwen35.py` - Evaluation script
- `explore_qwen35.py` - Architecture exploration script


## Phase 2: DartMoQ Hybrid MoE with FP16 Dequantization

**Goal:** Apply DartMoQ's mixed-precision quantization methodology to Qwen3.5 MoE, still evaluating in FP16 (with dequantization).

### Tasks:
- [ ] Adapt expert sensitivity analysis for Qwen3.5's merged weight format
- [ ] Implement neuron sorting and grouping within Qwen3.5's structure
- [ ] Create hybrid MoE wrapper that works with Qwen3.5's architecture
- [ ] Adapt dynamic programming bit allocation
- [ ] Verify numerical correctness with FP16 dequantized evaluation

### Key Design Decisions:
- **Option A:** Keep Qwen3.5's group format but split into bit-specific groups
  ```
  mlp.experts_bit2.gate_up_proj: (num_experts_bit2, 2 * inter_size_bit2, H)
  mlp.experts_bit3.gate_up_proj: (num_experts_bit3, 2 * inter_size_bit3, H)
  ...
  ```

- **Option B:** Convert to traditional format for quantization, then convert back
  (Simpler for Phase 2, but less efficient)

### Key Files:
- `qwen35_layer_reconstruct.py` - Layer reconstruction for Qwen3.5
- `qwen35_hybrid_moe.py` - Hybrid MoE wrapper for Qwen3.5
- `run_qwen35.py` - Main quantization script


## Phase 3: True Quantized Inference (RTL Kernel)

**Goal:** Implement actual quantized inference on RTX 5090 using Qwen3.5's group format.

### Tasks:
- [ ] Investigate PyTorch's support for group matrix multiplication
- [ ] Implement quantized weight storage format (int2/3/4 + scales/zeros)
- [ ] Add W4A16/W3A16/W2A16 group GEMM support
  - Option 1: Use existing kernels (AutoAWQ, AutoGPTQ, Marlin)
  - Option 2: Implement custom CUDA kernels for group GEMM
- [ ] Benchmark performance on RTX 5090

### Key Research Questions:
- Does PyTorch natively support group mm for our use case?
- Can we adapt existing MoE quantization kernels for Qwen3.5's format?
- What's the speedup potential vs dequantized FP16?

### Key Files:
- `qwen35_quant_kernels.py` - Quantization kernel wrappers
- `qwen35_inference.py` - Quantized inference engine


## Folder Structure

```
dartmoq-qwen3.5/
├── README.md
├── ROADMAP.md (this file)
├── explore_qwen35.py          # Phase 1: Architecture exploration
├── eval_qwen35.py             # Phase 1: FP16 evaluation
├── qwen35_utils.py            # Phase 1+: Qwen3.5 utilities
│
├── qwen35_layer_reconstruct.py  # Phase 2: Layer reconstruction
├── qwen35_hybrid_moe.py         # Phase 2: Hybrid MoE wrapper
├── run_qwen35.py                # Phase 2: Main quantization script
│
├── qwen35_quant_kernels.py    # Phase 3: Quantization kernels
├── qwen35_inference.py        # Phase 3: Quantized inference
│
└── test/                      # Test scripts
    ├── test_phase1.py
    ├── test_phase2.py
    └── test_phase3.py
```


## Key Challenges

### 1. Group Matrix Multiplication Support
- Does PyTorch have efficient group mm operations?
- Do we need custom CUDA kernels?

### 2. Quantization within Grouped Format
- How to apply per-group quantization while keeping the grouped structure?
- How to handle different bit widths within the same expert?

### 3. Memory Layout
- Qwen3.5's weight layout may be optimized for specific hardware
- Need to understand how to best adapt our quantization approach


## Success Metrics

### Phase 1
- [ ] Can load Qwen3.5 MoE model
- [ ] Can run sequential (CPU standby) evaluation
- [ ] PPL matches HuggingFace baseline

### Phase 2
- [ ] Can apply DartMoQ quantization
- [ ] PPL is within acceptable range of baseline
- [ ] Maintains hybrid MoE structure

### Phase 3
- [ ] Can run quantized inference on RTX 5090
- [ ] Significant speedup over Phase 2
- [ ] PPL remains acceptable


## Dependencies

Same as main DartMoQ project:
- PyTorch
- Transformers
- (Optional) CUDA Toolkit for Phase 3 custom kernels
- (Optional) lm_eval for zero-shot evaluation
