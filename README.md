# DartMoQ-Qwen3.5

DartMoQ mixed-precision quantization framework adapted for Qwen3.5 MoE architecture.

## Overview

Qwen3.5 MoE uses a fundamentally different weight storage format that enables efficient group matrix multiplication.
This project adapts the [DartMoQ](https://github.com/zzningxp/DartMoQ) framework to work with this new architecture.

## Quick Start

### Phase 1: Evaluate FP16 Baseline

```bash
# Evaluate with sequential mode (CPU standby, recommended for large models)
python eval_qwen35.py /path/to/qwen3.5/model --sequential-eval --standby-cpu

# Or evaluate normally
python eval_qwen35.py /path/to/qwen3.5/model
```

## Roadmap

- **Phase 1:** Baseline FP16 evaluation on Qwen3.5 MoE
- **Phase 2 (Current):** DartMoQ Hybrid MoE with FP16 dequantization
- **Phase 3:** Import quantized library
- **Phase 4:** True quantized inference on GPU like RTX 5090 (Blackwell)

See [ROADMAP.md](ROADMAP.md) for detailed plans.

## Requirements

Same as main DartMoQ project:
- PyTorch
- Transformers
- Datasets
