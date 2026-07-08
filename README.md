# DartMoQ-Qwen3.5

DartMoQ mixed-precision quantization framework adapted for Qwen3.5 MoE architecture.

## Overview

Qwen3.5 MoE uses a fundamentally different weight storage format that enables efficient group matrix multiplication.
This project adapts the DartMoQ framework to work with this new architecture.

## Quick Start

### Phase 1: Explore Architecture

```bash
cd dartmoq-qwen3.5
python explore_qwen35.py /path/to/qwen3.5/model --device cpu --detailed
```

### Phase 1: Evaluate FP16 Baseline

```bash
# Evaluate with sequential mode (CPU standby, recommended for large models)
python eval_qwen35.py /path/to/qwen3.5/model --sequential-eval --standby-cpu

# Or evaluate normally
python eval_qwen35.py /path/to/qwen3.5/model
```

## Phases

- **Phase 1 (Current):** Baseline FP16 evaluation on Qwen3.5 MoE
- **Phase 2:** DartMoQ Hybrid MoE with FP16 dequantization
- **Phase 3:** True quantized inference on RTX 5090

See [ROADMAP.md](ROADMAP.md) for detailed plans.

## Files

- `explore_qwen35.py` - Explore Qwen3.5 MoE architecture
- `eval_qwen35.py` - Evaluate Qwen3.5 MoE perplexity (Phase 1)
- `qwen35_utils.py` - Shared utilities
- `ROADMAP.md` - Long-term roadmap

## Requirements

Same as main DartMoQ project:
- PyTorch
- Transformers
- Datasets
