# DartMoQ-Qwen3.5

DartMoQ mixed-precision quantization framework adapted for Qwen3.5 MoE architecture.

## Overview

Qwen3.5 MoE uses a fundamentally different weight storage format that enables efficient group matrix multiplication.
This project adapts the [DartMoQ](https://github.com/zzningxp/DartMoQ) framework to work with this new architecture.

## Quick Start

## Evaluate FP16 Baseline

```bash
# Or evaluate normally
python eval_qwen35.py /path/to/qwen3.5/model
```

## WxA16 triton inference

```bash
python run_qwen35.py $modelname wikitext2 
        --wxa16 \
        --nsamples 64 \
        --slices 4 \
        --quant-scheme global-a8s8m2bpw \
        --rank-mode turboquant_innerproduct \
        --quantmode turboquant \
        --standby-layer-cpu
```

## several layer quantization inference

```bash
python run_qwen35.py $modelname wikitext2 
        --quant-layers 0-4 \
        --wxa16 \
        --nsamples 64 \
        --slices 4 \
        --quant-scheme global-a8s8m2bpw \
        --rank-mode turboquant_innerproduct \
        --quantmode turboquant \
        --standby-layer-cpu
```

## Requirements

Same as main DartMoQ project:
- PyTorch
- Transformers
- Datasets
