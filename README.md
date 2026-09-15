# vLLM FP8 Inference Benchmark

A homework assignment investigating LLM inference performance across
**prefill-dominant** and **decode-dominant** workloads using vLLM.

## Overview

This project benchmarks three language models with different architectures
and parameter counts to study how model architecture and FP8 inference affect
the two major phases of LLM inference: **prefilling** and **decoding**.

The experiments use vLLM as the inference engine and evaluate the models on
workloads designed to isolate the computational characteristics of each phase.

## Models

| Model | Architecture | Parameters |
|---|---|---|
| `allenai/OLMo-2-0425-1B-Instruct` | Dense | 1B |
| `allenai/OLMoE-1B-7B-0924-Instruct` | Mixture-of-Experts (MoE) | ~1B active / 7B total |
| `allenai/OLMo-2-1124-7B` | Dense | 7B |

The models provide a comparison between a small dense model, a sparse MoE
model with approximately 1B active parameters, and a larger 7B dense model.

## Workload Design

The benchmark uses the **GovReport** dataset, a long-document summarization
dataset containing reports from the U.S. Government Accountability Office
(GAO) and Congressional Research Service (CRS). GovReport was specifically
developed for long-document summarization and contains substantially longer
documents than many existing summarization datasets. :contentReference[oaicite:1]{index=1}

Two workloads are constructed from the dataset:

### 1. Prefill-Dominant: Long-Document Summarization

The model receives a long GovReport document and produces a relatively short
summary.

```text
Long document
     │
     │  ~thousands of input tokens
     ▼
  PREFILL
     │
     ▼
Short summary