# vLLM FP8 Inference Benchmark

A homework assignment investigating LLM inference performance across
**prefill-dominant** and **decode-dominant** workloads using vLLM.

## Homework Context

This work was completed as part of **CS 601.768: Language Model Agents**,
a graduate-level course taught by **Benjamin Van Durme** at Johns Hopkins
University during **Fall 2026**.

This repository contains the implementation and experiments for a **homework
assignment** in the course. It is intended as an empirical investigation of
LLM inference behavior rather than as a general-purpose benchmarking
framework.

## Overview

This homework assignment benchmarks three language models with different
architectures and parameter counts to investigate how model architecture and
FP8 inference affect the two major phases of LLM inference: **prefilling** and
**decoding**.

The experiments use vLLM as the inference engine and evaluate the models on
workloads designed to emphasize each phase separately.

## Models

| Model | Architecture | Parameters |
|---|---|---|
| `allenai/OLMo-2-0425-1B-Instruct` | Dense | 1B |
| `allenai/OLMoE-1B-7B-0924-Instruct` | Mixture-of-Experts (MoE) | ~1B active / 7B total |
| `allenai/OLMo-2-1124-7B` | Dense | 7B |

These models provide comparisons across both **model architecture** and
**model scale**: a small dense model, a sparse MoE model with approximately
1B active parameters and 7B total parameters, and a larger 7B dense model.

## Dataset

The experiments use the **GovReport** dataset, a long-document
summarization dataset containing reports from the U.S. Government
Accountability Office (GAO) and Congressional Research Service (CRS).

GovReport was designed for long-document summarization and contains
substantially longer documents than many existing summarization datasets.

## Workload Design

Two workloads are constructed from the GovReport dataset in order to
separately investigate **prefilling** and **decoding**.

### 1. Prefill-Dominant: Long-Document Summarization

The model receives a long GovReport document and produces a relatively short
summary.

```text
Long GovReport document
        │
        │  Many input tokens
        ▼
     PREFILL
        │
        ▼
   Short summary