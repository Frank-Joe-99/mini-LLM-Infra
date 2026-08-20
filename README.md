# mini-LLM-infra

**English** | [简体中文](./README.zh-CN.md)

> Building a small but complete LLM inference stack from scratch — from naive PyTorch inference to profiling, optimization, concurrent serving, and production-oriented experiments.

## Overview

`mini-LLM-infra` is a hands-on learning project for understanding **LLM inference infrastructure** through implementation and measurement.

Instead of treating an inference engine as a black box, this project starts from a simple ~2B decoder-only language model and gradually builds the components required by a practical inference system.

The project follows a simple philosophy:

**Build → Measure → Analyze → Optimize → Serve → Stress Test → Compare**

The goal is not to build another production-grade inference framework.

The goal is to understand **why modern LLM inference engines are designed the way they are**.

---

## Goals

This project aims to answer questions such as:

* What actually happens during LLM inference?
* What is the difference between **prefill** and **decode**?
* Where does inference latency come from?
* Where is GPU memory consumed?
* How does **KV Cache** reduce computation?
* Why does KV Cache become a memory-management problem?
* How do batch size and sequence length affect performance?
* How can multiple requests be executed efficiently?
* What is the difference between static batching and continuous batching?
* How should requests with different prompt/output lengths be scheduled?
* How can we reduce **TTFT** and **TPOT**?
* How do throughput and latency trade off with each other?
* What causes high P95/P99 latency under concurrency?
* How do modern engines such as vLLM and SGLang improve inference performance?

---

# Project Scope

## Model

The primary model is currently **Qwen3-1.7B**, a decoder-only LLM in the
1–2B parameter range.

The same primary model should be used across most experiments so that performance changes mainly come from infrastructure modifications rather than model differences.

Model weights are stored locally under `models/` and are intentionally excluded
from version control. Most scripts default to
`/workspace/mini-LLM-Infra/models/Qwen3-1.7B`; use `--model-path` when the
checkpoint is stored elsewhere.

---

## Hardware

Initial target:

* Single NVIDIA GPU

Possible future extensions:

* Multi-GPU inference
* Tensor parallelism
* Prefill/Decode disaggregation
* Distributed serving

The project intentionally starts from single-GPU inference because most fundamental inference-system problems can already be studied at this scale.

---

## Software Stack

The current baseline environment uses:

* Python 3.12
* uv
* PyTorch
* Hugging Face Transformers
* Accelerate
* Triton
* NVIDIA CUDA
* NumPy
* pytest

Later stages may also use:

* PyTorch Profiler
* NVIDIA Nsight Systems
* NVIDIA Nsight Compute
* FastAPI / HTTP serving tools
* vLLM
* SGLang

---

# Quick Start

Synchronize the Python environment from the repository root:

```bash
uv sync
```

Check that PyTorch can access the GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Run the shortest Hugging Face baseline smoke test:

```bash
uv run python experiments/01_default_infer_baseline/benchmark_hf_generate.py \
  --suite smoke \
  --no-profile
```

If the model is not in the default location, add:

```bash
--model-path /path/to/Qwen3-1.7B
```

See the experiment documentation for details:

* [First model load and structure export](./experiments/00_first_load_and_infra/README.md)
* [Hugging Face generation benchmark](./experiments/01_default_infer_baseline/README-benchmark-hf-generate.md)
* [Benchmark workload configuration](./experiments/01_default_infer_baseline/README-benchmark-config.md)

---

# Architecture

The project is organized into three abstraction levels.

```text
Level 1
Reference Implementation
        │
        ├── Hugging Face
        └── PyTorch
                │
                ▼
Level 2
mini-LLM Runtime
        │
        ├── manual decoding loop
        ├── KV cache
        ├── batching
        ├── scheduler
        ├── memory management
        └── serving
                │
                ▼
Level 3
Production Engine Comparison
        │
        ├── vLLM
        └── SGLang
```

The production engines are used as **references and comparison targets**, rather than as the starting point of the project.

---

# Roadmap

## Phase 0 — Reproducible Baseline

Goal: establish a simple and reliable inference baseline.

* [x] Create a reproducible Python environment with uv
* [x] Load the Qwen3-1.7B tokenizer and model from local files
* [x] Run single-request greedy text generation
* [x] Fix random seeds in the benchmark path
* [x] Record the software and hardware environment
* [x] Record tokenizer and model loading time
* [x] Measure peak allocated and reserved GPU memory
* [x] Build configurable smoke, prefill, decode, mixed, and full benchmark suites
* [x] Save raw measurements and generated outputs
* [x] Export PyTorch Profiler traces and operator tables

Initial baseline:

```text
Prompt
  ↓
Tokenizer
  ↓
Hugging Face / PyTorch Model
  ↓
generate()
  ↓
Generated Tokens
```

### Deliverable

```text
experiments/
├── 00_first_load_and_infra/
└── 01_default_infer_baseline/
```

with:

* a runnable first-load and generation script
* a saved Qwen3-1.7B module structure
* a configurable Hugging Face `model.generate()` benchmark
* environment, loading-time, latency, throughput, and GPU-memory collection
* JSON and text result reports
* optional PyTorch Profiler traces

The baseline tooling is implemented. Reproducible benchmark runs and analyzed
result reports are the next evidence-gathering step; no performance conclusion
should be drawn until those measurements are recorded and compared.

---

# Phase 1 — Understand Autoregressive Inference

Goal: stop treating `model.generate()` as a black box.

Implement the generation loop manually.

```text
Prompt
  ↓
Prefill
  ↓
logits
  ↓
sampling
  ↓
token
  ↓
Decode
  ↓
sampling
  ↓
token
  ↓
...
```

Tasks:

* [ ] Implement manual autoregressive decoding
* [ ] Understand logits generation
* [ ] Implement greedy decoding
* [ ] Implement temperature sampling
* [ ] Implement Top-K sampling
* [ ] Implement Top-P sampling
* [ ] Separate prefill and decode
* [ ] Measure prefill latency
* [ ] Measure decode latency
* [ ] Measure per-token latency

Important metrics:

```text
TTFT  = Time To First Token

TPOT  = Time Per Output Token

ITL   = Inter-Token Latency

E2E Latency = End-to-End Request Latency
```

---

# Phase 2 — KV Cache

Goal: understand the first major optimization in autoregressive inference.

Experiments:

* [ ] Decode without KV Cache
* [ ] Decode with KV Cache
* [ ] Compare computation
* [ ] Compare latency
* [ ] Compare GPU memory consumption
* [ ] Analyze KV Cache growth with sequence length
* [ ] Analyze KV Cache layout
* [ ] Study MHA / GQA / MQA differences

Questions to answer:

```text
Why cache K and V?

How much computation does KV Cache save?

How much memory does KV Cache consume?

How does sequence length affect KV Cache size?

Why does serving multiple users turn KV Cache into
a memory-management problem?
```

---

# Phase 3 — Performance Measurement and Profiling

Goal: learn how to diagnose inference performance before optimizing it.

## Macro-level metrics

Measure:

* TTFT
* TPOT
* ITL
* End-to-end latency
* Requests / second
* Input tokens / second
* Output tokens / second
* Total tokens / second
* GPU utilization
* GPU memory usage

Latency distributions:

* P50
* P90
* P95
* P99

## Profiling tools

### PyTorch Profiler

Use for:

* operator-level latency
* CPU/GPU execution timeline
* CUDA kernel activity
* operator shapes

### Nsight Systems

Use for:

* CPU ↔ GPU execution timeline
* CUDA kernel launches
* synchronization
* kernel gaps
* CUDA Graph behavior

### Nsight Compute

Use for:

* individual kernel analysis
* memory throughput
* compute utilization
* occupancy
* roofline analysis

Tasks:

* [ ] Profile prefill
* [ ] Profile decode
* [ ] Find major operators
* [ ] Identify GPU idle time
* [ ] Identify synchronization overhead
* [ ] Analyze memory bandwidth utilization
* [ ] Analyze compute utilization

---

# Phase 4 — Kernel-Level Optimization

Goal: connect LLM inference with GPU kernel optimization.

Candidate operators:

```text
RMSNorm
RoPE
Softmax
SwiGLU
Attention
Matrix Multiplication
Sampling
```

Experiments:

* [ ] PyTorch baseline
* [ ] Triton implementation
* [ ] Correctness test
* [ ] Performance benchmark
* [ ] Profile custom kernels
* [ ] Analyze arithmetic intensity
* [ ] Analyze memory traffic

Possible optimizations:

* kernel fusion
* reduced intermediate tensors
* better memory access
* online softmax
* fused normalization
* fused activation
* FlashAttention-style IO reduction

Each kernel experiment should answer:

```text
1. What is the baseline?
2. Where is the bottleneck?
3. What optimization is applied?
4. Why should it work?
5. Does profiling confirm the hypothesis?
6. How much speedup is achieved?
```

---

# Phase 5 — Batch Inference

Goal: move from single-request inference to multiple requests.

Start with static batching.

```text
Request A ┐
Request B ├── Batch ── GPU
Request C ┘
```

Experiments:

* [ ] Batch size = 1
* [ ] Batch size = 2
* [ ] Batch size = 4
* [ ] Batch size = 8
* [ ] Batch size = 16
* [ ] Measure throughput
* [ ] Measure latency
* [ ] Measure GPU utilization
* [ ] Measure memory usage

Study the trade-off:

```text
larger batch
    ↓
higher GPU utilization
    ↓
higher throughput

but potentially

larger batch
    ↓
higher waiting time
    ↓
higher request latency
```

---

# Phase 6 — Continuous Batching and Scheduling

Goal: understand how an inference server handles requests with different arrival times and sequence lengths.

Build a minimal scheduler.

```text
                 ┌── Request A
Incoming Queue ──┼── Request B
                 ├── Request C
                 └── Request D
                       │
                       ▼
                  Scheduler
                       │
                       ▼
                  Active Batch
                       │
                       ▼
                      GPU
```

Tasks:

* [ ] Request queue
* [ ] Active sequence tracking
* [ ] Dynamic batch construction
* [ ] Add newly arriving requests
* [ ] Remove completed requests
* [ ] Handle different prompt lengths
* [ ] Handle different output lengths
* [ ] Implement continuous batching

Scheduling questions:

* Should short requests wait for long requests?
* How should newly arrived requests enter the batch?
* How should prefill and decode compete for GPU time?
* How do we prevent starvation?
* How do we balance throughput and latency?
* How does scheduling affect TTFT?

---

# Phase 7 — KV Cache Management

Goal: move from KV Cache as a tensor to KV Cache as a system resource.

Study:

* [ ] Per-request KV Cache allocation
* [ ] KV Cache lifetime
* [ ] Memory fragmentation
* [ ] Block-based KV Cache allocation
* [ ] KV Cache block reuse
* [ ] Prefix sharing
* [ ] Prefix caching
* [ ] KV Cache eviction

Possible mini implementation:

```text
Logical KV Cache

Request A:
[0][1][2][3][4]

Request B:
[0][1][2]

        ↓

Physical KV Blocks

Block Pool:
[7][2][9][1][5][8]...
```

This phase can gradually introduce the ideas behind paged KV-cache management.

---

# Phase 8 — LLM Serving

Goal: expose the mini runtime as a real inference service.

Build an HTTP inference server.

Features:

* [ ] `/generate`
* [ ] `/chat`
* [ ] streaming output
* [ ] asynchronous requests
* [ ] concurrent users
* [ ] request timeout
* [ ] request cancellation
* [ ] maximum sequence length
* [ ] admission control
* [ ] backpressure
* [ ] error handling

Possible architecture:

```text
Client
  │
  ▼
HTTP API
  │
  ▼
Request Queue
  │
  ▼
Scheduler
  │
  ▼
Inference Runtime
  │
  ▼
GPU
```

Later, implement an OpenAI-compatible API if useful.

---

# Phase 9 — Multi-User Concurrency

Goal: study inference as an online system rather than an offline benchmark.

Create a load generator.

Example workloads:

```text
Concurrency = 1
Concurrency = 2
Concurrency = 4
Concurrency = 8
Concurrency = 16
Concurrency = 32
...
```

Measure:

| Metric             | Meaning                                  |
| ------------------ | ---------------------------------------- |
| TTFT               | Time to first generated token            |
| TPOT               | Average generation time per output token |
| E2E latency        | Total request latency                    |
| P50 latency        | Median latency                           |
| P95 latency        | Tail latency                             |
| P99 latency        | Extreme tail latency                     |
| Request throughput | Requests completed per second            |
| Token throughput   | Tokens generated per second              |
| Queue time         | Time waiting before execution            |
| GPU utilization    | GPU compute utilization                  |
| GPU memory         | Runtime memory consumption               |

Study:

```text
QPS
 │
 │                 overload
 │                /
 │              /
 │            /
 │___________/________ latency
```

Important engineering questions:

* At what concurrency does the server saturate?
* When does queueing dominate latency?
* Why does P99 increase?
* What is the maximum sustainable QPS?
* How does request length distribution affect throughput?
* Can one long request hurt all other users?

---

# Phase 10 — Advanced Inference Optimizations

After the basic runtime works, gradually study production techniques.

## Chunked Prefill

* [ ] Large prompt experiment
* [ ] Prefill chunking
* [ ] Compare TTFT
* [ ] Compare decode interference

## Prefix Caching

* [ ] Shared system prompts
* [ ] Cache reuse
* [ ] Measure prefill reduction

## CUDA Graph

* [ ] Identify CPU/kernel-launch overhead
* [ ] Capture stable execution paths
* [ ] Compare latency before/after CUDA Graph

## Quantization

Possible experiments:

```text
FP32
 ↓
FP16 / BF16
 ↓
INT8
 ↓
INT4
```

Measure:

* memory
* latency
* throughput
* output quality

## Speculative Decoding

Study:

```text
Draft Model
    │
    ├── token 1
    ├── token 2
    ├── token 3
    └── token 4
          │
          ▼
     Target Model
          │
       Verify
```

Measure:

* acceptance rate
* TPOT
* total throughput
* additional compute overhead

---

# Phase 11 — Compare with Production Inference Engines

Goal: understand the gap between a learning-oriented runtime and production systems.

Comparison targets:

* Hugging Face baseline
* mini-LLM-infra
* vLLM
* SGLang

Keep the following fixed:

```text
Same Model
Same GPU
Same Prompt Dataset
Same Input Length
Same Output Length
Same Concurrency
Same Sampling Parameters
```

Compare:

| System         | TTFT | TPOT | Output tok/s | P95 | GPU Mem |
| -------------- | ---: | ---: | -----------: | --: | ------: |
| Hugging Face   |      |      |              |     |         |
| mini-LLM-infra |      |      |              |     |         |
| vLLM           |      |      |              |     |         |
| SGLang         |      |      |              |     |         |

The most important part is not the ranking.

The important part is explaining:

> **Why is one system faster than another?**

---

# Phase 12 — Reliability and Production-Oriented Problems

Explore problems that do not appear in simple offline benchmarks.

Topics:

* [ ] GPU OOM handling
* [ ] request admission control
* [ ] overload protection
* [ ] request cancellation
* [ ] timeouts
* [ ] queue limits
* [ ] memory leaks
* [ ] malformed inputs
* [ ] server warm-up
* [ ] cold-start latency
* [ ] graceful shutdown
* [ ] metrics collection
* [ ] logging
* [ ] reproducible benchmarks

Possible service metrics:

```text
request_count
request_latency
queue_latency
ttft
tpot
tokens_per_second
active_requests
waiting_requests
kv_cache_usage
gpu_memory_usage
gpu_utilization
```

---

# Bonus — Distributed Inference

This is intentionally not part of the initial project.

Possible future topics:

* Tensor Parallelism
* Pipeline Parallelism
* Prefill/Decode disaggregation
* Multi-GPU KV Cache
* Distributed scheduling
* Request routing
* Load balancing
* Multi-node inference

---

# Repository Structure

The repository currently contains the Phase 0 environment and baseline
experiments. Later directories will be added only when their corresponding
development phase begins.

```text
mini-LLM-infra/
├── AGENTS.md
├── README.md
├── README.zh-CN.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── models/
│   └── .gitkeep                 # local model weights are ignored
├── src/
│   └── mini_llm_infra/
│       └── __init__.py
├── experiments/
│   ├── 00_first_load_and_infra/
│   │   ├── README.md
│   │   ├── first-load.py
│   │   └── qwen3-1.7b-structure.txt
│   └── 01_default_infer_baseline/
│       ├── README-benchmark-config.md
│       ├── README-benchmark-hf-generate.md
│       ├── benchmark_config.json
│       └── benchmark_hf_generate.py
```

Generated benchmark `results/` and profiler `profiles/` directories live under
the relevant experiment. Shared `benchmarks/`, `docs/`, `profiles/`, and
`results/` directories, plus `tests/`, will be introduced when reusable
infrastructure and correctness coverage appear.

---

# Experiment Template

Every experiment should contain four parts.

## 1. Question

Example:

> How much latency does KV Cache save during autoregressive decoding?

## 2. Hypothesis

Example:

> KV Cache should significantly reduce redundant computation during decode, while increasing GPU memory consumption.

## 3. Experiment

Record:

```text
Model
GPU
PyTorch version
CUDA version
dtype
batch size
prompt length
output length
warm-up iterations
measurement iterations
```

## 4. Result

Report:

```text
Baseline
Optimization
Speedup
Memory Change
Profiler Evidence
Explanation
```

The project should prioritize **explainable performance improvements**, rather than reporting speedup numbers alone.

---

# Benchmarking Principles

All performance comparisons should follow several rules:

1. Warm up the GPU before timing.
2. Synchronize CUDA when necessary.
3. Use the same model and inputs for comparisons.
4. Separate prefill and decode measurements.
5. Report distributions instead of only averages.
6. Record the complete hardware/software environment.
7. Repeat experiments multiple times.
8. Save raw benchmark data.
9. Profile before optimizing.
10. Explain why an optimization works.

---

# Milestones

## M0 — Baseline

```text
Hugging Face inference works
+
basic benchmark works
```

## M1 — Understand Inference

```text
manual autoregressive decoding
+
prefill/decode separation
+
KV Cache
```

## M2 — Understand Performance

```text
benchmark
+
PyTorch Profiler
+
Nsight
+
bottleneck analysis
```

## M3 — Optimize Single Request

```text
Triton kernels
+
attention optimization
+
memory optimization
```

## M4 — Build Mini Runtime

```text
batching
+
KV Cache manager
+
scheduler
```

## M5 — Build Serving System

```text
HTTP server
+
streaming
+
concurrency
+
load testing
```

## M6 — Production Comparison

```text
mini-LLM-infra
vs
Hugging Face
vs
vLLM
vs
SGLang
```

---

# What This Project Is Not

This repository is **not** intended to:

* train a foundation model from scratch
* implement every CUDA kernel manually
* replace vLLM or SGLang
* immediately support large distributed clusters
* chase benchmark numbers without understanding their causes

The emphasis is on:

> **understanding the full path from a model checkpoint to an efficient online inference service.**

---

# Current Status

🚧 Work in progress.

Completed baseline infrastructure:

* [x] Python 3.12 and uv environment with CUDA-enabled PyTorch
* [x] Local Qwen3-1.7B loading and model-structure export
* [x] Single-request greedy generation
* [x] Config-driven Hugging Face `generate()` benchmark suites
* [x] Warm-up, synchronized timing, latency distribution, throughput, and GPU-memory collection
* [x] JSON/text result output and optional PyTorch Profiler export

Current focus:

* [ ] Run and archive reproducible baseline measurements on the target GPU
* [ ] Review generated outputs and write the first benchmark analysis
* [ ] Add correctness tests for configuration and metric helpers
* [ ] Begin manual autoregressive decoding
* [ ] Separate strict Prefill, TTFT, Decode, TPOT, and ITL measurements

---

# Long-Term Goal

By the end of the project, `mini-LLM-infra` should provide a small but understandable inference stack:

```text
Model
  ↓
Inference Runtime
  ↓
KV Cache Manager
  ↓
Scheduler
  ↓
Continuous Batching
  ↓
HTTP Serving
  ↓
Concurrent Users
  ↓
Profiling
  ↓
Optimization
```

More importantly, every major optimization should be supported by:

**measurement, profiling, implementation, and explanation.**
