# AGENTS.md

## Project Goal

mini-LLM-infra is a learning-oriented LLM inference infrastructure project.

The project starts from a simple ~1–2B decoder-only language model and gradually explores:

- baseline inference
- autoregressive decoding
- prefill and decode
- KV cache
- profiling
- Triton kernel optimization
- batching
- continuous batching
- request scheduling
- concurrent serving
- latency and throughput optimization
- comparison with production inference engines such as vLLM and SGLang

The goal is not to build a production-grade inference engine immediately.
The priority is understanding, correctness, measurement, and explainability.

## Development Environment

This project uses:

- Python 3.12
- uv for Python/project dependency management
- PyTorch
- Hugging Face Transformers
- Triton
- NVIDIA CUDA

Use:

    uv sync

to synchronize the environment.

Run Python programs using:

    uv run python <script>

Run tests using:

    uv run pytest

Add dependencies using:

    uv add <package>

Do not use pip or uv pip install for persistent project dependencies unless explicitly requested.

## Repository Structure

- `src/mini_llm_infra/`: reusable project source code
- `experiments/`: exploratory experiments and learning code
- `benchmarks/`: standardized performance benchmarks
- `tests/`: correctness tests
- `docs/`: technical notes and architecture documentation
- `profiles/`: profiler outputs when present
- `results/`: benchmark results when present

## Development Principles

1. Correctness before performance.
2. Establish a baseline before optimizing.
3. Profile before making performance claims.
4. Keep experiments reproducible.
5. Avoid premature abstraction.
6. Prefer small, reviewable changes.
7. Do not silently change unrelated files.
8. Do not add large dependencies unless necessary.
9. New optimizations should include correctness validation.
10. Performance changes should ideally include benchmark evidence.

## GPU Optimization Work

For Triton/CUDA-related changes:

- always compare against a PyTorch reference implementation
- add numerical correctness tests
- record tensor shapes and dtypes
- distinguish warm-up from benchmark iterations
- synchronize CUDA when timing manually
- do not claim speedup from a single measurement
- identify whether the workload is compute-bound or memory-bound when relevant

## Coding Style

Prefer clear and readable code over clever abstractions.

Use explicit names such as:

- `prefill_latency_ms`
- `decode_latency_ms`
- `kv_cache`
- `input_seq_len`
- `output_seq_len`

Avoid unnecessary architecture complexity during early project phases.

## Current Development Stage

The project is currently in the baseline inference stage.

Before implementing advanced features such as continuous batching or paged KV cache, make sure the basic inference and benchmark infrastructure are correct.