# mini-LLM-infra

[English](./README.md) | **简体中文**

> 从 Hugging Face / PyTorch 原生推理出发，逐步构建、测量并理解一个小型但完整的 LLM 推理系统。

## 项目简介

`mini-LLM-infra` 是一个面向 **LLM Inference / AI Infra** 的学习与实践项目。
项目以约 1–2B 参数的 Decoder-only 模型为起点，通过实际实现和可复现实验，逐步研究：

* 基础推理与自回归生成
* Prefill、Decode 与 KV Cache
* 延迟、吞吐和 GPU 显存测量
* PyTorch Profiler、Nsight 与瓶颈分析
* Triton Kernel 优化
* Static / Continuous Batching
* 请求调度、KV Cache 管理和并发服务
* 与 vLLM、SGLang 等生产级推理引擎的公平对比

项目遵循以下迭代方式：

**Build → Measure → Analyze → Optimize → Serve → Stress Test → Compare**

即：

**实现 → 测量 → 分析 → 优化 → 服务化 → 压测 → 对比**

本项目不以立即实现生产级推理引擎为目标。当前优先级是：**正确性、可复现性、测量和可解释性**。

---

## 当前阶段

项目当前处于 **Phase 1：手写自回归解码**（Phase 0 已完成）。

### Phase 0 — 已完成

* [x] Python 3.12 + uv 项目环境
* [x] CUDA 13.0 版 PyTorch、Transformers、Accelerate、Triton、matplotlib 与 pytest 依赖
* [x] 从本地加载 Qwen3-1.7B Tokenizer 和模型权重
* [x] 导出模型结构与参数统计
* [x] 单请求贪心生成
* [x] 配置驱动的 `smoke`、`prefill`、`decode`、`mixed` 和 `full` 测试套件
* [x] Warm-up 与 CUDA 同步计时
* [x] 延迟（mean/P50/P95）、输出 tokens/s、ms/token 和 GPU 显存峰值采集
* [x] 环境信息、Tokenizer / 模型加载耗时记录
* [x] JSON / TXT 结果保存与可选的 PyTorch Profiler 导出

### Phase 1 — 进行中

* [x] 手写自回归解码循环（`src/mini_llm_infra/generation/manual.py` → `manual_generate()`）
* [x] 严格拆分 Prefill 与 Decode —— TTFT / TPOT / ITL / TOT
* [x] KV Cache 开/关对比（`--use-cache` / `--no-use-cache`）
* [x] 可复用模型加载器（`src/mini_llm_infra/model/loader.py`）
* [x] 手写解码正确性测试（`tests/test_autoregressive.py`）
* [ ] Temperature / Top-K / Top-P 采样（当前仅 Greedy）
* [ ] 与 HF `generate()` 逐 token 一致性验证（`compare_greedy_with_hf.py`）
* [ ] 在目标 GPU 上保存可复现的 Phase 1 测量结果
* [ ] 首份 Prefill / Decode 分析报告

当前工作重点：

* [ ] 完成 `compare_greedy_with_hf.py` 一致性对比
* [ ] 运行并保存可复现的 Phase 1 数据
* [ ] 撰写首份 Prefill / Decode 分析
* [ ] 接入 PyTorch Profiler 对 Phase 1 工作负载进行 Profiling

---

## 项目目标

本项目希望通过实现和实验逐步回答：

* LLM 如何完成一次完整推理？
* Prefill 和 Decode 的计算特征有什么区别？
* 延迟和显存分别消耗在哪里？
* KV Cache 为什么能减少计算，又为什么会成为显存管理问题？
* Batch Size 和序列长度如何影响吞吐、延迟与 GPU 利用率？
* 如何测量 TTFT、TPOT、ITL、P95 和 P99？
* 不同到达时间和不同长度的请求应该如何调度？
* Continuous Batching 如何提高 GPU 利用率？
* 为什么 vLLM、SGLang 等推理引擎能获得更好的性能？

---

## 项目范围

### 模型

当前主要模型为 **Qwen3-1.7B**。大部分实验会固定使用同一个模型，使性能变化尽可能来自推理基础设施，而不是模型差异。

模型权重放在本地 `models/` 目录中，并由 Git 忽略。脚本默认模型路径为：

```text
/workspace/mini-LLM-Infra/models/Qwen3-1.7B
```

模型位于其他位置时，可向基准脚本传入 `--model-path`。

### 硬件

初始目标是单张 NVIDIA GPU。多 GPU、Tensor Parallelism、Prefill/Decode 分离和分布式推理属于后续扩展，不是当前阶段的工作重点。

### 软件栈

当前基线环境使用：

* Python 3.12
* uv
* PyTorch（CUDA 13.0）
* Hugging Face Transformers / Hub
* Accelerate
* Triton
* NVIDIA CUDA
* NumPy
* matplotlib
* pytest

后续阶段计划使用 PyTorch Profiler、Nsight Systems、Nsight Compute、HTTP 服务工具、vLLM 和 SGLang。

---

## 快速开始

在仓库根目录同步环境：

```bash
uv sync
```

确认 PyTorch 可以访问 GPU：

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

运行最短的 Hugging Face 基线 Smoke Test：

```bash
uv run python experiments/01_default_infer_baseline/benchmark_hf_generate.py \
  --suite smoke \
  --no-profile
```

如果模型不在默认目录，增加：

```bash
--model-path /path/to/Qwen3-1.7B
```

运行 Phase 1 手写自回归解码 Smoke Test：

```bash
uv run python experiments/02_autoregressive_decoding/benchmark_manual_decode.py \
  --suite smoke
```

对比开/关 KV Cache 时的 Decode 延迟：

```bash
# 开启 KV Cache（默认）
uv run python experiments/02_autoregressive_decoding/benchmark_manual_decode.py \
  --suite decode

# 关闭 KV Cache ——每步重算完整历史，观察延迟变化
uv run python experiments/02_autoregressive_decoding/benchmark_manual_decode.py \
  --suite decode \
  --no-use-cache
```

### 实验文档

* [首次加载、生成与模型结构导出](./experiments/00_first_load_and_infra/README.md)
* [Hugging Face 原生生成基线](./experiments/01_default_infer_baseline/README-benchmark-hf-generate.md)
* [Benchmark 工作负载配置说明](./experiments/01_default_infer_baseline/README-benchmark-config.md)
* [手写自回归解码（Phase 1）](./experiments/02_autoregressive_decoding/README.md)

---

## 架构分层

```text
Level 1：参考实现
Hugging Face / PyTorch
          │
          ▼
Level 2：mini-LLM Runtime
手写 Decode、KV Cache、Batching、Scheduler、Serving
          │
          ▼
Level 3：生产引擎对比
vLLM / SGLang
```

生产级引擎是后续的参考和对比对象，不是项目起点。

---

## 路线图

### Phase 0 — Reproducible Baseline

目标：建立简单、可靠、可复现的单请求推理基线。

```text
Prompt → Tokenizer → Hugging Face / PyTorch Model → generate() → Generated Tokens
```

当前实现位于：

```text
experiments/
├── 00_first_load_and_infra/
└── 01_default_infer_baseline/
```

实现已覆盖模型加载、文本生成、固定随机种子、环境记录、加载耗时、GPU 显存、基础 Benchmark、生成结果保存和 Profiler 导出。

### Phase 1 — Autoregressive Decoding（进行中）

手写 `manual_generate()` 自回归生成循环，以 TTFT 衡量 Prefill、以 TPOT/ITL 衡量 Decode。已实现 Greedy 解码、KV Cache 开/关对比、TTFT / TPOT / ITL / TOT 计时，以及手写解码正确性测试。Temperature / Top-K / Top-P 采样和与 HF `generate()` 的一致性验证尚未完成。

重点指标：TTFT、TPOT、ITL 和端到端延迟。

### Phase 2 — KV Cache（部分实现）

已通过 `--use-cache` / `--no-use-cache` 实现 KV Cache 开/关对比，可测量两种模式的计算量、延迟与显存差异。KV Cache 随序列长度增长的规律分析、MHA / GQA / MQA 缓存布局对比尚未完成。

### Phase 3 — Profiling

使用 PyTorch Profiler、Nsight Systems 和 Nsight Compute 分析 Prefill、Decode、GPU 空闲、同步开销、显存带宽、计算利用率和单 Kernel 行为。

### Phase 4 — Kernel Optimization

为 RMSNorm、RoPE、Softmax、SwiGLU、Attention 和 Sampling 建立 PyTorch 参考实现、Triton 实现、正确性测试与性能基准。

### Phase 5 — Batch Inference

研究静态 Batch Size 对吞吐、延迟、GPU 利用率和显存的影响，理解吞吐与请求延迟之间的权衡。

### Phase 6 — Continuous Batching and Scheduling

实现 Request Queue、Active Batch、动态加入和移除请求，并处理不同 Prompt / Output Length。

### Phase 7 — KV Cache Management

把 KV Cache 从单个 Tensor 提升为系统资源，研究生命周期、碎片、Block 分配、复用、Prefix Cache 和淘汰策略。

### Phase 8 — LLM Serving

实现 HTTP 推理服务、流式输出、异步请求、并发、超时、取消、Admission Control、Backpressure 和错误处理。

### Phase 9 — Multi-User Concurrency

构造并发负载，测量请求吞吐、Token 吞吐、排队时间以及 P50 / P95 / P99 延迟，确定系统饱和点。

### Phase 10 — Advanced Optimizations

研究 Chunked Prefill、Prefix Caching、CUDA Graph、量化和 Speculative Decoding。

### Phase 11 — Production Engine Comparison

在模型、GPU、Prompt、输入/输出长度、并发和采样参数一致的条件下，对比 Hugging Face、mini-LLM-infra、vLLM 与 SGLang，并解释性能差异的原因。

### Phase 12 — Reliability

研究 GPU OOM、Admission Control、过载保护、请求取消、超时、队列限制、冷启动、优雅退出、日志与指标等在线系统问题。

---

## 当前仓库结构

```text
mini-LLM-infra/
├── AGENTS.md
├── README.md
├── README.zh-CN.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── models/
│   └── .gitkeep
├── src/
│   └── mini_llm_infra/
│       ├── __init__.py
│       ├── generation/
│       │   └── manual.py          # ManualGenerateConfig / GenerationMetrics / manual_generate()
│       ├── model/
│       │   └── loader.py          # load_tokenizer / load_causal_lm（支持 eager/sdpa）
│       └── utils/
│           ├── cuda.py            # configure_cuda_compatibility / select_dtype
│           └── loadconfig.py      # 配置校验、套件与 Prompt 解析
├── tests/
│   └── test_autoregressive.py     # 手写解码正确性测试
└── experiments/
    ├── 00_first_load_and_infra/
    │   ├── README.md
    │   ├── first-load.py
    │   └── qwen3-1.7b-structure.txt
    ├── 01_default_infer_baseline/
    │   ├── README-benchmark-config.md
    │   ├── README-benchmark-hf-generate.md
    │   ├── benchmark_config.json
    │   └── benchmark_hf_generate.py
    └── 02_autoregressive_decoding/
        ├── README.md
        ├── benchmark_config.json
        ├── benchmark_manual_decode.py   # 手写 Greedy 解码 + TTFT/TPOT/ITL/TOT
        └── compare_greedy_with_hf.py    # （占位）与 HF generate() 的一致性对比
```

运行基准后，`results/` 和 `profiles/` 会生成在对应实验目录中。随着可复用代码和正确性覆盖增加，再逐步引入顶层 `benchmarks/`、`tests/`、`docs/`、`profiles/` 和 `results/`。

---

## 实验与 Benchmark 原则

每个实验应明确记录：

1. 要回答的问题和假设。
2. 模型、GPU、软件版本、dtype、输入/输出长度和 Batch Size。
3. Warm-up 与正式测量次数。
4. 原始数据、延迟分布、吞吐和显存变化。
5. Profiler 证据和对结果的解释。

性能比较必须遵守：

* 计时前预热 GPU。
* 手工计时时正确同步 CUDA。
* 使用相同模型、输入和生成参数。
* 区分 Prefill 与 Decode。
* 重复测量并报告分布，而不是只报告一次或只给平均值。
* 保存原始结果并记录完整软硬件环境。
* 先分析瓶颈，再优化；先验证正确性，再讨论加速。

---

## 当前工作重点

* [ ] 完成 `compare_greedy_with_hf.py` 与 HF `generate()` 的逐 token 一致性对比
* [ ] 在目标 GPU 上运行并保存可复现的 Phase 1 测量数据
* [ ] 撰写首份 Prefill / TTFT / Decode / TPOT / ITL 分析报告
* [ ] 接入 PyTorch Profiler 对手写解码工作负载进行 Profiling
* [ ] 实现 Temperature / Top-K / Top-P 采样

---

## 本项目不做什么

本仓库不计划：

* 从零训练基础模型
* 手写所有 CUDA Kernel
* 替代 vLLM 或 SGLang
* 在早期阶段支持大型分布式集群
* 在没有正确性验证和分析证据时追逐 Benchmark 数字

长期目标是形成一个小型、清晰、可测量的推理栈：

```text
Model
  ↓
Inference Runtime
  ↓
KV Cache Manager
  ↓
Scheduler / Continuous Batching
  ↓
HTTP Serving / Concurrent Users
  ↓
Profiling / Optimization
```

每项重要优化都应由 **测量、Profiling、实现和解释** 共同支撑。
