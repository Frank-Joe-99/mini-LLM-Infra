# mini-LLM-infra

[English](./README.md) | **简体中文**

> 从零开始学习并构建一个小型 LLM 推理系统，逐步理解大模型部署、推理、性能分析、推理优化、多用户并发与在线服务中的核心问题。

---

## 项目简介

`mini-LLM-infra` 是一个面向 **LLM Inference / AI Infra** 学习与实践的个人项目。

项目计划以 **1–2B 参数规模的 Decoder-only LLM** 为主要实验对象，从最基础的 PyTorch / Hugging Face 推理开始，逐步深入到：

* 自回归推理过程
* Prefill / Decode
* KV Cache
* GPU 性能分析
* Triton Kernel 优化
* Batch Inference
* Continuous Batching
* KV Cache 管理
* Request Scheduling
* 多用户并发
* 在线推理服务
* 延迟与吞吐优化
* vLLM / SGLang 等工业级推理引擎

项目的核心思路是：

**Build → Measure → Analyze → Optimize → Serve**

即：

**实现 → 测量 → 分析 → 优化 → 服务化**

本项目并不是为了重新实现一个完整的 vLLM，而是希望通过逐步实现一个简单的推理系统，理解现代 LLM 推理引擎背后的设计思想和性能优化方法。

---

## 项目目标

通过这个项目，希望逐步回答以下问题：

* LLM 是如何完成一次完整推理的？
* Prefill 和 Decode 有什么区别？
* KV Cache 为什么能够加速推理？
* LLM 推理的主要 GPU 性能瓶颈在哪里？
* 如何分析 TTFT、TPOT 和吞吐？
* Batch Size 为什么会影响 GPU 利用率？
* 多个用户同时请求时应该如何调度？
* 什么是 Continuous Batching？
* KV Cache 如何进行高效的显存管理？
* 为什么在线服务需要关注 P95 / P99 延迟？
* 如何同时平衡吞吐量与请求延迟？
* vLLM、SGLang 等推理引擎为什么能够获得更高性能？

---

## 技术栈

计划使用：

* Python
* PyTorch
* Hugging Face Transformers
* Triton
* CUDA
* PyTorch Profiler
* NVIDIA Nsight Systems
* NVIDIA Nsight Compute
* vLLM
* SGLang
* FastAPI / HTTP Serving

---

## 学习路线

### Phase 0 — Baseline Inference

建立最基础、可复现的 LLM 推理环境。

* [ ] 模型加载
* [ ] Tokenizer
* [ ] Hugging Face `generate()`
* [ ] 单请求推理
* [ ] GPU 显存统计
* [ ] 基础延迟测试

---

### Phase 1 — Autoregressive Decoding

理解并手动实现 LLM 自回归生成过程。

* [ ] 手写 Decode Loop
* [ ] Greedy Sampling
* [ ] Temperature
* [ ] Top-K / Top-P
* [ ] 区分 Prefill 和 Decode
* [ ] 测量 TTFT / TPOT

---

### Phase 2 — KV Cache

研究 KV Cache 的原理、性能收益和显存开销。

* [ ] 无 KV Cache 推理
* [ ] KV Cache 推理
* [ ] 延迟对比
* [ ] 计算量对比
* [ ] 显存占用分析
* [ ] 不同 Sequence Length 测试

---

### Phase 3 — Profiling

建立完整的性能分析流程。

* [ ] PyTorch Profiler
* [ ] Nsight Systems
* [ ] Nsight Compute
* [ ] Prefill 性能分析
* [ ] Decode 性能分析
* [ ] GPU 利用率分析
* [ ] Memory Bandwidth 分析
* [ ] Roofline 分析

---

### Phase 4 — Kernel Optimization

使用 Triton / CUDA 对关键算子进行实验和优化。

计划包括：

* RMSNorm
* RoPE
* Softmax
* SwiGLU
* Attention
* Sampling

关注：

* Memory Traffic
* Arithmetic Intensity
* Kernel Fusion
* Memory Access
* GPU Occupancy

---

### Phase 5 — Batch Inference

研究 Batch Size 对推理性能的影响。

* [ ] Static Batching
* [ ] Batch Size 测试
* [ ] Throughput
* [ ] Latency
* [ ] GPU Utilization
* [ ] 显存使用

重点理解：

> Throughput 与 Latency 之间的 Trade-off。

---

### Phase 6 — Continuous Batching

构建一个简单的请求调度器。

* [ ] Request Queue
* [ ] Active Batch
* [ ] 动态加入请求
* [ ] 动态移除完成请求
* [ ] 不同 Prompt Length
* [ ] 不同 Output Length
* [ ] Continuous Batching

---

## Status

🚧 **Work in Progress**

这是一个持续学习和迭代中的项目。
