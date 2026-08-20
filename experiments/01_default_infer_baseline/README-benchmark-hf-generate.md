# Qwen3 Hugging Face 原生推理基线

`benchmark_hf_generate.py` 用于测量 Qwen3-1.7B 在 Hugging Face
`model.generate()` 路径下的单请求推理性能，并使用 `torch.profiler`
生成算子统计和 CUDA 时间线。

本实验中的“原生基线”表示：

- 单张 NVIDIA GPU、batch size 1；
- Hugging Face `model.generate()`；
- 默认使用 eager attention；
- 不使用 `torch.compile`、FlashAttention、量化或自定义 Kernel；
- 默认启用 Hugging Face 标准 KV Cache。

PyTorch、cuBLAS 和 CUDA 自身的基础优化不会被关闭。

## 文件

```text
experiments/01_default_infer_baseline/
├── benchmark_hf_generate.py
├── benchmark_config.json
├── results/
└── profiles/
```

`benchmark_config.json` 集中保存 Prompt、测试套件、输出长度、预热次数和
测量次数。脚本不再硬编码测试 Prompt。

## 环境

在仓库根目录同步环境：

```bash
uv sync
```

确认 CUDA 可用：

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

默认模型目录为：

```text
/workspace/mini-LLM-Infra/models/Qwen3-1.7B
```

可以通过 `--model-path` 覆盖。

## 测试套件

| Suite | 用途 | 典型输入 | 默认输出 |
|---|---|---|---:|
| `smoke` | 快速验证环境 | 短 Prompt | 32 |
| `prefill` | 观察长输入 Prefill | 长/超长 Prompt | 1 |
| `decode` | 观察稳态 Decode | 短 Prompt | 512 |
| `mixed` | 完整请求性能 | 中长 Prompt | 512 |
| `full` | 执行全部 Prompt | 全部长度 | 512 |

Prompt 长度由配置文件中的 `context_repeat` 控制。脚本会记录经过 Qwen3
Chat Template 和 Tokenizer 后的实际输入 token 数。

## 运行

快速验证：

```bash
uv run python experiments/01_default_infer_baseline/benchmark_hf_generate.py \
  --suite smoke \
  --no-profile
```

Decode 基准：

```bash
uv run python experiments/01_default_infer_baseline/benchmark_hf_generate.py \
  --suite decode \
  --no-profile
```

Prefill 基准：

```bash
uv run python experiments/01_default_infer_baseline/benchmark_hf_generate.py \
  --suite prefill \
  --no-profile
```

混合负载并运行 Profiler：

```bash
uv run python experiments/01_default_infer_baseline/benchmark_hf_generate.py \
  --suite mixed \
  --profile \
  --profile-prompt-id medium_generation \
  --profile-new-tokens 16
```

把 Decode 输出长度提高到 1024：

```bash
uv run python experiments/01_default_infer_baseline/benchmark_hf_generate.py \
  --suite decode \
  --max-new-tokens 1024 \
  --no-profile
```

只运行指定 Prompt：

```bash
uv run python experiments/01_default_infer_baseline/benchmark_hf_generate.py \
  --suite full \
  --prompt-id short_kv_cache \
  --prompt-id long_attention \
  --no-profile
```

关闭 KV Cache 进行对照：

```bash
uv run python experiments/01_default_infer_baseline/benchmark_hf_generate.py \
  --suite decode \
  --no-use-cache \
  --no-profile
```

在原生支持 BF16 的 GPU 上测试 BF16：

```bash
uv run python experiments/01_default_infer_baseline/benchmark_hf_generate.py \
  --suite decode \
  --dtype bfloat16 \
  --no-profile
```

## 输出

每次执行都会在 `results/` 中生成两个同名结果文件：

```text
results/qwen3_baseline_<suite>_<timestamp>.json
results/qwen3_baseline_<suite>_<timestamp>.txt
```

TXT 报告包含：

- 运行环境和 GPU 信息；
- 每个 Prompt 的实际输入/输出 token 数；
- 每轮延迟、输出 tokens/s 和平均 ms/token；
- P50/P95 延迟；
- GPU peak allocated/reserved memory；
- 整个测试套件的加权输出速度；
- 最后一轮生成文本。

JSON 保存相同指标及所有原始测量值，适合后续绘图和自动比较。

启用 Profiler 后还会生成：

```text
profiles/qwen3_<prompt-id>_<timestamp>_trace.json
profiles/qwen3_<prompt-id>_<timestamp>_ops.txt
```

Trace 可以上传到 [Perfetto](https://ui.perfetto.dev/) 查看 CPU/CUDA 时间线。

## 指标解释

脚本计时范围只包含 `model.generate()`，不包含：

- 模型与 Tokenizer 加载；
- Prompt 格式化和 Tokenization；
- CPU 到 GPU 的输入传输；
- Warm-up；
- Profiler 运行。

`output_tokens_per_second` 的计算方式是：

```text
实际输出 token 数 / model.generate() 总耗时
```

该指标包含 Prefill 和全部 Decode，因此不是纯 Decode throughput。
`average_ms_per_output_token` 同样包含 Prefill 摊销。

`prefill` 套件只生成一个 token，主要观察长输入请求的整体首步延迟；严格的
TTFT、TPOT 和 ITL 测量需要在后续阶段手写 Prefill/Decode 循环。

## 固定输出长度

Decode、Mixed 和 Full 套件默认令：

```text
min_new_tokens = max_new_tokens
```

这样模型不会因为提前产生 EOS 而导致各轮输出长度不同。它适合性能测试，
但可能让模型在自然回答结束后继续生成，因此不能用于评价回答质量。

## Profiler 注意事项

1. 不要把 Profiler 中测得的时间作为正式 tokens/s。`record_shapes` 和
   `profile_memory` 会带来额外开销。
2. Profiler 默认只生成 16 个 token。直接记录 512 或 1024 个 Decode step
   会产生很大的 Trace 文件。
3. 正式计时前需要 Warm-up，并在计时前后执行 `torch.cuda.synchronize()`。
4. `self_cuda_time_total` 表示算子自身 GPU 时间；`cuda_time_total` 还包含
   子算子时间。
5. `model.generate()` 会把 Prefill 和 Decode 混在同一条 Trace 中。当前可以
   通过输入形状区分首次长序列 Forward 与后续单 token Forward，但后续最好
   使用手写解码循环并添加 `record_function("prefill")` 和
   `record_function("decode_step")`。
6. `torch.profiler` 适合分析 PyTorch 算子；GPU 空闲、Kernel launch 间隙和
   系统级时间线后续应使用 Nsight Systems，单 Kernel 的带宽和 occupancy
   应使用 Nsight Compute。

## T4 兼容处理

T4 的 Compute Capability 为 7.5。当前 PyTorch 可能把部分 `bmm` 调用路由
到 Triton，而当前官方 Triton 不支持该架构。脚本会在 Compute Capability
低于 8.0 时禁用这个 `bmm` override，使其回退到 ATen/CUDA 实现。

T4 默认使用 FP16。RTX 4060 Laptop 等原生支持 BF16 的 GPU 可以通过
`--dtype bfloat16` 测试 BF16。
