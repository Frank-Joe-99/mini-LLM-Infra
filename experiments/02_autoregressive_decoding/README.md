# 手写自回归解码（Autoregressive Decoding）

`benchmark_manual_decode.py` 不再把 Hugging Face 的 `model.generate()` 当作
黑盒，而是用 PyTorch 手写逐 token 的 Greedy 解码循环，并在每个输出 token
就绪的时刻打点计时，从而把一次推理严格拆分为 **Prefill（首 token）** 与
**Decode（后续 token）** 两段，分别测量 TTFT、TPOT、ITL 和 TOT。

对应项目路线图中的：

* `README.zh-CN.md` → **Phase 1 — Autoregressive Decoding**（以及
  Phase 2 — KV Cache 的开/关对比能力）
* `README.md` → **Phase 1 — Understand Autoregressive Inference**

## 目的

本项目希望逐步回答“LLM 如何完成一次完整推理”。在此之前（Phase 0），
`model.generate()` 仍是黑盒：我们只知道输入和输出，却不知道：

* Prefill 和 Decode 的延迟分别花在哪里？
* TTFT 到底有多长？
* 每生成一个 token 的稳定延迟（TPOT）是多少？
* KV Cache 为什么能减少 Decode 计算？

本实验通过手写解码循环回答上述问题，并为后续阶段（KV Cache 管理、
Batching、调度、Serving）提供可以逐步替换的 Runtime 基础。

## 实现了项目 README 中的哪些内容

### 1. Phase 1：理解自回归推理（核心）

对应 `README.md` 的 “Phase 1 — Understand Autoregressive Inference” 与
`README.zh-CN.md` 的 “Phase 1 — Autoregressive Decoding”：

| README 中的任务 | 实现情况 |
|---|---|
| 手写自回归解码（manual autoregressive decoding） | [x] `manual_generate()` 逐 token 循环 |
| 理解 logits 生成 | [x] 每次取 `logits[:, -1, :]` 作为下一个 token 分布 |
| 实现 Greedy 解码 | [x] `torch.argmax` 选最大概率 token |
| 区分 Prefill 与 Decode | [x] 第一步输入完整 Prompt（Prefill），后续步骤只输入最新 token（Decode） |
| 测量 Prefill 延迟 | [x] 以 TTFT（首 token 就绪时间）衡量 |
| 测量 Decode 延迟 | [x] 以 TPOT / ITL 衡量 |
| 测量逐 token 延迟 | [x] 每个 token-ready 时刻都记录 ITL |
| 计算 TTFT / TPOT / ITL / E2E | [x] 全部输出到 JSON / TXT |

指标实现位于 `src/mini_llm_infra/generation/manual.py` 的
`GenerationMetrics.from_step_latencies()`：

```text
TTFT = 生成开始 -> 第一个输出 token 就绪
ITL  = 相邻两个 token 就绪时刻的间隔
TPOT = ITL 的均值，即 (TOT - TTFT) / (output_tokens - 1)
TOT  = 生成开始 -> 最后一个输出 token 就绪
```

### 2. Phase 2：KV Cache（部分实现）

对应 `README.md` 的 “Phase 2 — KV Cache”：

| README 中的实验 | 实现情况 |
|---|---|
| 无 KV Cache 解码 | [x] `use_cache=False`：每一步重新计算完整历史序列 |
| 有 KV Cache 解码 | [x] `use_cache=True`：第一步全量 Prefill，后续复用 `past_key_values` |
| 对比计算 / 延迟 / 显存 | [x] 同一脚本通过 `--use-cache` / `--no-use-cache` 分别测量，输出 GPU peak allocated/reserved 显存 |
| KV Cache 随序列长度增长的规律 | [ ] 需要后续基于已保存结果分析 |
| MHA / GQA / MQA 缓存布局 | [ ] 未实现 |

### 3. 复用的 Phase 0 基础设施

本实验直接复用并继续遵守 Phase 0 已建立的基准原则
（`README.zh-CN.md`“实验与 Benchmark 原则” / `README.md`
“Benchmarking Principles”）：

* [x] 配置驱动的 `smoke` / `prefill` / `decode` / `mixed` / `full` 测试套件
  （`benchmark_config.json` + `resolve_prompt_cases()`）
* [x] 计时前 Warm-up，计时前后 CUDA 同步
* [x] 延迟分布（mean / P50 / P95）、输出 tokens/s、GPU 显存峰值
* [x] 记录完整软硬件环境与 Tokenizer / 模型加载耗时
* [x] 固定随机种子，保证可复现
* [x] JSON / TXT 结果保存（不做比较分析时不下性能结论）

### 4. 新增的可复用基础设施

```text
src/mini_llm_infra/
├── generation/
│   └── manual.py          # 手写解码核心
│       ├── ManualGenerateConfig    # max_new_tokens / eos / force_exact_output_length / use_cache
│       ├── GenerationMetrics       # TTFT / TPOT / ITL / TOT
│       ├── ManualGenerationResult  # 序列、finish_reason、指标
│       └── manual_generate()       # Greedy 逐 token 推理循环
├── model/
│   └── loader.py          # load_tokenizer / load_causal_lm（支持 eager/sdpa）
└── utils/
    ├── cuda.py            # configure_cuda_compatibility / select_dtype
    └── loadconfig.py      # 配置校验、套件与 Prompt 解析
```

## 本实验尚未实现的内容

以下内容在项目 README 的 Phase 1 / 2 中提到，但本次**没有**实现：

* Temperature / Top-K / Top-P 采样（当前只有 Greedy）
* `compare_greedy_with_hf.py` 目前只是占位文件（仅 `import torch`），
  与 HF `generate()` 的逐 token 一致性对比尚未编写
* 严格独立的 Prefill 与 Decode Kernel 分离（当前是同一个循环：第一轮
  吃完整 Prompt，后续轮只吃新 token）
* 本实验未接入 PyTorch Profiler / Nsight（属于后续 Phase 3）

## 文件结构

```text
experiments/02_autoregressive_decoding/
├── README.md
├── benchmark_manual_decode.py   # 手写解码基准入口
├── benchmark_config.json        # Prompt 与测试套件配置
├── compare_greedy_with_hf.py    # 占位：与 HF generate() 的一致性对比（未完成）
└── results/                     # 运行后生成
```

## 测试套件

| Suite | 用途 | 典型 Prompt | 默认输出 |
|---|---|---|---:|
| `smoke` | 快速验证环境 | 短 | 32 |
| `prefill` | 观察长输入 Prefill（TTFT） | 长/超长 | 1 |
| `decode` | 观察稳态 Decode（TPOT/ITL） | 短 | 512 |
| `mixed` | 完整请求性能 | 中/长 | 512 |
| `full` | 执行全部 12 个 Prompt | 全部长度 | 512 |

Prompt 长度由配置文件中的 `context_repeat` 控制；脚本会记录经过 Qwen3
Chat Template 和 Tokenizer 后的实际输入 token 数，并在输入+输出总长度
超过 `max_position_embeddings` 时直接报错，防止越界生成。

## 运行

在仓库根目录同步环境：

```bash
uv sync
```

快速验证：

```bash
uv run python experiments/02_autoregressive_decoding/benchmark_manual_decode.py \
  --suite smoke
```

Prefill 基准（只生成 1 个 token，主要看 TTFT）：

```bash
uv run python experiments/02_autoregressive_decoding/benchmark_manual_decode.py \
  --suite prefill
```

Decode 基准（默认开启 KV Cache、固定输出长度 512）：

```bash
uv run python experiments/02_autoregressive_decoding/benchmark_manual_decode.py \
  --suite decode
```

关闭 KV Cache 对照（每步重算完整历史，观察 Decode 变慢多少）：

```bash
uv run python experiments/02_autoregressive_decoding/benchmark_manual_decode.py \
  --suite decode \
  --no-use-cache
```

关闭“固定输出长度”开关，让 EOS 提前结束：

```bash
uv run python experiments/02_autoregressive_decoding/benchmark_manual_decode.py \
  --suite decode \
  --no-force-exact-output-length
```

其他常用参数：`--prompt-id`（可多次传入）、`--max-new-tokens`、
`--warmup-runs`、`--benchmark-runs`、`--dtype`、`--attention-backend`、
`--seed`。

## 输出

每次执行在 `results/` 生成两个同名结果文件：

```text
results/manual_decode_<suite>_<timestamp>.json
results/manual_decode_<suite>_<timestamp>.txt
```

TXT 报告包含：运行环境、配置、每个 Prompt 的 TTFT/TPOT/ITL/TOT
（mean/P50/P95）、输出速度、显存峰值，以及最后一轮生成文本。
JSON 保存全部原始测量值，适合后续绘图与自动比较。

## 计时口径与注意事项

* 计时范围只包含 `manual_generate()` 内部的生成循环（含首个 Prefill
  Forward），不包含模型加载、Tokenization、Warm-up 和结果序列化。
* 每个 token-ready 点都会做 CUDA 同步，确保异步 kernel 完成后才记录时间，
  因此 TTFT / ITL 是“token 已就绪”的真实延迟。
* 显存变化通过每次测量前 `reset_peak_memory_stats` + 结束后
  `max_memory_allocated` 得到，`generation_peak_delta_mib` 与测量前基线比较。
* `force_exact_output_length` 开启时忽略 EOS，始终生成满
  `max_new_tokens` 个 token，保证各轮输出长度一致，适合性能测试；
  关闭时才报告 `finish_reason = "eos" / "length"`。
