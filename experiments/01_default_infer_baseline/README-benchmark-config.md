# `benchmark_config.json` 配置说明

`benchmark_config.json` 是 Qwen3 Hugging Face 基线实验的统一工作负载配置文件，
用于管理：

- 测试使用哪些 Prompt；
- Prompt 的相对长度；
- 每组测试生成多少 token；
- 预热和正式测量次数；
- 是否启用 KV Cache；
- 是否强制生成固定数量的输出 token；
- Profiler 默认记录多少个输出 token。

把工作负载放在配置文件中，可以避免在 Python 脚本中硬编码 Prompt，也方便在
不同 GPU、不同实现和不同优化版本之间重复使用完全相同的测试输入。

## 配置层级

配置文件包含四个顶层字段：

```json
{
  "version": 1,
  "defaults": {},
  "suites": {},
  "prompts": []
}
```

它们分别表示：

| 字段 | 类型 | 含义 |
|---|---|---|
| `version` | 整数 | 配置文件格式版本，便于未来调整结构 |
| `defaults` | 对象 | 所有测试套件共享的默认参数 |
| `suites` | 对象 | 按测试目标组织的工作负载集合 |
| `prompts` | 数组 | 具体 Prompt 内容及长度扩展方式 |

参数优先级从高到低为：

```text
命令行参数 > suite 中的参数 > defaults 中的参数
```

例如：

```bash
uv run python experiments/00_baseline/benchmark_hf_generate.py \
  --suite decode \
  --max-new-tokens 1024
```

即使 `decode` 在配置文件中设置为 512，命令行中的 1024 仍会覆盖它。

## `defaults`：全局默认设置

当前配置为：

```json
"defaults": {
  "warmup_runs": 1,
  "benchmark_runs": 5,
  "use_cache": true,
  "force_exact_output_length": true,
  "profile_new_tokens": 16
}
```

### `warmup_runs`

每个 Prompt 在正式计时前执行的预热次数。

```json
"warmup_runs": 1
```

预热可以触发：

- CUDA Context 初始化；
- CUDA Kernel 首次加载；
- PyTorch 内部缓存初始化；
- 显存分配器缓存；
- 可能存在的 JIT 或 Kernel 选择过程。

预热结果不会写入正式的延迟和 tokens/s 统计。

正式 Benchmark 建议至少预热 1～3 次。如果一次生成长度较长，预热本身也会耗费
较多时间，可以保留 1 次。

### `benchmark_runs`

每个 Prompt 正式测量的次数。

```json
"benchmark_runs": 5
```

脚本会保留每一轮的原始数据，并计算：

- 平均延迟；
- P50 延迟；
- P95 延迟；
- 平均输出 tokens/s；
- 平均 ms/token；
- 显存峰值。

5 次适合开发阶段快速比较。正式报告建议增加到 10～30 次，尤其是分析 P95/P99
时。只有 5 次样本时，P95 只能作为初步参考。

可以通过命令行覆盖：

```bash
--benchmark-runs 20
```

### `use_cache`

是否在 `model.generate()` 中启用标准 KV Cache。

```json
"use_cache": true
```

启用时，Decode 阶段会复用历史 token 的 Key/Value，避免每一步重新计算完整历史
序列。这是 Hugging Face 自回归生成的标准行为，因此默认基线保持开启。

关闭 KV Cache 可以用于对照实验：

```bash
--no-use-cache
```

关闭后通常会出现：

- Decode 延迟明显增加；
- 输出越长，速度下降越明显；
- KV Cache 显存消失或减少；
- 每一步重新计算历史 token。

### `force_exact_output_length`

是否强制每轮生成相同数量的输出 token。

```json
"force_exact_output_length": true
```

开启后，脚本会同时设置：

```text
min_new_tokens = max_new_tokens
max_new_tokens = max_new_tokens
```

这样模型即使提前生成 EOS，也会继续生成到指定长度，避免不同测试轮次因为输出
token 数不同而无法公平比较。

它适合性能测试，但不适合评价回答质量，因为模型可能在自然回答结束后继续生成。

`smoke` 和 `prefill` 套件会覆盖这个设置并关闭固定输出长度。

### `profile_new_tokens`

Profiler 单独运行时默认生成的 token 数。

```json
"profile_new_tokens": 16
```

这个参数不影响正式 Benchmark 的输出长度。它只影响 `torch.profiler` 那一次运行。

Profiler 会为大量 CPU 算子、CUDA Runtime 调用和 CUDA Kernel 保存事件。如果直接
Profile 512 或 1024 个 Decode step，Trace 可能非常大，Profiler 开销也会严重干扰
时间线。因此默认只记录 16 个输出 token。

## `suites`：测试套件

Suite 是一组针对同一测试目标组织的 Prompt 和生成参数。

运行时通过以下参数选择：

```bash
--suite <suite-name>
```

当前包含五个 Suite：

| Suite | 主要目的 | 输入特征 | 默认输出长度 |
|---|---|---|---:|
| `smoke` | 验证环境和代码能否正常执行 | 单个短 Prompt | 32 |
| `prefill` | 观察长输入 Prefill 成本 | 长/超长 Prompt | 1 |
| `decode` | 观察稳态 Decode 性能 | 多个短 Prompt | 512 |
| `mixed` | 观察完整请求的综合性能 | 中等到长 Prompt | 512 |
| `full` | 执行配置中的全部 Prompt | 全部长度 | 512 |

### `prompt_ids`

`prompt_ids` 决定当前 Suite 使用哪些 Prompt：

```json
"prompt_ids": [
  "short_kv_cache",
  "short_prefill_decode"
]
```

这里的每个字符串必须与 `prompts` 数组中某个对象的 `id` 完全一致。

特殊写法：

```json
"prompt_ids": ["*"]
```

表示使用 `prompts` 中定义的全部 Prompt。当前只有 `full` Suite 使用这个写法。

如果配置了不存在的 ID，脚本会在加载配置时报告错误，而不是静默跳过。

### `max_new_tokens`

限制每个请求最多生成多少个新 token，不包括输入 token。

例如：

```json
"max_new_tokens": 512
```

需要同时满足：

```text
实际输入 token 数 + max_new_tokens <= model.config.max_position_embeddings
```

脚本会在运行前检查这个限制。

### `smoke`

```json
"smoke": {
  "prompt_ids": ["short_kv_cache"],
  "max_new_tokens": 32,
  "force_exact_output_length": false
}
```

用途：

- 确认模型和 Tokenizer 能加载；
- 确认 CUDA 可用；
- 确认生成逻辑和结果文件正常；
- 快速发现依赖、模型路径或 T4 兼容问题。

它不是正式性能测试。首次修改代码后应优先运行这个 Suite。

### `prefill`

```json
"prefill": {
  "prompt_ids": [
    "long_attention",
    "long_profiler",
    "xlong_architecture",
    "xlong_concurrency"
  ],
  "max_new_tokens": 1,
  "force_exact_output_length": false
}
```

用途：观察输入长度增长对首次 Forward 和首个输出 token 的影响。

选择长 Prompt，同时只生成 1 个 token，是为了让总耗时主要由长输入的 Prefill
主导，避免数百步 Decode 掩盖 Prefill 差异。

注意：当前脚本测量的是 `model.generate()` 总时间，因此这里得到的是接近 TTFT 的
整体指标，但仍包含生成框架开销和第一个 token 的采样过程，不是严格定义的纯
Prefill Kernel 时间。

### `decode`

```json
"decode": {
  "prompt_ids": [
    "short_kv_cache",
    "short_prefill_decode",
    "short_attention"
  ],
  "max_new_tokens": 512,
  "force_exact_output_length": true
}
```

用途：观察较长 Decode 过程中的稳态性能。

使用短 Prompt 可以降低 Prefill 在总时间中的占比，512 个固定输出 token 可以让：

- KV Cache 持续增长；
- Attention 读取的历史序列持续变长；
- Decode 的平均 tokens/s 和 ms/token 更稳定；
- 不同 Decode Attention、Kernel、KV Cache 布局的差异更明显。

该 Suite 适合后续比较：

- eager Attention 与其他 Attention backend；
- 开启/关闭 KV Cache；
- FP16 与 BF16；
- PyTorch 原生实现与自定义 Decode 实现。

### `mixed`

```json
"mixed": {
  "prompt_ids": [
    "medium_generation",
    "medium_scheduler",
    "medium_cache",
    "long_attention",
    "long_serving"
  ],
  "max_new_tokens": 512,
  "force_exact_output_length": true
}
```

用途：测试更接近真实请求的“中长输入 + 长输出”。

该 Suite 同时包含明显的 Prefill 和 Decode 成本，适合比较完整请求的：

- 总延迟；
- 输出 tokens/s；
- 显存峰值；
- 输入长度对整体吞吐的影响。

它不适合单独证明某个 Prefill 或 Decode 优化有效，因为两个阶段的时间混合在一起。

### `full`

```json
"full": {
  "prompt_ids": ["*"],
  "max_new_tokens": 512,
  "force_exact_output_length": true
}
```

用途：执行全部 12 个 Prompt，得到覆盖多种输入长度的完整数据集。

由于每个 Prompt 都会预热并正式运行多次，`full` 在 T4 上可能运行很长时间。
开发阶段建议先使用 `smoke`、`prefill`、`decode` 或 `mixed`，准备生成正式报告时
再执行 `full`。

## `prompts`：Prompt 定义

每个 Prompt 对象的格式为：

```json
{
  "id": "short_kv_cache",
  "length_group": "short",
  "prompt": "请解释什么是 KV Cache。",
  "context": "",
  "context_repeat": 0
}
```

### `id`

Prompt 的唯一标识符，用于：

- 在 Suite 的 `prompt_ids` 中引用；
- 通过 `--prompt-id` 单独选择；
- 标记 JSON/TXT 结果；
- 命名 Profiler 输出文件。

建议只使用小写英文字母、数字和下划线，并保持名称稳定。修改已有 ID 会影响历史
结果之间的对应关系。

### `length_group`

人工定义的相对长度分组：

```text
short < medium < long < xlong
```

它用于结果分类，不会直接控制 Tokenizer，也不保证固定 token 数。实际长度由：

- `prompt` 内容；
- `context` 内容；
- `context_repeat`；
- Qwen3 Chat Template；
- Qwen3 Tokenizer

共同决定。

正式分析必须使用结果文件中的 `input_tokens`，不能只根据 `length_group` 推断长度。

### `prompt`

发送给模型的主要问题或任务说明。

```json
"prompt": "请解释什么是 KV Cache。"
```

### `context`

用于扩展输入长度的背景材料。

```json
"context": "自回归语言模型根据已有 token 计算隐藏状态……"
```

脚本会把背景材料放在主要问题之后，并添加“背景资料”标题。

### `context_repeat`

背景材料重复次数：

```json
"context_repeat": 8
```

脚本的等价逻辑为：

```python
repeated_context = "\n".join(
    context for _ in range(context_repeat)
)
```

这个字段用于生成不同长度的合成输入，避免在 JSON 中保存几千字的重复文本。

它不会产生精确的 128、512 或 2048 token。如果后续需要严格控制输入 token 数，
应增加专门的 token-length workload 生成器，而不是继续依赖重复次数。

## 每个 Prompt ID 的含义

### Short Prompt

| Prompt ID | 测试内容 | 主要用途 |
|---|---|---|
| `short_kv_cache` | 询问 KV Cache 的定义 | 最小 Smoke Test；短输入 Decode 基准 |
| `short_prefill_decode` | 询问 Prefill 与 Decode 区别 | 短输入 Decode 基准；检查模型基础推理知识 |
| `short_attention` | 询问自注意力计算过程 | 短输入 Attention/Decode 基准 |

这三条 Prompt 的输入很短，主要用于降低 Prefill 占比。它们共同组成 `decode`
Suite，适合测量长输出时的稳态 Decode 性能。

### Medium Prompt

| Prompt ID | 测试内容 | 主要用途 |
|---|---|---|
| `medium_generation` | 根据背景解释一个 token 的生成过程 | 中等输入的完整生成路径测试 |
| `medium_scheduler` | 根据背景设计请求调度器 | 更长输入、结构化回答和整体延迟测试 |
| `medium_cache` | 分析多请求下 KV Cache 显存增长 | 中等输入下的 KV Cache 与长输出测试 |

它们通过 2、4、6 次背景重复逐步增加输入长度，用于观察输入长度增长对完整请求
延迟和显存的影响。

### Long Prompt

| Prompt ID | 测试内容 | 主要用途 |
|---|---|---|
| `long_attention` | 对比 Prefill/Decode Attention 特征 | 长输入 Prefill；Attention 分阶段优化测试 |
| `long_profiler` | 设计 PyTorch Profiler 分析方案 | 长输入算子和时间线分析 |
| `long_serving` | 设计在线 LLM 推理服务 | 长输入 + 长输出 Mixed workload |

`long_attention` 和 `long_profiler` 被放入 `prefill` Suite，因为它们有明显的长输入。
`long_serving` 被放入 `mixed` Suite，用于观察较真实的完整请求。

### Extra-long Prompt

| Prompt ID | 测试内容 | 主要用途 |
|---|---|---|
| `xlong_architecture` | 设计完整 LLM Runtime 架构 | 超长输入 Prefill 和显存测试 |
| `xlong_concurrency` | 分析多用户并发和尾延迟 | 超长输入 Prefill；并发主题工作负载 |
| `xlong_engine_comparison` | 设计多个推理引擎的公平对比 | Full Suite 的超长输入综合测试 |

这些 Prompt 使用 24、32、48 次背景重复，主要用于放大输入序列长度对 Prefill 的
影响。它们不是为了评估模型回答质量，而是为了构造稳定、可重复的长输入。

## 如何选择测试套件

### 修改代码后快速检查

```bash
uv run python experiments/00_baseline/benchmark_hf_generate.py \
  --suite smoke \
  --no-profile
```

### 比较 Prefill Attention

```bash
uv run python experiments/00_baseline/benchmark_hf_generate.py \
  --suite prefill \
  --benchmark-runs 10 \
  --no-profile
```

### 比较 Decode Attention 或 KV Cache

```bash
uv run python experiments/00_baseline/benchmark_hf_generate.py \
  --suite decode \
  --max-new-tokens 512 \
  --benchmark-runs 10 \
  --no-profile
```

### 测试长 Decode

```bash
uv run python experiments/00_baseline/benchmark_hf_generate.py \
  --suite decode \
  --max-new-tokens 1024 \
  --benchmark-runs 10 \
  --no-profile
```

### 只运行一个 Prompt

```bash
uv run python experiments/00_baseline/benchmark_hf_generate.py \
  --suite full \
  --prompt-id long_attention \
  --no-profile
```

`--prompt-id` 可以传入多次：

```bash
--prompt-id short_attention \
--prompt-id long_attention
```

## 如何添加新 Prompt

在 `prompts` 数组中添加一个具有唯一 `id` 的对象：

```json
{
  "id": "long_rope_analysis",
  "length_group": "long",
  "prompt": "分析 RoPE 在长上下文中的作用。",
  "context": "RoPE 使用旋转位置编码把位置信息加入 Query 和 Key。",
  "context_repeat": 16
}
```

然后把 ID 添加到需要的 Suite：

```json
"prompt_ids": [
  "long_attention",
  "long_rope_analysis"
]
```

修改后建议先执行：

```bash
uv run python experiments/00_baseline/benchmark_hf_generate.py \
  --suite smoke \
  --prompt-id long_rope_analysis \
  --max-new-tokens 8 \
  --no-profile
```

## 配置注意事项

1. JSON 不支持注释，也不允许最后一个字段后带多余逗号。
2. `prompt_ids` 必须引用存在的 Prompt `id`，`id` 必须唯一。
3. `context_repeat` 只控制相对长度，正式实验应以 `input_tokens` 为准。
4. 输入 token 数加输出 token 数不能超过模型最大上下文长度。
5. `force_exact_output_length=true` 适合性能测试，不适合生成质量测试。
6. 不要用 Profiler 的运行时间报告正式 tokens/s。
7. 比较两个实现时必须使用同一个配置文件、Suite、dtype、GPU、KV Cache 设置和
   Benchmark 次数。
8. Laptop GPU 会受到功耗和温度影响，运行长 Suite 时应保持插电和固定性能模式。
9. `full` Suite 成本很高。先使用小 Suite 验证，再运行完整测试。
10. 当前 Prompt 是相对长度工作负载。严格的固定 token 长度矩阵应作为后续独立
    Benchmark 实现。
