# Baseline：首次加载 Qwen3-1.7B 与 ONNX 导出

本目录属于项目路线图的 Phase 0（Baseline）。目标是在动手写自己的推理
Runtime 之前，先把 Hugging Face 生态的「加载 → 生成」链路当作外部依赖跑通，
并验证「模型能否被导出为标准 ONNX 计算图」，为后续阶段提供对照基线与可复用
素材。

## 目录文件

| 文件 | 作用 |
|---|---|
| `first-load.py` | 在 GPU 上加载本地 Qwen3-1.7B，导出模型结构，完成一次基础文本生成 |
| `hf-tutorial-load.py` | 官方 ChatML 风格示例：开启 Thinking 模式，拆分「思考内容」与「正文」 |
| `onnx-model-flow.py` | 将 Qwen3-1.7B 导出为 ONNX 计算图 |
| `qwen3-1.7b-structure.txt` | `first-load.py` 生成的模型结构与参数统计 |
| `qwen3-1.7b.onnx` / `qwen3-1.7b.onnx.data` | `onnx-model-flow.py` 生成的 ONNX 图与外部权重文件 |
| `README.md` | 本说明 |

## 环境要求

- Python 3.12
- NVIDIA GPU（`first-load.py` / `hf-tutorial-load.py` 需要；`onnx-model-flow.py` 在 CPU 上即可完成导出）
- CUDA 可用的 PyTorch
- Transformers、Accelerate
- onnx、onnxscript（新版 ONNX 导出器依赖）
- uv

同步项目环境（`onnx` / `onnxscript` 已声明在 `pyproject.toml`）：

```bash
uv sync
```

检查 CUDA：

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 模型目录

脚本默认从以下目录读取模型：

```text
/workspace/mini-LLM-Infra/models/Qwen3-1.7B
```

本地模型目录应至少包含：

```text
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
model.safetensors
```

如果模型位于其他位置，请修改各脚本中的：

```python
model_path = project_root / "models" / "Qwen3-1.7B"
```

## first-load.py：加载 + 基础生成

### 实验内容

`first-load.py` 会依次完成：

1. 检测当前 CUDA GPU。
2. 在 T4 等 Compute Capability 低于 8.0 的 GPU 上禁用 PyTorch Triton `bmm` override。
3. 使用 Accelerate 在 `meta device` 上构造空模型。
4. 将模型结构和参数统计保存为 `qwen3-1.7b-structure.txt`。
5. 从本地目录加载 Tokenizer 和模型权重。
6. 使用 FP16 在 GPU 上执行一次贪心生成。
7. 在终端打印生成结果。

### 运行

在仓库根目录执行：

```bash
uv run python experiments/00_first_load_and_infra/first-load.py
```

### 输出

模型结构会保存到：

```text
experiments/00_first_load_and_infra/qwen3-1.7b-structure.txt
```

该文件包含：

- 模型类型
- 模型类名
- 总参数量
- PyTorch 模块结构
- 展开的模块树
- 各模块直接持有的参数量

生成的文本会直接打印到终端。

### 常用配置

修改输入问题：

```python
messages = [
    {"role": "user", "content": "请简要解释什么是 KV Cache。"},
]
```

修改最大输出长度：

```python
max_new_tokens=128
```

脚本中的 `12800` 会导致生成时间很长，首次验证建议使用 `128`。

修改推理精度：

```python
dtype=torch.float16
```

T4 建议使用 FP16；支持原生 BF16 的 Ampere、Ada 或更新 GPU 可以尝试：

```python
dtype=torch.bfloat16
```

## hf-tutorial-load.py：Thinking 模式示例

`hf-tutorial-load.py` 是 Transformers 官方教程风格的示例：加载模型后使用
`enable_thinking=True` 构造 ChatML 输入，调用 `model.generate` 完成生成，
再按 Qwen3 的 `</think>` 结束符（token 151668）把输出拆成两部分：

```text
thinking content:  …（模型的思考过程）
content:           …（最终回答正文）
```

运行：

```bash
uv run python experiments/00_first_load_and_infra/hf-tutorial-load.py
```

注意该脚本的 `max_new_tokens=32768` 会生成非常久，快速验证时可调小。
`torch_dtype="auto"` 会按模型配置自动选择精度（Qwen3-1.7B 默认 BF16）。

## onnx-model-flow.py：导出 ONNX

### 目的

后续要在自己的推理引擎上运行模型，因此先验证「同一个模型能否被 PyTorch
生态导出为标准 ONNX 图」，得到 `qwen3-1.7b.onnx` 作为后续部署 / 校验的素材；
这一步也会第一次暴露出动态形状、算子覆盖（opset coverage）等真实问题。

### 流程

`onnx-model-flow.py` 会依次完成：

1. 使用 FP32 在 CPU 上加载模型（导出不需要 GPU，内存足够即可）。
2. `model.eval()` 并将 `model.config.use_cache = False`：KV Cache 属于自回归
   运行期的状态，不属于静态计算图，导出阶段先关闭。
3. 随机构造 `shape=(1, 16)` 的 `input_ids`，作为让 `torch.onnx.export`
   走一遍计算图的示例输入。
4. 调用 `torch.onnx.export(..., dynamo=True)` 导出：

   ```python
   torch.onnx.export(
       model,
       (input_ids,),
       onnx_output_path,
       input_names=["input_ids"],
       dynamo=True,
   )
   ```

   `dynamo=True` 走的是 PyTorch 2.6+ 引入的 TorchDynamo 导出器（也是新版
   默认路径），它依赖 `onnx` 与 `onnxscript` 两个包。
5. 打印导出完成信息。

### 运行

在仓库根目录执行：

```bash
uv run python experiments/00_first_load_and_infra/onnx-model-flow.py
```

### 输出

ONNX 文件会写到脚本所在目录：

```text
experiments/00_first_load_and_infra/
├── qwen3-1.7b.onnx         # ONNX 图（protobuf）
└── qwen3-1.7b.onnx.data    # 外部权重（超过 2GB 时自动拆分）
```

主文件描述计算图结构；FP32 权重（约 1.7B 参数）存放在外部数据文件 `.data`
中，导出/使用这两个文件需保持同一目录。导出需要几 GB 磁盘空间，CPU 上完成
一次导出需要数分钟到更久，属正常现象。

### 查看导出的 ONNX 文件

使用 Netron 可视化计算图结构（算子、张量形状、数据流）：

- 在线打开：<https://netron.app/>，把 `qwen3-1.7b.onnx` 拖入页面即可；
- 或本地安装后启动（浏览器访问 <http://localhost:8080>）：

  ```bash
  uv pip install netron
  uv run netron experiments/00_first_load_and_infra/qwen3-1.7b.onnx
  ```

提示：ONNX 文件很大且权重存放在 `.data` 外部文件中，用 Netron 查看时
需要把 `qwen3-1.7b.onnx` 与 `qwen3-1.7b.onnx.data` 放在同一目录下。

### 已知问题与排查

1. `ModuleNotFoundError: No module named 'onnxscript'`

   新版 Dynamo 导出器依赖 `onnxscript`（以及 `onnx`）。这两个包已声明在
   `pyproject.toml`（`onnx>=1.22.0`、`onnxscript>=0.7.1`），执行前先同步环境：

   ```bash
   uv sync
   ```

   确认安装成功：

   ```bash
   uv run python -c "import onnx, onnxscript; print(onnx.__version__, onnxscript.__version__)"
   ```

2. `[transformers] torch_dtype is deprecated! Use dtype instead!`

   Transformers ≥ 4.51 中 `from_pretrained(..., dtype=...)` 取代了
   `torch_dtype`，脚本已使用新参数。旧写法仍可用但会打印该告警。

3. 示例输入长度固定为 16

   它只是为了让导出器走通一次计算图，不代表模型只能处理 16 个 token；
   实际部署时的序列长度由推理引擎 / ORT 的输入动态形状决定。

## 已知问题（加载环境）

如果遇到：

```text
fatal error: Python.h: No such file or directory
```

说明 Triton 编译环境缺少 Python 开发头文件，可在 Ubuntu/WSL 中安装：

```bash
sudo apt update
sudo apt install -y python3.12-dev build-essential
```

本目录实验仅用于验证模型能否正确加载、生成文本以及能否导出为 ONNX，
尚未进行严格的延迟、吞吐量和显存 Benchmark（那是 `01`、`02` 等实验的内容）。
