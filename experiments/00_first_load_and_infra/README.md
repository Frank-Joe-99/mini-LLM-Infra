# Baseline：首次加载 Qwen3-1.7B

本实验使用 Hugging Face Transformers 在单张 NVIDIA GPU 上加载本地 Qwen3-1.7B 模型，导出模型结构，并完成一次基础文本生成。

## 实验内容

`first-load.py` 会依次完成：

1. 检测当前 CUDA GPU。
2. 在 T4 等 Compute Capability 低于 8.0 的 GPU 上禁用 PyTorch Triton `bmm` override。
3. 使用 Accelerate 在 `meta device` 上构造空模型。
4. 将模型结构和参数统计保存为 `qwen3-1.7b-structure.txt`。
5. 从本地目录加载 Tokenizer 和模型权重。
6. 使用 FP16 在 GPU 上执行一次贪心生成。
7. 在终端打印生成结果。

## 环境要求

- Python 3.12
- NVIDIA GPU
- CUDA 可用的 PyTorch
- Transformers
- Accelerate
- uv

同步项目环境：

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

如果模型位于其他位置，请修改 `first-load.py` 中的：

```python
model_path = Path("/workspace/mini-LLM-Infra/models/Qwen3-1.7B")
```

## 运行实验

在仓库根目录执行：

```bash
uv run python experiments/00_first_load_and_infra/first-load.py
```

## 输出

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

## 常用配置

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

当前脚本中的 `12800` 会导致生成时间很长，首次验证建议使用 `128`。

修改推理精度：

```python
dtype=torch.float16
```

T4 建议使用 FP16；支持原生 BF16 的 Ampere、Ada 或更新 GPU 可以尝试：

```python
dtype=torch.bfloat16
```

## 已知问题

如果遇到：

```text
fatal error: Python.h: No such file or directory
```

说明 Triton 编译环境缺少 Python 开发头文件，可在 Ubuntu/WSL 中安装：

```bash
sudo apt update
sudo apt install -y python3.12-dev build-essential
```

本实验目前仅用于验证模型能否正确加载和生成文本，尚未进行严格的延迟、吞吐量和显存 Benchmark。
