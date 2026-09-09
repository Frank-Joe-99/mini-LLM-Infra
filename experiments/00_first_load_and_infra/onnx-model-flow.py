import torch
from transformers import AutoModelForCausalLM
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "models" / "Qwen3-1.7B"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
)

model.eval()
model.config.use_cache = False

# 随便构造一个输入，只是为了让 torch.onnx 知道计算图怎么走
input_ids = torch.randint(
    low=0,
    high=model.config.vocab_size,
    size=(1, 16),
    dtype=torch.long,
)

# dynamo=True 走 PyTorch 2.6+ 的新版 ONNX 导出器，依赖 onnx + onnxscript，
# 这两个包需在 pyproject.toml 中声明（uv add onnx onnxscript）。
# 输出固定写到脚本所在目录，避免在任意 CWD 下生成散落文件。
onnx_output_path = "/workspace/experiments/00_first_load_and_infra/qwen3-1.7b.onnx"

torch.onnx.export(
    model,
    (input_ids,),
    onnx_output_path,
    input_names=["input_ids"],
    dynamo=True,
)

print(f"Export finished: {onnx_output_path}")