from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 在 T4 上禁用 PyTorch 2.13 新增的 bmm Triton override，让它回退到传统 ATen/CUDA 实现。
if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()

    if major < 8:
        import torch._native
        from torch._native.registry import deregister_op_overrides

        deregister_op_overrides(disable_op_symbols="bmm")


# 用 Accelerate 的 init_empty_weights() 在 meta device 上构造模型结构，
# 不读取真实权重，然后把结构保存到脚本所在的 experiments/00_baseline/ 目录。
def save_model_structure(model_path: Path) -> Path:
    """构造空模型并将模块结构保存到当前脚本目录。"""

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
    )

    # 参数只创建在 meta device，不读取真实权重、不占用 GPU 显存。
    with init_empty_weights(include_buffers=True):
        structure_model = AutoModelForCausalLM.from_config(config)

    total_parameters = sum(
        parameter.numel()
        for parameter in structure_model.parameters()
    )

    lines = [
        f"Model path: {model_path}",
        f"Model type: {config.model_type}",
        f"Model class: {structure_model.__class__.__name__}",
        f"Total parameters: {total_parameters:,}",
        "",
        "=" * 80,
        "PYTORCH MODULE REPRESENTATION",
        "=" * 80,
        repr(structure_model),
        "",
        "=" * 80,
        "FLATTENED MODULE TREE",
        "=" * 80,
    ]

    # named_modules() 会把每一层完整列出来，避免 ModuleList 折叠显示。
    for module_name, module in structure_model.named_modules():
        display_name = module_name or "<root>"

        direct_parameters = sum(
            parameter.numel()
            for parameter in module.parameters(recurse=False)
        )

        lines.append(
            f"{display_name:<70} "
            f"{module.__class__.__name__:<30} "
            f"direct_parameters={direct_parameters:,}"
        )

    structure_text = "\n".join(lines)

    # 保存到 first-load.py 所在目录。
    output_path = (
        Path(__file__).resolve().parent
        / "qwen3-1.7b-structure.txt"
    )

    output_path.write_text(
        structure_text,
        encoding="utf-8",
    )

    print(structure_text)
    print(f"\nModel structure saved to: {output_path}")

    return output_path

model_path = Path("/workspace/mini-LLM-Infra/models/Qwen3-1.7B")

save_model_structure(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    local_files_only=True,
).to("cuda").eval()

messages = [
    {"role": "user", "content": "请简要解释什么是 KV Cache。"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=12800,
        do_sample=False,
    )

generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated_ids, skip_special_tokens=True))