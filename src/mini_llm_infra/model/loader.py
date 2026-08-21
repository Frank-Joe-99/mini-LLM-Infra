from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from accelerate import init_empty_weights


def validate_model_path(model_path: Path) -> Path:
    """检查并规范化本地模型目录。"""

    resolved_model_path = model_path.expanduser().resolve()

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"模型目录不存在：{resolved_model_path}"
        )

    if not resolved_model_path.is_dir():
        raise NotADirectoryError(
            f"模型路径不是目录：{resolved_model_path}"
        )

    return resolved_model_path


def load_tokenizer(
        model_path: Path,
        *,
        local_files_only: bool = True,
) -> PreTrainedTokenizerBase:
    """从本地模型目录加载 Tokenizer。"""
    resolved_model_path = validate_model_path(model_path)

    return AutoTokenizer.from_pretrained(
        resolved_model_path, 
        local_files_only=local_files_only,
    )

def load_causal_lm(
    model_path: Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    attention_backend: str | None = None,
    local_files_only: bool = True,
) -> PreTrainedModel:
    """加载 Causal Language Model，并移动到目标设备。"""

    resolved_model_path = validate_model_path(model_path)

    model_arguments: dict[str, Any] = {
        "dtype": dtype,
        "local_files_only": local_files_only,
    }

    # first-load.py 当前没有显式指定 Attention backend，
    # 因此这里允许不传，保持 Transformers 默认行为。
    if attention_backend is not None:
        model_arguments["attn_implementation"] = attention_backend

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        # avoid the memory problem in the low mem conditon
        low_cpu_mem_usage=True,
        **model_arguments,
    )

    return model.to(device).eval()


# 用 Accelerate 的 init_empty_weights() 在 meta device 上构造模型结构，
# 不读取真实权重，然后把结构保存到脚本所在的 experiments/00_baseline/ 目录。
def save_model_structure(model_path: Path) -> Path:
    """构造空模型并将模块结构保存到当前脚本目录。"""
    model_path = validate_model_path(model_path)
    
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