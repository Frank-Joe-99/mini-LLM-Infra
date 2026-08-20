from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


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
        local_files_only=local_files_only
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
        **model_arguments,
    )

    return model.to(device).eval()
