from __future__ import annotations
import torch


def configure_cuda_compatibility() -> tuple[int, int]:
    """检查 CUDA，并应用当前项目需要的 GPU 兼容处理。"""

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 CUDA GPU。")

    compute_capability = torch.cuda.get_device_capability()
    major, _ = compute_capability

    # T4 等 Compute Capability 低于 8.0 的 GPU，
    # 禁用当前 PyTorch 原生 Triton bmm override，
    # 使其回退到 ATen/CUDA 实现。
    if major < 8:
        from torch._native.registry import deregister_op_overrides

        deregister_op_overrides(
            disable_op_symbols="bmm",
        )

        print(
            "Disabled PyTorch native Triton bmm "
            "override for this GPU."
        )

    return compute_capability

def select_dtype(dtype_name: str) -> torch.dtype:
    """把命令行中的 dtype 名称转换为 torch.dtype。"""

    if dtype_name == "float16":
        return torch.float16

    if dtype_name == "bfloat16":
        native_bf16_supported = torch.cuda.is_bf16_supported(
            including_emulation=False,
        )

        if not native_bf16_supported:
            print(
                "Warning: 当前 GPU 没有原生 BF16 支持，"
                "BF16 可能回退或无法用于部分编译 Kernel。"
            )

        return torch.bfloat16

    raise ValueError(
        f"不支持的 dtype：{dtype_name!r}。"
        "当前仅支持 'float16' 和 'bfloat16'。"
    )
