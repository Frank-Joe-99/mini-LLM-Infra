from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch


EosTokenId: TypeAlias = int | tuple[int, ...] | None
FinishReason: TypeAlias = Literal["eos", "length"]


@dataclass(frozen=True, slots=True)
class ManualGenerateConfig:
    """手写 Greedy 解码所需的最小配置。"""

    max_new_tokens: int
    eos_token_id: EosTokenId
    force_exact_output_length: bool = False
    use_cache: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0。")

        eos_token_ids = self.eos_token_ids
        if any(token_id < 0 for token_id in eos_token_ids):
            raise ValueError("eos_token_id 不能为负数。")

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        if self.eos_token_id is None:
            return ()
        if isinstance(self.eos_token_id, int):
            return (self.eos_token_id,)
        return self.eos_token_id


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """单次请求的逐 token 延迟指标，单位均为毫秒。"""

    ttft_ms: float
    tpot_ms: float | None
    itl_ms: tuple[float, ...]
    tot_ms: float

    @classmethod
    def from_step_latencies(
        cls,
        step_latencies_ms: list[float],
    ) -> GenerationMetrics:
        """从每个 token-ready 间隔构造 TTFT/TPOT/ITL/TOT。

        第一个间隔是 TTFT；其余间隔是 ITL。TPOT 是所有 ITL 的均值，
        即 ``(TOT - TTFT) / (output_tokens - 1)``。TOT 表示从生成开始到
        最后一个输出 token 就绪的 Total Output Time。只生成一个 token 时
        没有 Decode 间隔，因此 TPOT 为 ``None``。
        """

        if not step_latencies_ms:
            raise ValueError("至少需要一个 token latency。")
        if any(latency < 0 for latency in step_latencies_ms):
            raise ValueError("token latency 不能为负数。")

        ttft_ms = step_latencies_ms[0]
        itl_ms = tuple(step_latencies_ms[1:])
        tpot_ms = statistics.mean(itl_ms) if itl_ms else None

        return cls(
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            itl_ms=itl_ms,
            tot_ms=sum(step_latencies_ms),
        )

    def to_dict(self) -> dict[str, float | list[float] | None]:
        return {
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "itl_ms": list(self.itl_ms),
            "tot_ms": self.tot_ms,
        }


@dataclass(frozen=True, slots=True)
class ManualGenerationResult:
    sequence_ids: torch.Tensor
    generated_ids: torch.Tensor
    finish_reason: FinishReason
    metrics: GenerationMetrics


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids 和 attention_mask 必须是二维张量。")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids 和 attention_mask 的形状必须相同。")
    if input_ids.shape[0] != 1:
        raise ValueError("当前指标实现一次只测量一个请求（batch size = 1）。")
    if input_ids.shape[1] == 0:
        raise ValueError("输入 Prompt 不能为空。")
    if input_ids.device != attention_mask.device:
        raise ValueError("input_ids 和 attention_mask 必须位于同一设备。")


def _is_eos(next_token_id: torch.Tensor, eos_token_ids: tuple[int, ...]) -> bool:
    if not eos_token_ids:
        return False

    eos_match = torch.zeros_like(next_token_id, dtype=torch.bool)
    for eos_token_id in eos_token_ids:
        eos_match |= next_token_id == eos_token_id
    return bool(torch.all(eos_match).item())


def manual_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: ManualGenerateConfig,
) -> ManualGenerationResult:
    """用 Greedy 策略逐 token 推理，并记录 token-ready 延迟。

    当 ``use_cache=False`` 时，每一步都会重新计算完整历史序列；开启缓存后，
    第一步处理完整 Prompt，后续步骤只把最新 token 和 ``past_key_values``
    传给模型。

    计时从第一次模型调用前开始，在每个 token 的 argmax 结果可用后记录时间。
    CUDA 上会在计时起点和每个 token-ready 点同步，确保异步 kernel 已完成。
    """

    _validate_inputs(input_ids, attention_mask)

    device = input_ids.device
    sequence_ids = input_ids.clone()
    current_attention_mask = attention_mask.clone()
    generated_tokens: list[torch.Tensor] = []
    step_latencies_ms: list[float] = []
    past_key_values = None
    finish_reason: FinishReason = "length"

    _synchronize_device(device)
    previous_token_ready_at = time.perf_counter()

    with torch.inference_mode():
        for step_index in range(config.max_new_tokens):
            if config.use_cache and step_index > 0:
                model_input_ids = generated_tokens[-1]
            else:
                model_input_ids = sequence_ids

            model_arguments = {
                "input_ids": model_input_ids,
                "attention_mask": current_attention_mask,
                "use_cache": config.use_cache,
                "return_dict": True,
            }
            if config.use_cache and past_key_values is not None:
                model_arguments["past_key_values"] = past_key_values

            outputs = model(**model_arguments)
            next_token_id = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            if config.use_cache:
                past_key_values = getattr(outputs, "past_key_values", None)
                if past_key_values is None:
                    raise RuntimeError(
                        "模型在 use_cache=True 时没有返回 past_key_values。"
                    )

            _synchronize_device(device)
            token_ready_at = time.perf_counter()
            step_latencies_ms.append(
                (token_ready_at - previous_token_ready_at) * 1000.0
            )
            previous_token_ready_at = token_ready_at

            generated_tokens.append(next_token_id)
            if not config.use_cache:
                sequence_ids = torch.cat((sequence_ids, next_token_id), dim=-1)
            new_attention_position = torch.ones(
                (current_attention_mask.shape[0], 1),
                dtype=current_attention_mask.dtype,
                device=device,
            )
            current_attention_mask = torch.cat(
                (current_attention_mask, new_attention_position),
                dim=-1,
            )

            if (
                not config.force_exact_output_length
                and _is_eos(next_token_id, config.eos_token_ids)
            ):
                finish_reason = "eos"
                break

    generated_ids = torch.cat(generated_tokens, dim=-1)
    if config.use_cache:
        sequence_ids = torch.cat((input_ids, generated_ids), dim=-1)
    metrics = GenerationMetrics.from_step_latencies(step_latencies_ms)

    return ManualGenerationResult(
        sequence_ids=sequence_ids,
        generated_ids=generated_ids,
        finish_reason=finish_reason,
        metrics=metrics,
    )
