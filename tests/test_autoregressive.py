from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mini_llm_infra.generation.manual import (
    GenerationMetrics,
    ManualGenerateConfig,
    manual_generate,
)


class FakeCausalLM(torch.nn.Module):
    def __init__(self, generated_token_ids: list[int], vocab_size: int = 16) -> None:
        super().__init__()
        self.generated_token_ids = generated_token_ids
        self.vocab_size = vocab_size
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        past_key_values: object | None = None,
    ) -> SimpleNamespace:
        call_index = len(self.calls)
        self.calls.append(
            {
                "input_length": input_ids.shape[-1],
                "attention_length": attention_mask.shape[-1],
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "return_dict": return_dict,
            }
        )

        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[-1], self.vocab_size),
            -torch.inf,
        )
        logits[:, -1, self.generated_token_ids[call_index]] = 0.0
        next_cache = object() if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=next_cache)


def test_generation_metrics_follow_token_ready_definitions() -> None:
    metrics = GenerationMetrics.from_step_latencies([10.0, 4.0, 6.0])

    assert metrics.ttft_ms == 10.0
    assert metrics.itl_ms == (4.0, 6.0)
    assert metrics.tpot_ms == 5.0
    assert metrics.tot_ms == 20.0


def test_generation_metrics_have_no_tpot_for_one_token() -> None:
    metrics = GenerationMetrics.from_step_latencies([10.0])

    assert metrics.itl_ms == ()
    assert metrics.tpot_ms is None
    assert metrics.tot_ms == 10.0


def test_manual_generate_without_cache_recomputes_full_sequence() -> None:
    model = FakeCausalLM([3, 4, 5])
    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.ones_like(input_ids)

    result = manual_generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        config=ManualGenerateConfig(
            max_new_tokens=5,
            eos_token_id=5,
            use_cache=False,
        ),
    )

    assert result.generated_ids.tolist() == [[3, 4, 5]]
    assert result.sequence_ids.tolist() == [[1, 2, 3, 4, 5]]
    assert result.finish_reason == "eos"
    assert [call["input_length"] for call in model.calls] == [2, 3, 4]
    assert [call["attention_length"] for call in model.calls] == [2, 3, 4]
    assert all(call["past_key_values"] is None for call in model.calls)
    assert len(result.metrics.itl_ms) == 2
    assert result.metrics.tot_ms == pytest.approx(
        result.metrics.ttft_ms + sum(result.metrics.itl_ms)
    )


def test_manual_generate_with_cache_only_decodes_latest_token() -> None:
    model = FakeCausalLM([3, 5, 6, 7])
    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.ones_like(input_ids)

    result = manual_generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        config=ManualGenerateConfig(
            max_new_tokens=4,
            eos_token_id=(5, 9),
            force_exact_output_length=True,
            use_cache=True,
        ),
    )

    assert result.generated_ids.tolist() == [[3, 5, 6, 7]]
    assert result.finish_reason == "length"
    assert [call["input_length"] for call in model.calls] == [2, 1, 1, 1]
    assert [call["attention_length"] for call in model.calls] == [2, 3, 4, 5]
    assert model.calls[0]["past_key_values"] is None
    assert all(call["past_key_values"] is not None for call in model.calls[1:])


def test_manual_generate_requires_single_request_batch() -> None:
    model = FakeCausalLM([3])
    input_ids = torch.tensor([[1, 2], [1, 2]])

    with pytest.raises(ValueError, match="batch size = 1"):
        manual_generate(
            model=model,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            config=ManualGenerateConfig(
                max_new_tokens=1,
                eos_token_id=None,
            ),
        )
