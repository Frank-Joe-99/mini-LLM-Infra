from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import PreTrainedTokenizerBase

from mini_llm_infra.generation.manual import (
    EosTokenId,
    ManualGenerateConfig,
    ManualGenerationResult,
    manual_generate,
)
from mini_llm_infra.model.loader import load_causal_lm, load_tokenizer
from mini_llm_infra.utils.cuda import configure_cuda_compatibility, select_dtype
from mini_llm_infra.utils.loadconfig import (
    load_benchmark_config,
    resolve_prompt_cases,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "benchmark_config.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-1.7B"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark manual greedy autoregressive decoding with optional KV Cache."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--suite",
        choices=["smoke", "prefill", "decode", "mixed", "full"],
        default="smoke",
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=None,
        help="只运行指定 Prompt；可多次传入该参数。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="覆盖配置文件中测试套件的 max_new_tokens。",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=None,
        help="覆盖配置文件中的预热次数。",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=None,
        help="覆盖配置文件中的正式测量次数。",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["eager", "sdpa"],
        default="eager",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖配置文件中的 KV Cache 开关。",
    )
    parser.add_argument(
        "--force-exact-output-length",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="忽略 EOS，始终生成 max_new_tokens 个 token。",
    )
    parser.add_argument("--seed", type=int, default=990808)
    return parser.parse_args()


def build_prompt_text(prompt_case: dict[str, Any]) -> str:
    sections = [str(prompt_case["prompt"])]
    context = str(prompt_case.get("context", ""))
    context_repeat = int(prompt_case.get("context_repeat", 0))

    if context and context_repeat > 0:
        repeated_context = "\n".join(context for _ in range(context_repeat))
        sections.append(f"背景资料：\n{repeated_context}")

    return "\n\n".join(sections)


def prepare_inputs(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(prompt_text, return_tensors="pt")

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def resolve_eos_token_id(
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
) -> EosTokenId:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        generation_config = getattr(model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)

    if eos_token_id is None or isinstance(eos_token_id, int):
        return eos_token_id
    return tuple(int(token_id) for token_id in eos_token_id)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
        }

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def benchmark_manual_generation(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    generation_config: ManualGenerateConfig,
    warmup_runs: int,
    benchmark_runs: int,
) -> tuple[list[dict[str, Any]], ManualGenerationResult]:
    device = inputs["input_ids"].device

    for _ in range(warmup_runs):
        manual_generate(model=model, config=generation_config, **inputs)

    _synchronize_if_cuda(device)
    run_results: list[dict[str, Any]] = []
    last_generation: ManualGenerationResult | None = None

    for run_index in range(benchmark_runs):
        if device.type == "cuda":
            baseline_allocated_bytes = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_allocated_bytes = 0

        generation = manual_generate(
            model=model,
            config=generation_config,
            **inputs,
        )
        output_token_count = int(generation.generated_ids.shape[-1])
        metrics = generation.metrics.to_dict()
        tot_seconds = generation.metrics.tot_ms / 1000.0
        output_tokens_per_second = output_token_count / tot_seconds
        decode_tokens_per_second = (
            1000.0 / generation.metrics.tpot_ms
            if generation.metrics.tpot_ms is not None
            and generation.metrics.tpot_ms > 0
            else None
        )

        if device.type == "cuda":
            peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
            peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
        else:
            peak_allocated_bytes = 0
            peak_reserved_bytes = 0

        run_result = {
            "run": run_index + 1,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": output_token_count,
            "finish_reason": generation.finish_reason,
            "metrics": metrics,
            "output_tokens_per_second": output_tokens_per_second,
            "decode_tokens_per_second": decode_tokens_per_second,
            "peak_allocated_mib": peak_allocated_bytes / 1024**2,
            "peak_reserved_mib": peak_reserved_bytes / 1024**2,
            "generation_peak_delta_mib": (
                max(0, peak_allocated_bytes - baseline_allocated_bytes) / 1024**2
            ),
        }
        run_results.append(run_result)
        last_generation = generation

        tpot_text = (
            f"{generation.metrics.tpot_ms:.3f} ms"
            if generation.metrics.tpot_ms is not None
            else "N/A"
        )
        print(
            f"  Run {run_index + 1:02d}: "
            f"TTFT={generation.metrics.ttft_ms:.3f} ms | "
            f"TPOT={tpot_text} | "
            f"TOT={generation.metrics.tot_ms:.3f} ms | "
            f"output={output_token_count} tokens"
        )

    if last_generation is None:
        raise RuntimeError("benchmark_runs 必须大于 0。")
    return run_results, last_generation


def summarize_runs(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not run_results:
        raise ValueError("run_results 为空，无法计算统计汇总。")

    ttft_values = [float(run["metrics"]["ttft_ms"]) for run in run_results]
    tpot_values = [
        float(run["metrics"]["tpot_ms"])
        for run in run_results
        if run["metrics"]["tpot_ms"] is not None
    ]
    itl_values = [
        float(itl_ms)
        for run in run_results
        for itl_ms in run["metrics"]["itl_ms"]
    ]
    tot_values = [float(run["metrics"]["tot_ms"]) for run in run_results]
    output_throughputs = [
        float(run["output_tokens_per_second"]) for run in run_results
    ]
    decode_throughputs = [
        float(run["decode_tokens_per_second"])
        for run in run_results
        if run["decode_tokens_per_second"] is not None
    ]

    return {
        "ttft_ms": _distribution(ttft_values),
        "tpot_ms": _distribution(tpot_values),
        "itl_ms": _distribution(itl_values),
        "tot_ms": _distribution(tot_values),
        "output_tokens_per_second": _distribution(output_throughputs),
        "decode_tokens_per_second": _distribution(decode_throughputs),
        "peak_allocated_mib": max(
            float(run["peak_allocated_mib"]) for run in run_results
        ),
        "peak_reserved_mib": max(
            float(run["peak_reserved_mib"]) for run in run_results
        ),
    }


def summarize_suite(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_runs = [run for case in case_results for run in case["runs"]]
    total_output_tokens = sum(int(run["output_tokens"]) for run in all_runs)
    total_generation_seconds = sum(
        float(run["metrics"]["tot_ms"]) / 1000.0 for run in all_runs
    )

    return {
        "case_count": len(case_results),
        "run_count": len(all_runs),
        "total_output_tokens": total_output_tokens,
        "total_generation_seconds": total_generation_seconds,
        "weighted_output_tokens_per_second": (
            total_output_tokens / total_generation_seconds
        ),
        "ttft_ms": _distribution(
            [float(run["metrics"]["ttft_ms"]) for run in all_runs]
        ),
        "tpot_ms": _distribution(
            [
                float(run["metrics"]["tpot_ms"])
                for run in all_runs
                if run["metrics"]["tpot_ms"] is not None
            ]
        ),
        "tot_ms": _distribution(
            [float(run["metrics"]["tot_ms"]) for run in all_runs]
        ),
    }


def _format_metric(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f}{suffix}"


def render_report(complete_result: dict[str, Any]) -> str:
    configuration = complete_result["configuration"]
    environment = complete_result["environment"]
    suite_summary = complete_result["suite_summary"]

    lines = [
        "Manual Autoregressive Decode Benchmark",
        "=" * 88,
        f"Timestamp: {complete_result['timestamp']}",
        f"Model: {configuration['model_path']}",
        f"Suite: {configuration['suite']}",
        f"GPU: {environment['gpu']}",
        f"dtype: {environment['dtype']}",
        f"Attention backend: {environment['attention_backend']}",
        f"KV Cache: {configuration['use_cache']}",
        f"Force exact output length: {configuration['force_exact_output_length']}",
        f"Warm-up runs per prompt: {configuration['warmup_runs']}",
        f"Benchmark runs per prompt: {configuration['benchmark_runs']}",
        f"Max new tokens: {configuration['max_new_tokens']}",
        "",
        "Metric definitions",
        "-" * 88,
        "TTFT: generation start -> first output token ready.",
        "ITL: latency between two adjacent output-token-ready timestamps.",
        "TPOT: mean ITL, equal to (TOT - TTFT) / (output_tokens - 1).",
        "TOT (Total Output Time): generation start -> final output token ready.",
        "",
        "Suite summary",
        "-" * 88,
        f"Cases: {suite_summary['case_count']}",
        f"Measured runs: {suite_summary['run_count']}",
        f"Total output tokens: {suite_summary['total_output_tokens']}",
        "Weighted output speed: "
        f"{suite_summary['weighted_output_tokens_per_second']:.3f} tok/s",
        "Mean TTFT: "
        f"{_format_metric(suite_summary['ttft_ms']['mean'], ' ms')}",
        "Mean TPOT: "
        f"{_format_metric(suite_summary['tpot_ms']['mean'], ' ms')}",
        "Mean TOT: "
        f"{_format_metric(suite_summary['tot_ms']['mean'], ' ms')}",
        "",
    ]

    for case_result in complete_result["cases"]:
        summary = case_result["summary"]
        lines.extend(
            [
                f"Prompt: {case_result['prompt_id']} ({case_result['length_group']})",
                "-" * 88,
                f"Input tokens: {case_result['input_tokens']}",
                f"Actual output tokens: {case_result['actual_output_tokens']}",
                "TTFT mean/P50/P95: "
                f"{_format_metric(summary['ttft_ms']['mean'])} / "
                f"{_format_metric(summary['ttft_ms']['p50'])} / "
                f"{_format_metric(summary['ttft_ms']['p95'])} ms",
                "TPOT mean/P50/P95: "
                f"{_format_metric(summary['tpot_ms']['mean'])} / "
                f"{_format_metric(summary['tpot_ms']['p50'])} / "
                f"{_format_metric(summary['tpot_ms']['p95'])} ms",
                "ITL mean/P50/P95: "
                f"{_format_metric(summary['itl_ms']['mean'])} / "
                f"{_format_metric(summary['itl_ms']['p50'])} / "
                f"{_format_metric(summary['itl_ms']['p95'])} ms",
                "TOT mean/P50/P95: "
                f"{_format_metric(summary['tot_ms']['mean'])} / "
                f"{_format_metric(summary['tot_ms']['p50'])} / "
                f"{_format_metric(summary['tot_ms']['p95'])} ms",
                "Mean output speed: "
                f"{_format_metric(summary['output_tokens_per_second']['mean'])} tok/s",
                f"Peak allocated memory: {summary['peak_allocated_mib']:.3f} MiB",
                f"Peak reserved memory: {summary['peak_reserved_mib']:.3f} MiB",
                "",
                "Generated text from the last run:",
                case_result["generated_text"],
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(args.config)
    prompt_cases, suite_settings = resolve_prompt_cases(
        config=config,
        suite_name=args.suite,
        selected_prompt_ids=args.prompt_id,
    )

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(suite_settings["max_new_tokens"])
    )
    warmup_runs = (
        args.warmup_runs
        if args.warmup_runs is not None
        else int(suite_settings["warmup_runs"])
    )
    benchmark_runs = (
        args.benchmark_runs
        if args.benchmark_runs is not None
        else int(suite_settings["benchmark_runs"])
    )
    use_cache = (
        args.use_cache
        if args.use_cache is not None
        else bool(suite_settings["use_cache"])
    )
    force_exact_output_length = (
        args.force_exact_output_length
        if args.force_exact_output_length is not None
        else bool(suite_settings["force_exact_output_length"])
    )

    if max_new_tokens <= 0 or benchmark_runs <= 0 or warmup_runs < 0:
        raise ValueError("token 数和 benchmark_runs 必须为正，warmup_runs 可以为 0。")

    capability = configure_cuda_compatibility()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = select_dtype(args.dtype)
    device = torch.device("cuda")

    tokenizer_started_at = time.perf_counter()
    tokenizer = load_tokenizer(args.model_path, local_files_only=True)
    tokenizer_load_seconds = time.perf_counter() - tokenizer_started_at

    model_started_at = time.perf_counter()
    model = load_causal_lm(
        args.model_path,
        device=device,
        dtype=dtype,
        attention_backend=args.attention_backend,
        local_files_only=True,
    )
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - model_started_at
    eos_token_id = resolve_eos_token_id(tokenizer, model)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute capability: {capability}")
    print(f"dtype: {dtype}")
    print(f"Suite: {args.suite}")
    print(f"Prompt count: {len(prompt_cases)}")
    print(f"KV Cache: {use_cache}")
    print(f"Max new tokens: {max_new_tokens}")

    generation_config = ManualGenerateConfig(
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        force_exact_output_length=force_exact_output_length,
        use_cache=use_cache,
    )
    case_results: list[dict[str, Any]] = []

    for case_index, prompt_case in enumerate(prompt_cases, start=1):
        prompt_id = str(prompt_case["id"])
        prompt_text = build_prompt_text(prompt_case)
        inputs = prepare_inputs(tokenizer, prompt_text, device)
        input_token_count = int(inputs["input_ids"].shape[-1])
        total_requested_length = input_token_count + max_new_tokens
        maximum_context_length = getattr(
            model.config,
            "max_position_embeddings",
            None,
        )

        if (
            maximum_context_length is not None
            and total_requested_length > int(maximum_context_length)
        ):
            raise ValueError(
                f"{prompt_id} 的输入和输出总长度 {total_requested_length} "
                f"超过模型上限 {maximum_context_length}。"
            )

        print(
            f"\n[{case_index}/{len(prompt_cases)}] {prompt_id} | "
            f"input={input_token_count} | max_output={max_new_tokens}"
        )
        run_results, last_generation = benchmark_manual_generation(
            model=model,
            inputs=inputs,
            generation_config=generation_config,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )
        generated_text = tokenizer.decode(
            last_generation.generated_ids[0],
            skip_special_tokens=True,
        )
        case_results.append(
            {
                "prompt_id": prompt_id,
                "length_group": prompt_case.get("length_group", "unknown"),
                "prompt_text": prompt_text,
                "input_tokens": input_token_count,
                "requested_max_new_tokens": max_new_tokens,
                "actual_output_tokens": int(
                    last_generation.generated_ids.shape[-1]
                ),
                "generated_text": generated_text,
                "summary": summarize_runs(run_results),
                "runs": run_results,
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_properties = torch.cuda.get_device_properties(device)
    complete_result: dict[str, Any] = {
        "timestamp": timestamp,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
            "gpu_total_memory_mib": gpu_properties.total_memory / 1024**2,
            "dtype": str(dtype),
            "attention_backend": args.attention_backend,
        },
        "configuration": {
            "model_path": str(args.model_path.resolve()),
            "config_path": str(args.config.resolve()),
            "suite": args.suite,
            "max_new_tokens": max_new_tokens,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "use_cache": use_cache,
            "force_exact_output_length": force_exact_output_length,
            "eos_token_id": eos_token_id,
            "seed": args.seed,
            "measurement_scope": (
                "manual generation loop only; excludes model loading, "
                "tokenization, warm-up and result serialization"
            ),
        },
        "metric_definitions": {
            "ttft_ms": "generation start to first output token ready",
            "itl_ms": "latency between adjacent output token-ready timestamps",
            "tpot_ms": "mean ITL; (TOT - TTFT) / (output_tokens - 1)",
            "tot_ms": (
                "Total Output Time; generation start to final output token ready"
            ),
        },
        "loading": {
            "tokenizer_load_seconds": tokenizer_load_seconds,
            "model_load_seconds": model_load_seconds,
        },
        "suite_summary": summarize_suite(case_results),
        "cases": case_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = f"manual_decode_{args.suite}_{timestamp}"
    json_result_path = args.output_dir / f"{base_filename}.json"
    text_result_path = args.output_dir / f"{base_filename}.txt"
    json_result_path.write_text(
        json.dumps(complete_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text_result_path.write_text(
        render_report(complete_result),
        encoding="utf-8",
    )

    print(f"\nJSON result: {json_result_path.resolve()}")
    print(f"Text report: {text_result_path.resolve()}")


if __name__ == "__main__":
    main()
