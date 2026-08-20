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
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer


from mini_llm_infra.model.loader import (load_causal_lm, load_tokenizer)
from mini_llm_infra.utils.cuda import (configure_cuda_compatibility, select_dtype)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "benchmark_config.json"

project_root = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = project_root / "models" / "Qwen3-1.7B"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3 Hugging Face 原生推理基线与 PyTorch Profiler 分析"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
        "--profile-new-tokens",
        type=int,
        default=None,
        help="Profiler 生成的 token 数；应远小于正式 Benchmark。",
    )
    parser.add_argument(
        "--profile-prompt-id",
        type=str,
        default=None,
        help="指定用于 Profiler 的 Prompt；默认使用本次测试的第一个 Prompt。",
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
        help="原生基线默认使用 eager；sdpa 可作为后续对照。",
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
        help="令 min_new_tokens 等于 max_new_tokens，强制固定输出长度。",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_benchmark_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Benchmark 配置文件不存在：{config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_keys = {"defaults", "suites", "prompts"}
    missing_keys = required_keys - set(config)
    if missing_keys:
        raise ValueError(f"配置文件缺少字段：{sorted(missing_keys)}")

    prompt_ids = [prompt_case["id"] for prompt_case in config["prompts"]]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("配置文件中存在重复的 Prompt ID。")

    return config


def resolve_prompt_cases(
    config: dict[str, Any],
    suite_name: str,
    selected_prompt_ids: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suite = config["suites"][suite_name]
    settings = {**config["defaults"], **suite}
    prompt_by_id = {
        prompt_case["id"]: prompt_case for prompt_case in config["prompts"]
    }

    requested_ids = selected_prompt_ids or suite["prompt_ids"]
    if requested_ids == ["*"]:
        prompt_cases = list(config["prompts"])
    else:
        missing_ids = [
            prompt_id for prompt_id in requested_ids if prompt_id not in prompt_by_id
        ]
        if missing_ids:
            raise ValueError(f"找不到 Prompt ID：{missing_ids}")
        prompt_cases = [prompt_by_id[prompt_id] for prompt_id in requested_ids]

    if not prompt_cases:
        raise ValueError("当前测试套件没有可运行的 Prompt。")

    return prompt_cases, settings


def build_prompt_text(prompt_case: dict[str, Any]) -> str:
    sections = [prompt_case["prompt"]]
    context = prompt_case.get("context", "")
    context_repeat = int(prompt_case.get("context_repeat", 0))

    if context and context_repeat > 0:
        repeated_context = "\n".join(context for _ in range(context_repeat))
        sections.append(f"背景资料：\n{repeated_context}")

    return "\n\n".join(sections)


# def configure_cuda_compatibility() -> tuple[int, int]:
#     if not torch.cuda.is_available():
#         raise RuntimeError("没有检测到可用的 CUDA GPU。")

#     capability = torch.cuda.get_device_capability()
#     major, _ = capability

#     if major < 8:
#         from torch._native.registry import deregister_op_overrides

#         deregister_op_overrides(
#             disable_op_symbols="bmm"
#         )
#         print(
#             "Disabled PyTorch native Triton bmm "
#             "override for this GPU."
#         )

#     return capability


# def select_dtype(dtype_name: str) -> torch.dtype:
#     if dtype_name == "float16":
#         return torch.float16

#     native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
#     if not native_bf16:
#         print(
#             "Warning: 当前 GPU 没有原生 BF16 支持，"
#             "BF16 可能回退或无法用于部分编译 Kernel。"
#         )
#     return torch.bfloat16


def prepare_inputs(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


def run_generate(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    use_cache: bool,
    pad_token_id: int | None,
    force_exact_output_length: bool,
) -> torch.Tensor:
    generation_arguments: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": use_cache,
    }

    if force_exact_output_length:
        generation_arguments["min_new_tokens"] = max_new_tokens
    if pad_token_id is not None:
        generation_arguments["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        return model.generate(**inputs, **generation_arguments)


def benchmark_generate(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    warmup_runs: int,
    benchmark_runs: int,
    use_cache: bool,
    pad_token_id: int | None,
    force_exact_output_length: bool,
) -> tuple[list[dict[str, float | int]], torch.Tensor]:
    for _ in range(warmup_runs):
        run_generate(
            model=model,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            force_exact_output_length=force_exact_output_length,
        )

    torch.cuda.synchronize()
    input_token_count = int(inputs["input_ids"].shape[-1])
    run_results: list[dict[str, float | int]] = []
    last_outputs: torch.Tensor | None = None

    for run_index in range(benchmark_runs):
        baseline_allocated_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        started_at = time.perf_counter()
        outputs = run_generate(
            model=model,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            force_exact_output_length=force_exact_output_length,
        )
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started_at

        output_token_count = int(outputs.shape[-1] - input_token_count)
        output_tokens_per_second = output_token_count / elapsed_seconds
        average_ms_per_output_token = (
            elapsed_seconds * 1000.0 / output_token_count
            if output_token_count > 0
            else float("inf")
        )
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        peak_reserved_bytes = torch.cuda.max_memory_reserved()

        run_result: dict[str, float | int] = {
            "run": run_index + 1,
            "elapsed_seconds": elapsed_seconds,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "output_tokens_per_second": output_tokens_per_second,
            "average_ms_per_output_token": average_ms_per_output_token,
            "peak_allocated_mib": peak_allocated_bytes / 1024**2,
            "peak_reserved_mib": peak_reserved_bytes / 1024**2,
            "generation_peak_delta_mib": (
                peak_allocated_bytes - baseline_allocated_bytes
            )
            / 1024**2,
        }
        run_results.append(run_result)
        last_outputs = outputs

        print(
            f"  Run {run_index + 1:02d}: "
            f"{elapsed_seconds:.3f} s | "
            f"{output_token_count} tokens | "
            f"{output_tokens_per_second:.2f} tok/s | "
            f"{average_ms_per_output_token:.2f} ms/token"
        )

    if last_outputs is None:
        raise RuntimeError("benchmark_runs 必须大于 0。")
    return run_results, last_outputs


def summarize_results(
    run_results: list[dict[str, float | int]],
) -> dict[str, float]:
    if not run_results:
        raise ValueError("run_results 为空，无法计算统计汇总。")

    latencies = [float(result["elapsed_seconds"]) for result in run_results]
    throughputs = [
        float(result["output_tokens_per_second"]) for result in run_results
    ]
    milliseconds_per_token = [
        float(result["average_ms_per_output_token"]) for result in run_results
    ]

    return {
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_p50_seconds": float(np.percentile(latencies, 50)),
        "latency_p95_seconds": float(np.percentile(latencies, 95)),
        "output_tokens_per_second_mean": statistics.mean(throughputs),
        "output_tokens_per_second_p50": float(np.percentile(throughputs, 50)),
        "average_ms_per_output_token_mean": statistics.mean(
            milliseconds_per_token
        ),
        "peak_allocated_mib": max(
            float(result["peak_allocated_mib"]) for result in run_results
        ),
        "peak_reserved_mib": max(
            float(result["peak_reserved_mib"]) for result in run_results
        ),
    }


def summarize_suite(case_results: list[dict[str, Any]]) -> dict[str, float]:
    total_measured_seconds = sum(
        float(run["elapsed_seconds"])
        for case_result in case_results
        for run in case_result["runs"]
    )
    total_output_tokens = sum(
        int(run["output_tokens"])
        for case_result in case_results
        for run in case_result["runs"]
    )
    case_mean_throughputs = [
        float(case_result["summary"]["output_tokens_per_second_mean"])
        for case_result in case_results
    ]

    return {
        "case_count": float(len(case_results)),
        "total_measured_seconds": total_measured_seconds,
        "total_output_tokens": float(total_output_tokens),
        "weighted_output_tokens_per_second": (
            total_output_tokens / total_measured_seconds
        ),
        "mean_of_case_throughputs": statistics.mean(case_mean_throughputs),
    }


def run_torch_profiler(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    prompt_id: str,
    max_new_tokens: int,
    use_cache: bool,
    pad_token_id: int | None,
    force_exact_output_length: bool,
    output_directory: Path,
    timestamp: str,
) -> dict[str, str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    trace_path = output_directory / f"qwen3_{prompt_id}_{timestamp}_trace.json"
    table_path = output_directory / f"qwen3_{prompt_id}_{timestamp}_ops.txt"

    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=False,
    ) as profiler:
        with record_function(f"qwen3_hf_generate_{prompt_id}"):
            run_generate(
                model=model,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                pad_token_id=pad_token_id,
                force_exact_output_length=force_exact_output_length,
            )
        torch.cuda.synchronize()

    profiler.export_chrome_trace(str(trace_path))
    profiler_table = profiler.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total",
        row_limit=80,
    )
    table_path.write_text(profiler_table, encoding="utf-8")

    return {
        "prompt_id": prompt_id,
        "chrome_trace": str(trace_path),
        "profiler_table": str(table_path),
    }


def render_speed_report(complete_result: dict[str, Any]) -> str:
    environment = complete_result["environment"]
    configuration = complete_result["configuration"]
    suite_summary = complete_result["suite_summary"]

    lines = [
        "Qwen3 Hugging Face Baseline Speed Report",
        "=" * 88,
        f"Timestamp: {complete_result['timestamp']}",
        f"Suite: {configuration['suite']}",
        f"Model: {configuration['model_path']}",
        f"GPU: {environment['gpu']}",
        f"Compute capability: {environment['compute_capability']}",
        f"PyTorch: {environment['torch']}",
        f"Transformers: {environment['transformers']}",
        f"PyTorch CUDA: {environment['torch_cuda']}",
        f"dtype: {environment['dtype']}",
        f"Attention backend: {environment['attention_backend']}",
        f"KV Cache: {configuration['use_cache']}",
        f"Force exact output length: {configuration['force_exact_output_length']}",
        f"Warm-up runs per prompt: {configuration['warmup_runs']}",
        f"Benchmark runs per prompt: {configuration['benchmark_runs']}",
        f"Max new tokens: {configuration['max_new_tokens']}",
        "",
        "Suite summary",
        "-" * 88,
        f"Cases: {int(suite_summary['case_count'])}",
        f"Total measured time: {suite_summary['total_measured_seconds']:.3f} s",
        f"Total output tokens: {int(suite_summary['total_output_tokens'])}",
        "Weighted output speed: "
        f"{suite_summary['weighted_output_tokens_per_second']:.3f} tok/s",
        "Mean of case speeds: "
        f"{suite_summary['mean_of_case_throughputs']:.3f} tok/s",
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
                f"Mean latency: {summary['latency_mean_seconds']:.6f} s",
                f"P50 latency: {summary['latency_p50_seconds']:.6f} s",
                f"P95 latency: {summary['latency_p95_seconds']:.6f} s",
                "Mean output speed: "
                f"{summary['output_tokens_per_second_mean']:.3f} tok/s",
                "P50 output speed: "
                f"{summary['output_tokens_per_second_p50']:.3f} tok/s",
                "Mean time per output token: "
                f"{summary['average_ms_per_output_token_mean']:.3f} ms/token",
                f"Peak allocated memory: {summary['peak_allocated_mib']:.3f} MiB",
                f"Peak reserved memory: {summary['peak_reserved_mib']:.3f} MiB",
                "",
                "Run  Latency(s)  Output tokens  Tokens/s  Avg ms/token  "
                "Peak allocated(MiB)  Peak delta(MiB)",
            ]
        )

        for run in case_result["runs"]:
            lines.append(
                f"{int(run['run']):>3}  "
                f"{float(run['elapsed_seconds']):>10.4f}  "
                f"{int(run['output_tokens']):>13}  "
                f"{float(run['output_tokens_per_second']):>8.3f}  "
                f"{float(run['average_ms_per_output_token']):>12.3f}  "
                f"{float(run['peak_allocated_mib']):>19.3f}  "
                f"{float(run['generation_peak_delta_mib']):>15.3f}"
            )
        lines.extend(["", "Generated text from the last run:", case_result["generated_text"], ""])

    profiler_result = complete_result.get("profiler")
    if profiler_result:
        lines.extend(
            [
                "Profiler outputs",
                "-" * 88,
                f"Prompt ID: {profiler_result['prompt_id']}",
                f"Chrome trace: {profiler_result['chrome_trace']}",
                f"Operator table: {profiler_result['profiler_table']}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"模型目录不存在：{args.model_path}")

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
    profile_new_tokens = (
        args.profile_new_tokens
        if args.profile_new_tokens is not None
        else int(suite_settings["profile_new_tokens"])
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
        raise ValueError("token 数和运行次数必须为正数，warmup_runs 可以为 0。")

    if args.profile and profile_new_tokens <= 0:
        raise ValueError("profile_new_tokens 必须为正数。")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    capability = configure_cuda_compatibility()
    dtype = select_dtype(args.dtype)
    device = torch.device("cuda")

    tokenizer_started_at = time.perf_counter()

    tokenizer = load_tokenizer(
        args.model_path,
        local_files_only=True
    )
    tokenizer_load_seconds = time.perf_counter() - tokenizer_started_at

    model_started_at = time.perf_counter()
    model = load_causal_lm(
        args.model_path,
        device=device,
        dtype=dtype,
        attention_backend=args.attention_backend,
        local_files_only=True
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_started_at

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {capability}")
    print(f"dtype: {dtype}")
    print(f"Suite: {args.suite}")
    print(f"Prompt count: {len(prompt_cases)}")
    print(f"Max new tokens: {max_new_tokens}")

    case_results: list[dict[str, Any]] = []
    profile_inputs: dict[str, torch.Tensor] | None = None
    profile_prompt_id = args.profile_prompt_id or prompt_cases[0]["id"]

    selected_ids = {prompt_case["id"] for prompt_case in prompt_cases}
    if profile_prompt_id not in selected_ids:
        raise ValueError(
            f"Profiler Prompt {profile_prompt_id!r} 不在本次测试套件中。"
        )

    for case_index, prompt_case in enumerate(prompt_cases, start=1):
        prompt_id = prompt_case["id"]
        prompt_text = build_prompt_text(prompt_case)
        inputs = prepare_inputs(tokenizer, prompt_text, device)
        input_token_count = int(inputs["input_ids"].shape[-1])
        total_requested_length = input_token_count + max_new_tokens
        maximum_context_length = int(model.config.max_position_embeddings)

        if total_requested_length > maximum_context_length:
            raise ValueError(
                f"{prompt_id} 的输入和输出总长度 {total_requested_length} "
                f"超过模型上限 {maximum_context_length}。"
            )

        print(
            f"\n[{case_index}/{len(prompt_cases)}] {prompt_id} | "
            f"input={input_token_count} | max_output={max_new_tokens}"
        )
        run_results, last_outputs = benchmark_generate(
            model=model,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            force_exact_output_length=force_exact_output_length,
        )

        generated_ids = last_outputs[0, input_token_count:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        case_results.append(
            {
                "prompt_id": prompt_id,
                "length_group": prompt_case.get("length_group", "unknown"),
                "prompt_text": prompt_text,
                "input_tokens": input_token_count,
                "requested_max_new_tokens": max_new_tokens,
                "actual_output_tokens": int(generated_ids.shape[-1]),
                "summary": summarize_results(run_results),
                "runs": run_results,
                "generated_text": generated_text,
            }
        )

        if prompt_id == profile_prompt_id:
            profile_inputs = inputs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_directory = SCRIPT_DIR / "results"
    profiles_directory = SCRIPT_DIR / "profiles"
    results_directory.mkdir(parents=True, exist_ok=True)

    profiler_result: dict[str, str] | None = None
    if args.profile:
        if profile_inputs is None:
            raise RuntimeError("没有找到 Profiler 对应的输入。")
        print(
            f"\nProfiling prompt={profile_prompt_id}, "
            f"new_tokens={profile_new_tokens}..."
        )
        profiler_result = run_torch_profiler(
            model=model,
            inputs=profile_inputs,
            prompt_id=profile_prompt_id,
            max_new_tokens=profile_new_tokens,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            force_exact_output_length=force_exact_output_length,
            output_directory=profiles_directory,
            timestamp=timestamp,
        )

    gpu_properties = torch.cuda.get_device_properties(0)
    complete_result: dict[str, Any] = {
        "timestamp": timestamp,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(capability),
            "gpu_total_memory_mib": gpu_properties.total_memory / 1024**2,
            "dtype": str(dtype),
            "attention_backend": args.attention_backend,
        },
        "configuration": {
            "model_path": str(args.model_path),
            "config_path": str(args.config),
            "suite": args.suite,
            "max_new_tokens": max_new_tokens,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "use_cache": use_cache,
            "force_exact_output_length": force_exact_output_length,
            "seed": args.seed,
            "measurement_scope": (
                "model.generate only; excludes model loading, tokenization, "
                "warm-up and profiler"
            ),
        },
        "loading": {
            "tokenizer_load_seconds": tokenizer_load_seconds,
            "model_load_seconds": model_load_seconds,
        },
        "suite_summary": summarize_suite(case_results),
        "cases": case_results,
        "profiler": profiler_result,
    }

    base_filename = f"qwen3_baseline_{args.suite}_{timestamp}"
    json_result_path = results_directory / f"{base_filename}.json"
    text_result_path = results_directory / f"{base_filename}.txt"
    json_result_path.write_text(
        json.dumps(complete_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text_result_path.write_text(
        render_speed_report(complete_result),
        encoding="utf-8",
    )

    print(f"\nJSON result: {json_result_path}")
    print(f"Speed report: {text_result_path}")
    if profiler_result:
        print(f"Chrome trace: {profiler_result['chrome_trace']}")
        print(f"Profiler table: {profiler_result['profiler_table']}")


if __name__ == "__main__":
    main()
