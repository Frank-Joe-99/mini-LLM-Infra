from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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