from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
from mini_llm_infra.utils.cuda import (configure_cuda_compatibility,  select_dtype)
from mini_llm_infra.utils.loadconfig import (load_benchmark_config, resolve_prompt_cases)
from mini_llm_infra.model.loader import (load_tokenizer, load_causal_lm)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "benchmark_config.json"

project_root = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = project_root / "models" / "Qwen3-1.7B"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark manual autoregressive decoding without a KV cache."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--suite",
        choices=["smoke", "prefill", "decode", "mixed", "full"],
        default="smoke",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="覆盖配置文件中测试套件的 max_new_tokens。",
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=None,
        help="只运行指定 Prompt；可多次传入该参数。",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=None,
        help="覆盖配置文件中的预热次数。",
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def manual_generate(
        model: torch.nn.Module,
        input_ids,
        attention_mask,
        config: dict[str, Any],
):
    # TODO: 为 input_ids、attention_mask 和返回值补充类型标注。
    
    # TODO: 明确 config 的结构；建议使用 dataclass 或定义清晰的字典字段，
    # 至少包含 max_new_tokens、eos_token_id 和是否强制固定输出长度。

    sequence_ids = input_ids
    # TODO: 如需避免调用方后续原地修改，确认这里是否应使用 input_ids.clone()。

    # TODO: 从 config 中读取并校验 max_new_tokens；当前循环中的变量尚未定义。
    # TODO: 保存初始 input_seq_len，便于最终只截取新生成的 token。
    # TODO: 初始化 generated_ids、finish_reason、prefill_latency_ms 和
    # decode_step_latencies_ms 等结果字段。
    # TODO: 使用 torch.inference_mode() 包裹整个手写生成过程。

    for _ in range(max_new_tokens):
        # TODO: 调用 model.forward()。Phase 1 的第一个版本建议明确设置
        # use_cache=False，并把完整 sequence_ids 和 attention_mask 传入模型。
        # TODO: 第一次完整 Prompt Forward 记为 Prefill，后续 Forward 记为
        # Decode without KV Cache；计时时需要在开始和结束位置同步 CUDA。
        # TODO: 从 outputs.logits[:, -1, :] 取出最后一个位置的 logits。
        # TODO: 使用 torch.argmax(..., dim=-1, keepdim=True) 实现 Greedy 选词。
        # TODO: 将 next_token_id 拼接到 sequence_ids。
        # TODO: 为 attention_mask 拼接一个值为 1 的新位置，并保持 device/dtype 一致。
        # TODO: 将本轮 token 保存到 generated_ids 或最终通过 input_seq_len 截取。
        # TODO: 检查 next_token_id 是否等于 eos_token_id；命中时设置
        # finish_reason="eos" 并退出循环，否则在达到上限时标记为 "length"。
        pass

    # TODO: 返回结构化结果。至少包含完整 sequence_ids、generated_ids、
    # finish_reason、prefill_latency_ms 和逐步 decode latency。


def main() -> None:
    args = parse_args()
    configure_cuda_compatibility()
    tensor_dtype = select_dtype(args.dtype)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = load_benchmark_config(args.config)
    prompt_cases, suite_settings = resolve_prompt_cases(
        config=config,
        suite_name=args.suite,
        selected_prompt_ids=args.prompt_id,
    )
    # TODO: 加载并校验 DEFAULT_CONFIG_PATH；当前 json 已导入但没有使用。
    prompt_case = prompt_cases[0]
    prompt_parts = [prompt_case["prompt"]]
    context = prompt_case.get("context", "")
    context_repeat = int(prompt_case.get("context_repeat", 0))
    if context and context_repeat > 0:
        prompt_parts.append(f"背景资料：\n{'\n'.join([context] * context_repeat)}")
    prompt_text = "\n\n".join(prompt_parts)



    tokenizer = load_tokenizer(
        model_path=DEFAULT_MODEL_PATH,
        local_files_only=True
        )

    model = load_causal_lm(
        model_path=DEFAULT_MODEL_PATH,
        device='cuda',
        dtype=tensor_dtype,
        attention_backend=None,
        local_files_only=True
        )

    # TODO: 使用 tokenizer.apply_chat_template() 得到 input_ids 和
    # attention_mask，并将它们移动到 model.device。
    # TODO: 确定 eos_token_id；必要时使用 tokenizer.eos_token_id。
    # TODO: 为正式 Benchmark 增加 warm-up runs 和 benchmark runs，且二者分开统计。
    # TODO: 重置并记录 CUDA peak allocated/reserved memory。

    # TODO: 向 manual_generate() 传入 model、input_ids、attention_mask 和 config；
    # 当前无参数调用会在运行时触发 TypeError。
    manual_generate()

    # TODO: 解码 generated_ids 并打印生成文本。
    # TODO: 汇总 TTFT、E2E latency、TPOT、ITL、tokens/s 和显存指标；
    # 明确无 KV Cache 时 Decode 会重复计算全部历史序列。
    # TODO: 将原始逐轮数据和汇总结果保存为 JSON/TXT，便于与 HF baseline 对比。
    # TODO: transformers 当前已导入但未使用；后续可用于记录 Transformers 版本，
    # 否则应删除该导入。


# TODO: main() 入口尚未接入；完成 main 所需参数和输入准备后取消下面两行的注释。
# if __name__ == "__main__":
#     main()
