from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mini_llm_infra.utils.cuda import (configure_cuda_compatibility, )
from mini_llm_infra.model.loader import (save_model_structure, 
                                         load_tokenizer, 
                                         load_causal_lm)


def main() -> None:
    configure_cuda_compatibility()

    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "Qwen3-1.7B"

    #save_model_structure(model_path=model_path)

    tokenizer = load_tokenizer(
        model_path, local_files_only=True
    )
    model = load_causal_lm(
        model_path,
        device='cuda',
        dtype=torch.float16,
        attention_backend=None,
        local_files_only=True,
    )

    messages = [
        {"role": "user", "content": "请详细解释什么是 KV Cache。"},
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
            max_new_tokens=2048,
            do_sample=False,
        )

    input_seq_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, input_seq_len:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(generated_text)


if __name__ == "__main__":
    main()