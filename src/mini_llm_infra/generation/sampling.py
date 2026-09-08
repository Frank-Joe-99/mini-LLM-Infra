from __future__ import annotations

import torch


def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    """
    输入 [batch_size, vocab_size]，
    返回 [batch_size, 1]。
    """
    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
    return next_token_id


def top_k_sampling(
    logits: torch.Tensor, top_k: int, temperature: float = 1.0
) -> torch.Tensor:
    """
    输入 [batch_size, vocab_size]，
    返回 [batch_size, 1]。
    """
    if not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("top_k 必须在 1 到 vocab_size 之间")

    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
    probs = torch.softmax(top_k_logits / temperature, dim=-1)

    sampled_index = torch.multinomial(probs, num_samples=1)
    # 将候选集合中的位置转换成原始词表中的 token ID
    next_token_id = torch.gather(top_k_indices, dim=-1, index=sampled_index)

    return next_token_id


def top_p_sampling(
    logits: torch.Tensor, top_p: float, temperature: float = 1.0
) -> torch.Tensor:
    """
    输入 [batch_size, vocab_size]，
    返回 [batch_size, 1]。
    """
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 0 到 1 之间")

    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    # 按分数降序排列，保留原始词表索引。
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits.float() / temperature, dim=-1)

    if top_p < 1:
        cumulative_probs = torch.cumsum(probs, dim=-1)
        # 前面的概率和已达到阈值时，才移除当前 token。
        # 保留首次达到或超过阈值的 token，且至少保留一个。
        remove_mask = torch.zeros_like(probs, dtype=torch.bool)
        remove_mask[:, 1:] = cumulative_probs[:, :-1] >= top_p
        probs = probs.masked_fill(remove_mask, 0.0)

    # multinomial 支持未归一化权重，因此无需再次归一化。
    sampled_index = torch.multinomial(probs, num_samples=1)
    next_token_id = torch.gather(sorted_indices, dim=-1, index=sampled_index)

    return next_token_id
