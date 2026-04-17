"""Shared helpers for probability extraction and common-vocabulary alignment."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def tokenizer_vocab_size(model: Any) -> int:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return int(getattr(model, "vocab_size"))
    try:
        return int(len(tokenizer))
    except Exception:
        pass
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            return int(len(get_vocab()))
        except Exception:
            pass
    return int(getattr(model, "vocab_size"))


def common_vocab_size(target_model: Any, draft_model: Any) -> int:
    sizes = [
        int(getattr(target_model, "vocab_size")),
        int(getattr(draft_model, "vocab_size")),
        tokenizer_vocab_size(target_model),
        tokenizer_vocab_size(draft_model),
    ]
    common = min(sizes)
    if common <= 0:
        raise ValueError(f"Common vocabulary size must be positive, got {sizes}.")
    return common


def next_token_probs_tensor(
    model: Any,
    context_tokens: Sequence[int],
    vocab_size: int | None = None,
) -> torch.Tensor:
    probs = torch.tensor(model.next_token_probs(context_tokens), dtype=torch.float32)
    probs = torch.clamp(probs, min=0.0)
    if vocab_size is not None:
        probs = probs[:vocab_size]
    total = probs.sum()
    if total <= 0:
        raise ValueError("Model returned a probability vector with non-positive mass.")
    return probs / total
