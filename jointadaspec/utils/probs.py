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


def block_next_token_probs_tensor(
    model: Any,
    context_tokens: Sequence[int],
    continuation_tokens: Sequence[int],
    vocab_size: int | None = None,
) -> list[torch.Tensor]:
    """Return target distributions for a speculative block plus bonus token.

    The returned list has ``len(continuation_tokens) + 1`` entries. Entry ``i``
    predicts ``continuation_tokens[i]`` from ``context_tokens + continuation[:i]``.
    The final entry is the bonus distribution after the whole continuation.

    Models may implement ``next_token_probs_block`` to compute these
    distributions in one forward pass. Generic toy models fall back to repeated
    ``next_token_probs`` calls, preserving correctness without requiring a batch
    API.
    """
    block_fn = getattr(model, "next_token_probs_block", None)
    if callable(block_fn):
        raw_vectors = block_fn(list(context_tokens), list(continuation_tokens))
        vectors = [
            torch.tensor(raw, dtype=torch.float32).flatten()
            for raw in raw_vectors
        ]
    else:
        vectors = [
            next_token_probs_tensor(
                model,
                list(context_tokens) + list(continuation_tokens[:idx]),
                vocab_size=None,
            )
            for idx in range(len(continuation_tokens) + 1)
        ]

    expected = len(continuation_tokens) + 1
    if len(vectors) != expected:
        raise ValueError(
            f"Block probability API returned {len(vectors)} vectors, expected {expected}."
        )

    normalised: list[torch.Tensor] = []
    for probs in vectors:
        probs = torch.clamp(probs, min=0.0)
        if vocab_size is not None:
            probs = probs[:vocab_size]
        total = probs.sum()
        if total <= 0:
            raise ValueError("Model returned a probability vector with non-positive mass.")
        normalised.append(probs / total)
    return normalised
