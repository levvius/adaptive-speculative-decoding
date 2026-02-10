from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from .models import BaseModel


@dataclass
class SamplingStats:
    proposed: int = 0
    accepted: int = 0
    steps: int = 0
    target_tokens: int = 0
    rejections: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def avg_tokens_per_step(self) -> float:
        return self.target_tokens / self.steps if self.steps else 0.0


def _weighted_choice(
    rng: random.Random, weights: Sequence[float], total: Optional[float] = None
) -> int:
    if total is None:
        total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("Total weight must be positive.")
    r = rng.random() * total
    cumulative = 0.0
    last_index = len(weights) - 1
    for i, w in enumerate(weights):
        cumulative += float(w)
        if r < cumulative:
            return i
    return last_index


def sample_baseline(
    target_model: BaseModel,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    rng: Optional[random.Random] = None,
    eos_id: Optional[int] = None,
    return_stats: bool = False,
) -> Union[List[int], Tuple[List[int], SamplingStats]]:
    """Sample tokens from the target model only."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if rng is None:
        rng = random.Random()
    generated: List[int] = []
    stats = SamplingStats()
    for _ in range(max_new_tokens):
        stats.steps += 1
        context = list(prompt_tokens) + generated
        probs = target_model.next_token_probs(context)
        token = _weighted_choice(rng, probs)
        generated.append(token)
        stats.target_tokens += 1
        if eos_id is not None and token == eos_id:
            break
    if return_stats:
        stats.proposed = stats.target_tokens
        stats.accepted = stats.target_tokens
        return generated, stats
    return generated


def speculative_sample(
    target_model: BaseModel,
    draft_model: BaseModel,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    k: int,
    rng: Optional[random.Random] = None,
    eos_id: Optional[int] = None,
    eps: float = 1e-12,
    return_stats: bool = False,
) -> Union[List[int], Tuple[List[int], SamplingStats]]:
    """Speculative Sampling (SpS) reference implementation.

    Returns only the newly generated tokens (not including the prompt).
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if target_model.vocab_size != draft_model.vocab_size:
        raise ValueError("target and draft vocab sizes must match.")
    if rng is None:
        rng = random.Random()

    generated: List[int] = []
    stats = SamplingStats()
    vocab_size = target_model.vocab_size

    while len(generated) < max_new_tokens:
        stats.steps += 1
        remaining = max_new_tokens - len(generated)
        draft_steps = min(k, remaining)
        draft_tokens: List[int] = []
        draft_dists: List[List[float]] = []

        # 1) Draft proposes up to k tokens.
        for _ in range(draft_steps):
            context = list(prompt_tokens) + generated + draft_tokens
            q = draft_model.next_token_probs(context)
            if len(q) != vocab_size:
                raise ValueError("Draft distribution has wrong length.")
            token = _weighted_choice(rng, q)
            draft_tokens.append(token)
            draft_dists.append(list(q))
            if eos_id is not None and token == eos_id:
                break
        stats.proposed += len(draft_tokens)

        # 2) Target validates proposed tokens.
        accepted_all = True
        for i, token in enumerate(draft_tokens):
            context = list(prompt_tokens) + generated + draft_tokens[:i]
            p = target_model.next_token_probs(context)
            if len(p) != vocab_size:
                raise ValueError("Target distribution has wrong length.")
            q = draft_dists[i]
            q_token = float(q[token])
            p_token = float(p[token])
            alpha = min(1.0, p_token / max(q_token, eps))
            if rng.random() > alpha:
                # Reject and sample from (p - q)+
                accepted_all = False
                stats.rejections += 1
                stats.accepted += i
                residual = [max(float(pi) - float(qi), 0.0) for pi, qi in zip(p, q)]
                residual_total = sum(residual)
                if residual_total <= 0.0:
                    residual = list(p)
                    residual_total = sum(residual)
                token_new = _weighted_choice(rng, residual, total=residual_total)
                generated.append(token_new)
                stats.target_tokens += 1
                if eos_id is not None and token_new == eos_id:
                    if return_stats:
                        return generated, stats
                    return generated
                break
            else:
                generated.append(token)
                stats.target_tokens += 1
                if eos_id is not None and token == eos_id:
                    stats.accepted += i + 1
                    if return_stats:
                        return generated, stats
                    return generated
                if len(generated) >= max_new_tokens:
                    stats.accepted += i + 1
                    if return_stats:
                        return generated, stats
                    return generated

        # 3) If all accepted, sample one extra token from target.
        if accepted_all and len(generated) < max_new_tokens:
            stats.accepted += len(draft_tokens)
            context = list(prompt_tokens) + generated
            p = target_model.next_token_probs(context)
            if len(p) != vocab_size:
                raise ValueError("Target distribution has wrong length.")
            token = _weighted_choice(rng, p)
            generated.append(token)
            stats.target_tokens += 1
            if eos_id is not None and token == eos_id:
                if return_stats:
                    return generated, stats
                return generated

    if return_stats:
        return generated, stats
    return generated
