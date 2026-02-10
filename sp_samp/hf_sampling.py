from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch

from .hf_adapter import HFModel, KVCacheState
from .sampling import SamplingStats


def _torch_multinomial(probs: torch.Tensor) -> int:
    token = torch.multinomial(probs, num_samples=1)
    return int(token.item())


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def sample_baseline_hf(
    model: HFModel,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    seed: Optional[int] = None,
    return_stats: bool = False,
) -> Union[List[int], Tuple[List[int], SamplingStats]]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if seed is not None:
        torch.manual_seed(seed)
    stats = SamplingStats()
    generated: List[int] = []
    with torch.no_grad():
        state = model.prefill(prompt_tokens)
        for _ in range(max_new_tokens):
            stats.steps += 1
            probs = _softmax(state.logits).squeeze(0)
            token = _torch_multinomial(probs)
            generated.append(token)
            stats.target_tokens += 1
            state = model.step([token], state)
            if eos_id is not None and token == eos_id:
                break
    if return_stats:
        stats.proposed = stats.target_tokens
        stats.accepted = stats.target_tokens
        return generated, stats
    return generated


def speculative_sample_hf(
    target_model: HFModel,
    draft_model: HFModel,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    k: int,
    eos_id: Optional[int] = None,
    seed: Optional[int] = None,
    eps: float = 1e-12,
    return_stats: bool = False,
) -> Union[List[int], Tuple[List[int], SamplingStats]]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if target_model.vocab_size != draft_model.vocab_size:
        raise ValueError("target and draft vocab sizes must match.")
    if seed is not None:
        torch.manual_seed(seed)

    stats = SamplingStats()
    generated: List[int] = []

    with torch.no_grad():
        target_state = target_model.prefill(prompt_tokens)
        draft_state = draft_model.prefill(prompt_tokens)

        while len(generated) < max_new_tokens:
            stats.steps += 1
            remaining = max_new_tokens - len(generated)
            draft_steps = min(k, remaining)

            draft_tokens: List[int] = []
            draft_probs: List[torch.Tensor] = []

            for _ in range(draft_steps):
                q_probs = _softmax(draft_state.logits).squeeze(0)
                token = _torch_multinomial(q_probs)
                draft_tokens.append(token)
                draft_probs.append(q_probs)
                draft_state = draft_model.step([token], draft_state)
                if eos_id is not None and token == eos_id:
                    break

            stats.proposed += len(draft_tokens)
            accepted_all = True

            for i, token in enumerate(draft_tokens):
                p_probs = _softmax(target_state.logits).squeeze(0)
                q_probs = draft_probs[i]
                q_token = float(q_probs[token])
                p_token = float(p_probs[token])
                alpha = min(1.0, p_token / max(q_token, eps))
                if torch.rand(1).item() > alpha:
                    accepted_all = False
                    stats.rejections += 1
                    stats.accepted += i
                    residual = torch.clamp(p_probs - q_probs, min=0.0)
                    residual_sum = float(residual.sum())
                    if residual_sum <= 0.0:
                        residual = p_probs
                        residual_sum = float(residual.sum())
                    residual = residual / residual_sum
                    token_new = _torch_multinomial(residual)
                    generated.append(token_new)
                    stats.target_tokens += 1
                    target_state = target_model.step([token_new], target_state)
                    if eos_id is not None and token_new == eos_id:
                        if return_stats:
                            return generated, stats
                        return generated
                    break
                else:
                    generated.append(token)
                    stats.target_tokens += 1
                    target_state = target_model.step([token], target_state)
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

            if not accepted_all:
                # Reset draft cache to match new prefix.
                draft_state = draft_model.prefill(list(prompt_tokens) + generated)
                continue

            stats.accepted += len(draft_tokens)

            if len(generated) < max_new_tokens:
                p_probs = _softmax(target_state.logits).squeeze(0)
                token = _torch_multinomial(p_probs)
                generated.append(token)
                stats.target_tokens += 1
                target_state = target_model.step([token], target_state)
                draft_state = draft_model.step([token], draft_state)
                if eos_id is not None and token == eos_id:
                    if return_stats:
                        return generated, stats
                    return generated

    if return_stats:
        return generated, stats
    return generated
