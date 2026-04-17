"""Exact fixed-window speculative decoding baseline."""

from __future__ import annotations

from typing import Any

import torch

from jointadaspec.core.features import entropy, kl_divergence
from jointadaspec.core.sd_base import GenerationResult, SpeculativeDecoder
from jointadaspec.core.verification import verify_draft_chain
from jointadaspec.utils.probs import common_vocab_size, next_token_probs_tensor


def _generate_windowed_sd(
    *,
    target_model: Any,
    draft_model: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    generator: torch.Generator,
    gamma: int,
    threshold: float,
    eos_token_id: int | None,
    device: str,
    entropy_threshold: float | None = None,
    common_vocab_n: int | None = None,
) -> GenerationResult:
    context_tokens = list(prompt_ids)
    generated_ids: list[int] = []
    per_step_metrics: list[dict[str, float | int | str | bool]] = []
    proposed = 0
    accepted_total = 0
    n_target_calls = 0
    n_draft_calls = 0

    started = SpeculativeDecoder._start_timer(device)
    while len(generated_ids) < max_new_tokens:
        remaining = max_new_tokens - len(generated_ids)
        draft_tokens: list[int] = []
        q_list: list[torch.Tensor] = []
        p_list: list[torch.Tensor] = []
        should_force_target = False

        for k in range(min(gamma, remaining)):
            draft_context = context_tokens + draft_tokens
            q_probs = next_token_probs_tensor(draft_model, draft_context, common_vocab_n)
            n_draft_calls += 1
            H = entropy(q_probs)
            if entropy_threshold is not None and H > entropy_threshold and draft_tokens:
                break
            if entropy_threshold is not None and H > entropy_threshold and not draft_tokens:
                should_force_target = True
                break

            token = int(torch.multinomial(q_probs, num_samples=1, generator=generator).item())
            draft_tokens.append(token)
            q_list.append(q_probs)
            p_probs = next_token_probs_tensor(target_model, draft_context, common_vocab_n)
            p_list.append(p_probs)
            n_target_calls += 1

            per_step_metrics.append(
                {
                    "H": H,
                    "K": kl_divergence(q_probs, p_probs),
                    "k": k,
                    "action_length": "continue",
                    "threshold": threshold,
                    "accepted": None,
                }
            )

        if should_force_target or not draft_tokens:
            p_probs = next_token_probs_tensor(target_model, context_tokens, common_vocab_n)
            n_target_calls += 1
            token = int(torch.multinomial(p_probs, num_samples=1, generator=generator).item())
            context_tokens.append(token)
            generated_ids.append(token)
            per_step_metrics.append(
                {
                    "H": 0.0,
                    "K": 0.0,
                    "k": 0,
                    "action_length": "stop",
                    "threshold": threshold,
                    "accepted": False,
                }
            )
            if eos_token_id is not None and token == eos_token_id:
                break
            continue

        p_bonus = next_token_probs_tensor(target_model, context_tokens + draft_tokens, common_vocab_n)
        n_target_calls += 1
        proposed += len(draft_tokens)
        n_accepted, corrective_token = verify_draft_chain(
            p_list=p_list,
            q_list=q_list,
            draft_tokens=draft_tokens,
            T=threshold,
            generator=generator,
            p_bonus=p_bonus,
        )
        accepted_total += n_accepted

        for token in draft_tokens[:n_accepted]:
            if len(generated_ids) >= max_new_tokens:
                break
            context_tokens.append(token)
            generated_ids.append(token)
            if eos_token_id is not None and token == eos_token_id:
                total_time_ms = SpeculativeDecoder._stop_timer(started, device)
                return GenerationResult(
                    generated_ids=generated_ids,
                    acceptance_rate=(accepted_total / proposed) if proposed else 0.0,
                    total_time_ms=total_time_ms,
                    n_target_calls=n_target_calls,
                    n_draft_calls=n_draft_calls,
                    n_tokens_generated=len(generated_ids),
                    per_step_metrics=per_step_metrics,
                )

        if len(generated_ids) < max_new_tokens:
            context_tokens.append(corrective_token)
            generated_ids.append(corrective_token)
            if eos_token_id is not None and corrective_token == eos_token_id:
                break

    total_time_ms = SpeculativeDecoder._stop_timer(started, device)
    return GenerationResult(
        generated_ids=generated_ids,
        acceptance_rate=(accepted_total / proposed) if proposed else 0.0,
        total_time_ms=total_time_ms,
        n_target_calls=n_target_calls,
        n_draft_calls=n_draft_calls,
        n_tokens_generated=len(generated_ids),
        per_step_metrics=per_step_metrics,
    )


class FixedSDDecoder(SpeculativeDecoder):
    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        gamma: int,
        eos_token_id: int | None = None,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma
        self.eos_token_id = eos_token_id
        self.device = str(getattr(target_model, "device", "cpu"))
        self.common_vocab_n = common_vocab_size(target_model, draft_model)

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        generator: torch.Generator,
    ) -> GenerationResult:
        return _generate_windowed_sd(
            target_model=self.target_model,
            draft_model=self.draft_model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            generator=generator,
            gamma=self.gamma,
            threshold=1.0,
            eos_token_id=self.eos_token_id,
            device=self.device,
            common_vocab_n=self.common_vocab_n,
        )
