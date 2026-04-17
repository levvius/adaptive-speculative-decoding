"""Inference-time JointAdaSpec decoder."""

from __future__ import annotations

from typing import Any

import torch

from jointadaspec.core.features import entropy, kl_divergence
from jointadaspec.core.sd_base import GenerationResult, SpeculativeDecoder
from jointadaspec.core.verification import fuzzy_verification, tv_distance_step
from jointadaspec.inference.policy import JointAdaSpecPolicy
from jointadaspec.utils.probs import common_vocab_size, next_token_probs_tensor


def _sample_token(probs: torch.Tensor, generator: torch.Generator) -> int:
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


class JointAdaSpecDecoder(SpeculativeDecoder):
    """Online policy-controlled fuzzy speculative decoding."""

    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        policy: JointAdaSpecPolicy,
        eos_token_id: int | None = None,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.policy = policy
        self.eos_token_id = eos_token_id
        self.device = str(getattr(target_model, "device", "cpu"))
        self.common_vocab_n = common_vocab_size(target_model, draft_model)

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        generator: torch.Generator,
    ) -> GenerationResult:
        context_tokens = list(prompt_ids)
        generated_ids: list[int] = []
        per_step_metrics: list[dict[str, Any]] = []
        proposed = 0
        accepted = 0
        n_target_calls = 0
        n_draft_calls = 0
        k = 0

        started = self._start_timer(self.device)
        while len(generated_ids) < max_new_tokens:
            p_probs = next_token_probs_tensor(self.target_model, context_tokens, self.common_vocab_n)
            q_probs = next_token_probs_tensor(self.draft_model, context_tokens, self.common_vocab_n)
            n_target_calls += 1
            n_draft_calls += 1

            H = entropy(q_probs)
            K = kl_divergence(q_probs, p_probs)
            action_length, threshold = self.policy.get_action(H=H, K=K, k=k)

            if action_length == "stop":
                emitted_token = _sample_token(p_probs, generator)
                accepted_flag = False
                d_step = 0.0
                k = 0
            else:
                draft_token = _sample_token(q_probs, generator)
                proposed += 1
                accepted_flag, corrective_fn = fuzzy_verification(
                    p=p_probs,
                    q=q_probs,
                    draft_token=draft_token,
                    T=threshold,
                    generator=generator,
                )
                if accepted_flag:
                    emitted_token = draft_token
                    accepted += 1
                    k = min(k + 1, self.policy.config.gamma_max)
                else:
                    emitted_token = corrective_fn(generator)
                    k = 0
                d_step = tv_distance_step(p_probs, q_probs, threshold)

            context_tokens.append(int(emitted_token))
            generated_ids.append(int(emitted_token))
            per_step_metrics.append(
                {
                    "H": H,
                    "K": K,
                    "k": k,
                    "action_length": action_length,
                    "threshold": threshold,
                    "accepted": bool(accepted_flag),
                    "d_step": d_step,
                }
            )
            if self.eos_token_id is not None and emitted_token == self.eos_token_id:
                break

        total_time_ms = self._stop_timer(started, self.device)
        acceptance_rate = (accepted / proposed) if proposed else 0.0
        return GenerationResult(
            generated_ids=generated_ids,
            acceptance_rate=acceptance_rate,
            total_time_ms=total_time_ms,
            n_target_calls=n_target_calls,
            n_draft_calls=n_draft_calls,
            n_tokens_generated=len(generated_ids),
            per_step_metrics=per_step_metrics,
        )
