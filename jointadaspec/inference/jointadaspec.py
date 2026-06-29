"""Inference-time JointAdaSpec decoder."""

from __future__ import annotations

from typing import Any

import torch

from jointadaspec.core.features import draft_confidence, entropy, kl_divergence
from jointadaspec.core.sd_base import GenerationResult, SpeculativeDecoder
from jointadaspec.core.verification import tv_distance_step, verify_draft_chain
from jointadaspec.inference.policy import JointAdaSpecPolicy
from jointadaspec.semantics import BLOCK_DECODER_SEMANTICS
from jointadaspec.utils.probs import (
    block_next_token_probs_tensor,
    common_vocab_size,
    next_token_probs_tensor,
)


def _sample_token(probs: torch.Tensor, generator: torch.Generator) -> int:
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


class JointAdaSpecDecoder(SpeculativeDecoder):
    """Online policy-controlled block fuzzy speculative decoding."""

    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        policy: JointAdaSpecPolicy,
        eos_token_id: int | None = None,
        conf_gate_tau: float = 0.0,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.policy = policy
        self.eos_token_id = eos_token_id
        # Inference-time early-verify gate: when the next draft token's confidence
        # falls below this threshold, stop drafting and verify the accumulated block
        # early instead of over-trusting a weak draft. 0.0 disables the gate.
        self.conf_gate_tau = float(conf_gate_tau)
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
        n_target_verified_positions = 0
        draft_tokens: list[int] = []
        q_list: list[torch.Tensor] = []
        K_prev = float(self.policy.config.K_init)

        started = self._start_timer(self.device)
        while len(generated_ids) < max_new_tokens:
            remaining = max_new_tokens - len(generated_ids)
            k = len(draft_tokens)
            draft_context = context_tokens + draft_tokens
            q_probs = next_token_probs_tensor(self.draft_model, draft_context, self.common_vocab_n)
            n_draft_calls += 1

            H = entropy(q_probs)
            C = draft_confidence(q_probs, self.policy.config.draft_conf_feature)
            action_length, threshold = self.policy.get_action(H=H, K=K_prev, k=k, C=C)
            must_verify = k >= self.policy.config.gamma_max
            must_verify = must_verify or (bool(draft_tokens) and k >= remaining)
            # Confidence gate (inference-time safety override, default off): when the
            # next draft token's confidence drops below tau, stop drafting and verify
            # the accumulated block early instead of over-trusting a weak draft.
            if self.conf_gate_tau > 0.0 and C < self.conf_gate_tau:
                must_verify = True
            should_continue = action_length == "continue" and not must_verify

            if should_continue:
                draft_token = _sample_token(q_probs, generator)
                draft_tokens.append(draft_token)
                q_list.append(q_probs)
                proposed += 1
                per_step_metrics.append(
                    {
                        "H": H,
                        "K": K_prev,
                        "C": C,
                        "k": k,
                        "action_length": "continue",
                        "threshold": 1.0,
                        "accepted": None,
                        "d_step": 0.0,
                    }
                )
                continue

            if not draft_tokens:
                p_probs = next_token_probs_tensor(self.target_model, context_tokens, self.common_vocab_n)
                n_target_calls += 1
                n_target_verified_positions += 1
                emitted_tokens = [_sample_token(p_probs, generator)]
                K_prev = kl_divergence(q_probs, p_probs)
                accepted_now = 0
                d_step = 0.0
            else:
                p_block = block_next_token_probs_tensor(
                    self.target_model,
                    context_tokens,
                    draft_tokens,
                    self.common_vocab_n,
                )
                n_target_calls += 1
                n_target_verified_positions += len(draft_tokens) + 1
                p_list = p_block[:-1]
                p_bonus = p_block[-1]
                n_accepted, corrective_token = verify_draft_chain(
                    p_list=p_list,
                    q_list=q_list,
                    draft_tokens=draft_tokens,
                    T=threshold,
                    generator=generator,
                    p_bonus=p_bonus,
                )
                accepted += n_accepted
                accepted_now = n_accepted
                emitted_tokens = list(draft_tokens[:n_accepted]) + [int(corrective_token)]
                K_values = [kl_divergence(q, p) for q, p in zip(q_list, p_list, strict=True)]
                K_prev = float(sum(K_values) / len(K_values)) if K_values else K_prev
                d_step = float(
                    sum(tv_distance_step(p, q, threshold) for p, q in zip(p_list, q_list, strict=True))
                    / max(len(q_list), 1)
                )

            per_step_metrics.append(
                {
                    "H": H,
                    "K": K_prev,
                    "C": C,
                    "k": k,
                    "action_length": "verify",
                    "threshold": threshold,
                    "accepted": accepted_now,
                    "d_step": d_step,
                }
            )
            draft_tokens = []
            q_list = []

            for emitted_token in emitted_tokens:
                if len(generated_ids) >= max_new_tokens:
                    break
                context_tokens.append(int(emitted_token))
                generated_ids.append(int(emitted_token))
                if self.eos_token_id is not None and emitted_token == self.eos_token_id:
                    break
            if self.eos_token_id is not None and generated_ids[-1:] == [self.eos_token_id]:
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
            n_target_verified_positions=n_target_verified_positions,
            decoder_semantics=BLOCK_DECODER_SEMANTICS,
        )
