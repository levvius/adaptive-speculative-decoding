"""Target-only autoregressive baseline."""

from __future__ import annotations

from typing import Any

import torch

from jointadaspec.core.sd_base import GenerationResult, SpeculativeDecoder
from jointadaspec.utils.probs import next_token_probs_tensor


class VanillaARDecoder(SpeculativeDecoder):
    def __init__(self, target_model: Any, eos_token_id: int | None = None) -> None:
        self.target_model = target_model
        self.eos_token_id = eos_token_id
        self.device = str(getattr(target_model, "device", "cpu"))

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        generator: torch.Generator,
    ) -> GenerationResult:
        context_tokens = list(prompt_ids)
        generated_ids: list[int] = []
        started = self._start_timer(self.device)

        for _ in range(max_new_tokens):
            probs = next_token_probs_tensor(self.target_model, context_tokens)
            token = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
            context_tokens.append(token)
            generated_ids.append(token)
            if self.eos_token_id is not None and token == self.eos_token_id:
                break

        total_time_ms = self._stop_timer(started, self.device)
        return GenerationResult(
            generated_ids=generated_ids,
            acceptance_rate=0.0,
            total_time_ms=total_time_ms,
            n_target_calls=len(generated_ids),
            n_draft_calls=0,
            n_tokens_generated=len(generated_ids),
            per_step_metrics=[],
        )
