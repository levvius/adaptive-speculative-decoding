"""Entropy-threshold adaptive draft-length baseline."""

from __future__ import annotations

from typing import Any

import torch

from jointadaspec.baselines.fixed_sd import _generate_windowed_sd
from jointadaspec.core.sd_base import GenerationResult, SpeculativeDecoder
from jointadaspec.utils.probs import common_vocab_size


class SpecDecPPDecoder(SpeculativeDecoder):
    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        gamma_max: int,
        entropy_threshold: float,
        eos_token_id: int | None = None,
    ) -> None:
        if gamma_max <= 0:
            raise ValueError("gamma_max must be positive.")
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma_max = gamma_max
        self.entropy_threshold = entropy_threshold
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
            gamma=self.gamma_max,
            threshold=1.0,
            eos_token_id=self.eos_token_id,
            device=self.device,
            entropy_threshold=self.entropy_threshold,
            common_vocab_n=self.common_vocab_n,
        )
