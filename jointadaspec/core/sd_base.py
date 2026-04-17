"""Abstract base class and result dataclass for speculative decoders."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GenerationResult:
    """Output of a single :meth:`SpeculativeDecoder.generate` call."""

    generated_ids: list[int]
    """Token IDs produced (excluding the prompt)."""

    acceptance_rate: float
    """Fraction of draft tokens accepted (0.0 for vanilla AR)."""

    total_time_ms: float
    """Wall-clock time of the generate call in milliseconds."""

    n_target_calls: int
    """Number of target-model forward passes."""

    n_draft_calls: int
    """Number of draft-model forward passes (0 for vanilla AR)."""

    n_tokens_generated: int
    """Total tokens produced (len(generated_ids))."""

    per_step_metrics: list[dict[str, Any]] = field(default_factory=list)
    """Per-iteration diagnostics: H, K, k, action_length, T, accepted."""


class SpeculativeDecoder(ABC):
    """Common interface for all speculative-decoding variants.

    Subclasses must:
    - Accept an explicit ``torch.Generator`` for all random operations.
    - Measure time via ``torch.cuda.synchronize()`` + ``time.perf_counter()``.
    - Never call ``torch.manual_seed`` or ``random.seed`` internally.
    """

    @abstractmethod
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        generator: torch.Generator,
    ) -> GenerationResult:
        """Generate up to *max_new_tokens* tokens given *prompt_ids*.

        Parameters
        ----------
        prompt_ids:
            Tokenised prompt (list of integer token IDs).
        max_new_tokens:
            Hard limit on output length.
        generator:
            ``torch.Generator`` used for all stochastic operations.  Callers
            set the seed; implementations must forward it to every sampling
            call.

        Returns
        -------
        GenerationResult
            Generated tokens plus timing and acceptance diagnostics.
        """

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _start_timer(device: str | torch.device) -> float:
        """Synchronise GPU (if applicable) and return ``time.perf_counter()``."""
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _stop_timer(start: float, device: str | torch.device) -> float:
        """Synchronise GPU and return elapsed milliseconds since *start*."""
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0
