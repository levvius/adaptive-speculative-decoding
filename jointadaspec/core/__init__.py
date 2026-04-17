"""Core JointAdaSpec building blocks."""

from .features import dequantize, entropy, kl_divergence, quantize
from .sd_base import GenerationResult, SpeculativeDecoder
from .verification import (
    fuzzy_step_distribution,
    fuzzy_verification,
    modified_rejection_sampling,
    tv_distance_step,
    verify_draft_chain,
)

__all__ = [
    "GenerationResult",
    "SpeculativeDecoder",
    "dequantize",
    "entropy",
    "fuzzy_step_distribution",
    "fuzzy_verification",
    "kl_divergence",
    "modified_rejection_sampling",
    "quantize",
    "tv_distance_step",
    "verify_draft_chain",
]
