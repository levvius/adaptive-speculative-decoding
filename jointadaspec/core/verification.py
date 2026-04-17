"""Speculative-decoding verification rules.

All functions are pure (no side-effects beyond the provided ``torch.Generator``)
and work directly on probability tensors.
"""

from __future__ import annotations

from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_from(probs: torch.Tensor, generator: torch.Generator) -> int:
    """Multinomial sample from a 1-D probability vector."""
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def _normalise_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.detach().float().flatten()
    probs = torch.clamp(probs, min=0.0)
    total = probs.sum()
    if total <= 0:
        raise ValueError("Probability vector must have positive mass.")
    return probs / total


def _residual_sample(
    p: torch.Tensor,
    q: torch.Tensor,
    generator: torch.Generator,
) -> int:
    """Sample from the renormalised residual distribution max(0, p-q)/Z."""
    p = _normalise_probs(p)
    q = _normalise_probs(q)
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    if total < 1e-12:
        # Fallback: sample directly from p (edge case when p ≈ q everywhere)
        return _sample_from(p, generator)
    return _sample_from(residual / total, generator)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def modified_rejection_sampling(
    p: torch.Tensor,
    q: torch.Tensor,
    draft_token: int,
    generator: torch.Generator,
) -> tuple[bool, Callable[[torch.Generator], int]]:
    """Exact speculative verification (T = 1, lossless).

    Parameters
    ----------
    p:
        Target-model probability vector (vocab-size, float).
    q:
        Draft-model probability vector (vocab-size, float).
    draft_token:
        Token index proposed by the draft model.
    generator:
        PRNG state used for the Bernoulli acceptance draw.

    Returns
    -------
    accepted : bool
        Whether the draft token is accepted.
    corrective_sample_fn : Callable[[Generator], int]
        Called when ``accepted`` is False to draw the corrected token.
        (Also used for the bonus token when all k drafts are accepted.)
    """
    p_x = p[draft_token].item()
    q_x = q[draft_token].item()
    accept_prob = min(1.0, p_x / (q_x + 1e-12))

    u = torch.rand(1, generator=generator).item()
    accepted = u < accept_prob

    def corrective_sample_fn(gen: torch.Generator) -> int:
        return _residual_sample(p, q, gen)

    return accepted, corrective_sample_fn


def fuzzy_step_distribution(p: torch.Tensor, q: torch.Tensor, T: float) -> torch.Tensor:
    """Return the one-step output distribution induced by fuzzy verification.

    The returned vector is the exact distribution over the emitted token after:
    1. sampling a draft token from ``q``;
    2. accepting it with probability ``min(1, T * p(x) / q(x))``;
    3. otherwise sampling from the corrective residual distribution.
    """
    if T < 1.0:
        raise ValueError(f"T must be >= 1.0, got {T}")

    p = _normalise_probs(p)
    q = _normalise_probs(q)
    beta = torch.clamp(T * p / (q + 1e-12), max=1.0)
    accepted_mass = beta * q
    reject_prob = torch.clamp(1.0 - accepted_mass.sum(), min=0.0)
    residual = torch.clamp(p - accepted_mass, min=0.0)
    residual_mass = residual.sum()
    if residual_mass <= 1e-12 or reject_prob <= 1e-12:
        return _normalise_probs(accepted_mass)
    return _normalise_probs(accepted_mass + reject_prob * (residual / residual_mass))


def tv_distance_step(p: torch.Tensor, q: torch.Tensor, T: float) -> float:
    """Total variation distance between fuzzy one-step output and target ``p``."""
    p = _normalise_probs(p)
    output = fuzzy_step_distribution(p, q, T)
    return float(0.5 * torch.abs(output - p).sum().item())


def fuzzy_verification(
    p: torch.Tensor,
    q: torch.Tensor,
    draft_token: int,
    T: float,
    generator: torch.Generator,
) -> tuple[bool, Callable[[torch.Generator], int]]:
    """Fuzzy speculative verification with threshold T ≥ 1.

    Acceptance probability: β_T(x) = min(1, T · p(x) / q(x)).

    At T = 1 this reduces to standard modified rejection sampling.
    At T > 1 the acceptance rate is strictly higher, but the output
    distribution deviates from p (controlled trade-off).

    Parameters
    ----------
    p, q, draft_token, generator:
        Same as :func:`modified_rejection_sampling`.
    T:
        Verification threshold.  Must be ≥ 1.0.
    """
    if T < 1.0:
        raise ValueError(f"T must be >= 1.0, got {T}")

    p = _normalise_probs(p)
    q = _normalise_probs(q)
    p_x = p[draft_token].item()
    q_x = q[draft_token].item()
    accept_prob = min(1.0, T * p_x / (q_x + 1e-12))

    u = torch.rand(1, generator=generator).item()
    accepted = u < accept_prob

    # Corrective distribution for fuzzy SD:
    # residual_{T}(x) ∝ max(0, p(x) - β_T(x)·q(x))
    def corrective_sample_fn(gen: torch.Generator) -> int:
        beta = torch.clamp(T * p / (q + 1e-12), max=1.0)
        residual = torch.clamp(p - beta * q, min=0.0)
        total = residual.sum()
        if total < 1e-12:
            return _sample_from(p, gen)
        return _sample_from(residual / total, gen)

    return accepted, corrective_sample_fn


def verify_draft_chain(
    p_list: list[torch.Tensor],
    q_list: list[torch.Tensor],
    draft_tokens: list[int],
    T: float,
    generator: torch.Generator,
    p_bonus: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Sequentially verify a chain of draft tokens, stopping at first rejection.

    Parameters
    ----------
    p_list:
        List of target-model distributions, one per draft position.
    q_list:
        List of draft-model distributions, one per draft position.
    draft_tokens:
        Proposed token IDs, one per draft position.
    T:
        Verification threshold forwarded to :func:`fuzzy_verification`.
    generator:
        PRNG state.

    Returns
    -------
    n_accepted : int
        Number of accepted draft tokens (may be 0).
    corrective_token : int
        Corrective token drawn from the residual distribution at the first
        rejection position (or bonus token sampled from ``p`` if all accepted).
    """
    n = len(draft_tokens)
    if len(q_list) != n:
        raise ValueError("q_list and draft_tokens must have the same length")
    if len(p_list) == n + 1 and p_bonus is None:
        verify_p_list = p_list[:-1]
        p_bonus = p_list[-1]
    elif len(p_list) == n:
        verify_p_list = p_list
    else:
        raise ValueError("p_list must have length n or n + 1")
    if len(verify_p_list) != n:
        raise ValueError("p_list and draft_tokens must have the same length")
    if p_bonus is None:
        raise ValueError("p_bonus is required when p_list does not include a bonus distribution")

    for i in range(n):
        accepted, corrective_fn = fuzzy_verification(
            verify_p_list[i], q_list[i], draft_tokens[i], T, generator
        )
        if not accepted:
            return i, corrective_fn(generator)

    bonus = _sample_from(_normalise_probs(p_bonus), generator)
    return n, bonus
