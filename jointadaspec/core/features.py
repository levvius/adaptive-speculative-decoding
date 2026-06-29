"""Feature extraction for the MDP state representation.

``entropy`` and ``kl_divergence`` accept either logits or already normalised
probabilities. Quantisation maps ``(H, K, k)`` to a linear state index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jointadaspec.mdp.spaces import MDPConfig


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _as_probabilities(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        raise ValueError("Expected a non-empty tensor.")
    if torch.all(values >= 0):
        total = values.sum()
        if torch.isfinite(total) and torch.isclose(total, torch.tensor(1.0), atol=1e-4, rtol=1e-4):
            return values
    return torch.softmax(values, dim=-1)


def entropy(q: torch.Tensor) -> float:
    """Shannon entropy of a probability distribution q in nats.

    Uses the numerically stable log-softmax form so the function accepts
    both normalised and unnormalised inputs (logits).

    Parameters
    ----------
    q:
        1-D tensor of probabilities or logits.
    """
    probs = _as_probabilities(q)
    safe = torch.clamp(probs, min=1e-12)
    h = -(probs * safe.log()).sum()
    return float(h.clamp(min=0.0).item())


def kl_divergence(q: torch.Tensor, p: torch.Tensor) -> float:
    """KL divergence KL(q || p) in nats.

    Parameters
    ----------
    q:
        Draft model distribution (probabilities or logits).
    p:
        Target model distribution (probabilities or logits).
    """
    probs_q = _as_probabilities(q)
    probs_p = _as_probabilities(p)
    safe_q = torch.clamp(probs_q, min=1e-12)
    safe_p = torch.clamp(probs_p, min=1e-12)
    kl = (probs_q * (safe_q.log() - safe_p.log())).sum()
    return float(kl.clamp(min=0.0).item())


def draft_confidence(q: torch.Tensor, kind: str = "max_prob") -> float:
    """Draft-only acceptance proxy in [0, 1], computed from ``q`` alone.

    No target forward pass is required, so this is a zero-cost signal the policy
    can read to decide whether to keep drafting or verify/stop early. The draft's
    own token-level confidence is a strong predictor of whether the target will
    accept the drafted token.

    Parameters
    ----------
    q:
        1-D tensor of probabilities or logits (the draft next-token distribution).
    kind:
        ``"max_prob"`` — top-1 probability ``max_x q(x)`` (default);
        ``"margin"``   — gap between the top-1 and top-2 probabilities.
    """
    probs = _as_probabilities(q)
    if kind == "max_prob":
        return float(probs.max().clamp(0.0, 1.0).item())
    if kind == "margin":
        if probs.numel() == 1:
            return float(probs.max().clamp(0.0, 1.0).item())
        top2 = torch.topk(probs, k=2).values
        return float((top2[0] - top2[1]).clamp(0.0, 1.0).item())
    raise ValueError(f"Unknown draft_confidence kind: {kind!r}")


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

def quantize(H: float, K: float, k: int, config: "MDPConfig", C: float = 0.0) -> int:
    """Map continuous (H, K, C) and discrete k to a linear MDP state index.

    The optional draft-confidence axis ``C`` is the *outermost* dimension, so when
    ``config.N_C == 1`` (default) the returned index is identical to the legacy
    3-D (H, K, k) layout and :func:`dequantize` keeps returning a 3-tuple.

    Parameters
    ----------
    H:
        Entropy value (nats).  Clipped to [0, H_max].
    K:
        KL divergence (nats).  Clipped to [0, K_max].
    k:
        Draft position.  Clipped to [0, gamma_max].
    config:
        MDP configuration with grid parameters.
    C:
        Draft confidence in [0, C_max].  Ignored when ``config.N_C == 1``.

    Returns
    -------
    int
        Linear state index in [0, N_C * N_H * N_K * (gamma_max + 1)).
    """
    N_H, N_K = config.N_H, config.N_K
    N_C = getattr(config, "N_C", 1)
    C_max = getattr(config, "C_max", 1.0)
    gamma_max = config.gamma_max
    gp1 = gamma_max + 1

    # Bin indices (0-based, clipped to valid range)
    H = max(0.0, min(float(H), float(config.H_max)))
    K = max(0.0, min(float(K), float(config.K_max)))
    i_H = min(int(H / config.H_max * N_H), N_H - 1)
    i_K = min(int(K / config.K_max * N_K), N_K - 1)
    i_k = min(max(k, 0), gamma_max)

    base = i_H * (N_K * gp1) + i_K * gp1 + i_k
    if N_C <= 1:
        return base
    C = max(0.0, min(float(C), float(C_max)))
    i_C = min(int(C / C_max * N_C), N_C - 1)
    return i_C * (N_H * N_K * gp1) + base


def dequantize(state_idx: int, config: "MDPConfig") -> tuple[int, int, int]:
    """Invert :func:`quantize` — recover (i_H, i_K, i_k) bin indices.

    The draft-confidence axis is stripped via modulo, so this keeps returning a
    3-tuple regardless of ``config.N_C`` (legacy callers are unaffected). Use
    :func:`dequantize_full` when the confidence bin is needed.
    """
    base_size = config.N_H * config.N_K * (config.gamma_max + 1)
    base_idx = state_idx % base_size
    gp1 = config.gamma_max + 1  # stride for k axis
    stride_H = config.N_K * gp1

    i_H = base_idx // stride_H
    remainder = base_idx % stride_H
    i_K = remainder // gp1
    i_k = remainder % gp1

    return i_H, i_K, i_k


def dequantize_full(state_idx: int, config: "MDPConfig") -> tuple[int, int, int, int]:
    """Recover (i_H, i_K, i_C, i_k) including the outer draft-confidence bin."""
    base_size = config.N_H * config.N_K * (config.gamma_max + 1)
    i_C = state_idx // base_size
    i_H, i_K, i_k = dequantize(state_idx, config)
    return i_H, i_K, i_C, i_k


def confidence_center(i_C: int, config: "MDPConfig") -> float:
    """Bin-center draft confidence for confidence bin ``i_C`` (0.0 when N_C == 1)."""
    N_C = getattr(config, "N_C", 1)
    if N_C <= 1:
        return 0.0
    return (i_C + 0.5) * getattr(config, "C_max", 1.0) / N_C
