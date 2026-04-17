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


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

def quantize(H: float, K: float, k: int, config: "MDPConfig") -> int:
    """Map continuous (H, K) and discrete k to a linear MDP state index.

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

    Returns
    -------
    int
        Linear state index in [0, N_H * N_K * (gamma_max + 1)).
    """
    N_H, N_K = config.N_H, config.N_K
    gamma_max = config.gamma_max

    # Bin indices (0-based, clipped to valid range)
    H = max(0.0, min(float(H), float(config.H_max)))
    K = max(0.0, min(float(K), float(config.K_max)))
    i_H = min(int(H / config.H_max * N_H), N_H - 1)
    i_K = min(int(K / config.K_max * N_K), N_K - 1)
    i_k = min(max(k, 0), gamma_max)

    return i_H * (N_K * (gamma_max + 1)) + i_K * (gamma_max + 1) + i_k


def dequantize(state_idx: int, config: "MDPConfig") -> tuple[int, int, int]:
    """Invert :func:`quantize` — recover (i_H, i_K, i_k) bin indices.

    Parameters
    ----------
    state_idx:
        Linear state index in [0, N_H * N_K * (gamma_max + 1)).
    config:
        MDP configuration.

    Returns
    -------
    (i_H, i_K, i_k) : tuple[int, int, int]
        Bin indices along each axis.
    """
    gp1 = config.gamma_max + 1  # stride for k axis
    stride_H = config.N_K * gp1

    i_H = state_idx // stride_H
    remainder = state_idx % stride_H
    i_K = remainder // gp1
    i_k = remainder % gp1

    return i_H, i_K, i_k
