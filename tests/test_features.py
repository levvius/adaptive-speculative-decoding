from __future__ import annotations

import math

import torch

from jointadaspec.core.features import dequantize, entropy, kl_divergence, quantize
from jointadaspec.mdp.spaces import MDPConfig


def test_entropy_uniform() -> None:
    probs = torch.full((4,), 0.25)
    assert abs(entropy(probs) - math.log(4.0)) < 1e-6


def test_kl_identity() -> None:
    probs = torch.tensor([0.6, 0.3, 0.1])
    assert abs(kl_divergence(probs, probs)) < 1e-8


def test_quantize_roundtrip() -> None:
    config = MDPConfig(N_H=4, N_K=5, gamma_max=3)
    state_idx = quantize(H=2.2, K=3.1, k=2, config=config)
    i_H, i_K, i_k = dequantize(state_idx, config)
    assert i_k == 2
    assert 0 <= i_H < config.N_H
    assert 0 <= i_K < config.N_K


def test_quantize_clamp() -> None:
    config = MDPConfig(N_H=4, N_K=4, gamma_max=2)
    state_idx = quantize(H=999.0, K=-1.0, k=10, config=config)
    i_H, i_K, i_k = dequantize(state_idx, config)
    assert i_H == config.N_H - 1
    assert i_K == 0
    assert i_k == config.gamma_max
