"""Tests for the draft-confidence MDP state axis and the confidence gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jointadaspec.core.features import (
    confidence_center,
    dequantize,
    dequantize_full,
    draft_confidence,
    quantize,
)
from jointadaspec.inference import JointAdaSpecDecoder, JointAdaSpecPolicy
from jointadaspec.mdp import MDPConfig, collect_traces, estimate_mdp_parameters, solve_mdp
from jointadaspec.mdp.spaces import ActionSpace, StateSpace
from sp_samp.models import FixedModel


class ToyModelAdapter:
    def __init__(self, model: FixedModel) -> None:
        self.model = model
        self.vocab_size = model.vocab_size
        self.device = "cpu"

    def next_token_probs(self, context_tokens):
        return self.model.next_token_probs(context_tokens)


# ---------------------------------------------------------------------------
# draft_confidence feature
# ---------------------------------------------------------------------------

def test_draft_confidence_max_prob_and_margin() -> None:
    peaked = torch.tensor([0.7, 0.2, 0.1])
    uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
    assert draft_confidence(peaked, "max_prob") == pytest.approx(0.7)
    assert draft_confidence(uniform, "max_prob") == pytest.approx(0.25)
    # Margin: top1 - top2.
    assert draft_confidence(peaked, "margin") == pytest.approx(0.5)
    assert draft_confidence(uniform, "margin") == pytest.approx(0.0, abs=1e-6)


def test_draft_confidence_accepts_logits_and_rejects_unknown_kind() -> None:
    logits = torch.tensor([10.0, 0.0, 0.0])  # softmax is sharply peaked -> high conf
    assert draft_confidence(logits, "max_prob") > 0.99
    with pytest.raises(ValueError, match="Unknown draft_confidence"):
        draft_confidence(logits, "entropy")


# ---------------------------------------------------------------------------
# State-space confidence axis
# ---------------------------------------------------------------------------

def test_confidence_axis_disabled_is_backward_compatible() -> None:
    config = MDPConfig(N_H=5, N_K=4, gamma_max=3, T_levels=(1.0, 2.0))  # N_C defaults to 1
    base = MDPConfig(N_H=5, N_K=4, gamma_max=3, T_levels=(1.0, 2.0))
    assert config.num_states == base.N_H * base.N_K * (base.gamma_max + 1)
    # C is ignored when N_C == 1.
    for k in range(config.gamma_max + 1):
        idx_no_c = quantize(H=1.0, K=1.0, k=k, config=config)
        idx_with_c = quantize(H=1.0, K=1.0, k=k, config=config, C=0.99)
        assert idx_no_c == idx_with_c
    assert confidence_center(0, config) == 0.0


def test_confidence_axis_outermost_layout_and_roundtrip() -> None:
    config = MDPConfig(N_H=2, N_K=2, gamma_max=2, N_C=3, T_levels=(1.0, 2.0))
    base_size = config.N_H * config.N_K * (config.gamma_max + 1)
    assert config.num_states == config.N_C * base_size

    state_space = StateSpace(config)
    # Confidence bin 0 reproduces the legacy index; higher bins are offset blocks.
    legacy = quantize(H=1.0, K=1.0, k=1, config=config, C=0.0)
    assert legacy < base_size
    high = state_space.encode(H=1.0, K=1.0, k=1, C=state_space.C_center(2))
    assert high == 2 * base_size + legacy

    # decode() strips the confidence bin; decode_full() recovers it.
    assert state_space.decode(high) == state_space.decode(legacy)
    assert state_space.c_of(high) == 2
    assert state_space.c_of(legacy) == 0

    # Full round-trip across every axis.
    for i_C in range(config.N_C):
        for i_H in range(config.N_H):
            for i_K in range(config.N_K):
                for k in range(config.gamma_max + 1):
                    idx = state_space.encode(
                        H=(i_H + 0.5) * config.H_max / config.N_H,
                        K=(i_K + 0.5) * config.K_max / config.N_K,
                        k=k,
                        C=state_space.C_center(i_C),
                    )
                    assert dequantize_full(idx, config) == (i_H, i_K, i_C, k)
                    assert dequantize(idx, config) == (i_H, i_K, k)


def test_policy_get_action_routes_by_confidence_bin() -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=2, N_C=2, T_levels=(1.0, 2.0))
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    continue_idx = action_space.encode("continue", 1.0)
    verify_idx = action_space.encode("verify", 2.0)
    pi_star = np.zeros(config.num_states, dtype=np.int32)
    # Low-confidence bin -> verify; high-confidence bin -> continue.
    for state_idx in range(config.num_states):
        i_C = state_space.c_of(state_idx)
        pi_star[state_idx] = continue_idx if i_C == 1 else verify_idx
    policy = JointAdaSpecPolicy(config=config, pi_star=pi_star)

    low_conf = state_space.C_center(0)
    high_conf = state_space.C_center(1)
    assert policy.get_action(H=0.1, K=0.1, k=0, C=low_conf)[0] == "verify"
    assert policy.get_action(H=0.1, K=0.1, k=0, C=high_conf)[0] == "continue"


# ---------------------------------------------------------------------------
# Confidence gate (inference-time early-verify override)
# ---------------------------------------------------------------------------

def _always_continue_policy(config: MDPConfig) -> JointAdaSpecPolicy:
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    continue_idx = action_space.encode("continue", 1.0)
    stop_idx = action_space.encode("verify", 1.0)
    pi_star = np.zeros(config.num_states, dtype=np.int32)
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        pi_star[state_idx] = stop_idx if k >= config.gamma_max else continue_idx
    return JointAdaSpecPolicy(config=config, pi_star=pi_star)


def test_confidence_gate_forces_early_verify_on_weak_draft() -> None:
    # Draft top-prob is 0.8; a gate at tau=0.9 must fire on every step.
    draft = ToyModelAdapter(FixedModel([0.8, 0.2]))
    target = ToyModelAdapter(FixedModel([0.8, 0.2]))
    config = MDPConfig(N_H=1, N_K=1, gamma_max=3, T_levels=(1.0,))
    policy = _always_continue_policy(config)

    def run(conf_gate_tau: float):
        decoder = JointAdaSpecDecoder(
            target_model=target,
            draft_model=draft,
            policy=policy,
            conf_gate_tau=conf_gate_tau,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        return decoder.generate(prompt_ids=[0], max_new_tokens=4, generator=generator)

    open_result = run(0.0)
    gated_result = run(0.9)

    # Gate off: the policy drafts, so there are more draft calls than target calls.
    assert open_result.n_draft_calls > open_result.n_target_calls
    # Gate on: every step verifies immediately (no accepted drafts, one target call per token).
    assert {m["action_length"] for m in gated_result.per_step_metrics} == {"verify"}
    assert gated_result.acceptance_rate == 0.0
    assert gated_result.n_tokens_generated == 4


# ---------------------------------------------------------------------------
# End-to-end pipeline with the confidence axis enabled
# ---------------------------------------------------------------------------

def test_confidence_axis_end_to_end_collect_solve_generate(tmp_path) -> None:
    target = ToyModelAdapter(FixedModel([0.7, 0.2, 0.1]))
    draft = ToyModelAdapter(FixedModel([0.6, 0.3, 0.1]))
    config = MDPConfig(
        N_H=4, N_K=4, gamma_max=2, N_C=3, T_levels=(1.0, 2.0), nu_min=1
    )
    prompts = [[0], [1], [2], [0, 1]]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    traces_path = collect_traces(
        target_model=target,
        draft_model=draft,
        prompts=prompts,
        n_traces=20,
        output_path=tmp_path / "traces.parquet",
        config=config,
        generator=generator,
    )

    import pandas as pd

    frame = pd.read_parquet(traces_path)
    assert "C" in frame.columns and "next_C" in frame.columns
    assert frame["state_idx"].max() < config.num_states

    estimate = estimate_mdp_parameters(traces_path=traces_path, config=config)
    V_star, pi_star, _ = solve_mdp(estimate.transitions, estimate.rewards, config)
    assert pi_star.shape == (config.num_states,)
    assert V_star.shape == (config.num_states,)

    policy = JointAdaSpecPolicy(config=config, pi_star=pi_star, V_star=V_star)
    decoder = JointAdaSpecDecoder(target_model=target, draft_model=draft, policy=policy)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1)
    result = decoder.generate(prompt_ids=[0], max_new_tokens=5, generator=gen)
    assert result.n_tokens_generated == 5
    assert all("C" in metric for metric in result.per_step_metrics)
