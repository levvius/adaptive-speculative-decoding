from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from jointadaspec.inference import JointAdaSpecPolicy
from jointadaspec.mdp import MDPConfig, estimate_mdp_parameters, solve_mdp
from jointadaspec.mdp.spaces import ActionSpace, StateSpace
from jointadaspec.semantics import semantic_metadata, trace_metadata_path


def _deterministic_transitions(config: MDPConfig) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    state_space = StateSpace(config)
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        for action_idx in range(config.num_actions):
            next_state = state_idx
            if k >= config.gamma_max:
                next_state = state_space.encode(H=0.1, K=0.1, k=0)
            rows.append(state_idx * config.num_actions + action_idx)
            cols.append(next_state)
            data.append(1.0)
    return sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(config.num_states * config.num_actions, config.num_states),
    ).tocsr()


def test_state_action_counts() -> None:
    config = MDPConfig()
    assert config.num_states == 3600
    assert config.num_actions == 9


def test_vi_converges() -> None:
    config = MDPConfig(
        N_H=2,
        N_K=2,
        gamma_max=1,
        T_levels=(1.0, 2.0),
        lambda_discount=0.8,
        max_vi_iterations=500,
    )
    transitions = _deterministic_transitions(config)
    rewards = np.zeros((config.num_states, config.num_actions), dtype=np.float64)
    rewards[:, 0] = 0.5
    rewards[:, 1] = 1.0
    rewards[:, 2] = 0.2

    V_star, pi_star, solve_log = solve_mdp(transitions, rewards, config)

    assert V_star.shape == (config.num_states,)
    assert pi_star.shape == (config.num_states,)
    assert solve_log["iterations"] < config.max_vi_iterations
    assert solve_log["final_delta"] < 1e-3


def test_gamma_max_masks_continue_actions() -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=1, T_levels=(1.0, 2.0), max_vi_iterations=50)
    transitions = _deterministic_transitions(config)
    rewards = np.zeros((config.num_states, config.num_actions), dtype=np.float64)
    action_space = ActionSpace(config)
    rewards[:, action_space.continue_action_indices] = 100.0

    _, pi_star, _ = solve_mdp(transitions, rewards, config)
    state_space = StateSpace(config)
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        if k == config.gamma_max:
            assert int(pi_star[state_idx]) in action_space.stop_action_indices


def test_pi_star_shape() -> None:
    config = MDPConfig()
    transitions = sparse.eye(config.num_states * config.num_actions, config.num_states, format="csr")
    rewards = np.zeros((config.num_states, config.num_actions), dtype=np.float64)
    _, pi_star, _ = solve_mdp(transitions, rewards, config)
    assert pi_star.shape == (3600,)


def _single_reward_trace(tmp_path, config: MDPConfig, *, H: float, K: float, k: int, d_step: float):
    state_space = StateSpace(config)
    action_space = ActionSpace(config)
    state_idx = state_space.encode(H=H, K=K, k=k)
    action_idx = action_space.encode("verify", config.T_levels[-1])
    path = tmp_path / f"trace_{K}_{k}.parquet"
    frame = pd.DataFrame.from_records(
        [
            {
                "state_idx": state_idx,
                "action_idx": action_idx,
                "next_state_idx": state_idx,
                "accepted": 1,
                "step_time_ms": 10.0,
                "d_step": d_step,
            }
        ]
    )
    frame.to_parquet(path, index=False)
    metadata = semantic_metadata(config=config, num_actions=action_space.num_actions)
    metadata.update({"artifact_kind": "traces", "num_records": len(frame), "n_traces": 1})
    trace_metadata_path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path, state_idx, action_idx


def test_quality_risk_defaults_keep_legacy_reward(tmp_path) -> None:
    config = MDPConfig(
        N_H=2,
        N_K=2,
        gamma_max=2,
        T_levels=(1.0, 2.0),
        kappa=2.0,
        c_time=0.01,
        nu_min=1,
    )
    traces_path, state_idx, action_idx = _single_reward_trace(
        tmp_path,
        config,
        H=0.1,
        K=7.9,
        k=2,
        d_step=0.2,
    )

    estimate = estimate_mdp_parameters(traces_path=traces_path, config=config)

    assert estimate.rewards[state_idx, action_idx] == np.float64(1.0 - 0.1 - 0.4)


def test_quality_risk_additive_form_applies_state_penalty(tmp_path) -> None:
    """Additive form applies an explicit state penalty; it is not an invariance proof."""
    config = MDPConfig(
        N_H=2,
        N_K=2,
        gamma_max=2,
        T_levels=(1.0, 2.0),
        kappa=2.0,
        c_time=0.0,
        nu_min=1,
        quality_risk_K=0.4,
        quality_risk_k=0.2,
        quality_risk_form="additive",
    )
    traces_path, state_idx, action_idx = _single_reward_trace(
        tmp_path,
        config,
        H=0.1,
        K=7.9,
        k=2,
        d_step=0.2,
    )

    estimate = estimate_mdp_parameters(traces_path=traces_path, config=config)

    state_space = StateSpace(config)
    _, i_K_test, k_test = state_space.decode(state_idx)
    expected_penalty = 0.4 * (i_K_test / (config.N_K - 1)) + 0.2 * (k_test / config.gamma_max)
    expected_reward = 1.0 - 2.0 * 0.2 - expected_penalty
    assert estimate.rewards[state_idx, action_idx] == np.float64(expected_reward)


def test_quality_risk_form_validation() -> None:
    """quality_risk_form must be 'multiplicative' or 'additive'."""
    with pytest.raises(ValueError, match="quality_risk_form"):
        MDPConfig(quality_risk_form="invalid")


def test_estimate_mdp_rejects_trace_without_semantic_metadata(tmp_path) -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=1, T_levels=(1.0,), nu_min=1)
    state_space = StateSpace(config)
    action_space = ActionSpace(config)
    state_idx = state_space.encode(H=0.1, K=0.1, k=0)
    path = tmp_path / "legacy_trace.parquet"
    pd.DataFrame.from_records(
        [
            {
                "state_idx": state_idx,
                "action_idx": action_space.encode("verify", 1.0),
                "next_state_idx": state_idx,
                "accepted": 1,
                "step_time_ms": 1.0,
                "d_step": 0.0,
            }
        ]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Trace metadata sidecar not found"):
        estimate_mdp_parameters(traces_path=path, config=config)


def test_estimate_mdp_rejects_trace_with_mismatched_config_hash(tmp_path) -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=1, T_levels=(1.0,), nu_min=1)
    traces_path, _, _ = _single_reward_trace(
        tmp_path,
        config,
        H=0.1,
        K=0.1,
        k=0,
        d_step=0.0,
    )
    meta_path = trace_metadata_path(traces_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["config_hash"] = "not-the-right-config"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="config_hash"):
        estimate_mdp_parameters(traces_path=traces_path, config=config)


def test_quality_risk_penalizes_high_K_and_k_states(tmp_path) -> None:
    config = MDPConfig(
        N_H=2,
        N_K=4,
        gamma_max=4,
        T_levels=(1.0, 2.0),
        kappa=1.0,
        c_time=0.0,
        nu_min=1,
        quality_risk_K=1.0,
        quality_risk_k=0.5,
    )
    low_path, low_state_idx, action_idx = _single_reward_trace(
        tmp_path,
        config,
        H=0.1,
        K=0.1,
        k=0,
        d_step=0.2,
    )
    high_path, high_state_idx, _ = _single_reward_trace(
        tmp_path,
        config,
        H=0.1,
        K=7.9,
        k=4,
        d_step=0.2,
    )

    low_estimate = estimate_mdp_parameters(traces_path=low_path, config=config)
    high_estimate = estimate_mdp_parameters(traces_path=high_path, config=config)

    assert high_estimate.rewards[high_state_idx, action_idx] < low_estimate.rewards[low_state_idx, action_idx]


def test_policy_roundtrip_preserves_quality_risk_fields(tmp_path) -> None:
    config = MDPConfig(
        N_H=1,
        N_K=1,
        gamma_max=1,
        T_levels=(1.0,),
        quality_risk_K=0.75,
        quality_risk_k=0.25,
    )
    pi_star = np.zeros(config.num_states, dtype=np.int32)
    policy = JointAdaSpecPolicy(config=config, pi_star=pi_star)
    path = tmp_path / "policy_quality.npz"

    policy.save(path)
    loaded = JointAdaSpecPolicy.load(path)

    assert loaded.config.quality_risk_K == 0.75
    assert loaded.config.quality_risk_k == 0.25
