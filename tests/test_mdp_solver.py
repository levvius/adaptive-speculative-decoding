from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from jointadaspec.inference import JointAdaSpecPolicy
from jointadaspec.mdp import MDPConfig, estimate_mdp_parameters, solve_mdp
from jointadaspec.mdp.spaces import ActionSpace, StateSpace


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
    assert config.num_actions == 16


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
    rewards[:, 1] = 0.4
    rewards[:, 2] = 1.0
    rewards[:, 3] = 0.2

    V_star, pi_star, solve_log = solve_mdp(transitions, rewards, config)

    assert V_star.shape == (config.num_states,)
    assert pi_star.shape == (config.num_states,)
    assert solve_log["iterations"] < config.max_vi_iterations
    assert solve_log["final_delta"] < 1e-3


def test_gamma_max_masks_continue_actions() -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=1, T_levels=(1.0, 2.0), max_vi_iterations=50)
    transitions = _deterministic_transitions(config)
    rewards = np.zeros((config.num_states, config.num_actions), dtype=np.float64)
    rewards[:, 2:] = 100.0

    _, pi_star, _ = solve_mdp(transitions, rewards, config)
    action_space = ActionSpace(config)
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
    action_idx = action_space.encode("continue", config.T_levels[-1])
    path = tmp_path / f"trace_{K}_{k}.parquet"
    pd.DataFrame.from_records(
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
    ).to_parquet(path, index=False)
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
