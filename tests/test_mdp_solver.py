from __future__ import annotations

import numpy as np
from scipy import sparse

from jointadaspec.mdp import MDPConfig, solve_mdp
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
