from __future__ import annotations

import numpy as np
from scipy import sparse

from jointadaspec.baselines import (
    solve_cascade_length_then_verif,
    solve_cascade_verif_then_length,
)
from jointadaspec.mdp import MDPConfig, solve_mdp
from jointadaspec.mdp.spaces import ActionSpace, StateSpace


def _terminal_chain_transitions(config: MDPConfig, *, threshold_sensitive: bool) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    s0 = state_space.encode(H=0.1, K=0.1, k=0)
    s1 = state_space.encode(H=0.1, K=0.1, k=1)
    s2 = state_space.encode(H=0.1, K=0.1, k=2)
    for state_idx in range(config.num_states):
        for action_idx in range(config.num_actions):
            action = action_space.decode(action_idx)
            next_state = s2
            if action.length_action == "continue":
                if state_idx == s0:
                    if threshold_sensitive and action.threshold == 1.0:
                        next_state = s2
                    else:
                        next_state = s1
                elif state_idx == s1:
                    next_state = s2
            row_idx = state_idx * config.num_actions + action_idx
            rows.append(row_idx)
            cols.append(next_state)
            data.append(1.0)
    return sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(config.num_states * config.num_actions, config.num_states),
        dtype=np.float64,
    ).tocsr()


def _separable_rewards(config: MDPConfig) -> np.ndarray:
    rewards = np.zeros((config.num_states, config.num_actions), dtype=np.float64)
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    length_term = {
        0: {"stop": 1.0, "continue": 2.0},
        1: {"stop": 3.0, "continue": -1.0},
        2: {"stop": 0.0, "continue": 0.0},
    }
    threshold_bonus = {1.0: 0.0, 2.0: 0.25}
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        for action_idx in range(config.num_actions):
            action = action_space.decode(action_idx)
            rewards[state_idx, action_idx] = (
                length_term[k][action.length_action] + threshold_bonus[action.threshold]
            )
    return rewards


def test_cascade_equals_joint_on_separable_reward() -> None:
    config = MDPConfig(
        N_H=1,
        N_K=1,
        gamma_max=2,
        T_levels=(1.0, 2.0),
        lambda_discount=0.9,
        epsilon_convergence=1.0e-8,
        max_vi_iterations=500,
    )
    transitions = _terminal_chain_transitions(config, threshold_sensitive=False)
    rewards = _separable_rewards(config)

    V_joint, _, _ = solve_mdp(transitions, rewards, config)
    V_len_then, _, _ = solve_cascade_length_then_verif(transitions, rewards, config)
    V_verif_then, _, _ = solve_cascade_verif_then_length(transitions, rewards, config)

    assert np.allclose(V_joint, V_len_then, atol=1.0e-6)
    assert np.allclose(V_joint, V_verif_then, atol=1.0e-6)
