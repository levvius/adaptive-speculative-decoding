"""Value iteration for the JointAdaSpec tabular MDP."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from jointadaspec.mdp.spaces import ActionSpace, MDPConfig, StateSpace


def solve_mdp(
    transitions: sparse.csr_matrix,
    rewards: np.ndarray,
    config: MDPConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve the tabular MDP with sparse value iteration."""
    S = config.num_states
    A = config.num_actions
    if transitions.shape != (S * A, S):
        raise ValueError(
            f"Expected transitions of shape {(S * A, S)}, got {transitions.shape}."
        )
    if rewards.shape != (S, A):
        raise ValueError(f"Expected rewards of shape {(S, A)}, got {rewards.shape}.")

    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    V = np.zeros(S, dtype=np.float64)
    deltas: list[float] = []
    invalid_mask = np.zeros((S, A), dtype=bool)
    for state_idx in range(S):
        _, _, k = state_space.decode(state_idx)
        valid_actions = set(action_space.valid_action_indices(k))
        for action_idx in range(A):
            invalid_mask[state_idx, action_idx] = action_idx not in valid_actions

    Q = np.full((S, A), -np.inf, dtype=np.float64)

    for iteration in range(config.max_vi_iterations):
        expected_next = np.asarray(transitions.dot(V)).reshape(S, A)
        Q = rewards + config.lambda_discount * expected_next
        Q[invalid_mask] = -np.inf
        new_V = np.max(Q, axis=1)
        delta = float(np.max(np.abs(new_V - V)))
        deltas.append(delta)
        V = new_V
        if delta < config.epsilon_convergence:
            break
    else:
        iteration = config.max_vi_iterations - 1

    pi_star = np.argmax(Q, axis=1).astype(np.int32)
    solve_log: dict[str, Any] = {
        "iterations": iteration + 1,
        "final_delta": deltas[-1] if deltas else 0.0,
        "converged": bool(deltas and deltas[-1] < config.epsilon_convergence),
        "deltas": deltas,
    }
    return V, pi_star, solve_log
