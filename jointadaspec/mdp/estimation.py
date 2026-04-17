"""MDP parameter estimation from collected traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from jointadaspec.mdp.spaces import ActionSpace, JointAction, MDPConfig, StateSpace


@dataclass(frozen=True)
class EstimatedMDP:
    """Estimated transition model and expected rewards."""

    transitions: sparse.csr_matrix
    rewards: np.ndarray
    visit_counts: np.ndarray


def _neighbour_indices(i: int, size: int) -> range:
    start = max(0, i - 1)
    stop = min(size, i + 2)
    return range(start, stop)


def _prior_support(
    state_idx: int,
    action: JointAction,
    state_space: StateSpace,
) -> list[int]:
    i_H, i_K, k = state_space.decode(state_idx)
    if action.is_stop:
        return [
            state_space.encode(
                H=(next_i_H + 0.5) * state_space.config.H_max / state_space.config.N_H,
                K=(next_i_K + 0.5) * state_space.config.K_max / state_space.config.N_K,
                k=0,
            )
            for next_i_H in _neighbour_indices(i_H, state_space.config.N_H)
            for next_i_K in _neighbour_indices(i_K, state_space.config.N_K)
        ]

    next_k = min(k + 1, state_space.config.gamma_max)
    support: list[int] = []
    for next_i_H in _neighbour_indices(i_H, state_space.config.N_H):
        for next_i_K in _neighbour_indices(i_K, state_space.config.N_K):
            support.append(
                state_space.encode(
                    H=(next_i_H + 0.5) * state_space.config.H_max / state_space.config.N_H,
                    K=(next_i_K + 0.5) * state_space.config.K_max / state_space.config.N_K,
                    k=next_k,
                )
            )
    return support


def _prior_reward(action: JointAction, config: MDPConfig) -> float:
    if action.is_stop:
        return float(-config.c_time)
    quality_penalty = 0.0 if action.threshold == 1.0 else min(1.0, (action.threshold - 1.0) / 3.0)
    return float(0.25 - config.c_time - config.kappa * quality_penalty)


def estimate_mdp_parameters(traces_path: str | bytes | "os.PathLike[str]", config: MDPConfig) -> EstimatedMDP:
    """Estimate sparse transitions and rewards from trace Parquet."""
    state_space = StateSpace(config)
    action_space = ActionSpace(config)
    frame = pd.read_parquet(traces_path)

    if frame.empty:
        raise ValueError("Trace table is empty.")

    S = config.num_states
    A = config.num_actions
    visit_counts = np.zeros((S, A), dtype=np.int32)
    reward_sums = np.zeros((S, A), dtype=np.float64)
    grouped_counts: dict[tuple[int, int], dict[int, float]] = {}

    for row in frame.itertuples(index=False):
        state_idx = int(row.state_idx)
        action_idx = int(row.action_idx)
        next_state_idx = int(row.next_state_idx)
        reward = float(row.accepted) - config.c_time * float(row.step_time_ms) - config.kappa * float(row.d_step)

        visit_counts[state_idx, action_idx] += 1
        reward_sums[state_idx, action_idx] += reward
        grouped_counts.setdefault((state_idx, action_idx), {})
        grouped_counts[(state_idx, action_idx)][next_state_idx] = (
            grouped_counts[(state_idx, action_idx)].get(next_state_idx, 0.0) + 1.0
        )

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    rewards = np.zeros((S, A), dtype=np.float64)

    for state_idx in range(S):
        for action_idx in range(A):
            action = action_space.decode(action_idx)
            empirical = grouped_counts.get((state_idx, action_idx), {})
            visits = int(visit_counts[state_idx, action_idx])

            if visits > 0:
                rewards[state_idx, action_idx] = reward_sums[state_idx, action_idx] / visits
            else:
                rewards[state_idx, action_idx] = _prior_reward(action, config)

            probs: dict[int, float]
            if visits >= config.nu_min:
                probs = {next_state: count / visits for next_state, count in empirical.items()}
            else:
                support = _prior_support(state_idx, action, state_space)
                prior_mass = config.alpha_smooth / max(len(support), 1)
                smoothed_counts: dict[int, float] = {
                    next_state: prior_mass for next_state in support
                }
                for next_state, count in empirical.items():
                    smoothed_counts[next_state] = smoothed_counts.get(next_state, 0.0) + count
                denom = visits + config.alpha_smooth
                probs = {next_state: count / denom for next_state, count in smoothed_counts.items()}

            row_idx = state_idx * A + action_idx
            for next_state_idx, prob in probs.items():
                row_indices.append(row_idx)
                col_indices.append(int(next_state_idx))
                data.append(float(prob))

    transitions = sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(S * A, S),
        dtype=np.float64,
    ).tocsr()
    return EstimatedMDP(transitions=transitions, rewards=rewards, visit_counts=visit_counts)
