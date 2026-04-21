"""Cascade baseline: solve draft length first, then verification threshold."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from jointadaspec.baselines.cascade_common import CascadePolicy
from jointadaspec.mdp.spaces import ActionSpace, MDPConfig, StateSpace
from jointadaspec.mdp.value_iteration import solve_mdp


def _strict_threshold_mask(config: MDPConfig) -> np.ndarray:
    if 1.0 not in config.T_levels:
        raise ValueError("Cascade length-first solve requires T=1.0 in config.T_levels.")
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    mask = np.zeros((config.num_states, config.num_actions), dtype=bool)
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        for action_idx in action_space.valid_action_indices(k):
            action = action_space.decode(action_idx)
            if action.threshold == 1.0:
                mask[state_idx, action_idx] = True
    return mask


def _length_constrained_mask(config: MDPConfig, length_policy: np.ndarray) -> np.ndarray:
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    mask = np.zeros((config.num_states, config.num_actions), dtype=bool)
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        chosen_length = action_space.decode(int(length_policy[state_idx])).length_action
        for action_idx in action_space.valid_action_indices(k):
            action = action_space.decode(action_idx)
            if action.length_action == chosen_length:
                mask[state_idx, action_idx] = True
    return mask


def solve_cascade_length_then_verif(
    transitions: sparse.csr_matrix,
    rewards: np.ndarray,
    config: MDPConfig,
) -> tuple[np.ndarray, CascadePolicy, dict[str, Any]]:
    """Solve the staged cascade from theorem 2.3's baseline ordering."""
    length_mask = _strict_threshold_mask(config)
    _, length_pi, length_log = solve_mdp(
        transitions=transitions,
        rewards=rewards,
        config=config,
        action_mask=length_mask,
    )

    verif_mask = _length_constrained_mask(config, length_pi)
    V_star, pi_star, verif_log = solve_mdp(
        transitions=transitions,
        rewards=rewards,
        config=config,
        action_mask=verif_mask,
    )
    policy = CascadePolicy(
        config=config,
        cascade_order="length_then_verif",
        pi_star=pi_star,
        length_policy=length_pi,
        verif_policy=pi_star,
        V_star=V_star,
    )
    return V_star, policy, {"stage1": length_log, "stage2": verif_log}
