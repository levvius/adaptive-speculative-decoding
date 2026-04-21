"""Cascade baseline: solve verification threshold first, then draft length."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from jointadaspec.baselines.cascade_common import CascadePolicy
from jointadaspec.mdp.spaces import ActionSpace, MDPConfig, StateSpace
from jointadaspec.mdp.value_iteration import solve_mdp


def _verif_only_mask(config: MDPConfig) -> np.ndarray:
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    mask = np.zeros((config.num_states, config.num_actions), dtype=bool)
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        for action_idx in action_space.valid_action_indices(k):
            action = action_space.decode(action_idx)
            if k >= config.gamma_max:
                mask[state_idx, action_idx] = action.length_action == "stop"
            else:
                mask[state_idx, action_idx] = action.length_action == "continue"
    return mask


def _threshold_constrained_mask(config: MDPConfig, verif_policy: np.ndarray) -> np.ndarray:
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    mask = np.zeros((config.num_states, config.num_actions), dtype=bool)
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        chosen_threshold = action_space.decode(int(verif_policy[state_idx])).threshold
        for action_idx in action_space.valid_action_indices(k):
            action = action_space.decode(action_idx)
            if action.threshold == chosen_threshold:
                mask[state_idx, action_idx] = True
    return mask


def solve_cascade_verif_then_length(
    transitions: sparse.csr_matrix,
    rewards: np.ndarray,
    config: MDPConfig,
) -> tuple[np.ndarray, CascadePolicy, dict[str, Any]]:
    """Solve the symmetric staged cascade with threshold first."""
    verif_mask = _verif_only_mask(config)
    _, verif_pi, verif_log = solve_mdp(
        transitions=transitions,
        rewards=rewards,
        config=config,
        action_mask=verif_mask,
    )

    length_mask = _threshold_constrained_mask(config, verif_pi)
    V_star, pi_star, length_log = solve_mdp(
        transitions=transitions,
        rewards=rewards,
        config=config,
        action_mask=length_mask,
    )
    policy = CascadePolicy(
        config=config,
        cascade_order="verif_then_length",
        pi_star=pi_star,
        length_policy=pi_star,
        verif_policy=verif_pi,
        V_star=V_star,
    )
    return V_star, policy, {"stage1": verif_log, "stage2": length_log}
