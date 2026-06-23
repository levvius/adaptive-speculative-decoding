from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from jointadaspec.analysis.conditions import _compute_c4
from jointadaspec.mdp import MDPConfig
from jointadaspec.mdp.spaces import ActionSpace, StateSpace


def _load_theorem_gap_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_theorem_c_gap.py"
    spec = importlib.util.spec_from_file_location("analyze_theorem_c_gap_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_c4_uses_verify_threshold_monotonicity_only() -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=1, T_levels=(1.0, 2.0))
    action_space = ActionSpace(config)
    rewards = np.zeros((config.num_states, config.num_actions), dtype=np.float64)
    verify_low = action_space.encode("verify", 1.0)
    verify_high = action_space.encode("verify", 2.0)
    continue_idx = action_space.encode("continue", 999.0)

    rewards[:, continue_idx] = 100.0
    rewards[:, verify_low] = 2.0
    rewards[:, verify_high] = 1.0

    report = _compute_c4(rewards, config)

    assert report["fraction_nonnegative"] == 0.0
    assert report["criterion"] == "verify@T reward is nondecreasing over adjacent T levels"


def test_theorem_gap_c4_mask_matches_verify_threshold_semantics() -> None:
    module = _load_theorem_gap_module()
    config = MDPConfig(N_H=1, N_K=1, gamma_max=1, T_levels=(1.0, 2.0))
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    rewards = np.zeros((config.num_states, config.num_actions), dtype=np.float64)
    rewards[:, action_space.encode("continue", 1.0)] = -100.0
    rewards[:, action_space.encode("verify", 1.0)] = 3.0
    rewards[:, action_space.encode("verify", 2.0)] = 2.0

    mask = module._c4_violation_mask(rewards, config)

    violating_state = state_space.encode(H=0.1, K=0.1, k=0)
    terminal_k_state = state_space.encode(H=0.1, K=0.1, k=1)
    assert bool(mask[violating_state]) is True
    assert bool(mask[terminal_k_state]) is False
