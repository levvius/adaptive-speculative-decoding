from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from jointadaspec.analysis import evaluate_conditions
from jointadaspec.baselines import (
    solve_cascade_length_then_verif,
    solve_cascade_verif_then_length,
)
from jointadaspec.inference import JointAdaSpecPolicy
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


def _interaction_transitions(config: MDPConfig) -> sparse.csr_matrix:
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
            if action.length_action == "continue" and state_idx == s0 and action.threshold == 2.0:
                next_state = s1
            row_idx = state_idx * config.num_actions + action_idx
            rows.append(row_idx)
            cols.append(next_state)
            data.append(1.0)
    return sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(config.num_states * config.num_actions, config.num_states),
        dtype=np.float64,
    ).tocsr()


def _interaction_rewards(config: MDPConfig) -> np.ndarray:
    rewards = np.zeros((config.num_states, config.num_actions), dtype=np.float64)
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        for action_idx in range(config.num_actions):
            action = action_space.decode(action_idx)
            if k == 0 and action.length_action == "stop":
                rewards[state_idx, action_idx] = 1.0
            elif k == 0 and action.length_action == "continue" and action.threshold == 1.0:
                rewards[state_idx, action_idx] = 0.0
            elif k == 0 and action.length_action == "continue" and action.threshold == 2.0:
                rewards[state_idx, action_idx] = 1.0
            elif k == 1 and action.length_action == "stop":
                rewards[state_idx, action_idx] = 5.0
            elif k == 1 and action.length_action == "continue":
                rewards[state_idx, action_idx] = -5.0
            else:
                rewards[state_idx, action_idx] = 0.0
    return rewards


def test_cascade_dominated_by_joint_on_toy_mdp() -> None:
    config = MDPConfig(
        N_H=1,
        N_K=1,
        gamma_max=2,
        T_levels=(1.0, 2.0),
        lambda_discount=0.9,
        epsilon_convergence=1.0e-8,
        max_vi_iterations=500,
    )
    transitions = _interaction_transitions(config)
    rewards = _interaction_rewards(config)
    state_space = StateSpace(config)
    s0 = state_space.encode(H=0.1, K=0.1, k=0)

    V_joint, _, _ = solve_mdp(transitions, rewards, config)
    V_len_then, _, _ = solve_cascade_length_then_verif(transitions, rewards, config)
    V_verif_then, _, _ = solve_cascade_verif_then_length(transitions, rewards, config)

    assert V_joint[s0] > V_len_then[s0] + 1.0e-6
    assert V_joint[s0] > V_verif_then[s0] + 1.0e-6


def test_value_iteration_convergence() -> None:
    config = MDPConfig(
        N_H=10,
        N_K=1,
        gamma_max=0,
        T_levels=(1.0,),
        lambda_discount=0.75,
        epsilon_convergence=1.0e-10,
        max_vi_iterations=500,
    )
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for state_idx in range(config.num_states):
        for action_idx in range(config.num_actions):
            rows.append(state_idx * config.num_actions + action_idx)
            cols.append(state_idx)
            data.append(1.0)
    transitions = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(config.num_states * config.num_actions, config.num_states),
        dtype=np.float64,
    ).tocsr()
    rewards = np.zeros((config.num_states, config.num_actions), dtype=np.float64)
    rewards[:, 0] = 2.0
    rewards[:, 1] = 100.0

    V_star, pi_star, _ = solve_mdp(transitions, rewards, config)

    assert np.allclose(V_star, np.full(config.num_states, 8.0), atol=1.0e-6)
    assert np.all(pi_star == 0)


def test_policy_npz_roundtrip(tmp_path) -> None:
    config = MDPConfig(N_H=2, N_K=2, gamma_max=2, T_levels=(1.0, 2.0))
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    pi_star = np.zeros(config.num_states, dtype=np.int32)
    continue_idx = action_space.encode("continue", 2.0)
    stop_idx = action_space.encode("stop", 1.0)
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        pi_star[state_idx] = stop_idx if k >= config.gamma_max else continue_idx

    policy = JointAdaSpecPolicy(config=config, pi_star=pi_star, V_star=np.ones(config.num_states))
    path = tmp_path / "joint_policy_roundtrip.npz"
    policy.save(path)
    loaded = JointAdaSpecPolicy.load(path)

    assert np.array_equal(loaded.pi_star, policy.pi_star)
    assert np.array_equal(loaded.V_star, policy.V_star)
    assert loaded.config == policy.config


def test_value_iteration_seed_stability(tmp_path) -> None:
    target = ToyModelAdapter(FixedModel([0.7, 0.2, 0.1]))
    draft = ToyModelAdapter(FixedModel([0.6, 0.3, 0.1]))
    config = MDPConfig(N_H=4, N_K=4, gamma_max=2, T_levels=(1.0, 2.0), nu_min=1)
    prompts = [[0], [1], [2], [0, 1]]
    policies: list[np.ndarray] = []
    for seed in (7, 11):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        traces_path = collect_traces(
            target_model=target,
            draft_model=draft,
            prompts=prompts,
            n_traces=40,
            output_path=tmp_path / f"traces_seed_{seed}.parquet",
            config=config,
            generator=generator,
        )
        estimate = estimate_mdp_parameters(traces_path=traces_path, config=config)
        _, pi_star, _ = solve_mdp(estimate.transitions, estimate.rewards, config)
        policies.append(pi_star)

    agreement = float(np.mean(policies[0] == policies[1]))
    assert agreement >= 0.95


def test_verify_conditions_smoke(tmp_path) -> None:
    config = MDPConfig(N_H=4, N_K=1, gamma_max=1, T_levels=(1.0, 2.0))
    state_space = StateSpace(config)
    action_space = ActionSpace(config)
    stop_idx = action_space.encode("stop", 1.0)
    stop_any_idx = action_space.encode("stop", 2.0)
    pi_star = np.full(config.num_states, stop_idx, dtype=np.int32)
    policy = JointAdaSpecPolicy(config=config, pi_star=pi_star, V_star=np.zeros(config.num_states))
    policy_path = tmp_path / "policy.npz"
    policy.save(policy_path)

    rows: list[dict[str, float | int | str]] = []
    trace_idx = 0
    for H in (0.2, 1.2, 2.2, 3.2):
        state_idx = state_space.encode(H=H, K=0.1, k=0)
        for _ in range(12):
            rows.append(
                {
                    "trace_idx": trace_idx,
                    "rollout_step": 0,
                    "state_idx": state_idx,
                    "action_idx": stop_idx,
                    "action_length": "stop",
                    "threshold": 1.0,
                    "reward": 5.0 - H,
                    "next_state_idx": state_idx,
                    "accepted": 1,
                    "proposed": 0,
                    "emitted_token": 0,
                    "step_time_ms": 0.1,
                    "d_step": 0.0,
                    "H": H,
                    "K": 0.1,
                    "k": 0,
                    "next_H": max(0.0, H - 0.1),
                    "next_K": 0.1,
                    "next_k": 0,
                }
            )
            rows.append(
                {
                    "trace_idx": trace_idx,
                    "rollout_step": 0,
                    "state_idx": state_idx,
                    "action_idx": stop_any_idx,
                    "action_length": "stop",
                    "threshold": 2.0,
                    "reward": 5.0 - H,
                    "next_state_idx": state_idx,
                    "accepted": 1,
                    "proposed": 0,
                    "emitted_token": 0,
                    "step_time_ms": 0.1,
                    "d_step": 0.0,
                    "H": H,
                    "K": 0.1,
                    "k": 0,
                    "next_H": max(0.0, H - 0.1),
                    "next_K": 0.1,
                    "next_k": 0,
                }
            )
            trace_idx += 1
    traces_path = tmp_path / "synthetic_traces.parquet"
    pd.DataFrame.from_records(rows).to_parquet(traces_path, index=False)

    report = evaluate_conditions(
        traces_path=traces_path,
        policy_path=policy_path,
        out_path=tmp_path / "conditions_report.json",
        n_resamples=100,
        seed=42,
    )

    c1_stop = report["checks"]["c1"]["per_action"]["stop@1"]
    assert c1_stop["rho_H"] < 0.0
    assert c1_stop["rho_H_ci"][1] < 0.1
