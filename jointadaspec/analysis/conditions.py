"""Empirical verification of the JointAdaSpec MDP conditions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kendalltau, spearmanr

from jointadaspec.baselines import (
    CascadePolicy,
    solve_cascade_length_then_verif,
    solve_cascade_verif_then_length,
)
from jointadaspec.inference import JointAdaSpecPolicy
from jointadaspec.mdp import estimate_mdp_parameters
from jointadaspec.mdp.spaces import ActionSpace, MDPConfig, StateSpace


def _action_label(action_space: ActionSpace, action_idx: int) -> str:
    action = action_space.decode(int(action_idx))
    return f"{action.length_action}@{action.threshold:g}"


def _bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float]]:
    if x.size < 2 or y.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), (float("nan"), float("nan"))
    rho = float(spearmanr(x, y).statistic)
    samples = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        picks = rng.integers(0, x.size, size=x.size)
        x_resampled = x[picks]
        y_resampled = y[picks]
        if np.unique(x_resampled).size < 2 or np.unique(y_resampled).size < 2:
            samples[idx] = float("nan")
            continue
        samples[idx] = float(spearmanr(x_resampled, y_resampled).statistic)
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return rho, (float(low), float(high))


def _state_action_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["state_idx", "action_idx"], as_index=False).agg(
        reward_mean=("reward", "mean"),
        H_mean=("H", "mean"),
        K_mean=("K", "mean"),
        next_H_mean=("next_H", "mean"),
        next_K_mean=("next_K", "mean"),
        count=("trace_idx", "count"),
    )
    grouped["state_score"] = grouped["H_mean"] + grouped["K_mean"]
    grouped["f_next_mean"] = -(grouped["next_H_mean"] + grouped["next_K_mean"])
    return grouped


def _state_metadata(config: MDPConfig) -> pd.DataFrame:
    state_space = StateSpace(config)
    rows: list[dict[str, int]] = []
    for state_idx in range(config.num_states):
        i_H, i_K, k = state_space.decode(state_idx)
        rows.append({"state_idx": state_idx, "i_H": i_H, "i_K": i_K, "k": k})
    return pd.DataFrame.from_records(rows)


def _compute_c1(
    summary: pd.DataFrame,
    config: MDPConfig,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    action_space = ActionSpace(config)
    per_action: dict[str, Any] = {}
    passed = True
    for action_idx in range(config.num_actions):
        subset = summary.loc[summary["action_idx"] == action_idx].copy()
        label = _action_label(action_space, action_idx)
        rho_h, ci_h = _bootstrap_spearman(
            subset["reward_mean"].to_numpy(dtype=np.float64),
            subset["H_mean"].to_numpy(dtype=np.float64),
            n_resamples=n_resamples,
            rng=rng,
        )
        rho_k, ci_k = _bootstrap_spearman(
            subset["reward_mean"].to_numpy(dtype=np.float64),
            subset["K_mean"].to_numpy(dtype=np.float64),
            n_resamples=n_resamples,
            rng=rng,
        )
        action_passed = (
            np.isfinite(rho_h)
            and np.isfinite(rho_k)
            and rho_h <= 0.0
            and rho_k <= 0.0
            and ci_h[1] < 0.1
            and ci_k[1] < 0.1
        )
        passed = passed and action_passed
        per_action[label] = {
            "rho_H": rho_h,
            "rho_H_ci": list(ci_h),
            "rho_K": rho_k,
            "rho_K_ci": list(ci_k),
            "n_states": int(subset.shape[0]),
            "passed": bool(action_passed),
        }
    return {
        "passed": bool(passed),
        "criterion": "rho <= 0 and upper_ci < 0.1 for both H and K",
        "per_action": per_action,
    }


def _compute_c2(summary: pd.DataFrame, config: MDPConfig) -> dict[str, Any]:
    action_space = ActionSpace(config)
    per_action: dict[str, Any] = {}
    passed = True
    for action_idx in range(config.num_actions):
        subset = summary.loc[summary["action_idx"] == action_idx].sort_values(
            ["state_score", "state_idx"]
        )
        label = _action_label(action_space, action_idx)
        if subset.shape[0] < 2:
            tau = float("nan")
            action_passed = False
        else:
            tau = float(kendalltau(np.arange(subset.shape[0]), subset["f_next_mean"]).statistic)
            action_passed = np.isfinite(tau) and tau <= 0.0
        passed = passed and action_passed
        per_action[label] = {
            "kendall_tau": tau,
            "n_states": int(subset.shape[0]),
            "passed": bool(action_passed),
        }
    return {
        "passed": bool(passed),
        "criterion": "Kendall tau <= 0 for E[-H'-K' | s, a] over state rank H+K",
        "per_action": per_action,
    }


def _sorted_action_indices(config: MDPConfig) -> list[int]:
    action_space = ActionSpace(config)
    return sorted(
        range(config.num_actions),
        key=lambda idx: (
            0 if action_space.decode(idx).length_action == "stop" else 1,
            action_space.decode(idx).threshold,
        ),
    )


def _iter_state_edges(config: MDPConfig, *, k_limit: int) -> list[tuple[str, int, int]]:
    state_space = StateSpace(config)
    edges: list[tuple[str, int, int]] = []
    for i_H in range(config.N_H):
        for i_K in range(config.N_K):
            for k in range(k_limit):
                state_idx = state_space.encode(
                    H=(i_H + 0.5) * config.H_max / config.N_H,
                    K=(i_K + 0.5) * config.K_max / config.N_K,
                    k=k,
                )
                if i_H + 1 < config.N_H:
                    next_idx = state_space.encode(
                        H=(i_H + 1.5) * config.H_max / config.N_H,
                        K=(i_K + 0.5) * config.K_max / config.N_K,
                        k=k,
                    )
                    edges.append(("H", state_idx, next_idx))
                if i_K + 1 < config.N_K:
                    next_idx = state_space.encode(
                        H=(i_H + 0.5) * config.H_max / config.N_H,
                        K=(i_K + 1.5) * config.K_max / config.N_K,
                        k=k,
                    )
                    edges.append(("K", state_idx, next_idx))
                if k + 1 < k_limit and k + 1 <= config.gamma_max:
                    next_idx = state_space.encode(
                        H=(i_H + 0.5) * config.H_max / config.N_H,
                        K=(i_K + 0.5) * config.K_max / config.N_K,
                        k=k + 1,
                    )
                    edges.append(("k", state_idx, next_idx))
    return edges


def _compute_c3(rewards: np.ndarray, config: MDPConfig) -> dict[str, Any]:
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    sorted_actions = _sorted_action_indices(config)
    k_limit = min(config.gamma_max + 1, 4)
    nonnegative = 0
    total = 0
    by_dim = {"H": {"nonnegative": 0, "total": 0}, "K": {"nonnegative": 0, "total": 0}, "k": {"nonnegative": 0, "total": 0}}
    for dim, low_state, high_state in _iter_state_edges(config, k_limit=k_limit):
        _, _, low_k = state_space.decode(low_state)
        _, _, high_k = state_space.decode(high_state)
        valid_low = set(action_space.valid_action_indices(low_k))
        valid_high = set(action_space.valid_action_indices(high_k))
        valid_both = valid_low & valid_high
        for low_action, high_action in zip(sorted_actions[:-1], sorted_actions[1:]):
            if low_action not in valid_both or high_action not in valid_both:
                continue
            delta = (
                rewards[high_state, high_action]
                - rewards[high_state, low_action]
                - rewards[low_state, high_action]
                + rewards[low_state, low_action]
            )
            total += 1
            by_dim[dim]["total"] += 1
            if delta >= -1.0e-9:
                nonnegative += 1
                by_dim[dim]["nonnegative"] += 1
    fraction = 0.0 if total == 0 else float(nonnegative / total)
    per_dimension = {
        key: {
            "fraction_nonnegative": 0.0 if value["total"] == 0 else float(value["nonnegative"] / value["total"]),
            "boxes": int(value["total"]),
        }
        for key, value in by_dim.items()
    }
    return {
        "passed": bool(fraction >= 0.9),
        "fraction_nonnegative": fraction,
        "target": 0.9,
        "boxes": int(total),
        "per_dimension": per_dimension,
    }


def _compute_c4(rewards: np.ndarray, config: MDPConfig) -> dict[str, Any]:
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    nonnegative = 0
    total = 0
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        if k >= config.gamma_max:
            continue
        for low_threshold, high_threshold in zip(config.T_levels[:-1], config.T_levels[1:]):
            stop_low = action_space.encode("stop", low_threshold)
            stop_high = action_space.encode("stop", high_threshold)
            continue_low = action_space.encode("continue", low_threshold)
            continue_high = action_space.encode("continue", high_threshold)
            delta = (
                rewards[state_idx, continue_high]
                - rewards[state_idx, continue_low]
                - rewards[state_idx, stop_high]
                + rewards[state_idx, stop_low]
            )
            total += 1
            if delta >= -1.0e-9:
                nonnegative += 1
    fraction = 0.0 if total == 0 else float(nonnegative / total)
    return {
        "passed": bool(fraction >= 0.9),
        "fraction_nonnegative": fraction,
        "target": 0.9,
        "boxes": int(total),
    }


def _policy_transition_matrix(
    transitions: sparse.csr_matrix,
    pi_star: np.ndarray,
    config: MDPConfig,
) -> sparse.csr_matrix:
    row_indices = np.arange(config.num_states, dtype=np.int64) * config.num_actions + pi_star.astype(np.int64)
    return transitions[row_indices].tocsr()


def _stationary_distribution(
    policy_transitions: sparse.csr_matrix,
    *,
    max_iters: int = 200,
    tol: float = 1.0e-10,
) -> tuple[np.ndarray, dict[str, Any]]:
    num_states = policy_transitions.shape[0]
    dist = np.full(num_states, 1.0 / max(num_states, 1), dtype=np.float64)
    residual = float("inf")
    for iteration in range(max_iters):
        new_dist = np.asarray(policy_transitions.transpose().dot(dist)).reshape(-1)
        total = float(new_dist.sum())
        if total > 0.0:
            new_dist /= total
        residual = float(np.max(np.abs(new_dist - dist)))
        dist = new_dist
        if residual < tol:
            break
    return dist, {"iterations": iteration + 1, "residual": residual, "converged": bool(residual < tol)}


def _compute_n1_n2(
    transitions: sparse.csr_matrix,
    joint_policy: JointAdaSpecPolicy,
    cascade_len: CascadePolicy,
    cascade_verif: CascadePolicy,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    joint_pi = joint_policy.pi_star
    diff_len = joint_pi != cascade_len.pi_star
    diff_verif = joint_pi != cascade_verif.pi_star
    diff_union = diff_len | diff_verif
    policy_transitions = _policy_transition_matrix(transitions, joint_pi, joint_policy.config)
    stationary, stationarity_log = _stationary_distribution(policy_transitions)

    def _summary(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "count": int(mask.sum()),
            "percentage": float(mask.mean() * 100.0),
            "stationary_mass": float(stationary[mask].sum()),
        }

    n1_report = {
        "passed": bool(diff_union.any()),
        "against_length_then_verif": _summary(diff_len),
        "against_verif_then_length": _summary(diff_verif),
        "union": _summary(diff_union),
    }

    state_space = StateSpace(joint_policy.config)
    mass_by_k = np.zeros(joint_policy.config.gamma_max + 1, dtype=np.float64)
    for state_idx, mass in enumerate(stationary):
        _, _, k = state_space.decode(state_idx)
        if diff_union[state_idx]:
            mass_by_k[k] += mass
    n2_report = {
        "passed": bool(float(stationary[diff_union].sum()) > 0.0),
        "stationary_mass_on_divergence": float(stationary[diff_union].sum()),
        "stationary_mass_by_k": mass_by_k.tolist(),
        "power_iteration": stationarity_log,
    }
    return n1_report, n2_report, stationary


def _plot_c1(c1_report: dict[str, Any], *, field: str, out_path: Path) -> None:
    labels = list(c1_report["per_action"].keys())
    values = [c1_report["per_action"][label][field] for label in labels]
    cis = [c1_report["per_action"][label][f"{field}_ci"] for label in labels]
    lower = [value - ci[0] for value, ci in zip(values, cis)]
    upper = [ci[1] - value for value, ci in zip(values, cis)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, values, color="#4C72B0")
    ax.errorbar(x, values, yerr=np.array([lower, upper]), fmt="none", ecolor="black", capsize=3)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axhline(0.1, color="#C44E52", linestyle="--", linewidth=1.0)
    ax.set_title(f"C1 diagnostic: {field}")
    ax.set_ylabel("Spearman rho")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_fraction_bars(values: dict[str, float], *, title: str, out_path: Path) -> None:
    labels = list(values.keys())
    heights = [values[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, heights, color="#55A868")
    ax.axhline(0.9, color="#C44E52", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Fraction non-negative")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_n1_heatmap(diff_union: np.ndarray, config: MDPConfig, *, out_path: Path) -> None:
    state_space = StateSpace(config)
    heatmap = np.zeros((config.N_H, config.N_K), dtype=np.float64)
    counts = np.zeros((config.N_H, config.N_K), dtype=np.float64)
    for state_idx, is_divergent in enumerate(diff_union.astype(np.float64)):
        i_H, i_K, _ = state_space.decode(state_idx)
        heatmap[i_H, i_K] += is_divergent
        counts[i_H, i_K] += 1.0
    counts[counts == 0.0] = 1.0
    heatmap /= counts
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(heatmap, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_title("N1 divergence heatmap")
    ax.set_xlabel("K bin")
    ax.set_ylabel("H bin")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_n2_mass(n2_report: dict[str, Any], *, out_path: Path) -> None:
    masses = n2_report["stationary_mass_by_k"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(masses)), masses, color="#8172B3")
    ax.set_title("N2 stationary mass on divergence states")
    ax.set_xlabel("k")
    ax.set_ylabel("Stationary mass")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_plots(
    *,
    out_dir: Path,
    c1_report: dict[str, Any],
    c3_report: dict[str, Any],
    c4_report: dict[str, Any],
    diff_union: np.ndarray,
    n2_report: dict[str, Any],
    config: MDPConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_c1(c1_report, field="rho_H", out_path=out_dir / "c1_rho_H.png")
    _plot_c1(c1_report, field="rho_K", out_path=out_dir / "c1_rho_K.png")
    _plot_fraction_bars(
        {key: value["fraction_nonnegative"] for key, value in c3_report["per_dimension"].items()},
        title="C3 supermodularity fractions",
        out_path=out_dir / "c3_supermod_fractions.png",
    )
    _plot_fraction_bars(
        {"joint_length_x_threshold": c4_report["fraction_nonnegative"]},
        title="C4 supermodularity fractions",
        out_path=out_dir / "c4_supermod_fractions.png",
    )
    _plot_n1_heatmap(diff_union=diff_union, config=config, out_path=out_dir / "n1_divergence_heatmap.png")
    _plot_n2_mass(n2_report=n2_report, out_path=out_dir / "n2_stationary_mass.png")


def evaluate_conditions(
    *,
    traces_path: Path,
    policy_path: Path,
    out_path: Path,
    n_resamples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    joint_policy = JointAdaSpecPolicy.load(policy_path)
    config = joint_policy.config
    traces = pd.read_parquet(traces_path)
    estimate = estimate_mdp_parameters(traces_path=traces_path, config=config)
    summary = _state_action_summary(traces)
    summary = summary.merge(_state_metadata(config), on="state_idx", how="left")
    rng = np.random.default_rng(seed)

    _, cascade_len, _ = solve_cascade_length_then_verif(estimate.transitions, estimate.rewards, config)
    _, cascade_verif, _ = solve_cascade_verif_then_length(estimate.transitions, estimate.rewards, config)

    c1_report = _compute_c1(summary, config, n_resamples=n_resamples, rng=rng)
    c2_report = _compute_c2(summary, config)
    c3_report = _compute_c3(estimate.rewards, config)
    c4_report = _compute_c4(estimate.rewards, config)
    n1_report, n2_report, stationary = _compute_n1_n2(
        estimate.transitions,
        joint_policy,
        cascade_len,
        cascade_verif,
    )
    diff_union = (joint_policy.pi_star != cascade_len.pi_star) | (joint_policy.pi_star != cascade_verif.pi_star)

    date_tag = out_path.stem.split("_")[-1]
    plots_dir = out_path.parent / f"conditions_plots_{date_tag}"
    _write_plots(
        out_dir=plots_dir,
        c1_report=c1_report,
        c3_report=c3_report,
        c4_report=c4_report,
        diff_union=diff_union,
        n2_report=n2_report,
        config=config,
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "traces_path": str(traces_path),
        "policy_path": str(policy_path),
        "plots_dir": str(plots_dir),
        "config": asdict(config),
        "checks": {
            "c1": c1_report,
            "c2": c2_report,
            "c3": c3_report,
            "c4": c4_report,
            "n1": n1_report,
            "n2": n2_report,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
