#!/usr/bin/env python
"""Theorem 2.3 / Theorem C empirical analysis (CPU, no GPU).

For each (policy, traces) config:
  * Re-estimate the MDP (transitions, rewards) from the trace parquet.
  * Compute the C4-violating state set B and its π*_J stationary occupancy
    mass μ*_J(B)  — this is the quantity in the Theorem-C worst-case bound.
  * Compute the *exact* discounted value gap  E_μ[V^{π_J} − V^{π_C}]  via
    policy evaluation V^π = (I − γ P^π)^{-1} r^π. This is the true
    performance-difference, not its (loose) bound.

Headline finding (2026-05-25): μ*_J(B) ≈ 0.9 (C4 violated almost everywhere),
so the Theorem-C bound is vacuous (~360). But joint weakly dominates cascade
in MDP-value on 100% of states (Theorem 2.3, confirmed exactly) by a *small*
margin (14B: +0.005, 7B: +0.10), consistent with the empirically negligible
EM/speed difference. Honest reading: C4 is a very conservative sufficient
condition — pervasively violated but benign.

Usage: .venv/bin/python scripts/analyze_theorem_c_gap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import spsolve

from jointadaspec.analysis.conditions import (
    _policy_transition_matrix,
    _stationary_distribution,
)
from jointadaspec.baselines import CascadePolicy
from jointadaspec.inference import JointAdaSpecPolicy
from jointadaspec.mdp import estimate_mdp_parameters
from jointadaspec.mdp.spaces import ActionSpace, StateSpace

R_MAX = 2.0

CONFIGS = {
    "14B/0.5B": "outputs/jointadaspec_qwen14b_0p5b_2026-04-28",
    "7B/1.5B": "outputs/jointadaspec_qwen7b_1p5b_quality_2026-05-05",
}


def _c4_violation_mask(rewards: np.ndarray, cfg) -> np.ndarray:
    """State is in B if any threshold-pair supermodularity delta is negative."""
    aspace, sspace = ActionSpace(cfg), StateSpace(cfg)
    viol = np.zeros(cfg.num_states, dtype=bool)
    for s in range(cfg.num_states):
        _, _, k = sspace.decode(s)
        if k >= cfg.gamma_max:
            continue
        for lo, hi in zip(cfg.T_levels[:-1], cfg.T_levels[1:]):
            d = (
                rewards[s, aspace.encode("continue", hi)]
                - rewards[s, aspace.encode("continue", lo)]
                - rewards[s, aspace.encode("stop", hi)]
                + rewards[s, aspace.encode("stop", lo)]
            )
            if d < -1e-9:
                viol[s] = True
                break
    return viol


def _policy_value(transitions, rewards, pi, cfg) -> np.ndarray:
    P = _policy_transition_matrix(transitions, pi, cfg)
    r = rewards[np.arange(cfg.num_states), pi.astype(np.int64)]
    A = eye(cfg.num_states, format="csr") - cfg.lambda_discount * P
    return spsolve(csc_matrix(A), r)


def analyze(solve_dir: Path, traces_path: Path) -> dict:
    pol = JointAdaSpecPolicy.load(str(solve_dir / "policy.npz"))
    casc = CascadePolicy.load(str(solve_dir / "cascade_verif_then_length.npz"))
    cfg = pol.config
    est = estimate_mdp_parameters(traces_path=str(traces_path), config=cfg)

    viol = _c4_violation_mask(est.rewards, cfg)
    stat, _ = _stationary_distribution(_policy_transition_matrix(est.transitions, pol.pi_star, cfg))
    mu_B = float(stat[viol].sum())
    bound = 2 * R_MAX * mu_B / (1 - cfg.lambda_discount)

    vj = _policy_value(est.transitions, est.rewards, pol.pi_star, cfg)
    vc = _policy_value(est.transitions, est.rewards, casc.pi_star, cfg)
    gap = vj - vc
    return {
        "gamma": cfg.lambda_discount,
        "c4_violating_state_frac": float(viol.mean()),
        "mu_star_J_B": mu_B,
        "theorem_c_bound": bound,
        "exact_value_gap_stationary": float((stat * gap).sum()),
        "value_gap_mean": float(gap.mean()),
        "value_gap_max": float(gap.max()),
        "weak_dominance_frac_states": float((gap >= -1e-9).mean()),
    }


def main() -> int:
    out = {}
    for tag, base in CONFIGS.items():
        b = Path(base)
        res = analyze(b / "02_solve", b / "01_traces" / "traces.parquet")
        out[tag] = res
        print(f"== {tag} ==")
        for k, v in res.items():
            print(f"   {k}: {v:.5f}" if isinstance(v, float) else f"   {k}: {v}")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/theorem_c_gap_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nWrote reports/theorem_c_gap_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
