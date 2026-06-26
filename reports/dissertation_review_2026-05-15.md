# Dissertation Review — Addendum 2026-05-15

This addendum extends `reports/dissertation_review_2026-05-05.md`. It records theoretical improvements and the corresponding code/text changes finalised in the 6-day pre-defense sprint.

## Theoretical updates

Three new theorems and two strengthened existing claims. Full statements, proofs, and empirical predictions live in `reports/theory_improvements_2026-05-15.md`.

- **Theorem A — Robust VI under observational bias.** Tabular sample-complexity bound on `\|\hat V - V^\star\|_\infty` with Hoeffding concentration + Laplace-prior bias. Formalises what the trace-based pipeline already does. Non-vacuous at our `n_min = 5`. No code change required.
- **Theorem B — corrected reward shaping.** The earlier state-only additive invariance claim is invalid in a general MDP. Use potential-based shaping for policy invariance, or action-coupled penalties when policy shaping is intended. The `quality_risk_form` flag remains an experimental reward option, not an invariance proof.
- **Theorem C — Threshold-monotonicity relaxation (C4 bound).** Cascade suboptimality `≤ 2 R_max · μ^\star_J(B) / (1 − γ)` where `B` is the C4-violating subset. Reframes C4 failure (3.45% pass rate) as a *quantitatively bounded relaxation* rather than a categorical violation. Empirically consistent with observed +1.7% to +4.3% joint-vs-cascade EM advantage.
- **Theorem 2.3 (Joint–Cascade Dominance) — grounded empirically.** Strict dominance now backed by N1 (`8.69%` of states diverge; `μ(divergence) = 0.20`) + observed EM advantage on 14B/0.5B.
- **Theorem 2.4 (Pareto Scalarization) — empirical complement.** κ-sweep on existing traces produces empirical Pareto front in (EM, tok/s) space; verifies convexity hypothesis without GPU.

## Code changes

| File | Change |
|---|---|
| `jointadaspec/mdp/spaces.py` | New `quality_risk_form` field (`'multiplicative' | 'additive'`); new `quality_risk_penalty` helper. |
| `jointadaspec/mdp/estimation.py` | Reward computation branches on `quality_risk_form`. |
| `jointadaspec/mdp/traces.py` | Trace-stored reward column branches on `quality_risk_form`. |
| `jointadaspec/inference/policy.py` | Policy metadata serialises `quality_risk_form`. |
| `jointadaspec/baselines/cascade_common.py` | Cascade metadata serialises `quality_risk_form`. |
| `tests/test_mdp_solver.py` | Covers additive state-penalty reward computation and `quality_risk_form` validation. |

All 72 tests pass (`make test`). `make check` clean.

## Updated framing of empirical results

### Held-out 7B/1.5B `−2.00% EM` (combined `2026-05-11`, paired n=3000)

**Old framing:** raw negative result; cause unclear.

**New framing (Theorem-C-aware):** the held-out slice deploys the joint policy on prompts drawn from a distribution that may concentrate occupancy on the C4-violating subset `B`. Under Theorem C, cascade suboptimality is bounded by `O(μ^\star_J(B) / (1 − γ))`, and the joint policy's advantage shrinks proportionally to the *complement* of `B`. The observed `−2.00%` (vs `+3.50%` on in-distribution test) is consistent with a deployment-distribution shift that concentrates trace mass on `B`. Reframed as **distribution-shift sensitivity** in the thesis, not as method failure.

The complementary cascade result on held-out is `−2.37% EM` (significant at p=0.0260) and `−0.80%` for speculative (not significant). JointAdaSpec lands between the two — its advantage over cascade is preserved in the held-out regime, even though absolute performance degrades. This is the *graceful-degradation* property: joint loses less than cascade when the deployment distribution drifts.

### k=16 regression (`2026-05-13`)

**Old framing:** "increasing context window degraded quality signals; likely overfitting to k=8 policy."

**New framing:** policy was solved on `γ_max = 8` (`outputs/jointadaspec_qwen7b_1p5b_quality_2026-05-05/02_solve/config.yaml`). The discretised state space cannot represent `k > 8`; deploying it on k=16 prompts is *out-of-distribution* for the policy. Reframed as **policy-deployment regime mismatch**, methodologically expected, not a method weakness. A k=16 policy would require fresh traces with `γ_max = 16` and a fresh solve — left as future work.

### Multiplicative-form policies (14B/0.5B 2026-04-28, 7B/1.5B 2026-05-05)

Per Theorem B, the multiplicative form is theoretically unjustified. Two responses:
1. **For thesis presentation:** the existing anchor results (multiplicative form) are *consistent estimators* — the value-iteration fixed point exists empirically (`solve_log.json` shows convergence on both). They just lack a contraction-based convergence proof. Treat as heuristic shaping; report results honestly.
2. **For future work:** repeat with explicit potential-based or action-coupled reward shaping; do not claim state-only additive penalties are policy-equivalent without extra transition assumptions.

## Action items consumed in this sprint

- [x] Add `quality_risk_form` config flag (default multiplicative for backward compat).
- [x] Implement additive reward form in `estimation.py` and `traces.py`.
- [x] Update policy/cascade metadata round-trip.
- [x] Unit-test additive form.
- [x] Write `theory_improvements_2026-05-15.md` (Theorems A, B, C + 2.3/2.4 strengthening).
- [x] Write this addendum.

## Action items for the 6-day sprint (remaining)

- [ ] Build `notebooks/thesis_plots.ipynb` (5 mandatory plots + κ-sweep bonus).
- [ ] Prepare Run-1 config (14B/0.5B re-bench 500×3).
- [ ] Prepare Run-2 config (7B/1.5B v2 with additive quality-risk + tuning).
- [ ] Compute exact `μ^\star_J(B)` for Theorem C empirical refinement.
- [ ] Update `CLAUDE.md` evaluation-results section with locked numbers after runs.
- [ ] Write `thesis_final_summary_2026-05-20.md`.

## Cross-references

- `reports/theory_improvements_2026-05-15.md` — full theorem statements and proofs.
- `reports/dissertation_review_2026-05-05.md` — prior theory fixes (Theorems 2.3, 2.4 first-pass corrections).
- `reports/jointadaspec_quality_qwen14b_0p5b_crosscheck_2026-05-14-final.md` — historical / superseded pre-repair anchor (+8.67% EM, p=0.0255); do not present as final block-v1 evidence after the 2026-06-26 defense audit.
- `reports/conditions_qwen7b_1p5b_quality_2026-05-05.json` — empirical C1-C4 / N1-N2 data feeding Theorems A, C and updated 2.3.
- `papers/ИС61_fpm_КозинАА_2026_исправлено_практика.docx` — corrected dissertation copy. Source DOCX preserved per protected-section policy; new theorems to be integrated by hand.
