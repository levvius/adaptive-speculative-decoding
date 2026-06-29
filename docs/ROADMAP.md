# Roadmap

## Current Focus (defense stabilization)

1. Keep the defense snapshot internally consistent: README, docs, deck, QR, and agent guidance all mark historical JointAdaSpec numbers as legacy / pre-repair evidence.
2. Use `v1-defense` as the immutable repository snapshot for the Monday defense.
3. Defer final block-v1 benchmark claims until a fresh solve + benchmark with regenerated 9-action artifacts.

## In Progress

- Commission-facing documentation cleanup before defense.
- **Draft-confidence control (post-defense, `main`).** Added an optional outer
  draft-confidence state axis (`N_C`, `draft_conf_feature`) and an inference-time
  early-verify gate (`conf_gate_tau`, a decoder knob) so the policy can stop/verify
  when the draft loses confidence — targeting the Theorem-G non-monotonicity and the
  joint≈cascade tie. `N_C=1` reproduces the legacy 3-D MDP exactly. CPU-validated end
  to end (`tests/test_draft_confidence.py`); a powered GPU comparison is staged in
  `scripts/run_improvement_eval.sh` (conf vs 3-D vs cascade vs target, prompt-clustered
  stats via `scripts/merge_benchmark_runs.py` + `analyze_jointadaspec_quality.py`).
  No quality/speed claim yet — pending the GPU rerun.

## Recently Completed

### Thesis sprint (2026-05-14 — 2026-05-27)

- **Locked Run 1 (14B/0.5B, n=1500)**: historical / pre-repair evidence only; adaptive control family +4% EM vs target_only under old artifacts, p<0.05. Reports in `reports/*qwen14b_0p5b_lock_2026-05-14*`.
- **Locked Run 2 (7B/1.5B, n=1500)**: confirmed null on lock window (−1.47%, p=0.369).
- **Triangulation (7B/1.5B, start=600, n=1500)**: third independent window confirms slice-independent null (−2.53%, p=0.094).
- **Experiment E ablation (14B/0.5B, n=300)**: pre-repair evidence that adaptive control can beat fixed fuzzy_sd_T. Figure `fig_E_adaptivity_ablation.pdf`.
- **κ-sweep (7B/1.5B, 6κ × n=300)**: joint ≈ cascade robust across Lagrange knob. Figure `fig_bonus_kappa_sweep.pdf`.
- **Theorem D exact value gap**: advantage-weighted gap derivation, mean |A^πC| on B = 6e-5 (14B), 1e-3 (7B). Script `scripts/analyze_theorem_c_gap.py`. Figure `fig_D_advantage_on_B.pdf`.
- **Theorem G quality non-monotonicity**: Δ EM +7.4% at low acceptance, −2.6% at high acceptance.
- **Thesis defense guide**: `papers/ВКР_тезисы_и_структура.md`.

### Theory sprint (2026-05-15)

- **Theorem A**: sample-complexity bound on `‖V̂ − V*‖∞`, non-vacuous at n_min=5.
- **Theorem B correction**: state-only quality-risk is not generally Bellman-invariant; keep `quality_risk_form` as an experimental flag and prefer potential-based/action-coupled shaping for new theory.
- **Theorem C**: cascade suboptimality ≤ 2R_max·μ*_J(B)/(1−γ); C4 failure reframed quantitatively.
- **Theorems 2.3/2.4**: empirically grounded with N1/N2 conditions and κ-sweep Pareto front.

### Quality-aware hypothesis (2026-05-05 — 2026-05-08)

- Completed the `2026-05-08` held-out Qwen `7B -> 1.5B` validation (500 prompts, 3 seeds): primary method did not improve held-out quality (EM delta −2.27 pp, CI crossing zero).
- The `2026-05-05` first-slice improvement treated as non-generalizing hypothesis.
- Added `quality_risk_form` config flag (default `multiplicative` for backward compat).
- Added held-out benchmark workflow with `test_start_index`.

## Next Technical Steps

### Decoding and Modeling

- Evaluate larger speculative window (`k=8`, `k=16`) with fixed runtime budget.
- Prototype stronger judge backends (tree/boosting models) behind a stable interface.
- Add task-specific AutoJudge training data path for LiveCodeBench-like tasks.
- Keep GSM8K exact-match evaluation in the seeded `scripts/03_benchmark.py` path and use it in future multi-seed reports.
- Prototype a prompt/state-level fallback policy that switches to `target_only` in high-risk math-reasoning regions.
- Run offline error analysis on held-out prompt-seed rows before launching another expensive quality run.
- Compare future quality mechanisms against both `target_only` and ordinary `speculative`, not only against older JointAdaSpec variants.
- Promote smoke-only `2026-04-21` and `2026-04-28` results into a normal multi-seed/multi-sample run before claiming a new best result.

### Performance Engineering

- Reduce judge overhead by minimizing CPU roundtrips.
- Investigate GPU-resident judge path.
- Add profiling snapshots for mismatch-heavy regions.
- Profile JointAdaSpec trace collection and benchmark loops to identify target-only bottlenecks.

### Benchmark Quality

- Add comparable 48h profiles across model families with unified run matrix.
- Publish concise per-run summary cards (accuracy/speed/cost).
- Keep strict JSONL schema compatibility for downstream analysis.
- Generate markdown reports directly from JointAdaSpec `outputs/` directories.
- Keep held-out benchmark slices explicit via `datasets.test_start_index` in manifests and result metadata.
- Add a retry/resume wrapper around HuggingFace model metadata fetches for long benchmark stages.
- Prefer local model-pair configs for long Qwen reruns when full checkpoint shards are already present on disk.

## Project Hygiene

- Expand CI coverage with additional smoke checks for config presets.
- Keep docs synchronized with scripts and defaults.
- Maintain reproducibility-first defaults (explicit manifests, deterministic file naming).
