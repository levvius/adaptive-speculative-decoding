# Adaptive Speculative Decoding

[![CI](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml/badge.svg)](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](.python-version)

Research implementation and benchmark of adaptive speculative decoding methods for LLM inference acceleration.

This repository compares exact and approximate decoding strategies under a single benchmark harness, with reproducible configs, JSONL outputs, and report scripts.

`sp_samp/` remains the historical benchmark stack. `jointadaspec/` is the new thesis-focused implementation of JointAdaSpec: joint adaptive control of draft length and verification threshold through a tabular MDP.

## Why This Project

- Side-by-side comparison of Baseline, Speculative Sampling, AutoJudge, Consensus AutoJudge, Top-K, SpecExec, and JointAdaSpec.
- Paper-aligned AutoJudge implementation (GSM8K label mining + LogisticRegression calibration).
- New `jointadaspec/` stack for joint adaptive control of draft length and verification threshold via a tabular MDP.
- Long-run friendly workflow (resume keys, checkpoints, strict result schema validation).
- Real benchmark reports are versioned in `reports/`.

## Implemented Methods

| Method | Exact vs target distribution | Main idea |
|---|---|---|
| `baseline` | exact | Target-only decoding |
| `speculative` | exact | Draft proposes, target verifies |
| `autojudge` | approximate | Judge can accept some mismatches |
| `consensus_autojudge` | approximate | Two drafts + consensus gate decide accept / escalate / fallback |
| `topk` | approximate | Accept mismatch if target token in top-k |
| `specexec` | exact | Parallel speculative branches + cache reuse |
| `jointadaspec` | approximate | Policy jointly chooses draft stop/continue and fuzzy verification threshold |

## JointAdaSpec: Joint Adaptive Speculative Decoding

JointAdaSpec formulates speculative decoding control as a finite MDP over the state `s = (H, K_prev, k)` and solves the joint policy by sparse value iteration.

- `H`: entropy of the draft distribution
- `K_prev`: delayed `KL(q || p)` from the latest available target-verification step
- `k`: current position in the draft window
- `a_length ∈ {stop, continue}`
- `a_verif ∈ {1.00, 1.22, 1.49, 1.82, 2.22, 2.71, 3.30, 4.00}`

The implementation lives in `jointadaspec/` and is split into:

- `jointadaspec/core/`: features, verification rules, decoder base classes
- `jointadaspec/mdp/`: discrete spaces, trace collection, MDP estimation, value iteration
- `jointadaspec/inference/`: learned policy lookup and online JointAdaSpec decoder
- `jointadaspec/baselines/`: `target_only`, `speculative`, `fuzzy_sd`, `adaptive_length`, and both cascade baselines
- `jointadaspec/analysis/`: empirical checks for C1-C4 and N1-N2
- `jointadaspec/metrics/`: speed, quality, Pareto helpers
- `jointadaspec/utils/`: model loading, dataset loading, manifests, structured logging

The pipeline is stage-based:

1. `scripts/01_collect_traces.py` collects exploratory SD traces on held-out prompts.
2. `scripts/02_solve_mdp.py` estimates transitions/rewards, solves the joint MDP, and saves both cascade baselines.
3. `scripts/03_benchmark.py` benchmarks JointAdaSpec against target-only, speculative, fuzzy-SD, adaptive-length, and the two cascades with seeded resume-safe JSONL output.
4. `scripts/04_verify_conditions.py` checks empirical monotonicity/supermodularity conditions and writes diagnostic plots.
5. `scripts/05_write_manifest.py` writes reproducibility manifests with git SHA, config YAML, seeds, and artifact hashes.

Minimal reproduction on the Qwen 14B / 0.5B setup:

```bash
make jointadaspec-full MODEL_PAIR=qwen14b_0p5b JOINTADA_MAX_TRACES=10 JOINTADA_MAX_SAMPLES=10 JOINTADA_N_SEEDS=1
```

Outputs include traces, policy NPZ files, cascade policies, seeded benchmark JSONL, conditions JSON, manifests, Pareto plots, threshold-surface heatmaps, and ablation charts.

Conservative quality-aware local rerun on Qwen 7B / 1.5B:

```bash
make jointadaspec-full MODEL_PAIR=qwen7b_1p5b_quality JOINTADA_MAX_TRACES=500 JOINTADA_MAX_SAMPLES=100 JOINTADA_N_SEEDS=3 JOINTADA_MAX_NEW_TOKENS=256
```

This config uses local checkpoints under `models/` and adds state-dependent quality-risk reward shaping through `quality_risk_K` and `quality_risk_k`.

Held-out quality validation using the fixed `2026-05-05` policy:

```bash
make jointadaspec-quality-heldout-full JOINTADA_DATE=2026-05-08
```

The validation uses GSM8K `test_start_index=100`, `500` prompts, `3` seeds, and pre-registers `cascade_verif_then_length` as the primary quality decoder.

## Final Results (2026-05-14, locked)

### JointAdaSpec — Qwen 14B → 0.5B (primary, n=1500)

GSM8K zero-shot CoT, RTX 5090, 500 prompts × 3 seeds, McNemar paired test:

| Method | EM | tok/s | vs speculative | Δ vs target_only | p |
|---|---:|---:|---:|---:|---:|
| target_only | 52.93% | 14.45 | 2.96× | — | — |
| speculative | 53.27% | 4.88 | 1.00× | +0.33% | 0.877 |
| cascade_verif_then_length | 57.13% | 10.29 | 2.11× | +4.20% | 0.015 ✓ |
| **jointadaspec** | **57.00%** | **10.84** | **2.22×** | **+4.07%** | **0.020 ✓** |

The adaptive control family (joint + cascade) significantly outperforms `target_only` at p < 0.05. Joint = cascade (Δ = −0.13%, p = 0.96): predicted by Theorem D (advantage-weighted value gap ε-small, mean 6e-5). Speed note: `target_only` is the fastest method (14.45 tok/s); the 2.22× figure is relative to vanilla speculative, not to plain autoregressive decoding.

Reports: `reports/{pareto,ablation,jointadaspec_quality}_qwen14b_0p5b_lock_2026-05-14.*`, `reports/threshold_surface_qwen14b_0p5b_lock_2026-05-14/`.

### JointAdaSpec — Qwen 7B → 1.5B (secondary, null at power)

Three independent held-out windows confirm the null at n = 1500:

| Window | n | joint − target_only | p |
|---|---:|---:|---:|
| start=1100 (2026-05-12, original) | 200 | +3.50% | 0.146 |
| start=100 lock (2026-05-14) | 1500 | −1.47% | 0.369 |
| start=600 triangulation (2026-05-22) | 1500 | −2.53% | 0.094 |

The early +3.50% was small-n noise. Low model-power ratio (4.7×) on a single GPU does not generate sufficient acceptance to recover quality. Reports: `reports/jointadaspec_quality_qwen7b_1p5b_quality_lock_2026-05-14.md`, `reports/jointadaspec_quality_qwen7b_1p5b_quality_tri_2026-05-20.md`.

### Adaptive vs fixed fuzzy threshold (Theorem E, 14B/0.5B, n=300)

JointAdaSpec is **+4–8% more accurate and ~3.7× faster** than any fixed `fuzzy_sd_T` baseline, showing that adaptive joint control over (draft length, threshold) is non-trivial versus the natural non-adaptive alternative.

| Method | EM | tok/s | vs jointadaspec |
|---|---:|---:|---:|
| fuzzy_sd_T=1.0 | 53.67% | 2.85 | joint +4.33% |
| fuzzy_sd_T=1.25 | 50.33% | 2.95 | joint +7.67% |
| fuzzy_sd_T=1.5 | 51.67% | 3.00 | joint +6.33% |
| fuzzy_sd_T=2.0 | 53.33% | 3.06 | joint +4.67% |
| **jointadaspec** | **58.00%** | **11.12** | — |

Figure: `reports/thesis_figs/fig_E_adaptivity_ablation.pdf`.

### κ-sweep: Pareto trade-off (7B/1.5B, 6 κ values × n=300)

`tok/s` is flat across κ (16.28–16.55); EM varies non-monotonically (55.7–60.3%); joint ≈ cascade at every κ (gap ≤ 3%). The joint–cascade parity is robust to the Lagrange trade-off knob. Figure: `reports/thesis_figs/fig_bonus_kappa_sweep.pdf`.

### Historical `sp_samp` snapshot (Llama 8B/3B, 2026-03-28)

| Method | Accuracy (%) | Speed (tok/s) |
|---|---:|---:|
| Baseline | 70.89 | 72.68 |
| Speculative | 71.89 | 40.68 |
| AutoJudge (t=0.14) | 78.67 | 45.98 |
| Top-K (all) | 75.67 | 59.29 |

Reports: `reports/yandex_llama3_8b_3b_2026-03-28-llama-48h-cgrid8-{gsm8k,livecodebench}.md`.

More context and historical runs: `docs/RESULTS.md`.

## Quick Start (5 Minutes)

```bash
make setup
make check
make test
make bench-toy OUT=/tmp/bench_toy.jsonl
```

Optional tiny HF smoke:

```bash
make smoke-hf OUT=/tmp/smoke_hf.jsonl
```

JointAdaSpec toy validation:

```bash
.venv/bin/python -m pytest tests/test_features.py tests/test_verification.py tests/test_mdp_solver.py tests/test_inference.py tests/test_end_to_end.py -q
```

JointAdaSpec short smoke on the staged pipeline:

```bash
make jointadaspec-full MODEL_PAIR=qwen7b_1p5b JOINTADA_MAX_TRACES=10 JOINTADA_MAX_SAMPLES=10 JOINTADA_N_SEEDS=1
```

## Reproduce Main Runs

Paper-style Qwen sweep:

```bash
make paper-eval
```

Local Qwen 7B/1.5B sweep:

```bash
make local-eval
```

Local Llama 8B/3B sweep:

```bash
bash scripts/run_llama3_8b_3b_eval.sh
```

JointAdaSpec staged long-run on Qwen2.5 7B/1.5B for both GSM8K and LiveCodeBench:

```bash
bash scripts/run_jointadaspec_qwen_longrun.sh
```

Recommended monitoring:

```bash
tmux new -s jointadaspec48h
bash scripts/run_jointadaspec_qwen_longrun.sh | tee -a logs/jointadaspec_$(date +%F).log
```

Validate any JSONL output:

```bash
.venv/bin/python scripts/validate_results_jsonl.py --path datasets/results.jsonl --strict
```

## Project Structure

- `sp_samp/` core implementations and HF adapters
- `jointadaspec/` JointAdaSpec MDP training and inference stack
- `benchmarks/` benchmark entrypoint and result logging
- `configs/` model, method, and experiment presets
- `scripts/` orchestration, validation, and report generation
- `tests/` unit tests
- `reports/` tracked benchmark artifacts
- `datasets/` local datasets and run outputs (gitignored)

Method design notes:

- `docs/CONSENSUS_AUTOJUDGE.md` - disagreement-aware two-draft approximate decoding design

## Constraints and Repro Notes

- Draft and target must use tokenizer-compatible vocab mapping.
- AutoJudge paper C-grid policy is `1e-7..1e0` (8 values).
- Reusing the same output file enables automatic resume by `resume_key`.
- Llama checkpoints in this environment are gated on Hugging Face; use `experiments/qwen25_7b_1p5b_jointadaspec` for ungated JointAdaSpec smoke and long runs.
- JointAdaSpec trace collection is intentionally exact and CPU-heavy at the control layer; long runs should be started in `tmux`.

## For Reviewers and Contributors

- Contribution guide: `CONTRIBUTING.md`
- Open issues and feature proposals: GitHub issue templates
- Current priorities: `docs/ROADMAP.md`
- Repository presentation checklist: `docs/GITHUB_SETUP.md`

## License

MIT. See `LICENSE`.
