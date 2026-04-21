# Adaptive Speculative Decoding

[![CI](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml/badge.svg)](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](.python-version)

Pet project and research playground for LLM decoding acceleration.

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

JointAdaSpec formulates speculative decoding control as a finite MDP over the state `s = (H, K, k)` and solves the joint policy by value iteration.

- `H`: entropy of the draft distribution
- `K`: `KL(q || p)` between draft and target distributions
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

## Latest Benchmark Snapshot

### Latest JointAdaSpec Run

Latest real `jointadaspec/` run: `2026-04-14`, Qwen2.5 `7B -> 1.5B`, RTX 5090.

Artifacts:
- `reports/jointadaspec_qwen_7b_1p5b_2026-04-14.md`
- `outputs/jointadaspec_qwen_2026-04-14/01_traces_gsm8k/`
- `outputs/jointadaspec_qwen_2026-04-14/02_solve/`
- `outputs/jointadaspec_qwen_2026-04-14/03_bench_gsm8k/`
- `outputs/jointadaspec_qwen_2026-04-14/04_bench_livecodebench/`

Trace and solve snapshot:
- `3000` collected traces
- `479880` one-step transition records
- `kappa` sweep: `0.0, 0.5, 1.0, 2.0, 5.0`
- value iteration converged for all saved policies

GSM8K throughput snapshot (`100` prompts, throughput + acceptance only):

| Method | Speed (tok/s) | Acceptance | vs Vanilla |
|---|---:|---:|---:|
| Vanilla AR | 29.38 | 0.000 | 1.000 |
| Fixed SD | 17.26 | 0.801 | 0.588 |
| Fuzzy SD (`T=4.0`) | 19.09 | 0.915 | 0.650 |
| JointAdaSpec | 18.99 | 0.951 | 0.646 |
| SpecDecPP | 17.81 | 0.835 | 0.606 |

LiveCodeBench throughput snapshot (`100` prompts):

| Method | Speed (tok/s) | Acceptance | vs Vanilla |
|---|---:|---:|---:|
| Vanilla AR | 14.36 | 0.000 | 1.000 |
| Fixed SD | 8.05 | 0.659 | 0.560 |
| Fuzzy SD (`T=4.0`) | 9.62 | 0.843 | 0.670 |
| JointAdaSpec | 10.34 | 0.890 | 0.720 |
| SpecDecPP | 9.04 | 0.763 | 0.629 |

Current interpretation:
- `JointAdaSpec` is operational end to end on the Qwen `7B -> 1.5B` profile used for the first thesis traces.
- On this single-GPU ratio it improves acceptance over the fixed-window baselines, but it does not yet outperform target-only decoding in throughput.
- The current `jointadaspec/` benchmark path now writes resumable seeded JSONL with bootstrap confidence intervals, plus legacy `run.jsonl` prompt records for backward compatibility.

### Historical `sp_samp` Snapshot

Latest historical `sp_samp/` full Llama run: `2026-03-28-llama-48h-cgrid8` on RTX 5090.

Source reports:
- `reports/yandex_llama3_8b_3b_2026-03-28-llama-48h-cgrid8-gsm8k.md`
- `reports/yandex_llama3_8b_3b_2026-03-28-llama-48h-cgrid8-livecodebench.md`

GSM8K highlights (k=4):

| Method | Accuracy (%) | Speed (tok/s) |
|---|---:|---:|
| Baseline | 70.89 | 72.68 |
| Speculative | 71.89 | 40.68 |
| AutoJudge (t=0.14) | 78.67 | 45.98 |
| Top-K (all) | 75.67 | 59.29 |

LiveCodeBench highlights (throughput only):

| Method | Speed (tok/s) |
|---|---:|
| Baseline | 71.52 |
| Speculative | 34.80 |
| AutoJudge (t=1.0) | 29.27 |
| Top-K (all) | 36.53 |

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
