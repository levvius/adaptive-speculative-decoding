# Adaptive Speculative Decoding

[![CI](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml/badge.svg)](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](.python-version)

Pet project and research playground for LLM decoding acceleration.

This repository compares exact and approximate decoding strategies under a single benchmark harness, with reproducible configs, JSONL outputs, and report scripts.

`sp_samp/` remains the historical benchmark stack. `jointadaspec/` is the new thesis-focused implementation of JointAdaSpec: joint adaptive control of draft length and verification threshold through a tabular MDP.

## Why This Project

- Side-by-side comparison of Baseline, Speculative Sampling, AutoJudge, Consensus AutoJudge, Top-K, and SpecExec.
- Paper-aligned AutoJudge implementation (GSM8K label mining + LogisticRegression calibration).
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

## JointAdaSpec

JointAdaSpec models speculative decoding as a finite MDP over the state `s = (H, K, k)`:

- `H`: entropy of the draft distribution
- `K`: `KL(q || p)` between draft and target distributions
- `k`: current position in the draft window

The action is joint:

- `a_length ∈ {stop, continue}`
- `a_verif ∈ {1.00, 1.22, 1.49, 1.82, 2.22, 2.71, 3.30, 4.00}`

The implementation lives in `jointadaspec/` and is split into:

- `jointadaspec/core/`: features, verification rules, decoder base classes
- `jointadaspec/mdp/`: discrete spaces, trace collection, MDP estimation, value iteration
- `jointadaspec/inference/`: learned policy lookup and online JointAdaSpec decoder
- `jointadaspec/baselines/`: `vanilla_ar`, `fixed_sd`, `fuzzy_sd`, `specdecpp`
- `jointadaspec/metrics/`: speed, quality, Pareto helpers
- `jointadaspec/utils/`: model loading, dataset loading, structured JSON logging

The new pipeline is stage-based:

1. `scripts/01_collect_traces.py`
2. `scripts/02_solve_mdp.py`
3. `scripts/03_benchmark.py`

## Latest Benchmark Snapshot

Latest full Llama run: `2026-03-28-llama-48h-cgrid8` on RTX 5090.

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

JointAdaSpec short smoke on an ungated remote pair:

```bash
.venv/bin/python scripts/01_collect_traces.py \
  --config-name experiments/qwen25_7b_1p5b_jointadaspec \
  experiments.output_dir=outputs/jointadaspec_smoke/01_traces \
  experiments.n_traces=2 \
  experiments.datasets.train_max_samples=2

.venv/bin/python scripts/02_solve_mdp.py \
  --config-name experiments/qwen25_7b_1p5b_jointadaspec \
  experiments.output_dir=outputs/jointadaspec_smoke/02_solve \
  experiments.traces_path=outputs/jointadaspec_smoke/01_traces/traces.parquet

.venv/bin/python scripts/03_benchmark.py \
  --config-name experiments/qwen25_7b_1p5b_jointadaspec \
  experiments.output_dir=outputs/jointadaspec_smoke/03_bench \
  experiments.policy_path=outputs/jointadaspec_smoke/02_solve/policy.npz \
  experiments.datasets.test_max_samples=2

.venv/bin/python scripts/03_benchmark.py \
  --config-name experiments/qwen25_7b_1p5b_jointadaspec_livecodebench \
  experiments.output_dir=outputs/jointadaspec_smoke/04_bench_livecodebench \
  experiments.policy_path=outputs/jointadaspec_smoke/02_solve/policy.npz \
  experiments.datasets.test_max_samples=2
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
