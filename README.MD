# Adaptive Speculative Decoding

[![CI](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml/badge.svg)](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](.python-version)

Pet project and research playground for LLM decoding acceleration.

This repository compares exact and approximate decoding strategies under a single benchmark harness, with reproducible configs, JSONL outputs, and report scripts.

## Why This Project

- Side-by-side comparison of Baseline, Speculative Sampling, AutoJudge, Top-K, and SpecExec.
- Paper-aligned AutoJudge implementation (GSM8K label mining + LogisticRegression calibration).
- Long-run friendly workflow (resume keys, checkpoints, strict result schema validation).
- Real benchmark reports are versioned in `reports/`.

## Implemented Methods

| Method | Exact vs target distribution | Main idea |
|---|---|---|
| `baseline` | exact | Target-only decoding |
| `speculative` | exact | Draft proposes, target verifies |
| `autojudge` | approximate | Judge can accept some mismatches |
| `topk` | approximate | Accept mismatch if target token in top-k |
| `specexec` | exact | Parallel speculative branches + cache reuse |

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

Validate any JSONL output:

```bash
.venv/bin/python scripts/validate_results_jsonl.py --path datasets/results.jsonl --strict
```

## Project Structure

- `sp_samp/` core implementations and HF adapters
- `benchmarks/` benchmark entrypoint and result logging
- `configs/` model, method, and experiment presets
- `scripts/` orchestration, validation, and report generation
- `tests/` unit tests
- `reports/` tracked benchmark artifacts
- `datasets/` local datasets and run outputs (gitignored)

## Constraints and Repro Notes

- Draft and target must use tokenizer-compatible vocab mapping.
- AutoJudge paper C-grid policy is `1e-7..1e0` (8 values).
- Reusing the same output file enables automatic resume by `resume_key`.

## For Reviewers and Contributors

- Contribution guide: `CONTRIBUTING.md`
- Open issues and feature proposals: GitHub issue templates
- Current priorities: `docs/ROADMAP.md`
- Repository presentation checklist: `docs/GITHUB_SETUP.md`

## License

MIT. See `LICENSE`.
