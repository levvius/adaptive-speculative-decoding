# Adaptive Speculative Decoding

[![CI](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml/badge.svg)](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](.python-version)

Implementation and benchmark of **joint adaptive control over draft length and verification
threshold** for speculative LLM decoding, formalised as a tabular MDP. Companion code for the
bachelor's thesis *JointAdaSpec* (Kozin A. A., 2026).

## Headline result

Qwen2.5 14B → 0.5B, GSM8K, n = 1500 paired (McNemar), RTX 5090:

| Method | EM | tok/s | Δ vs target_only | p |
|---|---:|---:|---:|---:|
| target_only | 52.93% | 14.45 | — | — |
| speculative | 53.27% | 4.88 | +0.33% | 0.877 |
| cascade_verif_then_length | 57.13% | 10.29 | +4.20% | 0.015 ✓ |
| **jointadaspec** | **57.00%** | **10.84** | **+4.07%** | **0.020 ✓** |

JointAdaSpec significantly outperforms `target_only` at p < 0.05 and recovers **2.22×** throughput
versus vanilla speculative. Joint ≈ cascade (Δ = −0.13%, p = 0.96) is the analytical prediction
of Theorem D (advantage on the C4-violating set is ε-small).

Full tables, 7B/1.5B null result, Theorem E ablation, and κ-sweep — see
[`docs/RESULTS.md`](docs/RESULTS.md).

## Methods implemented

| Method | Distribution | Idea |
|---|---|---|
| `baseline` | exact | Target-only autoregressive decoding |
| `speculative` | exact | Draft proposes, target verifies (Leviathan 2022) |
| `autojudge` | approximate | Trained classifier accepts mismatches |
| `topk` | approximate | Accept mismatch if target token in draft top-K |
| `specexec` | exact | Parallel speculative branches + cache reuse |
| **`jointadaspec`** | approximate | **MDP-controlled (draft length, fuzzy threshold)** |

## Quick start

```bash
make setup        # CPU install (uses pip + requirements.txt)
make test         # 72 unit tests, ~5 s
make bench-toy    # toy benchmark without HF models
```

Full JointAdaSpec pipeline (requires GPU):

```bash
make jointadaspec-full MODEL_PAIR=qwen14b_0p5b
```

Pipeline stages: `01_collect_traces` → `02_solve_mdp` → `03_benchmark` →
`04_verify_conditions` → `05_write_manifest`.

## Repository structure

| Directory | Contents |
|---|---|
| `jointadaspec/` | MDP framework — state/action spaces, VI solver, decoder, baselines |
| `sp_samp/` | Legacy speculative / AutoJudge / SpecExec / Top-K implementations |
| `benchmarks/` | Unified benchmark runner (`bench_speculative.py`) |
| `scripts/` | Pipeline stages and orchestration shell scripts |
| `configs/` | Model / method / experiment JSON presets + Hydra YAML |
| `tests/` | 72 unit tests covering all methods and MDP solver |
| `notebooks/thesis_plots.ipynb` | Defence figures with Russian explanations |
| `reports/`, `outputs/` | Benchmark artifacts, manifests, figures |
| `papers/` | Thesis DOCX and defence guide (Russian) |

## Documentation

- [`docs/RESULTS.md`](docs/RESULTS.md) — full benchmark tables (locked + historical runs)
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — post-thesis future work
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — local checks and PR workflow
- [`papers/ВКР_тезисы_и_структура.md`](papers/ВКР_тезисы_и_структура.md) — thesis defence guide (RU)
- [`CODEX.MD`](CODEX.MD) — chronological change log with file-level detail

## Key constraints

- Draft and target models must share an identical tokenizer vocabulary.
- AutoJudge requires GSM8K-format datasets (`question` + `answer` fields).
- RTX 50-series (Blackwell, sm_120) requires `torch>=2.7+cu128` for GPU paths.
- Benchmark resume is automatic when reusing the same `--out` JSONL file.

## License

MIT — see [LICENSE](LICENSE).
