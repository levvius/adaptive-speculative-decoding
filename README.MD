# Adaptive Speculative Decoding

[![CI](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml/badge.svg)](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](.python-version)

Implementation and benchmark of **joint adaptive control over draft length and verification
threshold** for speculative LLM decoding, formalised as a tabular MDP. Companion code for the
master's thesis (магистерская диссертация) *JointAdaSpec* (Kozin A. A., 2026).

## Current status

This repository has gone through a thesis/code repair pass. The current
JointAdaSpec code path now uses block-verification semantics (`block_verify_v1`)
and rejects legacy policy artifacts that lack semantic metadata.

The historical large-scale numbers below are still marked as **legacy /
pre-repair evidence**: the locked Run 1 report points to a pre-repair 16-action
policy artifact, and the raw lock `benchmark.csv` is not present in this working
copy for prompt-clustered reanalysis. These numbers are preserved for
traceability, but they must not be presented as final evidence for block
speculative-decoding speedup until the benchmark suite is rerun with regenerated
`block_verify_v1` artifacts.

Repair plan: [`docs/REPAIR_PLAN_REPO_THESIS.md`](docs/REPAIR_PLAN_REPO_THESIS.md).
Defense audit: [`docs/DEFENSE_AUDIT_2026-06-26.md`](docs/DEFENSE_AUDIT_2026-06-26.md).

## Defense snapshot

What is safe to defend from this snapshot:

- a finite MDP formulation for joint control over draft length and fuzzy
  verification threshold;
- theory explaining sample complexity, reward-shaping limits, and why joint and
  cascade policies can be nearly tied;
- a repaired block-v1 implementation with action-space and artifact metadata
  safeguards;
- legacy benchmark evidence as motivation and historical traceability;
- a reproducible rerun plan for turning the repaired implementation into final
  block-v1 benchmark claims.

## Legacy / pre-repair headline result

Qwen2.5 14B → 0.5B, GSM8K, n = 1500 paired (McNemar), RTX 5090:

| Method | EM | tok/s | Δ vs target_only | p |
|---|---:|---:|---:|---:|
| target_only | 52.93% | 14.45 | — | — |
| speculative | 53.27% | 4.88 | +0.33% | 0.877 |
| cascade_verif_then_length | 57.13% | 10.29 | +4.20% | 0.015 ✓ |
| **jointadaspec** | **57.00%** | **10.84** | **+4.07%** | **0.020 ✓** |

Under the legacy implementation, JointAdaSpec outperformed `target_only` at
p < 0.05 and recovered **2.22×** throughput versus vanilla speculative. This
result is **pre-repair evidence**, not a final block-v1 benchmark claim.

Full tables, 7B/1.5B null result, Experiment E ablation, and κ-sweep — see
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

Build the defence deck in the three most common formats (PPTX + PDF + HTML):

```bash
make slides   # -> papers/dist/JointAdaSpec_defense.{pptx,pdf,html}
```

The master deck is authored with `python-pptx` from
[`papers/pres.md`](papers/pres.md); the PDF and self-contained HTML are derived
via LibreOffice.

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
| `papers/` | Master's thesis ([DOCX](papers/Козин_newera.docx), clickable TOC · [PDF](papers/Козин_newera.pdf)), defence deck plan + speech, design brief, built deck in `papers/dist/` (Russian) |

## Documentation

- [`docs/RESULTS.md`](docs/RESULTS.md) — full benchmark tables (locked + historical runs)
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — post-thesis future work
- [`docs/REPRODUCIBILITY_RELEASE.md`](docs/REPRODUCIBILITY_RELEASE.md) — release checklist for rerun-ready thesis results
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — local checks and PR workflow
- [`papers/Козин_newera.pdf`](papers/Козин_newera.pdf) — full master's thesis (RU, PDF; DOCX alongside)
- [`papers/pres.md`](papers/pres.md) — defence presentation plan + 10-minute speech (RU)
- [`papers/claude_design.md`](papers/claude_design.md) — slide-deck design brief for building the visual deck
- [`papers/dist/`](papers/dist/) — built defence deck in three formats (PPTX · PDF · HTML); rebuild with `make slides`
- [`papers/ВКР_тезисы_и_структура.md`](papers/ВКР_тезисы_и_структура.md) — thesis defence guide (RU)
- [`CODEX.MD`](CODEX.MD) — chronological change log with file-level detail

## Key constraints

- Draft and target models must share an identical tokenizer vocabulary.
- AutoJudge requires GSM8K-format datasets (`question` + `answer` fields).
- RTX 50-series (Blackwell, sm_120) requires `torch>=2.7+cu128` for GPU paths.
- Benchmark resume is automatic when reusing the same `--out` JSONL file.

## License

MIT — see [LICENSE](LICENSE).
