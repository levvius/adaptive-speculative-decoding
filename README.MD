# Adaptive Speculative Decoding

[![CI](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml/badge.svg)](https://github.com/levvius/adaptive-speculative-decoding/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](.python-version)

Implementation and benchmark of **joint adaptive control over draft length and verification
threshold** for speculative LLM decoding, formalised as a tabular MDP. Companion code for the
master's thesis (магистерская диссертация) *JointAdaSpec* (Kozin A. A., 2026).

## ⚠️ Запуск и требования к ресурсам / Running this repo

**Полный исследовательский конвейер не запускается на ноутбуке или на CPU.** Воспроизведение
headline-результатов требует серьёзных вычислительных ресурсов и больших загрузок:

- топовый GPU (использовалась NVIDIA RTX 5090; нужно ≈ 24–32 ГБ VRAM);
- скачивание весов моделей на десятки гигабайт (Qwen2.5-14B ≈ 28 ГБ + draft-модель);
- датасет GSM8K и ≈ 14–18 ч расчёта на один полный прогон.

Без этих ресурсов сбор трейсов и бенчмарк (`make jointadaspec-full`, `make bench`, `make *-eval`)
работать не будут — это **ожидаемое поведение**, а не поломка репозитория.

**Что реально проверяется за минуты на CPU, без GPU и без скачивания моделей:**

| Команда | Что делает |
|---|---|
| `make test` | полный pytest-набор (94 теста) |
| `make bench-toy` | игрушечный бенчмарк без HF-моделей |
| `make check` | синтаксис + валидация конфигов |
| `make list-presets` | список моделей / методов / экспериментов |
| `make validate-configs` | проверка согласованности конфигов |

**EN (short).** The full pipeline needs a high-end GPU (RTX 5090, 24–32 GB VRAM), tens of GB of
model weights (Qwen2.5-14B ≈ 28 GB) and the GSM8K dataset; it does **not** run on a laptop or on
CPU. What *does* run on CPU in minutes without any downloads: `make test` (94 tests),
`make bench-toy`, `make check`, `make list-presets`, `make validate-configs`.

## Thesis snapshot status

The `v1-defense` tag is the immutable repository snapshot used for the thesis
defense. Its experimental results are artifact-backed thesis-snapshot results:
they correspond to the archived JointAdaSpec pipeline, reports, manifests, and
defense deck shipped in this branch.

The `main` branch is reserved for post-defense research: stricter artifact
metadata, fresh `block_verify_v1` reruns, prompt-clustered/Holm statistics, and
new benchmark claims. The results below should therefore be interpreted within
this archived thesis snapshot, not as a new post-defense block-v1 rerun.

Research continuation roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md).
Defense audit: [`docs/DEFENSE_AUDIT_2026-06-26.md`](docs/DEFENSE_AUDIT_2026-06-26.md).
Commission Q&A: [`docs/COMMISSION_QA_2026-06-26.md`](docs/COMMISSION_QA_2026-06-26.md).

## Defense snapshot

What is safe to defend from this snapshot:

- a finite MDP formulation for joint control over draft length and fuzzy
  verification threshold;
- theory explaining sample complexity, reward-shaping limits, and why joint and
  cascade policies can be nearly tied;
- a reproducible implementation with action-space and artifact metadata
  safeguards;
- artifact-backed thesis results from Run 1, Run 2, Experiment E, the κ-sweep,
  and the Theorem D analysis;
- a clear separation between the defense snapshot and future `main`-branch
  reruns.

## Thesis snapshot headline result

Qwen2.5 14B → 0.5B, GSM8K, n = 1500 paired (McNemar), RTX 5090:

| Method | EM | tok/s | Δ vs target_only | p |
|---|---:|---:|---:|---:|
| target_only | 52.93% | 14.45 | — | — |
| speculative | 53.27% | 4.88 | +0.33% | 0.877 |
| cascade_verif_then_length | 57.13% | 10.29 | +4.20% | 0.015 ✓ |
| **jointadaspec** | **57.00%** | **10.84** | **+4.07%** | **0.020 ✓** |

Within the archived thesis snapshot, JointAdaSpec outperformed `target_only` at
p < 0.05 and recovered **2.22×** throughput versus vanilla speculative. This is
the defended thesis-snapshot result; post-defense `main` work will rerun the
benchmark suite under the newest block-v1 artifact protocol before making new
claims.

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
make test         # full pytest suite; 94 passed on v1-defense
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
| `tests/` | Full pytest suite covering all methods and the MDP solver (94 passed on v1-defense) |
| `notebooks/thesis_plots.ipynb` | Defence figures with Russian explanations |
| `reports/`, `outputs/` | Benchmark artifacts, manifests, figures |
| `papers/` | Master's thesis materials, defence deck plan + speech, design brief, built deck in `papers/dist/` (Russian) |

## Documentation

- [`docs/RESULTS.md`](docs/RESULTS.md) — full benchmark tables (locked + historical runs)
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — post-thesis future work
- [`docs/REPRODUCIBILITY_RELEASE.md`](docs/REPRODUCIBILITY_RELEASE.md) — release checklist for rerun-ready thesis results
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — local checks and PR workflow
- [`papers/ИС61_fpm_КозинАА_2026.docx`](papers/ИС61_fpm_КозинАА_2026.docx) — master's thesis text (RU, DOCX)
- [`papers/pres.md`](papers/pres.md) — defence presentation plan + 5–5.5 minute speech (RU)
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
