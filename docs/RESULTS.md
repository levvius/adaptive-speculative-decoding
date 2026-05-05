# Results Overview

This page is a compact index of benchmark outcomes and where to find full artifacts.

## Latest Artifacts Through 2026-04-28

The latest tracked `jointadaspec/` artifacts include one substantive benchmark snapshot and several smoke or partial runs. Treat the smoke rows as regression and pipeline evidence, not as statistically stable best-of-run claims.

| Run | Model pair | Scope | Best observed result | Reliability |
|---|---|---|---|---|
| `2026-04-14` | Qwen2.5 `7B -> 1.5B` | `3000` traces, `100` GSM8K prompts, `100` LiveCodeBench prompts | GSM8K: `jointadaspec` `18.99` tok/s, acceptance `0.951`; LiveCodeBench: `jointadaspec` `10.34` tok/s, acceptance `0.890` | Best substantive snapshot |
| `2026-04-21` | Qwen2.5 `7B -> 1.5B` | seeded/schema smoke, `1` prompt, `1` seed, `8` new tokens | `cascade_verif_then_length` `18.47` tok/s, `1.204x` vs target-only | Complete smoke only |
| `2026-04-28` | Qwen2.5 `14B -> 0.5B` | smoke, `5` prompts, `1` seed, `64` new tokens | `cascade_verif_then_length` `13.15` tok/s, `0.725x`; `jointadaspec` `12.85` tok/s, `0.709x`; `target_only` `18.13` tok/s | Complete smoke only |
| `2026-04-28` | Qwen2.5 `7B -> 1.5B` | full trace/solve attempt, `500` traces | policy and condition artifacts saved; no benchmark summary | Partial, benchmark failed |

Curated roll-up report: `reports/jointadaspec_qwen_runs_through_2026-04-28.md`.

Next planned rerun: local quality-aware Qwen `7B -> 1.5B` with `MODEL_PAIR=qwen7b_1p5b_quality`, `500` traces, `100` GSM8K samples, `3` seeds, and `256` max new tokens. This rerun is intended to replace the failed HuggingFace-loading benchmark stage from `2026-04-28`, not to reinterpret the smoke rows above.

### 2026-04-28 failure note

The Qwen `7B -> 1.5B` full benchmark stage failed while loading `Qwen/Qwen2.5-7B-Instruct` from HuggingFace with an SSL EOF. The watcher script then refused to launch the dependent full Step 3 because it could not find `outputs/jointadaspec_qwen7b_1p5b_2026-04-28/03_bench_gsm8k/results.jsonl`, `reports/pareto_qwen7b_1p5b_2026-04-28.pdf`, `reports/ablation_qwen7b_1p5b_2026-04-28.pdf`, or `reports/threshold_surface_qwen7b_1p5b_2026-04-28/`.

### Condition checks

| Run | Passed | Failed |
|---|---|---|
| Qwen `7B -> 1.5B`, `2026-04-21` | `c3`, `n1`, `n2` | `c1`, `c2`, `c4` |
| Qwen `7B -> 1.5B`, `2026-04-28` | `n1`, `n2` | `c1`, `c2`, `c3`, `c4` |
| Qwen `14B -> 0.5B`, `2026-04-28` | `c2`, `c3`, `n1`, `n2` | `c1`, `c4` |

## Best Substantive JointAdaSpec Snapshot (Qwen 7B / 1.5B)

Run tag: `2026-04-14`

- Report: `reports/jointadaspec_qwen_7b_1p5b_2026-04-14.md`
- Trace artifacts: `outputs/jointadaspec_qwen_2026-04-14/01_traces_gsm8k/`
- Policy artifacts: `outputs/jointadaspec_qwen_2026-04-14/02_solve/`
- GSM8K benchmark: `outputs/jointadaspec_qwen_2026-04-14/03_bench_gsm8k/`
- LiveCodeBench benchmark: `outputs/jointadaspec_qwen_2026-04-14/04_bench_livecodebench/`

### Trace and solve snapshot

| Item | Value |
|---|---:|
| Model pair | `Qwen2.5-7B-Instruct -> Qwen2.5-1.5B-Instruct` |
| Trace count | 3000 |
| Recorded one-step transitions | 479880 |
| `kappa` sweep | `0.0, 0.5, 1.0, 2.0, 5.0` |
| VI iterations (`kappa=0.0`) | 834 |
| VI iterations (`kappa=0.5`) | 830 |
| VI iterations (`kappa=1.0`) | 827 |
| VI iterations (`kappa=2.0`) | 825 |
| VI iterations (`kappa=5.0`) | 824 |

### GSM8K throughput snapshot

This older run predates the seeded benchmark summary path that records task-level GSM8K exact match. It remains the best substantive throughput/acceptance snapshot because it used `100` prompts rather than a smoke-sized slice.

| Method | Speed (tok/s) | Acceptance | vs Vanilla |
|---|---:|---:|---:|
| vanilla_ar | 29.38 | 0.000 | 1.000 |
| fixed_sd | 17.26 | 0.801 | 0.588 |
| fuzzy_sd_T4 | 19.09 | 0.915 | 0.650 |
| jointadaspec | 18.99 | 0.951 | 0.646 |
| specdecpp | 17.81 | 0.835 | 0.606 |

### LiveCodeBench throughput snapshot

| Method | Speed (tok/s) | Acceptance | vs Vanilla |
|---|---:|---:|---:|
| vanilla_ar | 14.36 | 0.000 | 1.000 |
| fixed_sd | 8.05 | 0.659 | 0.560 |
| fuzzy_sd_T4 | 9.62 | 0.843 | 0.670 |
| jointadaspec | 10.34 | 0.890 | 0.720 |
| specdecpp | 9.04 | 0.763 | 0.629 |

### Repro command

```bash
bash scripts/run_jointadaspec_qwen_longrun.sh
```

## Latest Historical `sp_samp` Full Run (Llama 8B / 3B)

Run tag: `2026-03-28-llama-48h-cgrid8`

- Manifest: `reports/llama3_8b_3b_run_manifest_2026-03-28-llama-48h-cgrid8.json`
- GSM8K report: `reports/yandex_llama3_8b_3b_2026-03-28-llama-48h-cgrid8-gsm8k.md`
- LiveCodeBench report: `reports/yandex_llama3_8b_3b_2026-03-28-llama-48h-cgrid8-livecodebench.md`
- Raw outputs:
  - `datasets/results_llama3_8b_3b_gsm8k_2026-03-28-llama-48h-cgrid8.jsonl`
  - `datasets/results_llama3_8b_3b_lcb_2026-03-28-llama-48h-cgrid8.jsonl`

### GSM8K snapshot

| Method | Parameter | Accuracy (%) | Speed (tok/s) |
|---|---:|---:|---:|
| baseline | - | 70.89 | 72.68 |
| speculative | - | 71.89 | 40.68 |
| autojudge | 0.140 | 78.67 | 45.98 |
| topk | all | 75.67 | 59.29 |

### LiveCodeBench snapshot

| Method | Parameter | Speed (tok/s) |
|---|---:|---:|
| baseline | - | 71.52 |
| speculative | - | 34.80 |
| autojudge | 1.000 | 29.27 |
| topk | all | 36.53 |

## Historical Runs (Tracked)

- Qwen local 7B/1.5B:
  - `reports/yandex_local_7b_1p5b_2026-03-10-gsm8k.md`
  - `reports/yandex_local_7b_1p5b_2026-03-10-livecodebench.md`
- JointAdaSpec Qwen 7B/1.5B:
  - `reports/jointadaspec_qwen_7b_1p5b_2026-04-14.md`
- Mistral 8B/3B:
  - `reports/yandex_mistral3_8b_3b_2026-03-20-mistral-gsm8k.md`
  - `reports/yandex_mistral3_8b_3b_2026-03-20-mistral-livecodebench.md`
- Gemma-2 9B/2B:
  - `reports/yandex_gemma2_9b_2b_2026-03-16-gemma-gsm8k.md`
  - `reports/yandex_gemma2_9b_2b_2026-03-16-gemma-livecodebench.md`

## Repro Command (Latest Historical Llama Profile)

```bash
DATE_TAG="$(date +%F)-llama-48h-cgrid8"
CHECKPOINT_PATH="datasets/autojudge_llama3_3b_to_8b_${DATE_TAG}.pt" \
OUT_GSM8K="datasets/results_llama3_8b_3b_gsm8k_${DATE_TAG}.jsonl" \
OUT_LCB="datasets/results_llama3_8b_3b_lcb_${DATE_TAG}.jsonl" \
REPORT_PREFIX="reports/yandex_llama3_8b_3b_${DATE_TAG}" \
MANIFEST_PATH="reports/llama3_8b_3b_run_manifest_${DATE_TAG}.json" \
bash scripts/run_llama3_8b_3b_eval.sh
```

## Validation Commands

```bash
.venv/bin/python scripts/validate_results_jsonl.py --path datasets/results_llama3_8b_3b_gsm8k_2026-03-28-llama-48h-cgrid8.jsonl --strict
.venv/bin/python scripts/validate_results_jsonl.py --path datasets/results_llama3_8b_3b_lcb_2026-03-28-llama-48h-cgrid8.jsonl --strict
```
