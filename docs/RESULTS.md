# Results Overview

This page is a compact index of benchmark outcomes and where to find full artifacts.

## Latest Full Run (Llama 8B / 3B)

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
- Mistral 8B/3B:
  - `reports/yandex_mistral3_8b_3b_2026-03-20-mistral-gsm8k.md`
  - `reports/yandex_mistral3_8b_3b_2026-03-20-mistral-livecodebench.md`
- Gemma-2 9B/2B:
  - `reports/yandex_gemma2_9b_2b_2026-03-16-gemma-gsm8k.md`
  - `reports/yandex_gemma2_9b_2b_2026-03-16-gemma-livecodebench.md`

## Repro Command (Latest Llama Profile)

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
