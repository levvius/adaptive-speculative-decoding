#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Gemma-2 local sweep defaults; can still be overridden via env.
DATE_TAG="${DATE_TAG:-$(date +%F)}"

export OUT_GSM8K="${OUT_GSM8K:-datasets/results_gemma2_9b_2b_gsm8k_${DATE_TAG}.jsonl}"
export OUT_LCB="${OUT_LCB:-datasets/results_gemma2_9b_2b_lcb_${DATE_TAG}.jsonl}"
export REPORT_PREFIX="${REPORT_PREFIX:-reports/yandex_gemma2_9b_2b_${DATE_TAG}}"
export MANIFEST_PATH="${MANIFEST_PATH:-reports/gemma2_9b_2b_run_manifest_${DATE_TAG}.json}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-datasets/autojudge_gemma2_2b_to_9b.pt}"

export SPEC_EXPERIMENT="${SPEC_EXPERIMENT:-gemma2_9b_local_target_2b_local_speculative_k4}"
export AUTOJUDGE_EXPERIMENT="${AUTOJUDGE_EXPERIMENT:-gemma2_9b_local_target_2b_local_autojudge_k4}"
export TOPK_EXPERIMENT="${TOPK_EXPERIMENT:-gemma2_9b_local_target_2b_local_topk_k4}"

export MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-22000}"

exec bash scripts/run_local_7b_1p5b_eval.sh
