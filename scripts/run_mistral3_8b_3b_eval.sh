#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Mistral-3 local sweep defaults; can be overridden via env.
DATE_TAG="${DATE_TAG:-$(date +%F)}"
if [[ -x ".venv-mistral3/bin/python" ]]; then
  export PYTHON_BIN="${PYTHON_BIN:-.venv-mistral3/bin/python}"
fi

export OUT_GSM8K="${OUT_GSM8K:-datasets/results_mistral3_8b_3b_gsm8k_${DATE_TAG}.jsonl}"
export OUT_LCB="${OUT_LCB:-datasets/results_mistral3_8b_3b_lcb_${DATE_TAG}.jsonl}"
export REPORT_PREFIX="${REPORT_PREFIX:-reports/yandex_mistral3_8b_3b_${DATE_TAG}}"
export MANIFEST_PATH="${MANIFEST_PATH:-reports/mistral3_8b_3b_run_manifest_${DATE_TAG}.json}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-datasets/autojudge_mistral3_3b_to_8b.pt}"

export SPEC_EXPERIMENT="${SPEC_EXPERIMENT:-mistral_8b_local_target_3b_local_speculative_k4}"
export AUTOJUDGE_EXPERIMENT="${AUTOJUDGE_EXPERIMENT:-mistral_8b_local_target_3b_local_autojudge_k4}"
export TOPK_EXPERIMENT="${TOPK_EXPERIMENT:-mistral_8b_local_target_3b_local_topk_k4}"

export MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-24000}"

exec bash scripts/run_local_7b_1p5b_eval.sh
