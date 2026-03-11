#!/usr/bin/env bash
# Run only the AutoJudge + TopK methods for the local Qwen2.5-7B/1.5B GSM8K eval.
# Appends to the existing results file; benchmark resume mode skips completed entries.
#
# Usage (background, survives shell exit):
#   mkdir -p logs
#   nohup bash scripts/run_autojudge_topk_gsm8k_bg.sh > logs/aj_topk_2026-03-10.log 2>&1 &
#   echo "PID=$!"
#   tail -f logs/aj_topk_2026-03-10.log

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

DATE_TAG="${DATE_TAG:-2026-03-10}"
CONFIG_DIR="${CONFIG_DIR:-configs}"
TRAIN_DATASET="${TRAIN_DATASET:-datasets/gsm8k_train.jsonl}"
TEST_DATASET="${TEST_DATASET:-datasets/gsm8k_test.jsonl}"

# Append to the existing results file (resume mode handles deduplication).
OUT_GSM8K="${OUT_GSM8K:-datasets/results_local_7b_1p5b_gsm8k.jsonl}"
REPORT_PREFIX="${REPORT_PREFIX:-reports/yandex_local_7b_1p5b_${DATE_TAG}}"
MANIFEST_PATH="${MANIFEST_PATH:-reports/local_7b_1p5b_run_manifest_${DATE_TAG}.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-datasets/autojudge_qwen25_1p5b_to_7b.pt}"

AUTOJUDGE_EXPERIMENT="${AUTOJUDGE_EXPERIMENT:-qwen25_7b_local_target_1p5b_local_autojudge_k4}"
TOPK_EXPERIMENT="${TOPK_EXPERIMENT:-qwen25_7b_local_target_1p5b_local_topk_k4}"

MAX_SAMPLES="${MAX_SAMPLES:-100}"
RUNS="${RUNS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SPEC_K="${SPEC_K:-4}"
EVAL_MODE="${EVAL_MODE:-zero_shot_cot}"

HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_XET

# Optional quantization (e.g. QUANT=8bit DRAFT_QUANT=8bit to reduce VRAM usage).
QUANT="${QUANT:-}"
DRAFT_QUANT="${DRAFT_QUANT:-}"

# GPU pre-flight: wait until enough VRAM is free before loading models.
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-17000}"
GPU_WAIT_TIMEOUT_SECS="${GPU_WAIT_TIMEOUT_SECS:-43200}"
GPU_CHECK_INTERVAL_SECS="${GPU_CHECK_INTERVAL_SECS:-120}"

_gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -d ' '
}

_gpu_other_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -v "^$" || true
}

mkdir -p datasets reports logs

echo "[$(date)] Checking GPU availability (need ${MIN_FREE_VRAM_MIB} MiB free)..."
elapsed=0
while true; do
  free_mib=$(_gpu_free_mib)
  pids=$(_gpu_other_pids)
  if [[ -n "${free_mib}" && "${free_mib}" -ge "${MIN_FREE_VRAM_MIB}" ]]; then
    echo "[$(date)] GPU free: ${free_mib} MiB — proceeding."
    break
  fi
  if [[ "${elapsed}" -ge "${GPU_WAIT_TIMEOUT_SECS}" ]]; then
    echo "[$(date)] ERROR: GPU still busy after ${GPU_WAIT_TIMEOUT_SECS}s. Aborting."
    echo "  Free VRAM: ${free_mib} MiB, competing PIDs: ${pids}"
    exit 1
  fi
  echo "[$(date)] GPU busy (free: ${free_mib} MiB, PIDs: ${pids}). Waiting ${GPU_CHECK_INTERVAL_SECS}s..."
  sleep "${GPU_CHECK_INTERVAL_SECS}"
  elapsed=$(( elapsed + GPU_CHECK_INTERVAL_SECS ))
done

QUANT_ARGS=()
[[ -n "${QUANT}" ]] && QUANT_ARGS+=(--quant "${QUANT}")
[[ -n "${DRAFT_QUANT}" ]] && QUANT_ARGS+=(--draft-quant "${DRAFT_QUANT}")

COMMON_ARGS=(
  --config-dir "${CONFIG_DIR}"
  --eval-task gsm8k
  --gsm8k-eval-mode "${EVAL_MODE}"
  --dataset "${TEST_DATASET}"
  --runs "${RUNS}"
  --max-samples "${MAX_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --k "${SPEC_K}"
  --out "${OUT_GSM8K}"
)

echo "[$(date)] === Starting AutoJudge + TopK GSM8K run ==="
echo "[$(date)] OUT_GSM8K=${OUT_GSM8K}"
echo "[$(date)] CHECKPOINT_PATH=${CHECKPOINT_PATH}"

# --- AutoJudge thresholds ---
for threshold in 0.005 0.01 0.03 0.05 0.09 0.14 0.23 1.0; do
  echo "[$(date)] === AutoJudge threshold=${threshold} ==="
  "${PYTHON_BIN}" -m sp_samp.cli bench \
    --experiment "${AUTOJUDGE_EXPERIMENT}" \
    --method autojudge \
    --autojudge-checkpoint "${CHECKPOINT_PATH}" \
    --autojudge-train-dataset "${TRAIN_DATASET}" \
    --autojudge-threshold "${threshold}" \
    "${COMMON_ARGS[@]}" \
    "${QUANT_ARGS[@]}"
  echo "[$(date)] === AutoJudge threshold=${threshold} done ==="
done

# --- Top-K ranks ---
for rank in 2 4 8 16 32 all; do
  echo "[$(date)] === TopK rank=${rank} ==="
  "${PYTHON_BIN}" -m sp_samp.cli bench \
    --experiment "${TOPK_EXPERIMENT}" \
    --method topk \
    --topk-rank "${rank}" \
    "${COMMON_ARGS[@]}" \
    "${QUANT_ARGS[@]}"
  echo "[$(date)] === TopK rank=${rank} done ==="
done

echo "[$(date)] === All runs complete. Validating output schema... ==="
"${PYTHON_BIN}" scripts/validate_results_jsonl.py --path "${OUT_GSM8K}" --strict

echo "[$(date)] === Building Yandex-style report... ==="
"${PYTHON_BIN}" scripts/report_yandex_style.py \
  --input "${OUT_GSM8K}" \
  --eval-task gsm8k \
  --manifest "${MANIFEST_PATH}" \
  --out-prefix "${REPORT_PREFIX}-gsm8k"

echo "[$(date)] === Done. ==="
echo "  Results: ${OUT_GSM8K}"
echo "  Report:  ${REPORT_PREFIX}-gsm8k.md"
