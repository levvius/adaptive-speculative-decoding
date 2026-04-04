#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

DATE_TAG="${DATE_TAG:-$(date +%F)}"
CONFIG_DIR="${CONFIG_DIR:-configs}"
TRAIN_DATASET="${TRAIN_DATASET:-datasets/gsm8k_train.jsonl}"
TEST_DATASET="${TEST_DATASET:-datasets/gsm8k_test.jsonl}"
LCB_DATASET="${LCB_DATASET:-datasets/livecodebench.jsonl}"
LCB_VERSION="${LCB_VERSION:-release_v5}"

OUT_GSM8K="${OUT_GSM8K:-datasets/results_local_7b_1p5b_gsm8k_${DATE_TAG}.jsonl}"
OUT_LCB="${OUT_LCB:-datasets/results_local_7b_1p5b_lcb_${DATE_TAG}.jsonl}"
REPORT_PREFIX="${REPORT_PREFIX:-reports/yandex_local_7b_1p5b_${DATE_TAG}}"
MANIFEST_PATH="${MANIFEST_PATH:-reports/local_7b_1p5b_run_manifest_${DATE_TAG}.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-datasets/autojudge_qwen25_1p5b_to_7b.pt}"

SPEC_EXPERIMENT="${SPEC_EXPERIMENT:-qwen25_7b_local_target_1p5b_local_speculative_k4}"
AUTOJUDGE_EXPERIMENT="${AUTOJUDGE_EXPERIMENT:-qwen25_7b_local_target_1p5b_local_autojudge_k4}"
TOPK_EXPERIMENT="${TOPK_EXPERIMENT:-qwen25_7b_local_target_1p5b_local_topk_k4}"
AUTOJUDGE_CLASSIFIER="${AUTOJUDGE_CLASSIFIER:-logreg}"

MAX_SAMPLES="${MAX_SAMPLES:-300}"
RUNS="${RUNS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SPEC_K="${SPEC_K:-4}"
AUTOJUDGE_THRESHOLDS="${AUTOJUDGE_THRESHOLDS:-0.005,0.01,0.03,0.05,0.09,0.14,0.23,1.0}"
TOPK_GRID="${TOPK_GRID:-2,4,8,16,32,all}"
EVAL_MODE="${EVAL_MODE:-zero_shot_cot}"
RUN_CHECKS="${RUN_CHECKS:-1}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
RESET_OUT="${RESET_OUT:-0}"

export HF_HUB_DISABLE_XET

mkdir -p datasets reports

# Optional quantization
QUANT="${QUANT:-}"
DRAFT_QUANT="${DRAFT_QUANT:-}"

# GPU pre-flight
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

if [[ "${RESET_OUT}" == "1" ]]; then
  rm -f "${OUT_GSM8K}" "${OUT_LCB}"
fi

echo "[INFO] Writing manifest: ${MANIFEST_PATH}"
"${PYTHON_BIN}" scripts/write_run_manifest.py --out "${MANIFEST_PATH}"

# --- Download datasets if missing ---
if [[ ! -f "${TRAIN_DATASET}" ]]; then
  echo "[INFO] Downloading GSM8K train split to ${TRAIN_DATASET}"
  curl -fsSL "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl" -o "${TRAIN_DATASET}"
fi
if [[ ! -f "${TEST_DATASET}" ]]; then
  echo "[INFO] Downloading GSM8K test split to ${TEST_DATASET}"
  curl -fsSL "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl" -o "${TEST_DATASET}"
fi
if [[ ! -f "${LCB_DATASET}" ]]; then
  echo "[INFO] Downloading LiveCodeBench dataset to ${LCB_DATASET}"
  "${PYTHON_BIN}" -c "
from sp_samp.livecodebench import download_livecodebench
download_livecodebench('${LCB_DATASET}', version_tag='${LCB_VERSION}')
print('[INFO] LiveCodeBench download complete.')
"
fi

# --- Preflight checks ---
if [[ "${RUN_CHECKS}" == "1" ]]; then
  echo "[INFO] Running checks before benchmark"
  make check
  make test
fi

# --- GSM8K sweep ---
GSM8K_COMMON_ARGS=(
  --config-dir "${CONFIG_DIR}"
  --eval-task gsm8k
  --gsm8k-eval-mode "${EVAL_MODE}"
  --dataset "${TEST_DATASET}"
  --runs "${RUNS}"
  --max-samples "${MAX_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --k "${SPEC_K}"
  --out "${OUT_GSM8K}"
  --topk-grid "${TOPK_GRID}"
)

echo "[INFO] === GSM8K Sweep ==="

echo "[INFO] Running baseline"
"${PYTHON_BIN}" -m sp_samp.cli bench \
  --experiment "${SPEC_EXPERIMENT}" \
  --method baseline \
  "${GSM8K_COMMON_ARGS[@]}" \
  "${QUANT_ARGS[@]}"

echo "[INFO] Running speculative"
"${PYTHON_BIN}" -m sp_samp.cli bench \
  --experiment "${SPEC_EXPERIMENT}" \
  --method speculative \
  "${GSM8K_COMMON_ARGS[@]}" \
  "${QUANT_ARGS[@]}"

IFS=',' read -r -a threshold_values <<< "${AUTOJUDGE_THRESHOLDS}"
for threshold in "${threshold_values[@]}"; do
  echo "[INFO] Running AutoJudge threshold=${threshold}"
  "${PYTHON_BIN}" -m sp_samp.cli bench \
    --experiment "${AUTOJUDGE_EXPERIMENT}" \
    --method autojudge \
    --autojudge-checkpoint "${CHECKPOINT_PATH}" \
    --autojudge-classifier "${AUTOJUDGE_CLASSIFIER}" \
    --autojudge-train-dataset "${TRAIN_DATASET}" \
    --autojudge-threshold "${threshold}" \
    "${GSM8K_COMMON_ARGS[@]}" \
    "${QUANT_ARGS[@]}"
done

IFS=',' read -r -a topk_values <<< "${TOPK_GRID}"
for rank in "${topk_values[@]}"; do
  echo "[INFO] Running Top-K rank=${rank}"
  "${PYTHON_BIN}" -m sp_samp.cli bench \
    --experiment "${TOPK_EXPERIMENT}" \
    --method topk \
    --topk-rank "${rank}" \
    "${GSM8K_COMMON_ARGS[@]}" \
    "${QUANT_ARGS[@]}"
done

echo "[INFO] Validating GSM8K output schema"
"${PYTHON_BIN}" scripts/validate_results_jsonl.py --path "${OUT_GSM8K}" --strict

# --- LiveCodeBench sweep ---
LCB_COMMON_ARGS=(
  --config-dir "${CONFIG_DIR}"
  --eval-task livecodebench
  --dataset "${LCB_DATASET}"
  --runs "${RUNS}"
  --max-samples "${MAX_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --k "${SPEC_K}"
  --out "${OUT_LCB}"
  --topk-grid "${TOPK_GRID}"
)

echo "[INFO] === LiveCodeBench Sweep ==="

echo "[INFO] Running baseline"
"${PYTHON_BIN}" -m sp_samp.cli bench \
  --experiment "${SPEC_EXPERIMENT}" \
  --method baseline \
  "${LCB_COMMON_ARGS[@]}" \
  "${QUANT_ARGS[@]}"

echo "[INFO] Running speculative"
"${PYTHON_BIN}" -m sp_samp.cli bench \
  --experiment "${SPEC_EXPERIMENT}" \
  --method speculative \
  "${LCB_COMMON_ARGS[@]}" \
  "${QUANT_ARGS[@]}"

for threshold in "${threshold_values[@]}"; do
  echo "[INFO] Running AutoJudge threshold=${threshold}"
  "${PYTHON_BIN}" -m sp_samp.cli bench \
    --experiment "${AUTOJUDGE_EXPERIMENT}" \
    --method autojudge \
    --autojudge-checkpoint "${CHECKPOINT_PATH}" \
    --autojudge-classifier "${AUTOJUDGE_CLASSIFIER}" \
    --autojudge-train-dataset "${TRAIN_DATASET}" \
    --autojudge-threshold "${threshold}" \
    "${LCB_COMMON_ARGS[@]}" \
    "${QUANT_ARGS[@]}"
done

for rank in "${topk_values[@]}"; do
  echo "[INFO] Running Top-K rank=${rank}"
  "${PYTHON_BIN}" -m sp_samp.cli bench \
    --experiment "${TOPK_EXPERIMENT}" \
    --method topk \
    --topk-rank "${rank}" \
    "${LCB_COMMON_ARGS[@]}" \
    "${QUANT_ARGS[@]}"
done

echo "[INFO] Validating LiveCodeBench output schema"
"${PYTHON_BIN}" scripts/validate_results_jsonl.py --path "${OUT_LCB}" --strict

# --- Reports ---
echo "[INFO] Building Yandex-style reports"

"${PYTHON_BIN}" scripts/report_yandex_style.py \
  --input "${OUT_GSM8K}" \
  --eval-task gsm8k \
  --manifest "${MANIFEST_PATH}" \
  --out-prefix "${REPORT_PREFIX}-gsm8k"

"${PYTHON_BIN}" scripts/report_yandex_style.py \
  --input "${OUT_LCB}" \
  --eval-task livecodebench \
  --manifest "${MANIFEST_PATH}" \
  --out-prefix "${REPORT_PREFIX}-livecodebench"

echo "[INFO] Done."
echo "  GSM8K JSONL:        ${OUT_GSM8K}"
echo "  LiveCodeBench JSONL: ${OUT_LCB}"
echo "  GSM8K report:        ${REPORT_PREFIX}-gsm8k.md"
echo "  LCB report:          ${REPORT_PREFIX}-livecodebench.md"
