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
OUT_RAW="${OUT_RAW:-datasets/results_autojudge_qwen25_paper_${DATE_TAG}.jsonl}"
REPORT_PREFIX="${REPORT_PREFIX:-reports/autojudge_qwen25_paper_${DATE_TAG}}"
MANIFEST_PATH="${MANIFEST_PATH:-reports/autojudge_run_manifest_${DATE_TAG}.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-datasets/autojudge_qwen25_0p5b_to_7b.pt}"

MAX_SAMPLES="${MAX_SAMPLES:-300}"
RUNS="${RUNS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SPEC_K="${SPEC_K:-4}"
AUTOJUDGE_THRESHOLDS="${AUTOJUDGE_THRESHOLDS:-0.005,0.01,0.03,0.05,0.09,0.14,0.23,1.0}"
TOPK_GRID="${TOPK_GRID:-2,4,8,16,32,all}"
EVAL_MODE="${EVAL_MODE:-zero_shot_cot}"
RUN_CHECKS="${RUN_CHECKS:-1}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

export HF_HUB_DISABLE_XET

mkdir -p datasets reports
rm -f "${OUT_RAW}"

echo "[INFO] Writing manifest: ${MANIFEST_PATH}"
"${PYTHON_BIN}" scripts/write_run_manifest.py --out "${MANIFEST_PATH}"

if [[ ! -f "${TRAIN_DATASET}" ]]; then
  echo "[INFO] Downloading GSM8K train split to ${TRAIN_DATASET}"
  curl -fsSL "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl" -o "${TRAIN_DATASET}"
fi
if [[ ! -f "${TEST_DATASET}" ]]; then
  echo "[INFO] Downloading GSM8K test split to ${TEST_DATASET}"
  curl -fsSL "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl" -o "${TEST_DATASET}"
fi

if [[ "${RUN_CHECKS}" == "1" ]]; then
  echo "[INFO] Running checks before benchmark"
  make check
  make test
fi

COMMON_ARGS=(
  --config-dir "${CONFIG_DIR}"
  --eval-task gsm8k
  --gsm8k-eval-mode "${EVAL_MODE}"
  --dataset "${TEST_DATASET}"
  --runs "${RUNS}"
  --max-samples "${MAX_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --k "${SPEC_K}"
  --out "${OUT_RAW}"
  --topk-grid "${TOPK_GRID}"
)

echo "[INFO] Running baseline and speculative references"
"${PYTHON_BIN}" -m sp_samp.cli bench \
  --experiment qwen25_7b_target_qwen25_0p5b_speculative_k4 \
  --method baseline \
  "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" -m sp_samp.cli bench \
  --experiment qwen25_7b_target_qwen25_0p5b_speculative_k4 \
  --method speculative \
  "${COMMON_ARGS[@]}"

IFS=',' read -r -a threshold_values <<< "${AUTOJUDGE_THRESHOLDS}"
for threshold in "${threshold_values[@]}"; do
  echo "[INFO] Running AutoJudge threshold=${threshold}"
  "${PYTHON_BIN}" -m sp_samp.cli bench \
    --experiment qwen25_7b_target_qwen25_0p5b_autojudge_k4 \
    --method autojudge \
    --autojudge-checkpoint "${CHECKPOINT_PATH}" \
    --autojudge-train-dataset "${TRAIN_DATASET}" \
    --autojudge-threshold "${threshold}" \
    "${COMMON_ARGS[@]}"
done

IFS=',' read -r -a topk_values <<< "${TOPK_GRID}"
for rank in "${topk_values[@]}"; do
  echo "[INFO] Running Top-K rank=${rank}"
  "${PYTHON_BIN}" -m sp_samp.cli bench \
    --experiment qwen25_7b_target_qwen25_0p5b_topk_k4 \
    --method topk \
    --topk-rank "${rank}" \
    "${COMMON_ARGS[@]}"
done

echo "[INFO] Validating raw output schema"
"${PYTHON_BIN}" scripts/validate_results_jsonl.py --path "${OUT_RAW}" --strict

echo "[INFO] Building report artifacts"
"${PYTHON_BIN}" scripts/report_autojudge_paper.py \
  --input "${OUT_RAW}" \
  --manifest "${MANIFEST_PATH}" \
  --out-prefix "${REPORT_PREFIX}"

echo "[INFO] Done."
echo "  Raw JSONL:   ${OUT_RAW}"
echo "  Report MD:   ${REPORT_PREFIX}.md"
echo "  Report CSV:  ${REPORT_PREFIX}.csv"
echo "  Report JSON: ${REPORT_PREFIX}.json"
