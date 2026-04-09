#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

CONFIG_DIR="${CONFIG_DIR:-configs}"
TRAIN_DATASET="${TRAIN_DATASET:-datasets/gsm8k_train.jsonl}"
TEST_DATASET="${TEST_DATASET:-datasets/gsm8k_test.jsonl}"
OUT="${OUT:-datasets/results_consensus_reduced_gsm8k.jsonl}"
REPORT_PREFIX="${REPORT_PREFIX:-reports/consensus_reduced_gsm8k}"

AUTOJUDGE_EXPERIMENT="${AUTOJUDGE_EXPERIMENT:-qwen25_3b_target_qwen25_0p5b_autojudge_k4}"
CONSENSUS_EXPERIMENT="${CONSENSUS_EXPERIMENT:-qwen25_3b_target_qwen25_0p5b_1p5b_consensus_autojudge_k4}"

AUTOJUDGE_CHECKPOINT="${AUTOJUDGE_CHECKPOINT:-datasets/autojudge_qwen25_0p5b_to_3b_reduced.pt}"
CONSENSUS_CHECKPOINT="${CONSENSUS_CHECKPOINT:-datasets/consensus_qwen25_0p5b_1p5b_to_3b_ensemble_reduced.pt}"
D1ONLY_CHECKPOINT="${D1ONLY_CHECKPOINT:-datasets/consensus_qwen25_0p5b_1p5b_to_3b_d1only_reduced.pt}"

MAX_SAMPLES="${MAX_SAMPLES:-20}"
RUNS="${RUNS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-256}"
EVAL_MODE="${EVAL_MODE:-plain}"
RESET_OUT="${RESET_OUT:-0}"

mkdir -p datasets reports
if [[ "${RESET_OUT}" == "1" ]]; then
  rm -f "${OUT}"
fi

COMMON_ARGS=(
  --config-dir "${CONFIG_DIR}"
  --eval-task gsm8k
  --gsm8k-eval-mode "${EVAL_MODE}"
  --dataset "${TEST_DATASET}"
  --max-samples "${MAX_SAMPLES}"
  --runs "${RUNS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --out "${OUT}"
)

echo "[INFO] Running reduced AutoJudge baseline"
"${PYTHON_BIN}" -m sp_samp.cli bench \
  --experiment "${AUTOJUDGE_EXPERIMENT}" \
  --method autojudge \
  --autojudge-train-dataset "${TRAIN_DATASET}" \
  --autojudge-train-samples "${TRAIN_SAMPLES}" \
  --autojudge-checkpoint "${AUTOJUDGE_CHECKPOINT}" \
  "${COMMON_ARGS[@]}"

echo "[INFO] Running consensus rule baseline"
"${PYTHON_BIN}" -m sp_samp.cli bench \
  --experiment "${CONSENSUS_EXPERIMENT}" \
  --method consensus_autojudge \
  --consensus-gate rule \
  "${COMMON_ARGS[@]}"

echo "[INFO] Running learned consensus without escalation"
"${PYTHON_BIN}" -m sp_samp.cli bench \
  --experiment "${CONSENSUS_EXPERIMENT}" \
  --method consensus_autojudge \
  --consensus-gate learned \
  --consensus-disable-escalation \
  --consensus-train-dataset "${TRAIN_DATASET}" \
  --consensus-train-samples "${TRAIN_SAMPLES}" \
  --consensus-checkpoint "${CONSENSUS_CHECKPOINT}" \
  "${COMMON_ARGS[@]}"

for threshold in 0.5 0.65 0.8; do
  echo "[INFO] Running learned consensus threshold=${threshold}"
  "${PYTHON_BIN}" -m sp_samp.cli bench \
    --experiment "${CONSENSUS_EXPERIMENT}" \
    --method consensus_autojudge \
    --consensus-gate learned \
    --consensus-fallback-threshold "${threshold}" \
    --consensus-train-dataset "${TRAIN_DATASET}" \
    --consensus-train-samples "${TRAIN_SAMPLES}" \
    --consensus-checkpoint "${CONSENSUS_CHECKPOINT}" \
    "${COMMON_ARGS[@]}"
done

echo "[INFO] Running learned consensus with D1-only features"
"${PYTHON_BIN}" -m sp_samp.cli bench \
  --experiment "${CONSENSUS_EXPERIMENT}" \
  --method consensus_autojudge \
  --consensus-gate learned \
  --consensus-features d1_only \
  --consensus-train-dataset "${TRAIN_DATASET}" \
  --consensus-train-samples "${TRAIN_SAMPLES}" \
  --consensus-checkpoint "${D1ONLY_CHECKPOINT}" \
  "${COMMON_ARGS[@]}"

echo "[INFO] Validating output schema"
"${PYTHON_BIN}" scripts/validate_results_jsonl.py --path "${OUT}" --strict

echo "[INFO] Writing reduced report"
"${PYTHON_BIN}" scripts/report_yandex_style.py \
  --input "${OUT}" \
  --eval-task gsm8k \
  --out-prefix "${REPORT_PREFIX}"

echo "[INFO] Done."
echo "  JSONL:  ${OUT}"
echo "  Report: ${REPORT_PREFIX}.md"
