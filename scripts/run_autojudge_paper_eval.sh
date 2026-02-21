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

mkdir -p datasets reports
rm -f "${OUT_RAW}"

echo "[INFO] Writing manifest: ${MANIFEST_PATH}"
"${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

manifest_path = Path(sys.argv[1])
def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return None

torch_version = None
cuda_runtime = None
cuda_available = None
gpu_name = None
try:
    import torch
    torch_version = getattr(torch, "__version__", None)
    cuda_runtime = getattr(torch.version, "cuda", None)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
except Exception:
    pass

payload = {
    "generated_at": datetime.now().isoformat(),
    "platform": platform.platform(),
    "python": run(["python3", "--version"]),
    "kernel": run(["uname", "-a"]),
    "git_sha": run(["git", "rev-parse", "HEAD"]),
    "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"]),
    "torch_version": torch_version,
    "cuda_runtime": cuda_runtime,
    "cuda_available": cuda_available,
    "cuda_device_name": gpu_name,
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

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
