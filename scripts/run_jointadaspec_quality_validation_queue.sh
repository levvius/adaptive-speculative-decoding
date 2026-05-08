#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATE_TAG="${JOINTADA_DATE:-$(date +%F)}"
TOTAL_BUDGET_SECONDS="${TOTAL_BUDGET_SECONDS:-172800}"
SECONDARY_LAUNCH_LIMIT_SECONDS="${SECONDARY_LAUNCH_LIMIT_SECONDS:-129600}"
SECONDARY_TIMEOUT_SECONDS="${SECONDARY_TIMEOUT_SECONDS:-43200}"
MAX_NEW_TOKENS="${JOINTADA_MAX_NEW_TOKENS:-256}"
N_SEEDS="${JOINTADA_N_SEEDS:-3}"
LOG_PATH="${LOG_PATH:-logs/jointadaspec_quality_validation_queue_${DATE_TAG}.log}"

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

QUEUE_STARTED_AT="$(date +%s)"

remaining_seconds() {
  local now elapsed remaining
  now="$(date +%s)"
  elapsed=$((now - QUEUE_STARTED_AT))
  remaining=$((TOTAL_BUDGET_SECONDS - elapsed))
  if [ "$remaining" -lt 0 ]; then
    remaining=0
  fi
  echo "$remaining"
}

bench() {
  local config_name="$1"
  local output_dir="$2"
  local policy_path="$3"
  local start_index="$4"
  local max_samples="$5"
  "$PYTHON_BIN" scripts/03_benchmark.py --config-name "experiments/${config_name}" \
    "experiments.output_dir=${output_dir}" \
    "experiments.policy_path=${policy_path}" \
    "experiments.datasets.test_start_index=${start_index}" \
    "experiments.datasets.test_max_samples=${max_samples}" \
    "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}" \
    "experiments.n_seeds=${N_SEEDS}"
}

build_reports() {
  local tag="$1"
  local bench_dir="$2"
  local policy_path="$3"
  local manifest_path="reports/manifests/$(basename "$(dirname "$bench_dir")")_$(basename "$bench_dir").json"
  "$PYTHON_BIN" reports/templates/pareto_plot.py \
    --input "${bench_dir}/results.jsonl" \
    --out "reports/pareto_${tag}_${DATE_TAG}.pdf" \
    --manifest "$manifest_path"
  "$PYTHON_BIN" reports/templates/threshold_surface.py \
    --policy "$policy_path" \
    --out "reports/threshold_surface_${tag}_${DATE_TAG}" \
    --manifest "$manifest_path"
  "$PYTHON_BIN" reports/templates/ablation_bars.py \
    --input "${bench_dir}/results.jsonl" \
    --out "reports/ablation_${tag}_${DATE_TAG}.pdf" \
    --manifest "$manifest_path"
  "$PYTHON_BIN" scripts/analyze_jointadaspec_quality.py \
    --benchmark "${bench_dir}/benchmark.csv" \
    --out "reports/jointadaspec_quality_${tag}_${DATE_TAG}.md" \
    --primary cascade_verif_then_length \
    --baseline target_only \
    --controls target_only speculative jointadaspec
}

echo "[queue] started at $(date -Is)"
echo "[queue] log: ${LOG_PATH}"

QWEN7_POLICY="outputs/jointadaspec_qwen7b_1p5b_quality_2026-05-05/02_solve/policy.npz"
QWEN7_BENCH_DIR="outputs/jointadaspec_qwen7b_1p5b_quality_heldout_${DATE_TAG}/03_bench_gsm8k"

echo "[queue] primary: Qwen2.5 7B/1.5B quality-aware held-out GSM8K, start=100, samples=500"
bench "qwen25_7b_1p5b_jointadaspec_quality_heldout" "$QWEN7_BENCH_DIR" "$QWEN7_POLICY" 100 500
build_reports "qwen7b_1p5b_quality_heldout" "$QWEN7_BENCH_DIR" "$QWEN7_POLICY"

ELAPSED=$(($(date +%s) - QUEUE_STARTED_AT))
if [ "$ELAPSED" -gt "$SECONDARY_LAUNCH_LIMIT_SECONDS" ]; then
  echo "[queue] primary consumed ${ELAPSED}s; skipping secondary run."
  echo "[queue] finished at $(date -Is)"
  exit 0
fi

QWEN14_POLICY="outputs/jointadaspec_qwen14b_0p5b_2026-04-28/02_solve/policy.npz"
QWEN14_BENCH_DIR="outputs/jointadaspec_qwen14b_0p5b_crosscheck_${DATE_TAG}/03_bench_gsm8k"

if [ -f "$QWEN14_POLICY" ]; then
  SECONDARY_BUDGET="$(remaining_seconds)"
  if [ "$SECONDARY_BUDGET" -gt "$SECONDARY_TIMEOUT_SECONDS" ]; then
    SECONDARY_BUDGET="$SECONDARY_TIMEOUT_SECONDS"
  fi
  echo "[queue] secondary: Qwen2.5 14B/0.5B cross-check, start=100, samples=200, timeout=${SECONDARY_BUDGET}s"
  if timeout --foreground "${SECONDARY_BUDGET}s" "$PYTHON_BIN" scripts/03_benchmark.py --config-name experiments/qwen25_14b_0p5b_jointadaspec_crosscheck \
    "experiments.output_dir=${QWEN14_BENCH_DIR}" \
    "experiments.policy_path=${QWEN14_POLICY}" \
    "experiments.datasets.test_start_index=100" \
    "experiments.datasets.test_max_samples=200" \
    "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}" \
    "experiments.n_seeds=${N_SEEDS}"; then
    build_reports "qwen14b_0p5b_crosscheck" "$QWEN14_BENCH_DIR" "$QWEN14_POLICY"
    echo "[queue] finished at $(date -Is)"
    exit 0
  fi
  echo "[queue] secondary cross-check failed or timed out; considering fallback."
else
  echo "[queue] secondary policy not found: ${QWEN14_POLICY}; considering fallback."
fi

FALLBACK_REMAINING="$(remaining_seconds)"
if [ "$FALLBACK_REMAINING" -lt 14400 ]; then
  echo "[queue] less than 4h left; skipping fallback."
  echo "[queue] finished at $(date -Is)"
  exit 0
fi

QWEN7_EXT_BENCH_DIR="outputs/jointadaspec_qwen7b_1p5b_quality_heldout_extended_${DATE_TAG}/03_bench_gsm8k"
FALLBACK_TIMEOUT=$((FALLBACK_REMAINING - 1800))
echo "[queue] fallback: extend Qwen2.5 7B/1.5B held-out to 800 prompts, timeout=${FALLBACK_TIMEOUT}s"
if timeout --foreground "${FALLBACK_TIMEOUT}s" "$PYTHON_BIN" scripts/03_benchmark.py --config-name experiments/qwen25_7b_1p5b_jointadaspec_quality_heldout \
  "experiments.output_dir=${QWEN7_EXT_BENCH_DIR}" \
  "experiments.policy_path=${QWEN7_POLICY}" \
  "experiments.datasets.test_start_index=100" \
  "experiments.datasets.test_max_samples=800" \
  "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}" \
  "experiments.n_seeds=${N_SEEDS}"; then
  build_reports "qwen7b_1p5b_quality_heldout_extended" "$QWEN7_EXT_BENCH_DIR" "$QWEN7_POLICY"
fi

echo "[queue] finished at $(date -Is)"
