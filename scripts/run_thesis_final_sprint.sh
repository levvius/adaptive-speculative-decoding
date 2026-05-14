#!/usr/bin/env bash
# Sequenced JointAdaSpec final-sprint benchmark queue.
#
# Run 1: Qwen 14B/0.5B lock-in (re-bench anchor at 500 prompts x 3 seeds).
# Run 2: Qwen 7B/1.5B quality lock (re-bench borderline at 500 prompts x 3 seeds).
# Run 2 starts automatically after Run 1 finishes (success or failure — the two
# pairs are independent).
#
# Usage (in a fresh tmux session so it survives the SSH disconnect):
#   tmux new -s aj_sprint
#   bash scripts/run_thesis_final_sprint.sh
#
# Detach with Ctrl-b d. Reattach: tmux attach -t aj_sprint.
# Status file: logs/thesis_final_sprint_${DATE_TAG}.status

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATE_TAG="${SPRINT_DATE_TAG:-$(date +%F)}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/thesis_final_sprint_${DATE_TAG}.log}"
STATUS_PATH="${STATUS_PATH:-${LOG_DIR}/thesis_final_sprint_${DATE_TAG}.status}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
N_SEEDS="${N_SEEDS:-3}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-22000}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

write_status() {
  local phase="$1" state="$2" extra="${3:-}"
  printf '{"date":"%s","phase":"%s","state":"%s","timestamp":"%s","extra":"%s"}\n' \
    "$DATE_TAG" "$phase" "$state" "$(date -Is)" "$extra" > "$STATUS_PATH"
}

preflight() {
  if [ "$SKIP_PREFLIGHT" = "1" ]; then
    echo "[preflight] skipped via SKIP_PREFLIGHT=1"
    return 0
  fi
  echo "[preflight] checking GPU state..."
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[preflight] nvidia-smi not found; aborting." >&2
    return 1
  fi
  local active_pids
  active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' || true)"
  if [ -n "$active_pids" ]; then
    echo "[preflight] active compute PIDs on GPU: $active_pids"
    echo "[preflight] refusing to launch; stop these first or set SKIP_PREFLIGHT=1." >&2
    return 1
  fi
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  echo "[preflight] free VRAM: ${free_mib} MiB (need ${MIN_FREE_VRAM_MIB})"
  if [ -z "$free_mib" ] || [ "$free_mib" -lt "$MIN_FREE_VRAM_MIB" ]; then
    echo "[preflight] insufficient free VRAM; aborting." >&2
    return 1
  fi
  echo "[preflight] OK"
}

run_benchmark() {
  local config_name="$1" output_dir="$2"
  "$PYTHON_BIN" scripts/03_benchmark.py \
    --config-name "experiments/${config_name}" \
    "experiments.output_dir=${output_dir}" \
    "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}" \
    "experiments.n_seeds=${N_SEEDS}"
}

build_reports() {
  local tag="$1" bench_dir="$2" policy_path="$3"
  local results_jsonl="${bench_dir}/results.jsonl"
  local bench_csv="${bench_dir}/benchmark.csv"
  if [ ! -f "$results_jsonl" ] && [ ! -f "$bench_csv" ]; then
    echo "[reports] no benchmark artifacts found under ${bench_dir}; skipping."
    return 0
  fi
  local manifest_path="reports/manifests/$(basename "$(dirname "$bench_dir")")_$(basename "$bench_dir").json"
  if [ -f "$results_jsonl" ]; then
    "$PYTHON_BIN" reports/templates/pareto_plot.py \
      --input "$results_jsonl" \
      --out "reports/pareto_${tag}_${DATE_TAG}.pdf" \
      --manifest "$manifest_path" || echo "[reports] pareto_plot failed for ${tag}"
    "$PYTHON_BIN" reports/templates/ablation_bars.py \
      --input "$results_jsonl" \
      --out "reports/ablation_${tag}_${DATE_TAG}.pdf" \
      --manifest "$manifest_path" || echo "[reports] ablation_bars failed for ${tag}"
  fi
  if [ -f "$policy_path" ]; then
    "$PYTHON_BIN" reports/templates/threshold_surface.py \
      --policy "$policy_path" \
      --out "reports/threshold_surface_${tag}_${DATE_TAG}" \
      --manifest "$manifest_path" || echo "[reports] threshold_surface failed for ${tag}"
  fi
  if [ -f "$bench_csv" ]; then
    "$PYTHON_BIN" scripts/analyze_jointadaspec_quality.py \
      --benchmark "$bench_csv" \
      --out "reports/jointadaspec_quality_${tag}_${DATE_TAG}.md" \
      --primary cascade_verif_then_length \
      --baseline target_only \
      --controls target_only speculative jointadaspec || echo "[reports] analyze script failed for ${tag}"
  fi
}

verify_results() {
  local bench_dir="$1"
  local results_jsonl="${bench_dir}/results.jsonl"
  if [ ! -f "$results_jsonl" ]; then
    echo "[verify] missing ${results_jsonl}"
    return 1
  fi
  "$PYTHON_BIN" scripts/validate_results_jsonl.py --path "$results_jsonl" --strict
}

regenerate_notebook() {
  echo "[notebook] regenerating thesis_plots.ipynb (paths in generate_thesis_notebook.py
                may need manual update to point at new outputs/jointadaspec_*_lock_${DATE_TAG} dirs)"
  "$PYTHON_BIN" notebooks/generate_thesis_notebook.py || echo "[notebook] generation failed"
  "$PYTHON_BIN" -m jupyter nbconvert --to notebook --execute \
    notebooks/thesis_plots.ipynb --output thesis_plots.ipynb \
    2>&1 | tail -20 || echo "[notebook] execution failed (paths likely need update)"
}

echo "[sprint] start: $(date -Is)"
echo "[sprint] log:    ${LOG_PATH}"
echo "[sprint] status: ${STATUS_PATH}"
write_status "init" "starting"

if ! preflight; then
  write_status "preflight" "failed"
  exit 1
fi
write_status "preflight" "ok"

#######################################
# Run 1 — Qwen 14B/0.5B lock-in (anchor)
#######################################
RUN1_TAG="qwen14b_0p5b_lock"
RUN1_CONFIG="qwen25_14b_0p5b_jointadaspec_lock"
RUN1_POLICY="outputs/jointadaspec_qwen14b_0p5b_2026-04-28/02_solve/policy.npz"
RUN1_OUT_DIR="outputs/jointadaspec_qwen14b_0p5b_lock_${DATE_TAG}/03_bench_gsm8k"

if [ ! -f "$RUN1_POLICY" ]; then
  echo "[run1] anchor policy missing: ${RUN1_POLICY}" >&2
  write_status "run1" "skipped_missing_policy"
else
  echo "[run1] config=${RUN1_CONFIG} output=${RUN1_OUT_DIR}"
  write_status "run1" "running"
  RUN1_OK=0
  if run_benchmark "$RUN1_CONFIG" "$RUN1_OUT_DIR"; then
    RUN1_OK=1
  fi
  if [ "$RUN1_OK" = "1" ]; then
    if verify_results "$RUN1_OUT_DIR"; then
      build_reports "$RUN1_TAG" "$RUN1_OUT_DIR" "$RUN1_POLICY"
      write_status "run1" "succeeded"
      echo "[run1] succeeded at $(date -Is)"
    else
      write_status "run1" "validation_failed"
      echo "[run1] validation failed"
    fi
  else
    write_status "run1" "benchmark_failed"
    echo "[run1] benchmark stage failed"
  fi
fi

#######################################
# Run 2 — Qwen 7B/1.5B quality lock-in (secondary)
# Runs regardless of Run 1 outcome — the model pairs are independent.
#######################################
RUN2_TAG="qwen7b_1p5b_quality_lock"
RUN2_CONFIG="qwen25_7b_1p5b_jointadaspec_quality_lock"
RUN2_POLICY="outputs/jointadaspec_qwen7b_1p5b_quality_2026-05-05/02_solve/policy.npz"
RUN2_OUT_DIR="outputs/jointadaspec_qwen7b_1p5b_quality_lock_${DATE_TAG}/03_bench_gsm8k"

if [ ! -f "$RUN2_POLICY" ]; then
  echo "[run2] policy missing: ${RUN2_POLICY}" >&2
  write_status "run2" "skipped_missing_policy"
else
  echo "[run2] config=${RUN2_CONFIG} output=${RUN2_OUT_DIR}"
  write_status "run2" "running"
  RUN2_OK=0
  if run_benchmark "$RUN2_CONFIG" "$RUN2_OUT_DIR"; then
    RUN2_OK=1
  fi
  if [ "$RUN2_OK" = "1" ]; then
    if verify_results "$RUN2_OUT_DIR"; then
      build_reports "$RUN2_TAG" "$RUN2_OUT_DIR" "$RUN2_POLICY"
      write_status "run2" "succeeded"
      echo "[run2] succeeded at $(date -Is)"
    else
      write_status "run2" "validation_failed"
      echo "[run2] validation failed"
    fi
  else
    write_status "run2" "benchmark_failed"
    echo "[run2] benchmark stage failed"
  fi
fi

#######################################
# Wrap-up — regenerate thesis notebook with new data and finalise status.
#######################################
regenerate_notebook
write_status "wrap_up" "done"
echo "[sprint] finished at $(date -Is)"
