#!/usr/bin/env bash
# Staged evaluation of the draft-confidence improvement (block_verify_v1 + N_C axis).
#
# For each model pair this runs TWO self-consistent pipelines on the SAME held-out
# GSM8K window/seeds so the rows are paired by (seed, prompt_idx):
#   1. 3-D baseline policy  (qwen25_*_jointadaspec)        -> jointadaspec
#   2. draft-confidence policy (qwen25_*_jointadaspec_conf) -> jointadaspec_conf
# It then merges the runs and reports prompt-clustered paired statistics for
# jointadaspec_conf vs {target_only, speculative, cascade_verif_then_length,
# jointadaspec (3-D)}. An optional gate arm re-benchmarks the conf policy with the
# inference-time early-verify gate (DEFENSE_CONF_GATE_TAU > 0).
#
# This script is GPU-gated and SAFE to run when no GPU is present: the preflight
# detects a missing/!responding driver and exits cleanly without starting work.
# It is NOT a powered final benchmark unless DEFENSE_MAX_SAMPLES/N_SEEDS are set
# to the powered protocol (e.g. 500 samples x 3 seeds) on a real GPU.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATE_TAG="${DATE_TAG:-$(date +%F)}"
MODEL_PAIR="${MODEL_PAIR:-qwen14b_0p5b}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-22000}"

# Bounded by default; override for a powered run.
MAX_TRACES="${MAX_TRACES:-300}"
TEST_START_INDEX="${TEST_START_INDEX:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
N_SEEDS="${N_SEEDS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
CONF_GATE_TAU="${CONF_GATE_TAU:-0.0}"   # >0 adds a gated arm: jointadaspec_conf_gate
BASELINES="${BASELINES:-[vanilla_ar,fixed_sd,cascade_verif_then_length]}"

OUTPUT_ROOT="outputs/jointadaspec_${MODEL_PAIR}_improvement_${DATE_TAG}"
REPORT_DIR="reports/improvement_${MODEL_PAIR}_${DATE_TAG}"
LOG_PATH="${LOG_PATH:-logs/improvement_eval_${MODEL_PAIR}_${DATE_TAG}.log}"

mkdir -p "$(dirname "$LOG_PATH")" "$OUTPUT_ROOT" "$REPORT_DIR" reports/manifests
exec > >(tee -a "$LOG_PATH") 2>&1

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

experiments_for_pair() {
  # echoes "<base_config> <conf_config>"
  case "$1" in
    qwen14b_0p5b) echo "qwen25_14b_0p5b_jointadaspec qwen25_14b_0p5b_jointadaspec_conf" ;;
    qwen7b_1p5b)  echo "qwen25_7b_1p5b_jointadaspec qwen25_7b_1p5b_jointadaspec_conf" ;;
    *) echo "[config] unsupported MODEL_PAIR '$1'" >&2; return 2 ;;
  esac
}

preflight() {
  echo "[preflight] $(date -Is) pair=${MODEL_PAIR}"
  if [ ! -x "$PYTHON_BIN" ] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[preflight] Python not found: ${PYTHON_BIN}. Run 'make setup-gpu'." >&2
    exit 3
  fi
  for path in datasets/gsm8k_train.jsonl datasets/gsm8k_test.jsonl; do
    [ -f "$path" ] || { echo "[preflight] missing dataset ${path}" >&2; exit 3; }
  done
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "[preflight] nvidia-smi cannot reach the NVIDIA driver. GPU eval cannot run"
    echo "[preflight] on this machine. CPU tests + the merge/analysis helpers remain"
    echo "[preflight] runnable; rerun here once the GPU driver is restored."
    exit 3
  fi
  local active free_mib
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' || true)"
  if [ -n "$active" ] && [ "${ALLOW_ACTIVE_GPU:-0}" != "1" ]; then
    echo "[preflight] active GPU compute PIDs: ${active}. Stop them or set ALLOW_ACTIVE_GPU=1." >&2
    exit 3
  fi
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  echo "[preflight] free VRAM: ${free_mib:-unknown} MiB (need ${MIN_FREE_VRAM_MIB})"
  if [ -z "$free_mib" ] || [ "$free_mib" -lt "$MIN_FREE_VRAM_MIB" ]; then
    echo "[preflight] insufficient free VRAM." >&2
    exit 3
  fi
}

run_pipeline() {
  # args: <config> <out_subdir>
  local config="$1" sub="$2"
  local trace_dir="${OUTPUT_ROOT}/${sub}/01_traces"
  local solve_dir="${OUTPUT_ROOT}/${sub}/02_solve"
  local bench_dir="${OUTPUT_ROOT}/${sub}/03_bench_gsm8k"
  local policy_path="${solve_dir}/policy.npz"

  echo "[pipeline:${sub}] config=${config}"
  "$PYTHON_BIN" scripts/01_collect_traces.py --config-name "experiments/${config}" \
    "experiments.output_dir=${trace_dir}" \
    "experiments.n_traces=${MAX_TRACES}" \
    "experiments.datasets.train_max_samples=${MAX_TRACES}" \
    "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}"

  "$PYTHON_BIN" scripts/02_solve_mdp.py --config-name "experiments/${config}" \
    "experiments.output_dir=${solve_dir}" \
    "experiments.traces_path=${trace_dir}/traces.parquet"

  "$PYTHON_BIN" scripts/04_verify_conditions.py \
    --traces "${trace_dir}/traces.parquet" \
    --policy "${policy_path}" \
    --out "${REPORT_DIR}/conditions_${sub}.json" || true

  "$PYTHON_BIN" scripts/03_benchmark.py --config-name "experiments/${config}" \
    "experiments.output_dir=${bench_dir}" \
    "experiments.policy_path=${policy_path}" \
    "experiments.datasets.test_start_index=${TEST_START_INDEX}" \
    "experiments.datasets.test_max_samples=${MAX_SAMPLES}" \
    "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}" \
    "experiments.n_seeds=${N_SEEDS}" \
    "experiments.baselines=${BASELINES}"

  "$PYTHON_BIN" scripts/validate_results_jsonl.py --path "${bench_dir}/results.jsonl" --strict
  echo "${bench_dir}/benchmark.csv"
}

run_gate_bench() {
  # Re-benchmark the already-solved conf policy with the inference-time gate on.
  local config="$1"
  local solve_dir="${OUTPUT_ROOT}/conf/02_solve"
  local bench_dir="${OUTPUT_ROOT}/conf_gate/03_bench_gsm8k"
  "$PYTHON_BIN" scripts/03_benchmark.py --config-name "experiments/${config}" \
    "experiments.output_dir=${bench_dir}" \
    "experiments.policy_path=${solve_dir}/policy.npz" \
    "experiments.conf_gate_tau=${CONF_GATE_TAU}" \
    "experiments.datasets.test_start_index=${TEST_START_INDEX}" \
    "experiments.datasets.test_max_samples=${MAX_SAMPLES}" \
    "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}" \
    "experiments.n_seeds=${N_SEEDS}" \
    "experiments.baselines=[vanilla_ar]"
  echo "${bench_dir}/benchmark.csv"
}

main() {
  read -r BASE_CONFIG CONF_CONFIG < <(experiments_for_pair "$MODEL_PAIR")
  preflight

  echo "[improvement-eval] $(date -Is) base=${BASE_CONFIG} conf=${CONF_CONFIG}"
  BASE_CSV="$(run_pipeline "$BASE_CONFIG" base | tail -1)"
  CONF_CSV="$(run_pipeline "$CONF_CONFIG" conf | tail -1)"

  MERGED="${REPORT_DIR}/merged_benchmark.csv"
  VARIANTS=(--variant "${CONF_CSV}:jointadaspec:jointadaspec_conf")
  CONTROLS=(target_only speculative cascade_verif_then_length jointadaspec)

  if [ "$(awk "BEGIN{print ($CONF_GATE_TAU > 0.0)}")" = "1" ]; then
    GATE_CSV="$(run_gate_bench "$CONF_CONFIG" | tail -1)"
    VARIANTS+=(--variant "${GATE_CSV}:jointadaspec:jointadaspec_conf_gate")
    CONTROLS+=(jointadaspec_conf_gate)
  fi

  "$PYTHON_BIN" scripts/merge_benchmark_runs.py --base "$BASE_CSV" "${VARIANTS[@]}" --out "$MERGED"

  "$PYTHON_BIN" scripts/analyze_jointadaspec_quality.py \
    --benchmark "$MERGED" \
    --out "${REPORT_DIR}/improvement_quality.md" \
    --primary jointadaspec_conf \
    --baseline target_only \
    --controls "${CONTROLS[@]}"

  echo "[improvement-eval] done. Report: ${REPORT_DIR}/improvement_quality.md"
  echo "[improvement-eval] Treat as final ONLY at the powered protocol (>=500 samples x 3 seeds)."
}

main "$@"
