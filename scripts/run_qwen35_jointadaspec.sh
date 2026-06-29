#!/usr/bin/env bash
# Powered JointAdaSpec run for the local Qwen3.5 9B -> 2B pair.
#
# This script keeps the pinned thesis/CI environment untouched by defaulting to
# .venv-qwen35 (transformers 5.x), because qwen3_5 is not registered by the main
# transformers 4.57.x environment. It runs two paired pipelines on the same held-out
# GSM8K window/seeds:
#   1. base 3-D policy: target_only + speculative + cascade + jointadaspec
#   2. draft-confidence policy: jointadaspec only, relabelled jointadaspec_conf
#
# The final report is produced from a merged benchmark CSV so comparisons are
# paired by (seed, prompt_idx). The default protocol is powered but long-running:
# 500 traces, 500 prompts x 3 seeds, 256 new tokens. Smoke outputs are pipeline
# validation only; do not treat them as benchmark evidence.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv-qwen35/bin/python}"
DATE_TAG="${DATE_TAG:-$(date +%F)}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-qwen35_9b_2b_jointadaspec}"
CONF_EXPERIMENT="${CONF_EXPERIMENT:-qwen35_9b_2b_jointadaspec_conf}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-22000}"
REQUIRE_RTX_5090="${REQUIRE_RTX_5090:-1}"
DOWNLOAD_MISSING_MODELS="${DOWNLOAD_MISSING_MODELS:-1}"

# Powered defaults. Override downward only for smoke/debug runs.
MAX_TRACES="${MAX_TRACES:-500}"
TEST_START_INDEX="${TEST_START_INDEX:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
N_SEEDS="${N_SEEDS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
FIXED_SD_GAMMA="${FIXED_SD_GAMMA:-8}"
FUZZY_SD_GAMMA="${FUZZY_SD_GAMMA:-8}"
CONF_GATE_TAU="${CONF_GATE_TAU:-0.0}"
BASELINES="${BASELINES:-[vanilla_ar,fixed_sd,cascade_verif_then_length]}"
TRACE_RESUME="${TRACE_RESUME:-1}"
TRACE_CHECKPOINT_EVERY="${TRACE_CHECKPOINT_EVERY:-1}"
TRACE_PROGRESS_EVERY="${TRACE_PROGRESS_EVERY:-5}"

OUTPUT_ROOT="outputs/jointadaspec_qwen35_9b_2b_${DATE_TAG}"
REPORT_DIR="reports/qwen35_9b_2b_${DATE_TAG}"
LOG_PATH="${LOG_PATH:-logs/qwen35_jointadaspec_${DATE_TAG}.log}"

mkdir -p "$(dirname "$LOG_PATH")" "$OUTPUT_ROOT" "$REPORT_DIR" reports/manifests
exec > >(tee -a "$LOG_PATH") 2>&1

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TRACE_RESUME TRACE_CHECKPOINT_EVERY TRACE_PROGRESS_EVERY

download_model() {
  local repo="$1" dest="$2"
  if [ "${DOWNLOAD_MISSING_MODELS}" != "1" ]; then
    echo "[preflight] ${dest} is missing/incomplete and DOWNLOAD_MISSING_MODELS=0." >&2
    exit 4
  fi
  if ! command -v hf >/dev/null 2>&1; then
    echo "[preflight] hf CLI not found. Install it or download ${repo} to ${dest}." >&2
    exit 4
  fi
  echo "[preflight] repairing/downloading ${repo} -> ${dest}"
  hf download "${repo}" --local-dir "${dest}"
}

verify_or_download_model() {
  local repo="$1" dest="$2"
  if [ ! -d "$dest" ]; then
    download_model "$repo" "$dest"
  fi
  if ! "$PYTHON_BIN" scripts/verify_model_shards.py --model-dir "$dest"; then
    download_model "$repo" "$dest"
    "$PYTHON_BIN" scripts/verify_model_shards.py --model-dir "$dest"
  fi
}

check_python_env() {
  "$PYTHON_BIN" - <<'PY'
import sys
import torch
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

print(f"[preflight] torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"[preflight] transformers={transformers.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in PYTHON_BIN.")
if "qwen3_5" not in CONFIG_MAPPING_NAMES:
    raise SystemExit("transformers does not register qwen3_5.")
PY
}

preflight() {
  echo "[preflight] $(date -Is) base=${BASE_EXPERIMENT} conf=${CONF_EXPERIMENT}"
  if [ ! -x "$PYTHON_BIN" ] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[preflight] Python not found: ${PYTHON_BIN}." >&2
    echo "[preflight] Create/use the isolated Qwen3.5 env, e.g. PYTHON_BIN=.venv-qwen35/bin/python." >&2
    exit 3
  fi
  check_python_env
  for path in datasets/gsm8k_train.jsonl datasets/gsm8k_test.jsonl; do
    [ -f "$path" ] || { echo "[preflight] missing dataset ${path}" >&2; exit 3; }
  done
  verify_or_download_model "Qwen/Qwen3.5-9B" "models/Qwen3.5-9B"
  verify_or_download_model "Qwen/Qwen3.5-2B" "models/Qwen3.5-2B"
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "[preflight] nvidia-smi cannot reach the NVIDIA driver. Qwen3.5 eval cannot"
    echo "[preflight] run here; restore the GPU driver and retry."
    exit 3
  fi
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "[preflight] gpu=${gpu_name}"
  if [ "${REQUIRE_RTX_5090}" = "1" ] && [[ "${gpu_name}" != *"RTX 5090"* ]]; then
    echo "[preflight] expected RTX 5090. Set REQUIRE_RTX_5090=0 to override." >&2
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
  local experiment="$1" subdir="$2" baselines="$3"
  local trace_dir="${OUTPUT_ROOT}/${subdir}/01_traces"
  local solve_dir="${OUTPUT_ROOT}/${subdir}/02_solve"
  local bench_dir="${OUTPUT_ROOT}/${subdir}/03_bench_gsm8k"
  local policy_path="${solve_dir}/policy.npz"

  {
    echo "[pipeline:${subdir}] $(date -Is) experiment=${experiment}"
    if [ -f "${trace_dir}/traces.parquet" ]; then
      echo "[pipeline:${subdir}] skip collect_traces; found ${trace_dir}/traces.parquet"
    else
      "$PYTHON_BIN" scripts/01_collect_traces.py --config-name "experiments/${experiment}" \
        "experiments.output_dir=${trace_dir}" \
        "experiments.n_traces=${MAX_TRACES}" \
        "experiments.datasets.train_max_samples=${MAX_TRACES}" \
        "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}"
    fi

    if [ -f "${policy_path}" ]; then
      echo "[pipeline:${subdir}] skip solve_mdp; found ${policy_path}"
    else
      "$PYTHON_BIN" scripts/02_solve_mdp.py --config-name "experiments/${experiment}" \
        "experiments.output_dir=${solve_dir}" \
        "experiments.traces_path=${trace_dir}/traces.parquet"
    fi

    if [ -f "${REPORT_DIR}/conditions_${subdir}.json" ]; then
      echo "[pipeline:${subdir}] skip verify_conditions; found ${REPORT_DIR}/conditions_${subdir}.json"
    else
      "$PYTHON_BIN" scripts/04_verify_conditions.py \
        --traces "${trace_dir}/traces.parquet" \
        --policy "${policy_path}" \
        --out "${REPORT_DIR}/conditions_${subdir}.json" || true
    fi

    "$PYTHON_BIN" scripts/03_benchmark.py --config-name "experiments/${experiment}" \
      "experiments.output_dir=${bench_dir}" \
      "experiments.policy_path=${policy_path}" \
      "experiments.datasets.test_start_index=${TEST_START_INDEX}" \
      "experiments.datasets.test_max_samples=${MAX_SAMPLES}" \
      "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}" \
      "experiments.n_seeds=${N_SEEDS}" \
      "experiments.fixed_sd_gamma=${FIXED_SD_GAMMA}" \
      "experiments.fuzzy_sd_gamma=${FUZZY_SD_GAMMA}" \
      "experiments.baselines=${baselines}"

    "$PYTHON_BIN" scripts/validate_results_jsonl.py --path "${bench_dir}/results.jsonl" --strict
  } >&2
  echo "${bench_dir}/benchmark.csv"
}

run_gate_bench() {
  local solve_dir="${OUTPUT_ROOT}/conf/02_solve"
  local bench_dir="${OUTPUT_ROOT}/conf_gate/03_bench_gsm8k"

  {
    "$PYTHON_BIN" scripts/03_benchmark.py --config-name "experiments/${CONF_EXPERIMENT}" \
      "experiments.output_dir=${bench_dir}" \
      "experiments.policy_path=${solve_dir}/policy.npz" \
      "experiments.conf_gate_tau=${CONF_GATE_TAU}" \
      "experiments.datasets.test_start_index=${TEST_START_INDEX}" \
      "experiments.datasets.test_max_samples=${MAX_SAMPLES}" \
      "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}" \
      "experiments.n_seeds=${N_SEEDS}" \
      "experiments.fixed_sd_gamma=${FIXED_SD_GAMMA}" \
      "experiments.fuzzy_sd_gamma=${FUZZY_SD_GAMMA}" \
      "experiments.baselines=[]"

    "$PYTHON_BIN" scripts/validate_results_jsonl.py --path "${bench_dir}/results.jsonl" --strict
  } >&2
  echo "${bench_dir}/benchmark.csv"
}

main() {
  preflight

  echo "[qwen35] $(date -Is) powered paired run"
  echo "[qwen35] traces=${MAX_TRACES} prompts=${MAX_SAMPLES} seeds=${N_SEEDS} max_new_tokens=${MAX_NEW_TOKENS}"
  echo "[qwen35] fixed_sd_gamma=${FIXED_SD_GAMMA} fuzzy_sd_gamma=${FUZZY_SD_GAMMA} conf_gate_tau=${CONF_GATE_TAU}"
  echo "[qwen35] trace_resume=${TRACE_RESUME} checkpoint_every=${TRACE_CHECKPOINT_EVERY} progress_every=${TRACE_PROGRESS_EVERY}"

  local base_csv conf_csv merged_csv gate_csv
  base_csv="$(run_pipeline "${BASE_EXPERIMENT}" base "${BASELINES}" | tail -1)"
  conf_csv="$(run_pipeline "${CONF_EXPERIMENT}" conf "[]" | tail -1)"
  merged_csv="${REPORT_DIR}/merged_benchmark.csv"

  local variants=("--variant" "${conf_csv}:jointadaspec:jointadaspec_conf")
  local controls=(target_only speculative cascade_verif_then_length jointadaspec jointadaspec_conf)

  if "$PYTHON_BIN" - <<PY
raise SystemExit(0 if float("${CONF_GATE_TAU}") > 0.0 else 1)
PY
  then
    gate_csv="$(run_gate_bench | tail -1)"
    variants+=("--variant" "${gate_csv}:jointadaspec:jointadaspec_conf_gate")
    controls+=(jointadaspec_conf_gate)
  fi

  "$PYTHON_BIN" scripts/merge_benchmark_runs.py \
    --base "${base_csv}" \
    "${variants[@]}" \
    --out "${merged_csv}"

  "$PYTHON_BIN" scripts/analyze_jointadaspec_quality.py \
    --benchmark "${merged_csv}" \
    --out "${REPORT_DIR}/quality.md" \
    --primary jointadaspec_conf \
    --baseline target_only \
    --controls "${controls[@]}"

  echo "[qwen35] done. Reports under ${REPORT_DIR}."
  echo "[qwen35] Final interpretation requires strict validation plus paired/clustered analysis."
}

main "$@"
