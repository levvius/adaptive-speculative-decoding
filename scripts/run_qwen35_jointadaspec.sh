#!/usr/bin/env bash
# Staged JointAdaSpec run for the local Qwen3.5 9B -> 2B pair (large parameter gap).
#
# Qwen3.5 checkpoints are vision-language models whose text backbone we use for
# text-only speculative decoding. This run is GATED on two host conditions that are
# NOT met on the prep machine: (1) a working GPU, and (2) a transformers build that
# registers model_type "qwen3_5". The preflight checks both and exits cleanly with an
# actionable message, so this script is SAFE to invoke now (it will not start work).
#
# Pipeline (when the host is ready):
#   01_collect_traces -> 02_solve_mdp -> 04_verify_conditions -> 03_benchmark
# for qwen35_9b_2b_jointadaspec (set EXPERIMENT=qwen35_9b_2b_jointadaspec_conf for the
# draft-confidence arm). Bounded by env vars; NOT a powered final benchmark unless
# MAX_SAMPLES/N_SEEDS are raised to the powered protocol.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATE_TAG="${DATE_TAG:-$(date +%F)}"
EXPERIMENT="${EXPERIMENT:-qwen35_9b_2b_jointadaspec}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-22000}"

# Bounded defaults; override for a powered run.
MAX_TRACES="${MAX_TRACES:-300}"
TEST_START_INDEX="${TEST_START_INDEX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
N_SEEDS="${N_SEEDS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BASELINES="${BASELINES:-[vanilla_ar,fixed_sd,cascade_verif_then_length]}"

OUTPUT_ROOT="outputs/jointadaspec_qwen35_9b_2b_${DATE_TAG}"
REPORT_DIR="reports/qwen35_9b_2b_${DATE_TAG}"
LOG_PATH="${LOG_PATH:-logs/qwen35_jointadaspec_${DATE_TAG}.log}"

mkdir -p "$(dirname "$LOG_PATH")" "$OUTPUT_ROOT" "$REPORT_DIR" reports/manifests
exec > >(tee -a "$LOG_PATH") 2>&1

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

preflight() {
  echo "[preflight] $(date -Is) experiment=${EXPERIMENT}"
  if [ ! -x "$PYTHON_BIN" ] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[preflight] Python not found: ${PYTHON_BIN}. Run 'make setup-gpu'." >&2
    exit 3
  fi
  for path in datasets/gsm8k_train.jsonl datasets/gsm8k_test.jsonl; do
    [ -f "$path" ] || { echo "[preflight] missing dataset ${path}" >&2; exit 3; }
  done
  for d in models/Qwen3.5-9B models/Qwen3.5-2B; do
    [ -d "$d" ] || { echo "[preflight] missing model dir ${d}" >&2; exit 4; }
  done
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "[preflight] nvidia-smi cannot reach the NVIDIA driver. Qwen3.5 eval cannot"
    echo "[preflight] run here; restore the GPU driver and retry."
    exit 3
  fi
  # transformers must register qwen3_5 (Qwen3.5 is unsupported by older builds).
  if ! "$PYTHON_BIN" - <<'PY'
import sys
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
sys.exit(0 if "qwen3_5" in CONFIG_MAPPING_NAMES else 1)
PY
  then
    echo "[preflight] installed transformers does not register model_type 'qwen3_5'."
    echo "[preflight] Upgrade transformers to a build that ships Qwen3.5, then retry."
    echo "[preflight] (Upgrading is a deliberate, user-approved step: it may affect the"
    echo "[preflight]  pinned Qwen2.5/torch-cu128 environment.)"
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

main() {
  preflight

  local trace_dir="${OUTPUT_ROOT}/01_traces"
  local solve_dir="${OUTPUT_ROOT}/02_solve"
  local bench_dir="${OUTPUT_ROOT}/03_bench_gsm8k"
  local policy_path="${solve_dir}/policy.npz"

  echo "[qwen35] $(date -Is) experiment=${EXPERIMENT}"
  "$PYTHON_BIN" scripts/01_collect_traces.py --config-name "experiments/${EXPERIMENT}" \
    "experiments.output_dir=${trace_dir}" \
    "experiments.n_traces=${MAX_TRACES}" \
    "experiments.datasets.train_max_samples=${MAX_TRACES}" \
    "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}"

  "$PYTHON_BIN" scripts/02_solve_mdp.py --config-name "experiments/${EXPERIMENT}" \
    "experiments.output_dir=${solve_dir}" \
    "experiments.traces_path=${trace_dir}/traces.parquet"

  "$PYTHON_BIN" scripts/04_verify_conditions.py \
    --traces "${trace_dir}/traces.parquet" \
    --policy "${policy_path}" \
    --out "${REPORT_DIR}/conditions.json" || true

  "$PYTHON_BIN" scripts/03_benchmark.py --config-name "experiments/${EXPERIMENT}" \
    "experiments.output_dir=${bench_dir}" \
    "experiments.policy_path=${policy_path}" \
    "experiments.datasets.test_start_index=${TEST_START_INDEX}" \
    "experiments.datasets.test_max_samples=${MAX_SAMPLES}" \
    "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}" \
    "experiments.n_seeds=${N_SEEDS}" \
    "experiments.baselines=${BASELINES}"

  "$PYTHON_BIN" scripts/validate_results_jsonl.py --path "${bench_dir}/results.jsonl" --strict

  "$PYTHON_BIN" scripts/analyze_jointadaspec_quality.py \
    --benchmark "${bench_dir}/benchmark.csv" \
    --out "${REPORT_DIR}/quality.md" \
    --primary jointadaspec --baseline target_only \
    --controls target_only speculative cascade_verif_then_length jointadaspec || true

  echo "[qwen35] done. Reports under ${REPORT_DIR}."
  echo "[qwen35] Treat as final ONLY at the powered protocol (>=500 samples x 3 seeds)."
}

main "$@"
