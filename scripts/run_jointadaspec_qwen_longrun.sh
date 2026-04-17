#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-experiments/qwen25_7b_1p5b_jointadaspec}"
LCB_CONFIG_NAME="${LCB_CONFIG_NAME:-experiments/qwen25_7b_1p5b_jointadaspec_livecodebench}"
DATE_TAG="${DATE_TAG:-$(date +%F)}"
ROOT_DIR="${ROOT_DIR:-outputs/jointadaspec_qwen_${DATE_TAG}}"
TRACE_DIR="${TRACE_DIR:-${ROOT_DIR}/01_traces_gsm8k}"
SOLVE_DIR="${SOLVE_DIR:-${ROOT_DIR}/02_solve}"
BENCH_GSM8K_DIR="${BENCH_GSM8K_DIR:-${ROOT_DIR}/03_bench_gsm8k}"
BENCH_LCB_DIR="${BENCH_LCB_DIR:-${ROOT_DIR}/04_bench_livecodebench}"
SEED="${SEED:-0}"
N_TRACES="${N_TRACES:-400}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-100}"
GSM8K_TEST_MAX_SAMPLES="${GSM8K_TEST_MAX_SAMPLES:-100}"
LCB_TEST_MAX_SAMPLES="${LCB_TEST_MAX_SAMPLES:-50}"
KAPPA_VALUES="${KAPPA_VALUES:-[0.0,0.5,1.0,2.0,5.0]}"

mkdir -p "${ROOT_DIR}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

${PYTHON_BIN} scripts/01_collect_traces.py \
  --config-name "${CONFIG_NAME}" \
  experiments.output_dir="${TRACE_DIR}" \
  experiments.seed="${SEED}" \
  experiments.n_traces="${N_TRACES}" \
  experiments.datasets.train_max_samples="${TRAIN_MAX_SAMPLES}"

TRACE_PATH="${TRACE_DIR}/traces.parquet"

${PYTHON_BIN} scripts/02_solve_mdp.py \
  --config-name "${CONFIG_NAME}" \
  experiments.output_dir="${SOLVE_DIR}" \
  experiments.seed="${SEED}" \
  experiments.traces_path="${TRACE_PATH}" \
  +experiments.kappa_values="${KAPPA_VALUES}"

POLICY_PATH="${SOLVE_DIR}/policy_kappa_1.npz"
if [[ ! -f "${POLICY_PATH}" ]]; then
  POLICY_PATH="${SOLVE_DIR}/policy.npz"
fi

${PYTHON_BIN} scripts/03_benchmark.py \
  --config-name "${CONFIG_NAME}" \
  experiments.output_dir="${BENCH_GSM8K_DIR}" \
  experiments.seed="${SEED}" \
  experiments.policy_path="${POLICY_PATH}" \
  experiments.datasets.test_max_samples="${GSM8K_TEST_MAX_SAMPLES}"

${PYTHON_BIN} scripts/03_benchmark.py \
  --config-name "${LCB_CONFIG_NAME}" \
  experiments.output_dir="${BENCH_LCB_DIR}" \
  experiments.seed="${SEED}" \
  experiments.policy_path="${POLICY_PATH}" \
  experiments.datasets.test_max_samples="${LCB_TEST_MAX_SAMPLES}"

printf 'JointAdaSpec long-run outputs:\n'
printf '  traces: %s\n' "${TRACE_PATH}"
printf '  solve:  %s\n' "${SOLVE_DIR}"
printf '  gsm8k:  %s\n' "${BENCH_GSM8K_DIR}"
printf '  lcb:    %s\n' "${BENCH_LCB_DIR}"
