#!/usr/bin/env bash
# Bounded defense validation for the repaired JointAdaSpec block-v1 path.
#
# This script is intentionally conservative: it validates that fresh traces,
# a fresh 9-action policy, and a small benchmark can run end-to-end under the
# repaired block_verify_v1 semantics. It is not a powered benchmark run and
# must not be presented as final block-v1 evidence.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEFENSE_DATE="${DEFENSE_DATE:-$(date +%F)}"
MODEL_PAIR="${MODEL_PAIR:-qwen14b_0p5b}"
FALLBACK_MODEL_PAIR="${FALLBACK_MODEL_PAIR:-qwen7b_1p5b}"
TOTAL_BUDGET_SECONDS="${TOTAL_BUDGET_SECONDS:-82800}" # 23h, leaving room for inspection.
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-22000}"
MAX_TRACES="${DEFENSE_MAX_TRACES:-300}"
MAX_SAMPLES="${DEFENSE_MAX_SAMPLES:-60}"
MAX_NEW_TOKENS="${DEFENSE_MAX_NEW_TOKENS:-192}"
N_SEEDS="${DEFENSE_N_SEEDS:-1}"
TEST_START_INDEX="${DEFENSE_TEST_START_INDEX:-0}"
LOG_PATH="${LOG_PATH:-logs/defense_blockv1_${DEFENSE_DATE}.log}"
STATUS_PATH="${STATUS_PATH:-logs/defense_blockv1_${DEFENSE_DATE}.status.json}"
SUMMARY_PATH="${SUMMARY_PATH:-reports/defense_blockv1_${DEFENSE_DATE}.md}"

mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$STATUS_PATH")" "$(dirname "$SUMMARY_PATH")" reports/manifests
exec > >(tee -a "$LOG_PATH") 2>&1

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

START_TS="$(date +%s)"
DEADLINE_TS=$((START_TS + TOTAL_BUDGET_SECONDS))

json_escape() {
  printf '%s' "$1" | tr '\n' ' ' | sed 's/\\/\\\\/g; s/"/\\"/g'
}

write_status() {
  local phase="$1" state="$2" extra="${3:-}"
  printf '{"date":"%s","phase":"%s","state":"%s","timestamp":"%s","extra":"%s"}\n' \
    "$(json_escape "$DEFENSE_DATE")" \
    "$(json_escape "$phase")" \
    "$(json_escape "$state")" \
    "$(date -Is)" \
    "$(json_escape "$extra")" > "$STATUS_PATH"
}

write_summary() {
  local state="$1" pair="$2" output_root="$3" note="$4"
  local git_sha dirty
  git_sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  dirty="$(git diff --quiet && git diff --cached --quiet && echo false || echo true)"
  cat > "$SUMMARY_PATH" <<EOF
# Defense block-v1 bounded validation (${DEFENSE_DATE})

Status: ${state}

This run is a bounded validation of the repaired JointAdaSpec \`block_verify_v1\`
pipeline. It is **not final block-v1 benchmark evidence** and must not replace
the legacy/pre-repair boundary in the defense narrative.

## Configuration

- git SHA: \`${git_sha}\`
- dirty worktree at launch/end: \`${dirty}\`
- selected pair: \`${pair}\`
- output root: \`${output_root:-not created}\`
- max traces: \`${MAX_TRACES}\`
- max samples: \`${MAX_SAMPLES}\`
- seeds: \`${N_SEEDS}\`
- max new tokens: \`${MAX_NEW_TOKENS}\`
- wall-clock budget: \`${TOTAL_BUDGET_SECONDS}s\`
- log: \`${LOG_PATH}\`
- status: \`${STATUS_PATH}\`

## Interpretation

${note}

Use this artifact only as pipeline validation. Any final quality or speed claim
still requires a powered rerun with prompt-clustered statistics.
EOF
}

remaining_seconds() {
  local now remaining
  now="$(date +%s)"
  remaining=$((DEADLINE_TS - now))
  if [ "$remaining" -lt 0 ]; then
    remaining=0
  fi
  echo "$remaining"
}

run_with_budget() {
  local remaining
  remaining="$(remaining_seconds)"
  if [ "$remaining" -le 300 ]; then
    echo "[budget] less than 5 minutes remain; refusing to start next stage."
    return 124
  fi
  echo "[budget] remaining=${remaining}s :: $*"
  timeout --foreground "${remaining}s" "$@"
}

experiment_for_pair() {
  case "$1" in
    qwen14b_0p5b) echo "qwen25_14b_0p5b_jointadaspec" ;;
    qwen7b_1p5b) echo "qwen25_7b_1p5b_jointadaspec" ;;
    qwen7b_1p5b_quality) echo "qwen25_7b_1p5b_jointadaspec_quality" ;;
    *)
      echo "[config] unsupported MODEL_PAIR '$1'" >&2
      return 2
      ;;
  esac
}

check_python() {
  if [ -x "$PYTHON_BIN" ] || command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    "$PYTHON_BIN" - <<'PY'
import importlib
for name in ("torch", "pandas", "hydra", "transformers"):
    importlib.import_module(name)
PY
    return 0
  fi
  echo "[preflight] Python not found/executable: ${PYTHON_BIN}" >&2
  return 1
}

check_common_preflight() {
  write_status "preflight" "running"
  echo "[preflight] started at $(date -Is)"

  if ! check_python; then
    write_status "preflight" "failed" "Python environment is incomplete. Run make setup-gpu first."
    write_summary "not run" "$MODEL_PAIR" "" "GPU validation did not start: Python environment is incomplete. Run \`make setup-gpu\` first."
    exit 3
  fi

  for path in datasets/gsm8k_train.jsonl datasets/gsm8k_test.jsonl; do
    if [ ! -f "$path" ]; then
      write_status "preflight" "failed" "Missing dataset: ${path}"
      write_summary "not run" "$MODEL_PAIR" "" "GPU validation did not start: required dataset \`${path}\` is missing."
      exit 3
    fi
  done

  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    write_status "preflight" "failed" "nvidia-smi cannot communicate with the NVIDIA driver."
    write_summary "not run" "$MODEL_PAIR" "" "GPU validation did not start on this machine: \`nvidia-smi\` could not communicate with the NVIDIA driver. CPU checks and CI remain the defense fallback."
    exit 3
  fi

  local active_pids free_mib
  active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' || true)"
  if [ -n "$active_pids" ] && [ "${ALLOW_ACTIVE_GPU:-0}" != "1" ]; then
    write_status "preflight" "failed" "Active GPU compute PIDs: ${active_pids}"
    write_summary "not run" "$MODEL_PAIR" "" "GPU validation did not start because another compute process was active: \`${active_pids}\`."
    exit 3
  fi

  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  echo "[preflight] free VRAM: ${free_mib:-unknown} MiB (need ${MIN_FREE_VRAM_MIB})"
  if [ -z "$free_mib" ] || [ "$free_mib" -lt "$MIN_FREE_VRAM_MIB" ]; then
    write_status "preflight" "failed" "Insufficient free VRAM: ${free_mib:-unknown} MiB"
    write_summary "not run" "$MODEL_PAIR" "" "GPU validation did not start because free VRAM was below the configured threshold."
    exit 3
  fi
  write_status "preflight" "ok"
}

check_pair_assets() {
  local pair="$1"
  if [ "$pair" = "qwen14b_0p5b" ]; then
    for path in models/qwen2.5-14b-Instruct-model models/qwen2.5-0.5b-Instruct-model; do
      if [ ! -d "$path" ]; then
        echo "[assets] missing local model directory for ${pair}: ${path}" >&2
        return 4
      fi
    done
  fi
}

run_pipeline() {
  local pair="$1" label="$2"
  local config output_root trace_dir solve_dir bench_dir policy_path conditions_path manifest_path
  config="$(experiment_for_pair "$pair")" || return 2
  output_root="outputs/jointadaspec_${pair}_defense_blockv1_${DEFENSE_DATE}"
  trace_dir="${output_root}/01_traces"
  solve_dir="${output_root}/02_solve"
  bench_dir="${output_root}/03_bench_gsm8k"
  policy_path="${solve_dir}/policy.npz"
  conditions_path="reports/conditions_${pair}_defense_blockv1_${DEFENSE_DATE}.json"
  manifest_path="reports/manifests/$(basename "$output_root")_$(basename "$bench_dir").json"

  echo "[run:${label}] pair=${pair} config=${config} output=${output_root}"
  write_status "${label}" "checking_assets" "${pair}"
  check_pair_assets "$pair" || return 4

  write_status "${label}" "collect_traces" "${output_root}"
  run_with_budget "$PYTHON_BIN" scripts/01_collect_traces.py --config-name "experiments/${config}" \
    "experiments.output_dir=${trace_dir}" \
    "experiments.n_traces=${MAX_TRACES}" \
    "experiments.datasets.train_max_samples=${MAX_TRACES}" \
    "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}"

  write_status "${label}" "solve_mdp" "${policy_path}"
  run_with_budget "$PYTHON_BIN" scripts/02_solve_mdp.py --config-name "experiments/${config}" \
    "experiments.output_dir=${solve_dir}" \
    "experiments.traces_path=${trace_dir}/traces.parquet"

  write_status "${label}" "verify_conditions" "${conditions_path}"
  run_with_budget "$PYTHON_BIN" scripts/04_verify_conditions.py \
    --traces "${trace_dir}/traces.parquet" \
    --policy "$policy_path" \
    --out "$conditions_path"

  write_status "${label}" "benchmark" "${bench_dir}"
  run_with_budget "$PYTHON_BIN" scripts/03_benchmark.py --config-name "experiments/${config}" \
    "experiments.output_dir=${bench_dir}" \
    "experiments.policy_path=${policy_path}" \
    "experiments.datasets.test_start_index=${TEST_START_INDEX}" \
    "experiments.datasets.test_max_samples=${MAX_SAMPLES}" \
    "experiments.datasets.max_new_tokens=${MAX_NEW_TOKENS}" \
    "experiments.n_seeds=${N_SEEDS}" \
    "experiments.baselines=[vanilla_ar,fixed_sd,cascade_verif_then_length]"

  write_status "${label}" "validate_results" "${bench_dir}/results.jsonl"
  run_with_budget "$PYTHON_BIN" scripts/validate_results_jsonl.py \
    --path "${bench_dir}/results.jsonl" \
    --strict

  write_status "${label}" "reports" "$pair"
  run_with_budget "$PYTHON_BIN" reports/templates/pareto_plot.py \
    --input "${bench_dir}/results.jsonl" \
    --out "reports/pareto_${pair}_defense_blockv1_${DEFENSE_DATE}.pdf" \
    --manifest "$manifest_path"
  run_with_budget "$PYTHON_BIN" reports/templates/threshold_surface.py \
    --policy "$policy_path" \
    --out "reports/threshold_surface_${pair}_defense_blockv1_${DEFENSE_DATE}" \
    --manifest "$manifest_path"
  run_with_budget "$PYTHON_BIN" reports/templates/ablation_bars.py \
    --input "${bench_dir}/results.jsonl" \
    --out "reports/ablation_${pair}_defense_blockv1_${DEFENSE_DATE}.pdf" \
    --manifest "$manifest_path"
  if [ -f "${bench_dir}/benchmark.csv" ]; then
    run_with_budget "$PYTHON_BIN" scripts/analyze_jointadaspec_quality.py \
      --benchmark "${bench_dir}/benchmark.csv" \
      --out "reports/jointadaspec_quality_${pair}_defense_blockv1_${DEFENSE_DATE}.md" \
      --primary jointadaspec \
      --baseline target_only \
      --controls target_only speculative cascade_verif_then_length
  fi

  write_status "${label}" "succeeded" "$output_root"
  write_summary "succeeded" "$pair" "$output_root" "The bounded fresh block-v1 pipeline completed. Treat the result as end-to-end validation only; the sample size and wall-clock budget are intentionally not sufficient for final benchmark claims."
}

echo "[defense-blockv1] started at $(date -Is)"
echo "[defense-blockv1] log: ${LOG_PATH}"
echo "[defense-blockv1] status: ${STATUS_PATH}"
echo "[defense-blockv1] summary: ${SUMMARY_PATH}"

check_common_preflight

PRIMARY_OK=0
if run_pipeline "$MODEL_PAIR" "primary"; then
  PRIMARY_OK=1
fi

if [ "$PRIMARY_OK" = "1" ]; then
  echo "[defense-blockv1] primary run succeeded at $(date -Is)"
  exit 0
fi

echo "[defense-blockv1] primary run failed or was unavailable."
if [ "$MODEL_PAIR" = "$FALLBACK_MODEL_PAIR" ]; then
  write_status "fallback" "skipped" "Primary and fallback pairs are identical."
  write_summary "failed" "$MODEL_PAIR" "" "The selected block-v1 validation run failed, and no distinct fallback pair was configured."
  exit 1
fi

if [ "$(remaining_seconds)" -le 7200 ]; then
  write_status "fallback" "skipped" "Less than 2h remain."
  write_summary "failed" "$MODEL_PAIR" "" "The primary validation did not complete and less than two hours remained, so fallback was skipped."
  exit 1
fi

echo "[defense-blockv1] attempting fallback pair=${FALLBACK_MODEL_PAIR}"
if run_pipeline "$FALLBACK_MODEL_PAIR" "fallback"; then
  echo "[defense-blockv1] fallback run succeeded at $(date -Is)"
  exit 0
fi

write_status "fallback" "failed" "$FALLBACK_MODEL_PAIR"
write_summary "failed" "$FALLBACK_MODEL_PAIR" "" "Both the primary and fallback bounded block-v1 validation attempts failed. Keep the defense narrative on CPU/CI validation and the legacy/pre-repair boundary."
exit 1
