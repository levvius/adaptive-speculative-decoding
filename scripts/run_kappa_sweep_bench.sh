#!/usr/bin/env bash
# κ-sweep benchmark — Theorem 2.4 empirical Pareto front (Qwen 7B/1.5B).
#
# Benchmarks each pre-solved κ policy (jointadaspec + cascade_verif_then_length)
# on a fixed small slice so the (tokens/sec, GSM8K-EM) operating points can be
# traced as the speed/quality Lagrange weight κ varies — giving two Pareto
# fronts (joint vs cascade) to compare directly. Policies must already exist
# under outputs/jointadaspec_qwen7b_1p5b_kappa_sweep/02_solve/ (02_solve_mdp.py
# with kappa_values=[0,1,5,20,50,100]).
#
# GPU-only; run AFTER the triangulation has released the GPU (single-GPU rule).
# Resume-safe per κ via 03_benchmark.py. Usage:
#   bash scripts/run_kappa_sweep_bench.sh
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PY="${PYTHON_BIN:-.venv/bin/python}"
SWEEP_DIR="outputs/jointadaspec_qwen7b_1p5b_kappa_sweep"
SOLVE_DIR="${SWEEP_DIR}/02_solve"
KAPPAS=(0 1 5 20 50 100)
START_INDEX="${START_INDEX:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
N_SEEDS="${N_SEEDS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-22000}"
LOG="logs/kappa_sweep_$(date +%F).log"
STATUS="logs/kappa_sweep_$(date +%F).status"

mkdir -p logs
exec > >(tee -a "$LOG") 2>&1
echo "[kappa] start $(date -Is)"

# Single-GPU preflight: refuse to launch while another compute job holds the GPU.
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' || true)"
if [ -n "$pids" ]; then
  echo "[kappa] active compute PIDs on GPU: $pids — refusing to launch." >&2
  echo "blocked_gpu_busy" > "$STATUS"
  exit 1
fi
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
echo "[kappa] free VRAM: ${free_mib} MiB (need ${MIN_FREE_VRAM_MIB})"
if [ -z "$free_mib" ] || [ "$free_mib" -lt "$MIN_FREE_VRAM_MIB" ]; then
  echo "[kappa] insufficient free VRAM — aborting." >&2
  echo "blocked_low_vram" > "$STATUS"
  exit 1
fi

echo "running" > "$STATUS"
for K in "${KAPPAS[@]}"; do
  OUT="${SWEEP_DIR}/kappa_${K}/03_bench_gsm8k"
  POL="${SOLVE_DIR}/policy_kappa_${K}.npz"
  if [ ! -f "$POL" ]; then
    echo "[kappa] κ=${K}: policy missing ${POL} — skipping." >&2
    continue
  fi
  echo "[kappa] === κ=${K} -> ${OUT} ==="
  if "$PY" scripts/03_benchmark.py \
      --config-name experiments/qwen25_7b_1p5b_jointadaspec_quality \
      experiments.output_dir="$OUT" \
      experiments.policy_path="$POL" \
      'experiments.baselines=[cascade_verif_then_length]' \
      experiments.datasets.test_start_index="$START_INDEX" \
      experiments.datasets.test_max_samples="$MAX_SAMPLES" \
      experiments.datasets.max_new_tokens="$MAX_NEW_TOKENS" \
      experiments.n_seeds="$N_SEEDS"; then
    echo "[kappa] κ=${K} done $(date -Is)"
  else
    echo "[kappa] κ=${K} benchmark FAILED $(date -Is)" >&2
  fi
done

echo "done" > "$STATUS"
echo "[kappa] finished $(date -Is)"
