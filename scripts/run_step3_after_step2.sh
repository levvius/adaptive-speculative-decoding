#!/usr/bin/env bash
# Watcher: waits for Step 2 (7B/1.5B full sweep) to finish, validates its
# artifacts, then launches Step 3 (14B/0.5B full sweep).
#
# Step 2 is the running `make jointadaspec-full MODEL_PAIR=qwen7b_1p5b ...`
# process. We detect completion by absence of the matching python process AND
# presence of the final report PDFs.

set -u

LOG_DIR="logs"
# Match anything whose cmdline references the Step 2 output dir or the make
# target with MODEL_PAIR=qwen7b_1p5b. Robust to config-name aliasing
# (qwen25_7b_1p5b_jointadaspec) since the output dir is what each stage uses.
STEP2_PID_PATTERN="jointadaspec_qwen7b_1p5b_2026-04-28|MODEL_PAIR=qwen7b_1p5b"
STEP2_OUT="outputs/jointadaspec_qwen7b_1p5b_2026-04-28"
STEP2_PARETO="reports/pareto_qwen7b_1p5b_2026-04-28.pdf"
STEP2_ABLATION="reports/ablation_qwen7b_1p5b_2026-04-28.pdf"
STEP2_THRESHOLD_DIR="reports/threshold_surface_qwen7b_1p5b_2026-04-28"
STEP2_BENCH_RESULTS="${STEP2_OUT}/03_bench_gsm8k/results.jsonl"

STEP3_DATE="2026-04-29"
STEP3_LOG="${LOG_DIR}/jointadaspec_qwen14b_0p5b_full_${STEP3_DATE}.log"
WATCHER_LOG="${LOG_DIR}/run_step3_after_step2.log"

mkdir -p "$LOG_DIR"

log() {
  printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$WATCHER_LOG"
}

log "watcher started (pid=$$); polling every 60s"

# 1) Poll until no Step-2 python child is alive.
while pgrep -f "$STEP2_PID_PATTERN" >/dev/null 2>&1; do
  sleep 60
done

log "no Step 2 python processes detected; validating artifacts"

# 2) Validate artifacts. Refuse to launch Step 3 unless Step 2 produced its
#    canonical outputs.
fail=0
for path in "$STEP2_BENCH_RESULTS" "$STEP2_PARETO" "$STEP2_ABLATION"; do
  if [ ! -e "$path" ]; then
    log "MISSING $path"
    fail=1
  else
    log "OK      $path"
  fi
done
if [ ! -d "$STEP2_THRESHOLD_DIR" ] || [ -z "$(ls -A "$STEP2_THRESHOLD_DIR" 2>/dev/null)" ]; then
  log "MISSING $STEP2_THRESHOLD_DIR (empty or absent)"
  fail=1
else
  log "OK      $STEP2_THRESHOLD_DIR"
fi

if [ "$fail" -ne 0 ]; then
  log "Step 2 did not produce required artifacts; refusing to launch Step 3"
  exit 1
fi

# 3) GPU preflight before launching Step 3.
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
log "GPU free VRAM: ${free_mib} MiB"
if [ -n "$free_mib" ] && [ "$free_mib" -lt 22000 ]; then
  log "free VRAM (${free_mib} MiB) below 22000 MiB; refusing to launch Step 3"
  exit 2
fi

compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')
if [ -n "$compute_apps" ]; then
  log "compute apps still active: $compute_apps; refusing to launch Step 3"
  exit 3
fi

# 4) Launch Step 3 (14B/0.5B full sweep) under the same env as the plan.
log "launching Step 3: make jointadaspec-full MODEL_PAIR=qwen14b_0p5b JOINTADA_DATE=${STEP3_DATE} JOINTADA_MAX_TRACES=500 JOINTADA_MAX_SAMPLES=100 JOINTADA_N_SEEDS=3 JOINTADA_MAX_NEW_TOKENS=256"

export HF_HUB_DISABLE_XET=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

make jointadaspec-full \
  MODEL_PAIR=qwen14b_0p5b \
  JOINTADA_DATE="${STEP3_DATE}" \
  JOINTADA_MAX_TRACES=500 \
  JOINTADA_MAX_SAMPLES=100 \
  JOINTADA_N_SEEDS=3 \
  JOINTADA_MAX_NEW_TOKENS=256 \
  2>&1 | tee -a "$STEP3_LOG"

rc=${PIPESTATUS[0]}
log "Step 3 finished with rc=${rc}; tail of ${STEP3_LOG} below"
tail -20 "$STEP3_LOG" | sed 's/^/  /' | tee -a "$WATCHER_LOG"
exit "$rc"
