# Qwen3.5 9B->2B Powered JointAdaSpec Run

Date staged: 2026-06-29
Primary script: `scripts/run_qwen35_jointadaspec.sh`
Pair: `Qwen/Qwen3.5-9B` target -> `Qwen/Qwen3.5-2B` draft

## Purpose

This run is the first powered Qwen3.5 evaluation for the repaired block-v1
JointAdaSpec path. The target claim is confidence-aware adaptive-control
evidence on Qwen3.5, not a claim that speculative variants are faster than
plain target-only decoding.

The selected pair is the largest local Qwen3.5 gap available on disk
(~4.5x) and has compatible tokenizers. `Qwen3.5-4B` remains a fallback, not
part of the primary run.

## Environment

Use the isolated Qwen3.5 environment because the main pinned environment uses
`transformers 4.57.x`, which does not register `model_type: qwen3_5`.

Expected:

- Python: `.venv-qwen35/bin/python`
- PyTorch: `2.9.1+cu128`
- Transformers: `5.12.1`
- GPU: RTX 5090, at least `22000` MiB free VRAM
- No active compute PIDs unless `ALLOW_ACTIVE_GPU=1` is intentional

Check:

```bash
.venv-qwen35/bin/python - <<'PY'
import torch, transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("qwen3_5", "qwen3_5" in CONFIG_MAPPING_NAMES)
PY
```

## Model And Dataset Preflight

Datasets:

```bash
ls -lh datasets/gsm8k_train.jsonl datasets/gsm8k_test.jsonl
```

Model shards:

```bash
.venv-qwen35/bin/python scripts/verify_model_shards.py --model-dir models/Qwen3.5-9B
.venv-qwen35/bin/python scripts/verify_model_shards.py --model-dir models/Qwen3.5-2B
```

If missing or incomplete, repair with the Hugging Face CLI:

```bash
hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
```

The runner also performs these checks and can download missing/incomplete
checkpoints when `DOWNLOAD_MISSING_MODELS=1` (default).

## Powered Command

Run in tmux:

```bash
tmux new -s qwen35_powered
cd /home/robot/Project/adaptive-speculative-decoding
export HF_HUB_DISABLE_XET=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN=.venv-qwen35/bin/python bash scripts/run_qwen35_jointadaspec.sh
```

Defaults:

- `MAX_TRACES=500`
- `TEST_START_INDEX=100`
- `MAX_SAMPLES=500`
- `N_SEEDS=3`
- `MAX_NEW_TOKENS=256`
- `FIXED_SD_GAMMA=8`
- `FUZZY_SD_GAMMA=8`
- `CONF_GATE_TAU=0.0`

Expected wall time from the 2026-06-29 smoke rates is about 65-75 hours on the
RTX 5090.

## What The Script Runs

Base arm (`qwen35_9b_2b_jointadaspec`):

- `target_only`
- `speculative`
- `cascade_verif_then_length`
- `jointadaspec`

Confidence arm (`qwen35_9b_2b_jointadaspec_conf`):

- `jointadaspec` only, merged and relabelled as `jointadaspec_conf`

The merged report is written to:

```text
reports/qwen35_9b_2b_${DATE_TAG}/quality.md
```

The merged benchmark CSV is written to:

```text
reports/qwen35_9b_2b_${DATE_TAG}/merged_benchmark.csv
```

## Monitoring

```bash
tail -f logs/qwen35_jointadaspec_$(date +%F).log
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader
find outputs/jointadaspec_qwen35_9b_2b_$(date +%F) -maxdepth 3 -type f | sort
```

Detach tmux with `Ctrl-b d`; reattach with:

```bash
tmux attach -t qwen35_powered
```

## Validation

After completion:

```bash
.venv-qwen35/bin/python scripts/validate_results_jsonl.py \
  --path outputs/jointadaspec_qwen35_9b_2b_$(date +%F)/base/03_bench_gsm8k/results.jsonl \
  --strict

.venv-qwen35/bin/python scripts/validate_results_jsonl.py \
  --path outputs/jointadaspec_qwen35_9b_2b_$(date +%F)/conf/03_bench_gsm8k/results.jsonl \
  --strict
```

The final table must come from the merged CSV and
`scripts/analyze_jointadaspec_quality.py`, which reports paired and
prompt-clustered statistics.

## Interpretation Rules

- The 2026-06-29 smoke (`10` traces, `5` prompts x `1` seed, `128` new tokens)
  is pipeline validation only.
- Report speed relative to both `target_only` and vanilla speculative.
- Do not claim a Qwen3.5 quality win without the powered paired/clustered table.
- If `jointadaspec_conf` ties base `jointadaspec`, report the tie. That is still
  useful evidence about whether the confidence axis changes behavior on Qwen3.5.
- If the powered run fails, preserve logs and manifests and do not update
  `docs/RESULTS.md` with partial metrics.

Sources: Hugging Face model cards for
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B),
[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), and the Transformers
[Qwen3.5 docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
