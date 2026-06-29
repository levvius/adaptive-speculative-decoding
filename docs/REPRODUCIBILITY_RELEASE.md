# Reproducibility Release Checklist

Use this checklist before presenting repaired JointAdaSpec results as final thesis
evidence.

## Code State

1. Work from a named branch, not `main`.
2. Commit all intended code, docs, configs, and generated source artifacts.
3. Confirm no unrelated local edits are present:

```bash
git status --short
git diff --stat
```

4. Create a signed or annotated release tag after validation:

```bash
git tag -a jointadaspec-blocksd-v1 -m "JointAdaSpec block speculative decoding release"
```

## Environment

Record the exact environment in the run manifest:

- OS and hostname.
- GPU name, driver, CUDA runtime.
- Python, PyTorch, Transformers, NumPy, pandas, scipy versions.
- `CUBLAS_WORKSPACE_CONFIG`.
- `PYTORCH_ALLOC_CONF`.
- Quantization settings.

For RTX 50xx / Blackwell, use a PyTorch build with CUDA 12.8 support.

For Qwen3.5, keep the main thesis/CI environment pinned and use the isolated
Qwen3.5 environment:

```bash
.venv-qwen35/bin/python - <<'PY'
import torch, transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print(torch.__version__, torch.cuda.is_available())
print(transformers.__version__, "qwen3_5" in CONFIG_MAPPING_NAMES)
PY
```

The expected Qwen3.5 environment is `torch==2.9.1+cu128`,
`transformers==5.12.1`, CUDA available, and `qwen3_5` registered.

## Model And Dataset Revisions

Record exact model and dataset identifiers:

- Target model path or HF revision.
- Draft model path or HF revision.
- Tokenizer path or HF revision.
- Dataset file path and SHA256.
- Trace parquet SHA256.
- Policy `.npz` SHA256.

For Qwen3.5 9B->2B, verify local checkpoint completeness before launch:

```bash
.venv-qwen35/bin/python scripts/verify_model_shards.py --model-dir models/Qwen3.5-9B
.venv-qwen35/bin/python scripts/verify_model_shards.py --model-dir models/Qwen3.5-2B
```

If a local checkpoint is missing or incomplete, repair it idempotently:

```bash
hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
```

## Validation Commands

Run these on the machine that has the project environment installed:

```bash
make check
make test
make bench-toy
```

For GPU smoke only, use the smallest configured HF run before launching long
benchmarks. Do not use smoke metrics as final evidence.

For the powered Qwen3.5 comparison:

```bash
tmux new -s qwen35_powered
cd /home/robot/Project/adaptive-speculative-decoding
export HF_HUB_DISABLE_XET=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN=.venv-qwen35/bin/python bash scripts/run_qwen35_jointadaspec.sh
```

The default Qwen3.5 protocol is 500 traces, held-out GSM8K prompts 100-599,
3 seeds, 256 max new tokens, `fixed_sd_gamma=8`, and `fuzzy_sd_gamma=8`. The
runner benchmarks the base controls once and merges the draft-confidence arm as
`jointadaspec_conf`; expected wall time from the smoke rates is about 65-75
hours on the RTX 5090.

## RTX 5090 Rerun Boundary

The repaired implementation must be benchmarked on the RTX 5090 host before
updating final thesis claims:

```bash
make jointadaspec-full MODEL_PAIR=qwen14b_0p5b
make jointadaspec-full MODEL_PAIR=qwen7b_1p5b
```

If a run starts from existing traces or policies, the final report must state
which artifacts were reused and why reuse is methodologically valid.

## Reporting Rules

- Label pre-repair numbers as legacy.
- Report speed relative to both `target_only` and vanilla speculative.
- Use prompt-level clustered confidence intervals for GSM8K exact match.
- Correct for multiple comparisons when reporting secondary sweeps.
- Do not claim "first" without a fresh literature check.
- Treat Experiment E as an ablation, not a theorem.
- Treat the 2026-06-29 Qwen3.5 5-prompt smoke as pipeline validation only; final
  Qwen3.5 claims require strict JSONL validation and prompt-clustered paired
  analysis on the powered run.
