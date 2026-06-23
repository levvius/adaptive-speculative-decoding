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

## Model And Dataset Revisions

Record exact model and dataset identifiers:

- Target model path or HF revision.
- Draft model path or HF revision.
- Tokenizer path or HF revision.
- Dataset file path and SHA256.
- Trace parquet SHA256.
- Policy `.npz` SHA256.

## Validation Commands

Run these on the machine that has the project environment installed:

```bash
make check
make test
make bench-toy
```

For GPU smoke only, use the smallest configured HF run before launching long
benchmarks. Do not use smoke metrics as final evidence.

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
