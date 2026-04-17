# Roadmap

## Current Focus

1. Improve quality-speed tradeoff on local single-GPU runs.
2. Add task-level quality metrics to the new JointAdaSpec benchmark path.
3. Make experiment outputs easier to compare and review.

## In Progress

- Llama 8B/3B long-run profile with paper-aligned AutoJudge C-grid (`1e-7..1e0`).
- Better project presentation for external reviewers (docs/templates/results index).
- JointAdaSpec Qwen `7B -> 1.5B` documentation pass and report-card cleanup.

## Next Technical Steps

### Decoding and Modeling

- Evaluate larger speculative window (`k=8`, `k=16`) with fixed runtime budget.
- Prototype stronger judge backends (tree/boosting models) behind a stable interface.
- Add task-specific AutoJudge training data path for LiveCodeBench-like tasks.
- Add GSM8K exact-match evaluation to `scripts/03_benchmark.py` for the `jointadaspec/` stack.
- Compare `JointAdaSpec` policies across `kappa` values instead of benchmarking only a single selected policy.

### Performance Engineering

- Reduce judge overhead by minimizing CPU roundtrips.
- Investigate GPU-resident judge path.
- Add profiling snapshots for mismatch-heavy regions.
- Profile JointAdaSpec trace collection and benchmark loops to identify target-only bottlenecks.

### Benchmark Quality

- Add comparable 48h profiles across model families with unified run matrix.
- Publish concise per-run summary cards (accuracy/speed/cost).
- Keep strict JSONL schema compatibility for downstream analysis.
- Generate markdown reports directly from JointAdaSpec `outputs/` directories.

## Project Hygiene

- Expand CI coverage with additional smoke checks for config presets.
- Keep docs synchronized with scripts and defaults.
- Maintain reproducibility-first defaults (explicit manifests, deterministic file naming).
