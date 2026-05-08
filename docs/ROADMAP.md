# Roadmap

## Current Focus

1. Improve quality-speed tradeoff on local single-GPU runs.
2. Re-run JointAdaSpec full benchmarks with enough prompts/seeds to separate real speed wins from smoke noise.
3. Make experiment outputs easier to compare and review.

## In Progress

- Run the final 48-hour held-out quality validation for Qwen `7B -> 1.5B` using the fixed `2026-05-05` quality-aware policy.
- Treat `cascade_verif_then_length` as the pre-registered primary decoder for the held-out quality claim.
- Use `test_start_index=100`, `500` GSM8K prompts, `3` seeds, and `256` max new tokens for the main validation run.
- Better project presentation for external reviewers (docs/templates/results index).
- JointAdaSpec Qwen documentation pass and report-card cleanup through the `2026-05-05` artifacts.

## Next Technical Steps

### Decoding and Modeling

- Evaluate larger speculative window (`k=8`, `k=16`) with fixed runtime budget.
- Prototype stronger judge backends (tree/boosting models) behind a stable interface.
- Add task-specific AutoJudge training data path for LiveCodeBench-like tasks.
- Keep GSM8K exact-match evaluation in the seeded `scripts/03_benchmark.py` path and use it in future multi-seed reports.
- Compare `JointAdaSpec` policies across `kappa` values instead of benchmarking only a single selected policy.
- Compare baseline JointAdaSpec against quality-aware JointAdaSpec on the same Qwen `7B -> 1.5B` local model pair, same seeds, same prompt count, and same max token budget.
- Analyze held-out paired EM deltas against `target_only` and `speculative` before making a quality improvement claim.
- Promote smoke-only `2026-04-21` and `2026-04-28` results into a normal multi-seed/multi-sample run before claiming a new best result.

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
- Keep held-out benchmark slices explicit via `datasets.test_start_index` in manifests and result metadata.
- Add a retry/resume wrapper around HuggingFace model metadata fetches for long benchmark stages.
- Prefer local model-pair configs for long Qwen reruns when full checkpoint shards are already present on disk.

## Project Hygiene

- Expand CI coverage with additional smoke checks for config presets.
- Keep docs synchronized with scripts and defaults.
- Maintain reproducibility-first defaults (explicit manifests, deterministic file naming).
