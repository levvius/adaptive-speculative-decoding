# Roadmap

## Current Focus

1. Replace the current quality-risk reward shaping with a stronger quality mechanism.
2. Use held-out prompt slices before treating quality improvements as real.
3. Make experiment outputs easier to compare and review.

## In Progress

- Pivot from scalar `quality_risk_K/k` shaping to prompt/state-level fallback policies.
- Analyze where held-out `jointadaspec` and `cascade_verif_then_length` lose exact match against `target_only`.
- Better project presentation for external reviewers (docs/templates/results index).
- JointAdaSpec Qwen documentation pass and report-card cleanup through the `2026-05-08` artifacts.

## Recently Completed

- Completed the `2026-05-08` held-out Qwen `7B -> 1.5B` quality validation with `500` GSM8K prompts, `3` seeds, and `256` max new tokens.
- Primary method `cascade_verif_then_length` did not improve held-out quality: EM delta vs `target_only` was `-2.27` percentage points with CI crossing zero.
- The `2026-05-05` first-slice quality improvement is now treated as a non-generalizing hypothesis, not a final quality claim.
- Secondary Qwen `14B -> 0.5B` cross-check and extended fallback runs remained partial and are not used for benchmark claims.

## Next Technical Steps

### Decoding and Modeling

- Evaluate larger speculative window (`k=8`, `k=16`) with fixed runtime budget.
- Prototype stronger judge backends (tree/boosting models) behind a stable interface.
- Add task-specific AutoJudge training data path for LiveCodeBench-like tasks.
- Keep GSM8K exact-match evaluation in the seeded `scripts/03_benchmark.py` path and use it in future multi-seed reports.
- Prototype a prompt/state-level fallback policy that switches to `target_only` in high-risk math-reasoning regions.
- Run offline error analysis on held-out prompt-seed rows before launching another expensive quality run.
- Compare future quality mechanisms against both `target_only` and ordinary `speculative`, not only against older JointAdaSpec variants.
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
