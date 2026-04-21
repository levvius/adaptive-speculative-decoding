# 2026-04-21: JointAdaSpec Thesis Pipeline Upgrades

## Summary

Implemented the missing theorem-to-practice pieces for the JointAdaSpec thesis stack:

- cascade baselines for theorem 2.3 comparison
- empirical verification of C1-C4 and N1-N2
- seeded/resumable JointAdaSpec benchmark JSONL with backward compatibility
- Qwen2.5 14B/0.5B local presets and Hydra experiment configs
- reproducibility manifests
- Pareto / threshold-surface / ablation plotting templates
- parameterized `make jointadaspec-*` orchestration

## Files Changed

### `jointadaspec/mdp/value_iteration.py`
- Added optional `action_mask=None` support for staged solves.
- Kept the default path unchanged for existing joint solves.

### `jointadaspec/baselines/cascade_common.py` (new)
- Added `CascadePolicy` save/load wrapper over the shared state grid.

### `jointadaspec/baselines/cascade_length_then_verif.py` (new)
- Stage 1: solve length with strict `T=1.0`.
- Stage 2: solve verification threshold with the learned length policy fixed.

### `jointadaspec/baselines/cascade_verif_then_length.py` (new)
- Stage 1: solve verification threshold with forced-continue length policy.
- Stage 2: solve length with the learned threshold policy fixed.

### `jointadaspec/analysis/conditions.py` (new)
- Added empirical condition checker for:
  - C1 reward monotonicity
  - C2 transition monotonicity
  - C3 state × action supermodularity fraction
  - C4 length × threshold supermodularity fraction
  - N1 divergence-set size and weighted mass
  - N2 stationary mass on divergence states
- Added six diagnostic plot writers.

### `jointadaspec/utils/manifest.py` (new)
- Added reproducibility manifest builder with git SHA, dirty flag, config YAML, seed list, and artifact hashes.

### `scripts/02_solve_mdp.py`
- Save both cascade policies next to the joint policy.
- Write cascade solve logs alongside the main value-iteration log.

### `scripts/03_benchmark.py`
- Replaced prompt-only throughput logging with seeded per-run summary records in `results.jsonl`.
- Kept legacy `run.jsonl`, `benchmark.csv`, and `summary.csv` outputs for backward compatibility.
- Added resume keys, bootstrap 95% CIs, deterministic seeding, and manifest creation.
- Added support for cascade policy loading from the solve directory.

### `scripts/04_verify_conditions.py` (new)
- Added CLI wrapper for the condition-analysis module.
- Writes JSON report, non-blocking warnings, plots, and manifest.

### `scripts/05_write_manifest.py` (new)
- Added standalone CLI for writing JointAdaSpec manifests.

### `scripts/validate_results_jsonl.py`
- Extended method support for JointAdaSpec methods and cascades.
- Added optional CI fields and schema-version support.
- Added validation path for historical JointAdaSpec prompt-level `run.jsonl` records.

### `configs/models.json`
- Added local `qwen25_14b_instruct_local` and `qwen25_0p5b_instruct_local` presets.

### `configs/experiments.json`
- Added `sp_samp`-compatible 14B/0.5B local presets for baseline/speculative/AutoJudge/Top-K/SpecExec discovery.

### `configs/model_pairs/qwen25_14b_0p5b.yaml` (new)
- Added local Hydra model-pair config for the 14B/0.5B JointAdaSpec pipeline.

### `configs/experiments/qwen25_14b_0p5b_jointadaspec*.yaml` (new)
- Added Hydra execution configs for GSM8K and LiveCodeBench.

### `reports/templates/*.py` (new)
- Added Pareto plot, threshold-surface, and ablation bar chart generators.

### `Makefile`
- Added generic `jointadaspec-traces`, `jointadaspec-solve`, `jointadaspec-verify`, `jointadaspec-bench`, `jointadaspec-report`, and `jointadaspec-full` targets.
- Parameterized the pipeline with `MODEL_PAIR`.
- Expanded `make check` to compile `jointadaspec/`, `scripts/`, and `reports/templates/`.

### `requirements.txt`, `requirements-gpu.txt`
- Pinned the JointAdaSpec runtime/analysis stack:
  - `hydra-core==1.3.2`
  - `omegaconf==2.3.0`
  - `pandas==3.0.1`
  - `scipy==1.17.0`
  - `pyarrow==23.0.1`
  - `matplotlib==3.10.8`

### `tests/test_cascade.py` (new)
- Added separable-reward regression where cascade equals joint.

### `tests/test_jointadaspec.py` (new)
- Added theorem 2.3 toy dominance check.
- Added value-iteration closed-form convergence test.
- Added policy NPZ roundtrip test.
- Added seed-stability regression.
- Added `04_verify_conditions.py` smoke test on synthetic traces.

### Documentation
- Updated `README.MD` with the new JointAdaSpec pipeline and generic make targets.
- Updated `CLAUDE.md` with JointAdaSpec constraints, limitations, and determinism policy.

## Deviations / Notes

- The repository still intentionally keeps two config systems:
  - JSON presets for `sp_samp`
  - Hydra configs for executable JointAdaSpec runs
- JointAdaSpec/cascade experiment names were not added to `configs/experiments.json`; that file remains `sp_samp`-only by design.
- The local `models/qwen2.5-14b-Instruct-model/` directory currently contains tokenizer/config files but is missing the safetensor shard files required for a real 14B smoke/full run in this workspace. Smoke validation of the new seeded JSONL/manifest/report path was therefore performed on the local Qwen2.5 `7B/1.5B` pair instead.
- Historical outputs under `outputs/jointadaspec_qwen_2026-04-14/` were left untouched.
