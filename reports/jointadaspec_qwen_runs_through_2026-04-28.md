# JointAdaSpec Qwen Runs Through 2026-04-28

## Summary

This report rolls up the tracked Qwen JointAdaSpec artifacts through `2026-04-28`.

The best substantive benchmark snapshot remains the `2026-04-14` Qwen2.5 `7B -> 1.5B` run: it used `3000` traces and `100` prompts for both GSM8K and LiveCodeBench. The newer `2026-04-21` and `2026-04-28` runs are useful pipeline artifacts, but they are smoke-sized or partial and should not be treated as final best-of-run evidence.

## Result Snapshot

| Run | Model pair | Scope | Best observed result | Status |
|---|---|---|---|---|
| `2026-04-14` | Qwen2.5 `7B -> 1.5B` | `3000` traces, `100` GSM8K prompts, `100` LiveCodeBench prompts | GSM8K `jointadaspec`: `18.99` tok/s, acceptance `0.951`; LiveCodeBench `jointadaspec`: `10.34` tok/s, acceptance `0.890` | Best substantive snapshot |
| `2026-04-21` | Qwen2.5 `7B -> 1.5B` | seeded/schema smoke, `1` prompt, `1` seed, `8` new tokens | `cascade_verif_then_length`: `18.47` tok/s, `1.204x` vs target-only | Complete smoke benchmark |
| `2026-04-28` | Qwen2.5 `14B -> 0.5B` | smoke, `5` prompts, `1` seed, `64` new tokens | `cascade_verif_then_length`: `13.15` tok/s, `0.725x`; `jointadaspec`: `12.85` tok/s, `0.709x`; `target_only`: `18.13` tok/s | Complete smoke benchmark |
| `2026-04-28` | Qwen2.5 `7B -> 1.5B` | full trace/solve attempt, `500` traces | policy and condition artifacts saved | Benchmark stage failed |

## Artifacts

- `reports/jointadaspec_qwen_7b_1p5b_2026-04-14.md`
- `outputs/jointadaspec_qwen_2026-04-14/`
- `outputs/jointadaspec_qwen7b_1p5b_2026-04-21/`
- `outputs/jointadaspec_qwen7b_1p5b_2026-04-28/`
- `outputs/jointadaspec_qwen14b_0p5b_2026-04-28/`
- `reports/conditions_qwen7b_1p5b_2026-04-21.json`
- `reports/conditions_qwen7b_1p5b_2026-04-28.json`
- `reports/conditions_qwen14b_0p5b_2026-04-28.json`
- `reports/manifests/`
- `reports/conditions_plots_2026-04-21/`
- `reports/conditions_plots_2026-04-28/`

## Condition Checks

| Run | Passed | Failed |
|---|---|---|
| Qwen `7B -> 1.5B`, `2026-04-21` | `c3`, `n1`, `n2` | `c1`, `c2`, `c4` |
| Qwen `7B -> 1.5B`, `2026-04-28` | `n1`, `n2` | `c1`, `c2`, `c3`, `c4` |
| Qwen `14B -> 0.5B`, `2026-04-28` | `c2`, `c3`, `n1`, `n2` | `c1`, `c4` |

Condition failures are recorded as empirical findings, not runtime failures. The checker is intentionally non-blocking so these results can be discussed in the thesis analysis.

## 2026-04-28 Qwen 7B/1.5B Failure

The Qwen `7B -> 1.5B` full run collected `500` traces and saved joint plus cascade policy artifacts under `outputs/jointadaspec_qwen7b_1p5b_2026-04-28/02_solve/`.

The benchmark stage did not produce `03_bench_gsm8k` outputs. The log shows an SSL EOF while loading `Qwen/Qwen2.5-7B-Instruct` from HuggingFace. After the benchmark failed, `scripts/run_step3_after_step2.sh` checked for the expected Step 2 artifacts and refused to launch the dependent full Step 3 because these files were missing:

- `outputs/jointadaspec_qwen7b_1p5b_2026-04-28/03_bench_gsm8k/results.jsonl`
- `reports/pareto_qwen7b_1p5b_2026-04-28.pdf`
- `reports/ablation_qwen7b_1p5b_2026-04-28.pdf`
- `reports/threshold_surface_qwen7b_1p5b_2026-04-28/`

## Interpretation

- JointAdaSpec remains operational end to end on the Qwen `7B -> 1.5B` profile, with the `2026-04-14` run serving as the meaningful benchmark reference.
- The newer seeded benchmark path records GSM8K exact match, bootstrap confidence intervals, manifests, cascade baselines, and resume-safe JSONL.
- The latest smoke runs show the report and condition-verification pipeline working, but a multi-seed/multi-sample rerun is still required before claiming a new best result.
