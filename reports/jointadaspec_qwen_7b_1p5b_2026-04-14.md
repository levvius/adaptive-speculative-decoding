# JointAdaSpec Run Report

## Summary

- Run tag: `2026-04-14`
- Stack: `jointadaspec/`
- Model pair: `Qwen2.5-7B-Instruct -> Qwen2.5-1.5B-Instruct`
- GPU: RTX 5090
- Datasets: `GSM8K`, `LiveCodeBench`
- Scope: real HF end-to-end run with trace collection, MDP solving, and benchmark stages

This run validates that the new JointAdaSpec stack works end-to-end on a real target/draft model pair. The current benchmark implementation reports throughput and acceptance metrics; task-level GSM8K exact-match accuracy is not yet integrated into `scripts/03_benchmark.py`.

## Artifacts

- Traces: `outputs/jointadaspec_qwen_2026-04-14/01_traces_gsm8k/`
- Policies: `outputs/jointadaspec_qwen_2026-04-14/02_solve/`
- GSM8K benchmark: `outputs/jointadaspec_qwen_2026-04-14/03_bench_gsm8k/`
- LiveCodeBench benchmark: `outputs/jointadaspec_qwen_2026-04-14/04_bench_livecodebench/`

## Trace Collection

| Item | Value |
|---|---:|
| Trace count | 3000 |
| One-step transition records | 479880 |
| `gamma_max` | 8 |
| `T_levels` | `1.00, 1.22, 1.49, 1.82, 2.22, 2.71, 3.30, 4.00` |
| Discount | 0.99 |
| `kappa` used during trace logging | 1.0 |
| Git SHA at run time | `0bcce61e5eaee58f5ff0af7ad6c4680c112fcc4d` |

## MDP Solve Snapshot

| `kappa` | iterations | final delta | converged |
|---:|---:|---:|---:|
| 0.0 | 834 | `9.95e-05` | yes |
| 0.5 | 830 | `9.97e-05` | yes |
| 1.0 | 827 | `9.97e-05` | yes |
| 2.0 | 825 | `9.95e-05` | yes |
| 5.0 | 824 | `9.91e-05` | yes |

Policies were saved for all five `kappa` values under `outputs/jointadaspec_qwen_2026-04-14/02_solve/`.

## GSM8K Throughput Snapshot

| Method | Speed (tok/s) | Acceptance | vs Vanilla |
|---|---:|---:|---:|
| vanilla_ar | 29.38 | 0.000 | 1.000 |
| fixed_sd | 17.26 | 0.801 | 0.588 |
| fuzzy_sd_T1.5 | 18.17 | 0.858 | 0.619 |
| fuzzy_sd_T2 | 18.58 | 0.885 | 0.633 |
| fuzzy_sd_T2.5 | 18.74 | 0.895 | 0.638 |
| fuzzy_sd_T3 | 18.87 | 0.905 | 0.642 |
| fuzzy_sd_T3.5 | 19.01 | 0.912 | 0.647 |
| fuzzy_sd_T4 | 19.09 | 0.915 | 0.650 |
| jointadaspec | 18.99 | 0.951 | 0.646 |
| specdecpp | 17.81 | 0.835 | 0.606 |

Observations:
- `JointAdaSpec` achieves the highest acceptance rate in the table.
- On this single-GPU Qwen `7B -> 1.5B` pair, the fastest approximate baseline is still `fuzzy_sd_T4` by a small margin in throughput.
- `vanilla_ar` remains the throughput leader in this profile.

## LiveCodeBench Throughput Snapshot

| Method | Speed (tok/s) | Acceptance | vs Vanilla |
|---|---:|---:|---:|
| vanilla_ar | 14.36 | 0.000 | 1.000 |
| fixed_sd | 8.05 | 0.659 | 0.560 |
| fuzzy_sd_T1.5 | 8.63 | 0.728 | 0.601 |
| fuzzy_sd_T2 | 9.05 | 0.779 | 0.631 |
| fuzzy_sd_T2.5 | 9.27 | 0.803 | 0.646 |
| fuzzy_sd_T3 | 9.47 | 0.824 | 0.659 |
| fuzzy_sd_T3.5 | 9.54 | 0.835 | 0.664 |
| fuzzy_sd_T4 | 9.62 | 0.843 | 0.670 |
| jointadaspec | 10.34 | 0.890 | 0.720 |
| specdecpp | 9.04 | 0.763 | 0.629 |

Observations:
- `JointAdaSpec` is the fastest speculative-style method in this benchmark slice.
- `vanilla_ar` is still faster overall.
- Acceptance remains substantially higher for `JointAdaSpec` than for the fixed-window baselines.

## Interpretation

- The new stack is operational end-to-end on a real Hugging Face model pair.
- Joint policy control of draft length and verification threshold produces a clear acceptance-rate gain over the fixed-window baselines.
- The current Qwen `7B -> 1.5B` single-GPU profile does not yet recover enough runtime savings to beat target-only decoding.

## Limitations

- The current benchmark path does not compute task-level GSM8K exact-match accuracy for `jointadaspec/` runs.
- This run is a medium-scale real benchmark, not a 48-72h endurance profile by wall-clock duration.
- The Llama profile remains gated in this environment, so the practical JointAdaSpec path currently targets the ungated Qwen pair.
