# JointAdaSpec Quality Analysis

- Benchmark: `outputs/jointadaspec_qwen14b_0p5b_lock_2026-05-14/03_bench_gsm8k/benchmark.csv`
- Primary method: `cascade_verif_then_length`
- Baseline: `target_only`
- Primary success: `True`
- Strong success: `True`

## Method Summary

| Method | Runs | tok/s mean | Acceptance | GSM8K EM |
|---|---:|---:|---:|---:|
| cascade_verif_then_length | 1500 | 10.2916 | 54.93% | 57.13% |
| jointadaspec | 1500 | 10.8407 | 55.21% | 57.00% |
| speculative | 1500 | 4.8802 | 27.07% | 53.27% |
| target_only | 1500 | 14.4456 | 0.00% | 52.93% |

## Paired EM Comparisons

| Method | Baseline | Paired n | EM diff | 95% CI | Wins | Losses | Ties | p-value |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cascade_verif_then_length | target_only | 1500 | 4.20% | [0.87%, 7.67%] | 356 | 293 | 851 | 0.0149 |
| speculative | target_only | 1500 | 0.33% | [-3.00%, 3.73%] | 336 | 331 | 833 | 0.8769 |
| jointadaspec | target_only | 1500 | 4.07% | [0.67%, 7.47%] | 365 | 304 | 831 | 0.0203 |
