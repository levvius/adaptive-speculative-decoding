# JointAdaSpec Quality Analysis

- Benchmark: `outputs/jointadaspec_qwen7b_1p5b_quality_k16_2026-05-13/03_bench_gsm8k/benchmark.csv`
- Primary method: `cascade_verif_then_length`
- Baseline: `target_only`
- Primary success: `True`
- Strong success: `False`

## Method Summary

| Method | Runs | tok/s mean | Acceptance | GSM8K EM |
|---|---:|---:|---:|---:|
| cascade_verif_then_length | 600 | 15.6627 | 92.15% | 59.00% |
| jointadaspec | 600 | 15.6841 | 91.92% | 59.00% |
| speculative | 600 | 6.9770 | 39.92% | 62.17% |
| target_only | 600 | 24.8594 | 0.00% | 58.83% |

## Paired EM Comparisons

| Method | Baseline | Paired n | EM diff | 95% CI | Wins | Losses | Ties | p-value |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cascade_verif_then_length | target_only | 600 | 0.17% | [-4.50%, 4.67%] | 100 | 99 | 401 | 1.0000 |
| speculative | target_only | 600 | 3.33% | [-1.50%, 8.17%] | 125 | 105 | 370 | 0.2102 |
| jointadaspec | target_only | 600 | 0.17% | [-4.50%, 4.67%] | 100 | 99 | 401 | 1.0000 |
