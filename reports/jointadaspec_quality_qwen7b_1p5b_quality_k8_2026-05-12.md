# JointAdaSpec Quality Analysis

- Benchmark: `outputs/jointadaspec_qwen7b_1p5b_quality_k8_2026-05-12/03_bench_gsm8k/benchmark.csv`
- Primary method: `cascade_verif_then_length`
- Baseline: `target_only`
- Primary success: `True`
- Strong success: `False`

## Method Summary

| Method | Runs | tok/s mean | Acceptance | GSM8K EM |
|---|---:|---:|---:|---:|
| cascade_verif_then_length | 600 | 15.7161 | 92.02% | 61.33% |
| jointadaspec | 600 | 15.7250 | 91.56% | 62.33% |
| speculative | 600 | 10.0886 | 56.78% | 61.17% |
| target_only | 600 | 24.8645 | 0.00% | 58.83% |

## Paired EM Comparisons

| Method | Baseline | Paired n | EM diff | 95% CI | Wins | Losses | Ties | p-value |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cascade_verif_then_length | target_only | 600 | 2.50% | [-2.33%, 7.33%] | 115 | 100 | 385 | 0.3397 |
| speculative | target_only | 600 | 2.33% | [-2.50%, 7.17%] | 116 | 102 | 382 | 0.3786 |
| jointadaspec | target_only | 600 | 3.50% | [-1.33%, 8.33%] | 119 | 98 | 383 | 0.1744 |
