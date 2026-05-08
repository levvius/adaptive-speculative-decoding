# JointAdaSpec Quality Analysis

- Benchmark: `outputs/jointadaspec_qwen7b_1p5b_quality_2026-05-05/03_bench_gsm8k/benchmark.csv`
- Primary method: `cascade_verif_then_length`
- Baseline: `target_only`
- Primary success: `True`
- Strong success: `False`

## Method Summary

| Method | Runs | tok/s mean | Acceptance | GSM8K EM |
|---|---:|---:|---:|---:|
| adaptive_length | 300 | 14.2996 | 78.95% | 59.00% |
| cascade_length_then_verif | 300 | 15.7922 | 92.09% | 58.33% |
| cascade_verif_then_length | 300 | 15.7996 | 91.91% | 62.67% |
| fuzzy_sd_T1 | 300 | 13.2290 | 71.92% | 62.00% |
| fuzzy_sd_T1.5 | 300 | 14.1283 | 78.53% | 58.33% |
| fuzzy_sd_T2 | 300 | 14.4750 | 81.13% | 58.00% |
| fuzzy_sd_T2.5 | 300 | 14.6372 | 82.33% | 62.00% |
| fuzzy_sd_T3 | 300 | 14.8174 | 83.66% | 57.33% |
| fuzzy_sd_T3.5 | 300 | 14.9188 | 84.37% | 58.67% |
| fuzzy_sd_T4 | 300 | 15.0058 | 85.01% | 57.00% |
| jointadaspec | 300 | 15.7920 | 91.70% | 61.67% |
| speculative | 300 | 13.2283 | 71.92% | 62.00% |
| target_only | 300 | 25.0077 | 0.00% | 57.00% |

## Paired EM Comparisons

| Method | Baseline | Paired n | EM diff | 95% CI | Wins | Losses | Ties | p-value |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cascade_verif_then_length | target_only | 300 | 5.67% | [-0.67%, 12.00%] | 56 | 39 | 205 | 0.1002 |
| speculative | target_only | 300 | 5.00% | [-1.67%, 12.00%] | 64 | 49 | 187 | 0.1876 |
| jointadaspec | target_only | 300 | 4.67% | [-2.00%, 11.33%] | 60 | 46 | 194 | 0.2065 |
