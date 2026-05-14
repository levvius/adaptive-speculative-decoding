# JointAdaSpec Quality Analysis

- Benchmark: `outputs/jointadaspec_qwen14b_0p5b_crosscheck_2026-05-14-final/03_bench_gsm8k/benchmark.csv`
- Primary method: `cascade_verif_then_length`
- Baseline: `target_only`
- Primary success: `True`
- Strong success: `False`

## Method Summary

| Method | Runs | tok/s mean | Acceptance | GSM8K EM |
|---|---:|---:|---:|---:|
| cascade_verif_then_length | 300 | 10.2014 | 54.78% | 56.33% |
| jointadaspec | 300 | 10.2093 | 55.12% | 58.00% |
| speculative | 300 | 4.4901 | 27.27% | 55.00% |
| target_only | 300 | 13.0044 | 0.00% | 49.33% |

## Paired EM Comparisons

| Method | Baseline | Paired n | EM diff | 95% CI | Wins | Losses | Ties | p-value |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cascade_verif_then_length | target_only | 300 | 7.00% | [-0.33%, 14.33%] | 76 | 55 | 169 | 0.0802 |
| speculative | target_only | 300 | 5.67% | [-1.67%, 12.67%] | 71 | 54 | 175 | 0.1521 |
| jointadaspec | target_only | 300 | 8.67% | [1.33%, 16.00%] | 76 | 50 | 174 | 0.0255 |
