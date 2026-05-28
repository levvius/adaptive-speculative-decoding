# JointAdaSpec Quality Analysis

- Benchmark: `outputs/jointadaspec_qwen7b_1p5b_quality_lock_2026-05-14/03_bench_gsm8k/benchmark.csv`
- Primary method: `cascade_verif_then_length`
- Baseline: `target_only`
- Primary success: `False`
- Strong success: `False`

## Method Summary

| Method | Runs | tok/s mean | Acceptance | GSM8K EM |
|---|---:|---:|---:|---:|
| cascade_verif_then_length | 1500 | 15.7619 | 92.11% | 57.93% |
| jointadaspec | 1500 | 15.7678 | 91.73% | 58.73% |
| speculative | 1500 | 13.1250 | 71.52% | 60.60% |
| target_only | 1500 | 24.9324 | 0.00% | 60.20% |

## Paired EM Comparisons

| Method | Baseline | Paired n | EM diff | 95% CI | Wins | Losses | Ties | p-value |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cascade_verif_then_length | target_only | 1500 | -2.27% | [-5.27%, 0.60%] | 238 | 272 | 990 | 0.1439 |
| speculative | target_only | 1500 | 0.40% | [-2.60%, 3.47%] | 275 | 269 | 956 | 0.8303 |
| jointadaspec | target_only | 1500 | -1.47% | [-4.53%, 1.67%] | 262 | 284 | 954 | 0.3688 |
