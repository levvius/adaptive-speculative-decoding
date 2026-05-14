# JointAdaSpec Quality Analysis

- Benchmark: `outputs/jointadaspec_qwen7b_1p5b_quality_heldout_combined_2026-05-11/benchmark.csv`
- Primary method: `cascade_verif_then_length`
- Baseline: `target_only`
- Primary success: `False`
- Strong success: `False`

## Method Summary

| Method | Runs | tok/s mean | Acceptance | GSM8K EM |
|---|---:|---:|---:|---:|
| cascade_verif_then_length | 3000 | 15.7402 | 92.12% | 57.97% |
| jointadaspec | 3000 | 15.7398 | 91.76% | 58.33% |
| speculative | 3000 | 13.0852 | 71.40% | 59.53% |
| target_only | 3000 | 25.1869 | 0.00% | 60.33% |

## Paired EM Comparisons

| Method | Baseline | Paired n | EM diff | 95% CI | Wins | Losses | Ties | p-value |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cascade_verif_then_length | target_only | 3000 | -2.37% | [-4.40%, -0.37%] | 459 | 530 | 2011 | 0.0260 |
| speculative | target_only | 3000 | -0.80% | [-2.90%, 1.33%] | 507 | 531 | 1962 | 0.4753 |
| jointadaspec | target_only | 3000 | -2.00% | [-4.10%, 0.07%] | 488 | 548 | 1964 | 0.0667 |
