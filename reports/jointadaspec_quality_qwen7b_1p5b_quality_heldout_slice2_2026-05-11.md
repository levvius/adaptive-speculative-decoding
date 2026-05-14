# JointAdaSpec Quality Analysis

- Benchmark: `outputs/jointadaspec_qwen7b_1p5b_quality_heldout_2026-05-11-slice2/03_bench_gsm8k/benchmark.csv`
- Primary method: `cascade_verif_then_length`
- Baseline: `target_only`
- Primary success: `False`
- Strong success: `False`

## Method Summary

| Method | Runs | tok/s mean | Acceptance | GSM8K EM |
|---|---:|---:|---:|---:|
| cascade_verif_then_length | 1500 | 15.7009 | 92.14% | 58.00% |
| jointadaspec | 1500 | 15.6899 | 91.78% | 57.93% |
| speculative | 1500 | 13.0397 | 71.28% | 58.47% |
| target_only | 1500 | 24.8908 | 0.00% | 60.47% |

## Paired EM Comparisons

| Method | Baseline | Paired n | EM diff | 95% CI | Wins | Losses | Ties | p-value |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cascade_verif_then_length | target_only | 1500 | -2.47% | [-5.27%, 0.47%] | 221 | 258 | 1021 | 0.0999 |
| speculative | target_only | 1500 | -2.00% | [-4.87%, 0.73%] | 232 | 262 | 1006 | 0.1919 |
| jointadaspec | target_only | 1500 | -2.53% | [-5.33%, 0.40%] | 226 | 264 | 1010 | 0.0945 |
