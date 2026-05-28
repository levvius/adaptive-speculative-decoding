# JointAdaSpec Quality Analysis

- Benchmark: `outputs/jointadaspec_qwen7b_1p5b_quality_tri_2026-05-20/03_bench_gsm8k/benchmark.csv`
- Primary method: `cascade_verif_then_length`
- Baseline: `target_only`
- Primary success: `False`
- Strong success: `False`

## Method Summary

| Method | Runs | tok/s mean | Acceptance | GSM8K EM |
|---|---:|---:|---:|---:|
| cascade_verif_then_length | 1500 | 15.7132 | 92.14% | 58.00% |
| jointadaspec | 1500 | 15.7147 | 91.78% | 57.93% |
| speculative | 1500 | 10.1410 | 57.21% | 59.07% |
| target_only | 1500 | 24.9330 | 0.00% | 60.47% |

## Paired EM Comparisons

| Method | Baseline | Paired n | EM diff | 95% CI | Wins | Losses | Ties | p-value |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cascade_verif_then_length | target_only | 1500 | -2.47% | [-5.27%, 0.47%] | 221 | 258 | 1021 | 0.0999 |
| speculative | target_only | 1500 | -1.40% | [-4.33%, 1.53%] | 243 | 264 | 993 | 0.3744 |
| jointadaspec | target_only | 1500 | -2.53% | [-5.33%, 0.40%] | 226 | 264 | 1010 | 0.0945 |
