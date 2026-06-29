# Qwen3.5 9B->2B Smoke Quality Report

Status: **pipeline validation only**. Do not quote this file as a benchmark
result.

This report records the 2026-06-29 smoke that proved the Qwen3.5 9B->2B path
can load under `.venv-qwen35` (`transformers 5.12.1`) and complete
trace->solve->condition->benchmark with strict JSONL validation. The scope was
only `10` traces and `5` GSM8K prompts x `1` seed with `128` max new tokens.

The powered protocol is documented in `docs/QWEN35_POWERED_RUN_2026-06-29.md`
and must be used for any quality/speed claim.

## Smoke Artifacts

- Benchmark: `outputs/jointadaspec_qwen35_9b_2b_2026-06-29/03_bench_gsm8k/benchmark.csv`
- Results JSONL: `outputs/jointadaspec_qwen35_9b_2b_2026-06-29/03_bench_gsm8k/results.jsonl`
- Conditions: `reports/qwen35_9b_2b_2026-06-29/conditions.json`

## Diagnostic Method Summary

These numbers are useful only for sanity-checking that rows were produced.

| Method | Runs | tok/s mean | Acceptance | GSM8K EM |
|---|---:|---:|---:|---:|
| cascade_verif_then_length | 5 | 7.3232 | 90.22% | 20.00% |
| jointadaspec | 5 | 7.3365 | 90.22% | 20.00% |
| speculative | 5 | 8.5703 | 75.14% | 20.00% |
| target_only | 5 | 9.4298 | 0.00% | 0.00% |

## Non-Evidence Paired Diagnostics

The generated analyzer saw a positive smoke delta because one of five prompts
flipped relative to `target_only`. At `n=5`, this is not statistical evidence.

| Method | Baseline | Paired n | Clusters | EM diff | Cluster 95% CI | Wins | Losses | Ties | McNemar p | Cluster p | Cluster Holm p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jointadaspec | target_only | 5 | 5 | 20.00% | [0.00%, 60.00%] | 1 | 0 | 4 | 1.0000 | 0.5887 | 1.0000 |
| speculative | target_only | 5 | 5 | 20.00% | [0.00%, 60.00%] | 1 | 0 | 4 | 1.0000 | 0.5983 | 1.0000 |
| cascade_verif_then_length | target_only | 5 | 5 | 20.00% | [0.00%, 60.00%] | 1 | 0 | 4 | 1.0000 | 0.5922 | 1.0000 |
