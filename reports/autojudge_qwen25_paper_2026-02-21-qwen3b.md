# AutoJudge Paper-Style Report

## Environment snapshot

```json
{
  "generated_at": "2026-02-21T08:01:18.935384",
  "platform": "Linux-6.17.0-14-generic-x86_64-with-glibc2.39",
  "python": "Python 3.12.3",
  "kernel": "Linux robot-To-Be-Filled-By-O-E-M 6.17.0-14-generic #14~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Jan 15 15:52:10 UTC 2 x86_64 x86_64 x86_64 GNU/Linux",
  "git_sha": "700c04f5cd604a21ac1618896f44a65a3f8ee1c6",
  "git_branch": "main",
  "nvidia_smi": "NVIDIA GeForce RTX 5090, 590.48.01, 32607 MiB, 31289 MiB",
  "torch_version": null,
  "cuda_runtime": null,
  "cuda_available": null,
  "cuda_device_name": null
}
```

## Aggregated metrics

| Method | Setting | Runs | tok/s median | avg tok/step median | GSM8K EM mean | Speedup vs speculative | Accuracy delta vs speculative |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | default | 3 | 116.558 | 1.000 | 0.3478 | 2.490 | -0.0056 |
| speculative | default | 3 | 46.810 | 2.245 | 0.3533 | 1.000 | 0.0000 |
| autojudge | threshold=0.005 | 3 | 46.297 | 2.507 | 0.3933 | 0.989 | 0.0400 |
| autojudge | threshold=0.01 | 3 | 46.533 | 2.507 | 0.3933 | 0.994 | 0.0400 |
| autojudge | threshold=0.03 | 3 | 46.322 | 2.507 | 0.3933 | 0.990 | 0.0400 |
| autojudge | threshold=0.05 | 3 | 46.271 | 2.506 | 0.3933 | 0.989 | 0.0400 |
| autojudge | threshold=0.09 | 3 | 46.321 | 2.512 | 0.3900 | 0.990 | 0.0367 |
| autojudge | threshold=0.14 | 3 | 46.521 | 2.524 | 0.3933 | 0.994 | 0.0400 |
| autojudge | threshold=0.23 | 3 | 48.005 | 2.605 | 0.3833 | 1.026 | 0.0300 |
| autojudge | threshold=1 | 3 | 92.203 | 4.927 | 0.0167 | 1.970 | -0.3367 |
| topk | topk_rank=16 | 3 | 92.392 | 4.387 | 0.0300 | 1.974 | -0.3233 |
| topk | topk_rank=2 | 3 | 65.061 | 3.103 | 0.1233 | 1.390 | -0.2300 |
| topk | topk_rank=32 | 3 | 97.348 | 4.650 | 0.0133 | 2.080 | -0.3400 |
| topk | topk_rank=4 | 3 | 76.126 | 3.625 | 0.0300 | 1.626 | -0.3233 |
| topk | topk_rank=8 | 3 | 85.157 | 4.057 | 0.0167 | 1.819 | -0.3367 |
| topk | topk_rank=all | 3 | 104.766 | 4.927 | 0.0167 | 2.238 | -0.3367 |


## AutoJudge threshold sweep

| Method | Setting | Runs | tok/s median | avg tok/step median | GSM8K EM mean | Speedup vs speculative | Accuracy delta vs speculative |
|---|---|---:|---:|---:|---:|---:|---:|
| autojudge | threshold=0.005 | 3 | 46.297 | 2.507 | 0.3933 | 0.989 | 0.0400 |
| autojudge | threshold=0.01 | 3 | 46.533 | 2.507 | 0.3933 | 0.994 | 0.0400 |
| autojudge | threshold=0.03 | 3 | 46.322 | 2.507 | 0.3933 | 0.990 | 0.0400 |
| autojudge | threshold=0.05 | 3 | 46.271 | 2.506 | 0.3933 | 0.989 | 0.0400 |
| autojudge | threshold=0.09 | 3 | 46.321 | 2.512 | 0.3900 | 0.990 | 0.0367 |
| autojudge | threshold=0.14 | 3 | 46.521 | 2.524 | 0.3933 | 0.994 | 0.0400 |
| autojudge | threshold=0.23 | 3 | 48.005 | 2.605 | 0.3833 | 1.026 | 0.0300 |
| autojudge | threshold=1 | 3 | 92.203 | 4.927 | 0.0167 | 1.970 | -0.3367 |


## Top-K sweep

| Method | Setting | Runs | tok/s median | avg tok/step median | GSM8K EM mean | Speedup vs speculative | Accuracy delta vs speculative |
|---|---|---:|---:|---:|---:|---:|---:|
| topk | topk_rank=16 | 3 | 92.392 | 4.387 | 0.0300 | 1.974 | -0.3233 |
| topk | topk_rank=2 | 3 | 65.061 | 3.103 | 0.1233 | 1.390 | -0.2300 |
| topk | topk_rank=32 | 3 | 97.348 | 4.650 | 0.0133 | 2.080 | -0.3400 |
| topk | topk_rank=4 | 3 | 76.126 | 3.625 | 0.0300 | 1.626 | -0.3233 |
| topk | topk_rank=8 | 3 | 85.157 | 4.057 | 0.0167 | 1.819 | -0.3367 |
| topk | topk_rank=all | 3 | 104.766 | 4.927 | 0.0167 | 2.238 | -0.3367 |


## Pareto-like shortlist

| Method | Setting | Runs | tok/s median | avg tok/step median | GSM8K EM mean | Speedup vs speculative | Accuracy delta vs speculative |
|---|---|---:|---:|---:|---:|---:|---:|
| autojudge | threshold=0.01 | 3 | 46.533 | 2.507 | 0.3933 | 0.994 | 0.0400 |
| autojudge | threshold=0.23 | 3 | 48.005 | 2.605 | 0.3833 | 1.026 | 0.0300 |
| baseline | default | 3 | 116.558 | 1.000 | 0.3478 | 2.490 | -0.0056 |


## Notes

- `speedup_vs_speculative` is computed from median tokens/sec.
- `accuracy_delta_vs_speculative` is computed from mean GSM8K exact-match over run records.