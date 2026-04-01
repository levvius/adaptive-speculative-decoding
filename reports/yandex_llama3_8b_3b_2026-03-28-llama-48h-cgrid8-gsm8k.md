# Yandex-Style Report

## Environment snapshot

```json
{
  "generated_at": "2026-03-28T16:27:09.620429",
  "platform": "Linux-6.17.0-19-generic-x86_64-with-glibc2.39",
  "python": "Python 3.12.3",
  "kernel": "Linux robot-To-Be-Filled-By-O-E-M 6.17.0-19-generic #19~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC Fri Mar  6 23:08:46 UTC 2 x86_64 x86_64 x86_64 GNU/Linux",
  "git_sha": "694e39747cc2864d60e93eb5688e23e7c62898ee",
  "git_branch": "main",
  "nvidia_smi": "NVIDIA GeForce RTX 5090, 590.48.01, 32607 MiB, 31492 MiB",
  "torch_version": "2.9.1+cu128",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "cuda_device_name": "NVIDIA GeForce RTX 5090"
}
```

## Results

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| baseline | - | 70.89 | 72.68 | 40.68 | 1.787 |
| speculative | - | 71.89 | 40.68 | 40.68 | 1.000 |
| autojudge | 0.005 | 74.67 | 40.46 | 40.68 | 0.995 |
| autojudge | 0.010 | 74.67 | 40.51 | 40.68 | 0.996 |
| autojudge | 0.030 | 76.67 | 41.36 | 40.68 | 1.017 |
| autojudge | 0.050 | 77.33 | 42.28 | 40.68 | 1.039 |
| autojudge | 0.090 | 76.33 | 44.15 | 40.68 | 1.085 |
| autojudge | 0.140 | 78.67 | 45.98 | 40.68 | 1.130 |
| autojudge | 0.230 | 77.00 | 47.66 | 40.68 | 1.172 |
| autojudge | 1.000 | 75.67 | 49.27 | 40.68 | 1.211 |
| topk | 16 | 76.00 | 59.28 | 40.68 | 1.457 |
| topk | 2 | 73.33 | 56.27 | 40.68 | 1.383 |
| topk | 32 | 76.33 | 59.26 | 40.68 | 1.457 |
| topk | 4 | 73.67 | 58.60 | 40.68 | 1.441 |
| topk | 8 | 76.33 | 59.18 | 40.68 | 1.455 |
| topk | all | 75.67 | 59.29 | 40.68 | 1.458 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | 74.67 | 40.46 | 40.68 | 0.995 |
| autojudge | 0.010 | 74.67 | 40.51 | 40.68 | 0.996 |
| autojudge | 0.030 | 76.67 | 41.36 | 40.68 | 1.017 |
| autojudge | 0.050 | 77.33 | 42.28 | 40.68 | 1.039 |
| autojudge | 0.090 | 76.33 | 44.15 | 40.68 | 1.085 |
| autojudge | 0.140 | 78.67 | 45.98 | 40.68 | 1.130 |
| autojudge | 0.230 | 77.00 | 47.66 | 40.68 | 1.172 |
| autojudge | 1.000 | 75.67 | 49.27 | 40.68 | 1.211 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | 76.00 | 59.28 | 40.68 | 1.457 |
| topk | 2 | 73.33 | 56.27 | 40.68 | 1.383 |
| topk | 32 | 76.33 | 59.26 | 40.68 | 1.457 |
| topk | 4 | 73.67 | 58.60 | 40.68 | 1.441 |
| topk | 8 | 76.33 | 59.18 | 40.68 | 1.455 |
| topk | all | 75.67 | 59.29 | 40.68 | 1.458 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.