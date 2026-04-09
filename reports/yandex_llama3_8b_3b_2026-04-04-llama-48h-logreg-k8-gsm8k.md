# Yandex-Style Report

## Environment snapshot

```json
{
  "generated_at": "2026-04-04T13:45:05.201805",
  "platform": "Linux-6.17.0-19-generic-x86_64-with-glibc2.39",
  "python": "Python 3.12.3",
  "kernel": "Linux robot-To-Be-Filled-By-O-E-M 6.17.0-19-generic #19~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC Fri Mar  6 23:08:46 UTC 2 x86_64 x86_64 x86_64 GNU/Linux",
  "git_sha": "40fc797da4c8b0059e84b6d04bbe8944afb39e6b",
  "git_branch": "main",
  "nvidia_smi": "NVIDIA GeForce RTX 5090, 590.48.01, 32607 MiB, 31155 MiB",
  "torch_version": "2.9.1+cu128",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "cuda_device_name": "NVIDIA GeForce RTX 5090"
}
```

## Results

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| baseline | - | 70.89 | 70.58 | 36.62 | 1.927 |
| speculative | - | 71.44 | 36.62 | 36.62 | 1.000 |
| autojudge | 0.005 | 74.33 | 47.56 | 36.62 | 1.299 |
| autojudge | 0.010 | 74.33 | 47.82 | 36.62 | 1.306 |
| autojudge | 0.030 | 76.00 | 48.97 | 36.62 | 1.337 |
| autojudge | 0.050 | 77.33 | 51.09 | 36.62 | 1.395 |
| autojudge | 0.090 | 76.33 | 55.15 | 36.62 | 1.506 |
| autojudge | 0.140 | 72.00 | 59.50 | 36.62 | 1.625 |
| autojudge | 0.230 | 73.00 | 64.45 | 36.62 | 1.760 |
| autojudge | 1.000 | 72.33 | 68.85 | 36.62 | 1.880 |
| topk | 16 | 72.00 | 79.06 | 36.62 | 2.159 |
| topk | 2 | 74.67 | 70.75 | 36.62 | 1.932 |
| topk | 32 | 72.33 | 78.79 | 36.62 | 2.151 |
| topk | 4 | 71.00 | 77.44 | 36.62 | 2.115 |
| topk | 8 | 71.67 | 78.78 | 36.62 | 2.151 |
| topk | all | 72.33 | 78.55 | 36.62 | 2.145 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | 74.33 | 47.56 | 36.62 | 1.299 |
| autojudge | 0.010 | 74.33 | 47.82 | 36.62 | 1.306 |
| autojudge | 0.030 | 76.00 | 48.97 | 36.62 | 1.337 |
| autojudge | 0.050 | 77.33 | 51.09 | 36.62 | 1.395 |
| autojudge | 0.090 | 76.33 | 55.15 | 36.62 | 1.506 |
| autojudge | 0.140 | 72.00 | 59.50 | 36.62 | 1.625 |
| autojudge | 0.230 | 73.00 | 64.45 | 36.62 | 1.760 |
| autojudge | 1.000 | 72.33 | 68.85 | 36.62 | 1.880 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | 72.00 | 79.06 | 36.62 | 2.159 |
| topk | 2 | 74.67 | 70.75 | 36.62 | 1.932 |
| topk | 32 | 72.33 | 78.79 | 36.62 | 2.151 |
| topk | 4 | 71.00 | 77.44 | 36.62 | 2.115 |
| topk | 8 | 71.67 | 78.78 | 36.62 | 2.151 |
| topk | all | 72.33 | 78.55 | 36.62 | 2.145 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.