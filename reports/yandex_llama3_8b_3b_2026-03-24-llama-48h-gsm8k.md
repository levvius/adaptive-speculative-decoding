# Yandex-Style Report

## Environment snapshot

```json
{
  "generated_at": "2026-03-24T20:56:20.743617",
  "platform": "Linux-6.17.0-19-generic-x86_64-with-glibc2.39",
  "python": "Python 3.12.3",
  "kernel": "Linux robot-To-Be-Filled-By-O-E-M 6.17.0-19-generic #19~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC Fri Mar  6 23:08:46 UTC 2 x86_64 x86_64 x86_64 GNU/Linux",
  "git_sha": "4a7b46c00e4322a571c46238615b7e9e7caac6f4",
  "git_branch": "main",
  "nvidia_smi": "NVIDIA GeForce RTX 5090, 590.48.01, 32607 MiB, 30868 MiB",
  "torch_version": "2.9.1+cu128",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "cuda_device_name": "NVIDIA GeForce RTX 5090"
}
```

## Results

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| baseline | - | 70.89 | 72.63 | 40.60 | 1.789 |
| speculative | - | 71.89 | 40.60 | 40.60 | 1.000 |
| autojudge | 0.005 | 74.67 | 40.43 | 40.60 | 0.996 |
| autojudge | 0.010 | 74.67 | 40.48 | 40.60 | 0.997 |
| autojudge | 0.030 | 76.67 | 41.30 | 40.60 | 1.017 |
| autojudge | 0.050 | 77.33 | 42.23 | 40.60 | 1.040 |
| autojudge | 0.090 | 76.33 | 44.11 | 40.60 | 1.087 |
| autojudge | 0.140 | 78.67 | 45.92 | 40.60 | 1.131 |
| autojudge | 0.230 | 77.00 | 47.60 | 40.60 | 1.172 |
| autojudge | 1.000 | 75.67 | 49.20 | 40.60 | 1.212 |
| topk | 16 | 76.00 | 59.22 | 40.60 | 1.459 |
| topk | 2 | 73.33 | 56.21 | 40.60 | 1.385 |
| topk | 32 | 76.33 | 59.20 | 40.60 | 1.458 |
| topk | 4 | 73.67 | 58.55 | 40.60 | 1.442 |
| topk | 8 | 76.33 | 59.12 | 40.60 | 1.456 |
| topk | all | 75.67 | 59.24 | 40.60 | 1.459 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | 74.67 | 40.43 | 40.60 | 0.996 |
| autojudge | 0.010 | 74.67 | 40.48 | 40.60 | 0.997 |
| autojudge | 0.030 | 76.67 | 41.30 | 40.60 | 1.017 |
| autojudge | 0.050 | 77.33 | 42.23 | 40.60 | 1.040 |
| autojudge | 0.090 | 76.33 | 44.11 | 40.60 | 1.087 |
| autojudge | 0.140 | 78.67 | 45.92 | 40.60 | 1.131 |
| autojudge | 0.230 | 77.00 | 47.60 | 40.60 | 1.172 |
| autojudge | 1.000 | 75.67 | 49.20 | 40.60 | 1.212 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | 76.00 | 59.22 | 40.60 | 1.459 |
| topk | 2 | 73.33 | 56.21 | 40.60 | 1.385 |
| topk | 32 | 76.33 | 59.20 | 40.60 | 1.458 |
| topk | 4 | 73.67 | 58.55 | 40.60 | 1.442 |
| topk | 8 | 76.33 | 59.12 | 40.60 | 1.456 |
| topk | all | 75.67 | 59.24 | 40.60 | 1.459 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.