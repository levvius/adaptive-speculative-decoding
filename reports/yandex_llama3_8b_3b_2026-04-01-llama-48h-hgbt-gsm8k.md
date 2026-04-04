# Yandex-Style Report

## Environment snapshot

```json
{
  "generated_at": "2026-04-01T22:27:00.040119",
  "platform": "Linux-6.17.0-19-generic-x86_64-with-glibc2.39",
  "python": "Python 3.12.3",
  "kernel": "Linux robot-To-Be-Filled-By-O-E-M 6.17.0-19-generic #19~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC Fri Mar  6 23:08:46 UTC 2 x86_64 x86_64 x86_64 GNU/Linux",
  "git_sha": "7a5112ddead657b153c1b27b00d602c998e62433",
  "git_branch": "main",
  "nvidia_smi": "NVIDIA GeForce RTX 5090, 590.48.01, 32607 MiB, 31528 MiB",
  "torch_version": "2.9.1+cu128",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "cuda_device_name": "NVIDIA GeForce RTX 5090"
}
```

## Results

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| baseline | - | 70.89 | 72.72 | 40.68 | 1.787 |
| speculative | - | 71.89 | 40.68 | 40.68 | 1.000 |
| autojudge | 0.005 | 74.67 | 40.18 | 40.68 | 0.988 |
| autojudge | 0.010 | 74.67 | 40.17 | 40.68 | 0.987 |
| autojudge | 0.030 | 74.00 | 40.25 | 40.68 | 0.989 |
| autojudge | 0.050 | 73.33 | 41.05 | 40.68 | 1.009 |
| autojudge | 0.090 | 74.33 | 43.13 | 40.68 | 1.060 |
| autojudge | 0.140 | 73.33 | 45.34 | 40.68 | 1.114 |
| autojudge | 0.230 | 73.00 | 47.91 | 40.68 | 1.178 |
| autojudge | 1.000 | 75.67 | 48.91 | 40.68 | 1.202 |
| topk | 16 | 76.00 | 59.27 | 40.68 | 1.457 |
| topk | 2 | 73.33 | 56.30 | 40.68 | 1.384 |
| topk | 32 | 76.33 | 59.27 | 40.68 | 1.457 |
| topk | 4 | 73.67 | 58.62 | 40.68 | 1.441 |
| topk | 8 | 76.33 | 59.19 | 40.68 | 1.455 |
| topk | all | 75.67 | 59.30 | 40.68 | 1.458 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | 74.67 | 40.18 | 40.68 | 0.988 |
| autojudge | 0.010 | 74.67 | 40.17 | 40.68 | 0.987 |
| autojudge | 0.030 | 74.00 | 40.25 | 40.68 | 0.989 |
| autojudge | 0.050 | 73.33 | 41.05 | 40.68 | 1.009 |
| autojudge | 0.090 | 74.33 | 43.13 | 40.68 | 1.060 |
| autojudge | 0.140 | 73.33 | 45.34 | 40.68 | 1.114 |
| autojudge | 0.230 | 73.00 | 47.91 | 40.68 | 1.178 |
| autojudge | 1.000 | 75.67 | 48.91 | 40.68 | 1.202 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | 76.00 | 59.27 | 40.68 | 1.457 |
| topk | 2 | 73.33 | 56.30 | 40.68 | 1.384 |
| topk | 32 | 76.33 | 59.27 | 40.68 | 1.457 |
| topk | 4 | 73.67 | 58.62 | 40.68 | 1.441 |
| topk | 8 | 76.33 | 59.19 | 40.68 | 1.455 |
| topk | all | 75.67 | 59.30 | 40.68 | 1.458 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.