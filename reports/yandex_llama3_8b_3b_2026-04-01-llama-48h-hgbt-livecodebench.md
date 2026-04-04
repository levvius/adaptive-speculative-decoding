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
| baseline | - | - | 71.52 | 34.82 | 2.054 |
| speculative | - | - | 34.82 | 34.82 | 1.000 |
| autojudge | 0.005 | - | 22.78 | 34.82 | 0.654 |
| autojudge | 0.010 | - | 22.80 | 34.82 | 0.655 |
| autojudge | 0.030 | - | 22.81 | 34.82 | 0.655 |
| autojudge | 0.050 | - | 23.25 | 34.82 | 0.668 |
| autojudge | 0.090 | - | 24.73 | 34.82 | 0.710 |
| autojudge | 0.140 | - | 26.61 | 34.82 | 0.764 |
| autojudge | 0.230 | - | 28.38 | 34.82 | 0.815 |
| autojudge | 1.000 | - | 28.91 | 34.82 | 0.830 |
| topk | 16 | - | 36.18 | 34.82 | 1.039 |
| topk | 2 | - | 33.47 | 34.82 | 0.961 |
| topk | 32 | - | 36.32 | 34.82 | 1.043 |
| topk | 4 | - | 35.44 | 34.82 | 1.018 |
| topk | 8 | - | 35.99 | 34.82 | 1.034 |
| topk | all | - | 36.54 | 34.82 | 1.049 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | - | 22.78 | 34.82 | 0.654 |
| autojudge | 0.010 | - | 22.80 | 34.82 | 0.655 |
| autojudge | 0.030 | - | 22.81 | 34.82 | 0.655 |
| autojudge | 0.050 | - | 23.25 | 34.82 | 0.668 |
| autojudge | 0.090 | - | 24.73 | 34.82 | 0.710 |
| autojudge | 0.140 | - | 26.61 | 34.82 | 0.764 |
| autojudge | 0.230 | - | 28.38 | 34.82 | 0.815 |
| autojudge | 1.000 | - | 28.91 | 34.82 | 0.830 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | - | 36.18 | 34.82 | 1.039 |
| topk | 2 | - | 33.47 | 34.82 | 0.961 |
| topk | 32 | - | 36.32 | 34.82 | 1.043 |
| topk | 4 | - | 35.44 | 34.82 | 1.018 |
| topk | 8 | - | 35.99 | 34.82 | 1.034 |
| topk | all | - | 36.54 | 34.82 | 1.049 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.