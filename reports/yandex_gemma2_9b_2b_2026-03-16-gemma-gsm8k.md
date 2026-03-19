# Yandex-Style Report

## Environment snapshot

```json
{
  "generated_at": "2026-03-16T20:21:52.017717",
  "platform": "Linux-6.17.0-14-generic-x86_64-with-glibc2.39",
  "python": "Python 3.12.3",
  "kernel": "Linux robot-To-Be-Filled-By-O-E-M 6.17.0-14-generic #14~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Jan 15 15:52:10 UTC 2 x86_64 x86_64 x86_64 GNU/Linux",
  "git_sha": "3351ae8d069a4c8f7e59293ab7e38cf6ab334fc3",
  "git_branch": "main",
  "nvidia_smi": "NVIDIA GeForce RTX 5090, 590.48.01, 32607 MiB, 31735 MiB",
  "torch_version": "2.9.1+cu128",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "cuda_device_name": "NVIDIA GeForce RTX 5090"
}
```

## Results

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| baseline | - | 85.33 | 48.85 | 30.99 | 1.576 |
| speculative | - | 86.00 | 30.99 | 30.99 | 1.000 |
| autojudge | 0.005 | 81.00 | 37.52 | 30.99 | 1.211 |
| autojudge | 0.010 | 81.00 | 41.12 | 30.99 | 1.327 |
| autojudge | 0.030 | 82.00 | 42.45 | 30.99 | 1.370 |
| autojudge | 0.050 | 80.00 | 43.58 | 30.99 | 1.406 |
| autojudge | 0.090 | 83.00 | 45.16 | 30.99 | 1.457 |
| autojudge | 0.140 | 80.00 | 45.81 | 30.99 | 1.478 |
| autojudge | 0.230 | 81.00 | 46.71 | 30.99 | 1.507 |
| autojudge | 1.000 | 62.00 | 48.53 | 30.99 | 1.566 |
| topk | 16 | 65.00 | 56.26 | 30.99 | 1.816 |
| topk | 2 | 85.00 | 52.20 | 30.99 | 1.684 |
| topk | 32 | 63.00 | 56.32 | 30.99 | 1.817 |
| topk | 4 | 77.00 | 55.23 | 30.99 | 1.782 |
| topk | 8 | 68.00 | 56.01 | 30.99 | 1.808 |
| topk | all | 62.00 | 56.40 | 30.99 | 1.820 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | 81.00 | 37.52 | 30.99 | 1.211 |
| autojudge | 0.010 | 81.00 | 41.12 | 30.99 | 1.327 |
| autojudge | 0.030 | 82.00 | 42.45 | 30.99 | 1.370 |
| autojudge | 0.050 | 80.00 | 43.58 | 30.99 | 1.406 |
| autojudge | 0.090 | 83.00 | 45.16 | 30.99 | 1.457 |
| autojudge | 0.140 | 80.00 | 45.81 | 30.99 | 1.478 |
| autojudge | 0.230 | 81.00 | 46.71 | 30.99 | 1.507 |
| autojudge | 1.000 | 62.00 | 48.53 | 30.99 | 1.566 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | 65.00 | 56.26 | 30.99 | 1.816 |
| topk | 2 | 85.00 | 52.20 | 30.99 | 1.684 |
| topk | 32 | 63.00 | 56.32 | 30.99 | 1.817 |
| topk | 4 | 77.00 | 55.23 | 30.99 | 1.782 |
| topk | 8 | 68.00 | 56.01 | 30.99 | 1.808 |
| topk | all | 62.00 | 56.40 | 30.99 | 1.820 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.