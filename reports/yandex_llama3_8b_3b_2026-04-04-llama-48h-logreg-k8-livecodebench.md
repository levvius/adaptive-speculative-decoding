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
| baseline | - | - | 69.23 | 30.68 | 2.257 |
| speculative | - | - | 30.68 | 30.68 | 1.000 |
| autojudge | 0.005 | - | 28.30 | 30.68 | 0.922 |
| autojudge | 0.010 | - | 28.38 | 30.68 | 0.925 |
| autojudge | 0.030 | - | 29.04 | 30.68 | 0.947 |
| autojudge | 0.050 | - | 30.26 | 30.68 | 0.986 |
| autojudge | 0.090 | - | 33.52 | 30.68 | 1.093 |
| autojudge | 0.140 | - | 37.69 | 30.68 | 1.229 |
| autojudge | 0.230 | - | 41.29 | 30.68 | 1.346 |
| autojudge | 1.000 | - | 44.84 | 30.68 | 1.462 |
| topk | 16 | - | 52.40 | 30.68 | 1.708 |
| topk | 2 | - | 45.44 | 30.68 | 1.481 |
| topk | 32 | - | 52.63 | 30.68 | 1.716 |
| topk | 4 | - | 50.77 | 30.68 | 1.655 |
| topk | 8 | - | 52.35 | 30.68 | 1.706 |
| topk | all | - | 53.84 | 30.68 | 1.755 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | - | 28.30 | 30.68 | 0.922 |
| autojudge | 0.010 | - | 28.38 | 30.68 | 0.925 |
| autojudge | 0.030 | - | 29.04 | 30.68 | 0.947 |
| autojudge | 0.050 | - | 30.26 | 30.68 | 0.986 |
| autojudge | 0.090 | - | 33.52 | 30.68 | 1.093 |
| autojudge | 0.140 | - | 37.69 | 30.68 | 1.229 |
| autojudge | 0.230 | - | 41.29 | 30.68 | 1.346 |
| autojudge | 1.000 | - | 44.84 | 30.68 | 1.462 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | - | 52.40 | 30.68 | 1.708 |
| topk | 2 | - | 45.44 | 30.68 | 1.481 |
| topk | 32 | - | 52.63 | 30.68 | 1.716 |
| topk | 4 | - | 50.77 | 30.68 | 1.655 |
| topk | 8 | - | 52.35 | 30.68 | 1.706 |
| topk | all | - | 53.84 | 30.68 | 1.755 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.