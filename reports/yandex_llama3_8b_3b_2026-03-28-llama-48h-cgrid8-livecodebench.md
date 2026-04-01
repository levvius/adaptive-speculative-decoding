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
| baseline | - | - | 71.52 | 34.80 | 2.055 |
| speculative | - | - | 34.80 | 34.80 | 1.000 |
| autojudge | 0.005 | - | 23.02 | 34.80 | 0.661 |
| autojudge | 0.010 | - | 23.03 | 34.80 | 0.662 |
| autojudge | 0.030 | - | 23.35 | 34.80 | 0.671 |
| autojudge | 0.050 | - | 23.83 | 34.80 | 0.685 |
| autojudge | 0.090 | - | 25.13 | 34.80 | 0.722 |
| autojudge | 0.140 | - | 26.70 | 34.80 | 0.767 |
| autojudge | 0.230 | - | 28.20 | 34.80 | 0.810 |
| autojudge | 1.000 | - | 29.27 | 34.80 | 0.841 |
| topk | 16 | - | 36.17 | 34.80 | 1.039 |
| topk | 2 | - | 33.45 | 34.80 | 0.961 |
| topk | 32 | - | 36.28 | 34.80 | 1.042 |
| topk | 4 | - | 35.41 | 34.80 | 1.017 |
| topk | 8 | - | 35.96 | 34.80 | 1.033 |
| topk | all | - | 36.53 | 34.80 | 1.050 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | - | 23.02 | 34.80 | 0.661 |
| autojudge | 0.010 | - | 23.03 | 34.80 | 0.662 |
| autojudge | 0.030 | - | 23.35 | 34.80 | 0.671 |
| autojudge | 0.050 | - | 23.83 | 34.80 | 0.685 |
| autojudge | 0.090 | - | 25.13 | 34.80 | 0.722 |
| autojudge | 0.140 | - | 26.70 | 34.80 | 0.767 |
| autojudge | 0.230 | - | 28.20 | 34.80 | 0.810 |
| autojudge | 1.000 | - | 29.27 | 34.80 | 0.841 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | - | 36.17 | 34.80 | 1.039 |
| topk | 2 | - | 33.45 | 34.80 | 0.961 |
| topk | 32 | - | 36.28 | 34.80 | 1.042 |
| topk | 4 | - | 35.41 | 34.80 | 1.017 |
| topk | 8 | - | 35.96 | 34.80 | 1.033 |
| topk | all | - | 36.53 | 34.80 | 1.050 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.