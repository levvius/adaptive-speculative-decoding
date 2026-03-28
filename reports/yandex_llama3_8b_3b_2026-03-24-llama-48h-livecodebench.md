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
| baseline | - | - | 71.34 | 34.69 | 2.056 |
| speculative | - | - | 34.69 | 34.69 | 1.000 |
| autojudge | 0.005 | - | 22.95 | 34.69 | 0.662 |
| autojudge | 0.010 | - | 22.97 | 34.69 | 0.662 |
| autojudge | 0.030 | - | 23.28 | 34.69 | 0.671 |
| autojudge | 0.050 | - | 23.75 | 34.69 | 0.685 |
| autojudge | 0.090 | - | 25.06 | 34.69 | 0.722 |
| autojudge | 0.140 | - | 26.61 | 34.69 | 0.767 |
| autojudge | 0.230 | - | 28.11 | 34.69 | 0.810 |
| autojudge | 1.000 | - | 29.17 | 34.69 | 0.841 |
| topk | 16 | - | 36.09 | 34.69 | 1.040 |
| topk | 2 | - | 33.41 | 34.69 | 0.963 |
| topk | 32 | - | 36.21 | 34.69 | 1.044 |
| topk | 4 | - | 35.35 | 34.69 | 1.019 |
| topk | 8 | - | 35.90 | 34.69 | 1.035 |
| topk | all | - | 36.44 | 34.69 | 1.050 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | - | 22.95 | 34.69 | 0.662 |
| autojudge | 0.010 | - | 22.97 | 34.69 | 0.662 |
| autojudge | 0.030 | - | 23.28 | 34.69 | 0.671 |
| autojudge | 0.050 | - | 23.75 | 34.69 | 0.685 |
| autojudge | 0.090 | - | 25.06 | 34.69 | 0.722 |
| autojudge | 0.140 | - | 26.61 | 34.69 | 0.767 |
| autojudge | 0.230 | - | 28.11 | 34.69 | 0.810 |
| autojudge | 1.000 | - | 29.17 | 34.69 | 0.841 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | - | 36.09 | 34.69 | 1.040 |
| topk | 2 | - | 33.41 | 34.69 | 0.963 |
| topk | 32 | - | 36.21 | 34.69 | 1.044 |
| topk | 4 | - | 35.35 | 34.69 | 1.019 |
| topk | 8 | - | 35.90 | 34.69 | 1.035 |
| topk | all | - | 36.44 | 34.69 | 1.050 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.