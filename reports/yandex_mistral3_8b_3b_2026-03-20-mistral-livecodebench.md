# Yandex-Style Report

## Environment snapshot

```json
{
  "generated_at": "2026-03-20T23:55:57.218456",
  "platform": "Linux-6.17.0-19-generic-x86_64-with-glibc2.39",
  "python": "Python 3.12.3",
  "kernel": "Linux robot-To-Be-Filled-By-O-E-M 6.17.0-19-generic #19~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC Fri Mar  6 23:08:46 UTC 2 x86_64 x86_64 x86_64 GNU/Linux",
  "git_sha": "9571a26ae24690ca10bc071854d98b9647391a6d",
  "git_branch": "main",
  "nvidia_smi": "NVIDIA GeForce RTX 5090, 590.48.01, 32607 MiB, 31250 MiB",
  "torch_version": "2.9.1+cu128",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "cuda_device_name": "NVIDIA GeForce RTX 5090"
}
```

## Results

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| baseline | - | - | 63.80 | 31.58 | 2.020 |
| speculative | - | - | 31.58 | 31.58 | 1.000 |
| autojudge | 0.005 | - | 16.40 | 31.58 | 0.519 |
| autojudge | 0.010 | - | 16.40 | 31.58 | 0.519 |
| autojudge | 0.030 | - | 16.42 | 31.58 | 0.520 |
| autojudge | 0.050 | - | 16.42 | 31.58 | 0.520 |
| autojudge | 0.090 | - | 16.42 | 31.58 | 0.520 |
| autojudge | 0.140 | - | 16.46 | 31.58 | 0.521 |
| autojudge | 0.230 | - | 16.73 | 31.58 | 0.530 |
| autojudge | 1.000 | - | 20.21 | 31.58 | 0.640 |
| topk | 16 | - | 25.51 | 31.58 | 0.808 |
| topk | 2 | - | 24.12 | 31.58 | 0.764 |
| topk | 32 | - | 25.51 | 31.58 | 0.808 |
| topk | 4 | - | 25.37 | 31.58 | 0.803 |
| topk | 8 | - | 25.52 | 31.58 | 0.808 |
| topk | all | - | 25.53 | 31.58 | 0.808 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | - | 16.40 | 31.58 | 0.519 |
| autojudge | 0.010 | - | 16.40 | 31.58 | 0.519 |
| autojudge | 0.030 | - | 16.42 | 31.58 | 0.520 |
| autojudge | 0.050 | - | 16.42 | 31.58 | 0.520 |
| autojudge | 0.090 | - | 16.42 | 31.58 | 0.520 |
| autojudge | 0.140 | - | 16.46 | 31.58 | 0.521 |
| autojudge | 0.230 | - | 16.73 | 31.58 | 0.530 |
| autojudge | 1.000 | - | 20.21 | 31.58 | 0.640 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | - | 25.51 | 31.58 | 0.808 |
| topk | 2 | - | 24.12 | 31.58 | 0.764 |
| topk | 32 | - | 25.51 | 31.58 | 0.808 |
| topk | 4 | - | 25.37 | 31.58 | 0.803 |
| topk | 8 | - | 25.52 | 31.58 | 0.808 |
| topk | all | - | 25.53 | 31.58 | 0.808 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.