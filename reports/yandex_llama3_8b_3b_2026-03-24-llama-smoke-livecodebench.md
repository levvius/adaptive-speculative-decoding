# Yandex-Style Report

## Environment snapshot

```json
{
  "generated_at": "2026-03-24T19:06:37.733487",
  "platform": "Linux-6.17.0-19-generic-x86_64-with-glibc2.39",
  "python": "Python 3.12.3",
  "kernel": "Linux robot-To-Be-Filled-By-O-E-M 6.17.0-19-generic #19~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC Fri Mar  6 23:08:46 UTC 2 x86_64 x86_64 x86_64 GNU/Linux",
  "git_sha": "865d45fdc963e078ca97abf2512d4f06f0b60c7c",
  "git_branch": "main",
  "nvidia_smi": "NVIDIA GeForce RTX 5090, 590.48.01, 32607 MiB, 30746 MiB",
  "torch_version": "2.9.1+cu128",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "cuda_device_name": "NVIDIA GeForce RTX 5090"
}
```

## Results

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| baseline | - | - | 66.18 | 31.13 | 2.126 |
| speculative | - | - | 31.13 | 31.13 | 1.000 |
| autojudge | 0.294 | - | 23.71 | 31.13 | 0.762 |
| topk | 4 | - | 34.24 | 31.13 | 1.100 |
| specexec | - | - | 1.86 | 31.13 | 0.060 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.294 | - | 23.71 | 31.13 | 0.762 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 4 | - | 34.24 | 31.13 | 1.100 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.