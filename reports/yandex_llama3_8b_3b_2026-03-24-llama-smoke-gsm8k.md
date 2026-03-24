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
| baseline | - | 20.00 | 68.74 | 36.97 | 1.859 |
| speculative | - | 40.00 | 36.97 | 36.97 | 1.000 |
| autojudge | 0.294 | 40.00 | 43.84 | 36.97 | 1.186 |
| topk | 4 | 40.00 | 66.06 | 36.97 | 1.787 |
| specexec | - | 0.00 | 2.28 | 36.97 | 0.062 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.294 | 40.00 | 43.84 | 36.97 | 1.186 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 4 | 40.00 | 66.06 | 36.97 | 1.787 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.