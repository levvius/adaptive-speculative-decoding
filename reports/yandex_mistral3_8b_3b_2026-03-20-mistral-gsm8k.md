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
| baseline | - | 63.67 | 64.31 | 34.88 | 1.844 |
| speculative | - | 63.00 | 34.88 | 34.88 | 1.000 |
| autojudge | 0.005 | 61.00 | 21.21 | 34.88 | 0.608 |
| autojudge | 0.010 | 61.00 | 21.23 | 34.88 | 0.608 |
| autojudge | 0.030 | 60.00 | 21.24 | 34.88 | 0.609 |
| autojudge | 0.050 | 61.00 | 21.40 | 34.88 | 0.614 |
| autojudge | 0.090 | 63.00 | 21.75 | 34.88 | 0.624 |
| autojudge | 0.140 | 65.00 | 22.08 | 34.88 | 0.633 |
| autojudge | 0.230 | 63.00 | 22.59 | 34.88 | 0.648 |
| autojudge | 1.000 | 63.00 | 24.17 | 34.88 | 0.693 |
| topk | 16 | 63.00 | 29.89 | 34.88 | 0.857 |
| topk | 2 | 63.00 | 29.02 | 34.88 | 0.832 |
| topk | 32 | 63.00 | 29.88 | 34.88 | 0.857 |
| topk | 4 | 64.00 | 29.70 | 34.88 | 0.851 |
| topk | 8 | 63.00 | 29.85 | 34.88 | 0.856 |
| topk | all | 63.00 | 29.86 | 34.88 | 0.856 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | 61.00 | 21.21 | 34.88 | 0.608 |
| autojudge | 0.010 | 61.00 | 21.23 | 34.88 | 0.608 |
| autojudge | 0.030 | 60.00 | 21.24 | 34.88 | 0.609 |
| autojudge | 0.050 | 61.00 | 21.40 | 34.88 | 0.614 |
| autojudge | 0.090 | 63.00 | 21.75 | 34.88 | 0.624 |
| autojudge | 0.140 | 65.00 | 22.08 | 34.88 | 0.633 |
| autojudge | 0.230 | 63.00 | 22.59 | 34.88 | 0.648 |
| autojudge | 1.000 | 63.00 | 24.17 | 34.88 | 0.693 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | 63.00 | 29.89 | 34.88 | 0.857 |
| topk | 2 | 63.00 | 29.02 | 34.88 | 0.832 |
| topk | 32 | 63.00 | 29.88 | 34.88 | 0.857 |
| topk | 4 | 64.00 | 29.70 | 34.88 | 0.851 |
| topk | 8 | 63.00 | 29.85 | 34.88 | 0.856 |
| topk | all | 63.00 | 29.86 | 34.88 | 0.856 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.