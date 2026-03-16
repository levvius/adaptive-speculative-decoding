# Yandex-Style Report

## Environment snapshot

```json
{
  "generated_at": "2026-03-10T19:30:16.423041",
  "platform": "Linux-6.17.0-14-generic-x86_64-with-glibc2.39",
  "python": "Python 3.12.3",
  "kernel": "Linux robot-To-Be-Filled-By-O-E-M 6.17.0-14-generic #14~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Jan 15 15:52:10 UTC 2 x86_64 x86_64 x86_64 GNU/Linux",
  "git_sha": "b719f0953c44208f02cd501a559ba9b45a4711ef",
  "git_branch": "main",
  "nvidia_smi": "NVIDIA GeForce RTX 5090, 590.48.01, 32607 MiB, 31027 MiB",
  "torch_version": null,
  "cuda_runtime": null,
  "cuda_available": null,
  "cuda_device_name": null
}
```

## Results

| method | threshold | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| baseline | - | - | 76.41 | 41.41 | 1.845 |
| speculative | - | - | 41.41 | 41.41 | 1.000 |
| autojudge | 0.005 | - | 28.23 | 41.41 | 0.682 |
| autojudge | 0.010 | - | 28.25 | 41.41 | 0.682 |
| autojudge | 0.030 | - | 28.22 | 41.41 | 0.681 |
| autojudge | 0.050 | - | 28.16 | 41.41 | 0.680 |
| autojudge | 0.090 | - | 28.52 | 41.41 | 0.689 |
| autojudge | 0.140 | - | 28.60 | 41.41 | 0.691 |
| autojudge | 0.230 | - | 28.78 | 41.41 | 0.695 |
| autojudge | 1.000 | - | 37.13 | 41.41 | 0.897 |
| topk | - | - | 43.85 | 41.41 | 1.059 |
| topk | - | - | 39.92 | 41.41 | 0.964 |
| topk | - | - | 43.87 | 41.41 | 1.059 |
| topk | - | - | 42.83 | 41.41 | 1.034 |
| topk | - | - | 43.62 | 41.41 | 1.053 |
| topk | - | - | 44.02 | 41.41 | 1.063 |


## AutoJudge threshold sweep

| method | threshold | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | - | 28.23 | 41.41 | 0.682 |
| autojudge | 0.010 | - | 28.25 | 41.41 | 0.682 |
| autojudge | 0.030 | - | 28.22 | 41.41 | 0.681 |
| autojudge | 0.050 | - | 28.16 | 41.41 | 0.680 |
| autojudge | 0.090 | - | 28.52 | 41.41 | 0.689 |
| autojudge | 0.140 | - | 28.60 | 41.41 | 0.691 |
| autojudge | 0.230 | - | 28.78 | 41.41 | 0.695 |
| autojudge | 1.000 | - | 37.13 | 41.41 | 0.897 |


## Top-K sweep

| method | threshold | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | - | - | 43.85 | 41.41 | 1.059 |
| topk | - | - | 39.92 | 41.41 | 0.964 |
| topk | - | - | 43.87 | 41.41 | 1.059 |
| topk | - | - | 42.83 | 41.41 | 1.034 |
| topk | - | - | 43.62 | 41.41 | 1.053 |
| topk | - | - | 44.02 | 41.41 | 1.063 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.