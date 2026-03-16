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
| baseline | - | 58.11 | 78.56 | 47.16 | 1.666 |
| speculative | - | 56.89 | 47.16 | 47.16 | 1.000 |
| autojudge | 0.005 | 61.67 | 51.55 | 47.16 | 1.093 |
| autojudge | 0.010 | 61.67 | 54.50 | 47.16 | 1.156 |
| autojudge | 0.030 | 61.33 | 54.85 | 47.16 | 1.163 |
| autojudge | 0.050 | 61.67 | 54.73 | 47.16 | 1.160 |
| autojudge | 0.090 | 61.67 | 55.85 | 47.16 | 1.184 |
| autojudge | 0.140 | 59.67 | 56.35 | 47.16 | 1.195 |
| autojudge | 0.230 | 59.67 | 57.84 | 47.16 | 1.226 |
| autojudge | 1.000 | 52.33 | 63.38 | 47.16 | 1.344 |
| topk | - | 53.67 | 68.62 | 47.16 | 1.455 |
| topk | - | 54.00 | 71.53 | 47.16 | 1.517 |
| topk | - | 53.00 | 68.39 | 47.16 | 1.450 |
| topk | - | 54.33 | 71.53 | 47.16 | 1.517 |
| topk | - | 54.33 | 68.47 | 47.16 | 1.452 |
| topk | - | 52.33 | 68.47 | 47.16 | 1.452 |


## AutoJudge threshold sweep

| method | threshold | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | 61.67 | 51.55 | 47.16 | 1.093 |
| autojudge | 0.010 | 61.67 | 54.50 | 47.16 | 1.156 |
| autojudge | 0.030 | 61.33 | 54.85 | 47.16 | 1.163 |
| autojudge | 0.050 | 61.67 | 54.73 | 47.16 | 1.160 |
| autojudge | 0.090 | 61.67 | 55.85 | 47.16 | 1.184 |
| autojudge | 0.140 | 59.67 | 56.35 | 47.16 | 1.195 |
| autojudge | 0.230 | 59.67 | 57.84 | 47.16 | 1.226 |
| autojudge | 1.000 | 52.33 | 63.38 | 47.16 | 1.344 |


## Top-K sweep

| method | threshold | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | - | 53.67 | 68.62 | 47.16 | 1.455 |
| topk | - | 54.00 | 71.53 | 47.16 | 1.517 |
| topk | - | 53.00 | 68.39 | 47.16 | 1.450 |
| topk | - | 54.33 | 71.53 | 47.16 | 1.517 |
| topk | - | 54.33 | 68.47 | 47.16 | 1.452 |
| topk | - | 52.33 | 68.47 | 47.16 | 1.452 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.