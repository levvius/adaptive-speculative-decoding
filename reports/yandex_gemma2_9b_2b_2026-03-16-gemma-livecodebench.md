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
| baseline | - | - | 49.83 | 30.08 | 1.657 |
| speculative | - | - | 30.08 | 30.08 | 1.000 |
| autojudge | 0.005 | - | 25.34 | 30.08 | 0.842 |
| autojudge | 0.010 | - | 25.37 | 30.08 | 0.843 |
| autojudge | 0.030 | - | 25.39 | 30.08 | 0.844 |
| autojudge | 0.050 | - | 25.40 | 30.08 | 0.844 |
| autojudge | 0.090 | - | 26.22 | 30.08 | 0.872 |
| autojudge | 0.140 | - | 27.01 | 30.08 | 0.898 |
| autojudge | 0.230 | - | 28.17 | 30.08 | 0.936 |
| autojudge | 1.000 | - | 29.90 | 30.08 | 0.994 |
| topk | 16 | - | 35.61 | 30.08 | 1.184 |
| topk | 2 | - | 33.47 | 30.08 | 1.113 |
| topk | 32 | - | 35.68 | 30.08 | 1.186 |
| topk | 4 | - | 35.05 | 30.08 | 1.165 |
| topk | 8 | - | 35.52 | 30.08 | 1.181 |
| topk | all | - | 35.69 | 30.08 | 1.187 |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.005 | - | 25.34 | 30.08 | 0.842 |
| autojudge | 0.010 | - | 25.37 | 30.08 | 0.843 |
| autojudge | 0.030 | - | 25.39 | 30.08 | 0.844 |
| autojudge | 0.050 | - | 25.40 | 30.08 | 0.844 |
| autojudge | 0.090 | - | 26.22 | 30.08 | 0.872 |
| autojudge | 0.140 | - | 27.01 | 30.08 | 0.898 |
| autojudge | 0.230 | - | 28.17 | 30.08 | 0.936 |
| autojudge | 1.000 | - | 29.90 | 30.08 | 0.994 |


## Top-K sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| topk | 16 | - | 35.61 | 30.08 | 1.184 |
| topk | 2 | - | 33.47 | 30.08 | 1.113 |
| topk | 32 | - | 35.68 | 30.08 | 1.186 |
| topk | 4 | - | 35.05 | 30.08 | 1.165 |
| topk | 8 | - | 35.52 | 30.08 | 1.181 |
| topk | all | - | 35.69 | 30.08 | 1.187 |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.