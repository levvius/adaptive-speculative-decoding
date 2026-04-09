# Yandex-Style Report

## Environment snapshot

Manifest not provided.

## Results

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.352 | 5.00 | 47.70 | - | - |
| consensus_autojudge | 0.500/d1_only | 0.00 | 50.28 | - | - |
| consensus_autojudge | 0.500/no_esc | 0.00 | 46.62 | - | - |
| consensus_autojudge | 0.500 | 5.00 | 50.15 | - | - |
| consensus_autojudge | 0.650 | 5.00 | 52.41 | - | - |
| consensus_autojudge | 0.800 | 5.00 | 52.65 | - | - |
| consensus_autojudge | rule | 0.00 | 51.51 | - | - |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.352 | 5.00 | 47.70 | - | - |


## Consensus AutoJudge sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| consensus_autojudge | 0.500/d1_only | 0.00 | 50.28 | - | - |
| consensus_autojudge | 0.500/no_esc | 0.00 | 46.62 | - | - |
| consensus_autojudge | 0.500 | 5.00 | 50.15 | - | - |
| consensus_autojudge | 0.650 | 5.00 | 52.41 | - | - |
| consensus_autojudge | 0.800 | 5.00 | 52.65 | - | - |
| consensus_autojudge | rule | 0.00 | 51.51 | - | - |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.