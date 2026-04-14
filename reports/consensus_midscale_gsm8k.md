# Yandex-Style Report

## Environment snapshot

Manifest not provided.

## Results

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.291 | 0.00 | 51.22 | - | - |
| consensus_autojudge | 0.500/d1_only | 0.00 | 53.02 | - | - |
| consensus_autojudge | 0.500/no_esc | 2.00 | 42.79 | - | - |
| consensus_autojudge | 0.500 | 4.00 | 51.25 | - | - |
| consensus_autojudge | 0.650 | 2.00 | 52.44 | - | - |
| consensus_autojudge | 0.800 | 0.00 | 54.06 | - | - |
| consensus_autojudge | rule | 2.00 | 48.94 | - | - |


## AutoJudge threshold sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| autojudge | 0.291 | 0.00 | 51.22 | - | - |


## Consensus AutoJudge sweep

| method | parameter | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |
|---|---:|---:|---:|---:|---:|
| consensus_autojudge | 0.500/d1_only | 0.00 | 53.02 | - | - |
| consensus_autojudge | 0.500/no_esc | 2.00 | 42.79 | - | - |
| consensus_autojudge | 0.500 | 4.00 | 51.25 | - | - |
| consensus_autojudge | 0.650 | 2.00 | 52.44 | - | - |
| consensus_autojudge | 0.800 | 0.00 | 54.06 | - | - |
| consensus_autojudge | rule | 2.00 | 48.94 | - | - |


## Notes

- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).
- `speculative decoding` = reference speculative tok/s.
- `speedup (ours)` = row tok/s / speculative tok/s.