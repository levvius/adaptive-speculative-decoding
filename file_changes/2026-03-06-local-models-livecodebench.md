# 2026-03-06: Local Models + LiveCodeBench

## Summary

Added support for local pre-downloaded Qwen2.5 7B/1.5B model experiments, LiveCodeBench as an evaluation dataset, and Yandex-style report generation.

## Model Pair

- **Target**: `models/qwen2.5-7b-Instruct-model` (~14GB bfloat16)
- **Draft**: `models/qwen2.5-1.5b-Instruct-model` (~3GB bfloat16)
- **GPU**: RTX 5090, 32GB VRAM (sufficient for both models simultaneously)
- **Tokenizer**: Confirmed identical tokenizer files in both model directories

## Files Changed

### `.gitignore`
- Added `models/*` and `!models/.gitkeep` rules

### `models/.gitkeep`
- New placeholder file for the gitignored `models/` directory

### `configs/models.json`
- Added `qwen25_7b_instruct_local` preset (local path, bfloat16, cuda)
- Added `qwen25_1p5b_instruct_local` preset (local path, bfloat16, cuda, uses 7B tokenizer for compatibility)

### `configs/experiments.json`
- Added `qwen25_7b_local_target_1p5b_local_speculative_k4`
- Added `qwen25_7b_local_target_1p5b_local_autojudge_k4`
- Added `qwen25_7b_local_target_1p5b_local_topk_k4`
- Added `qwen25_7b_local_target_1p5b_local_all_paper`
- Added `qwen25_7b_local_baseline`

### `sp_samp/livecodebench.py` (new)
- `load_livecodebench(path, max_samples)`: loads prompts from local JSONL
- `download_livecodebench(output_path, version_tag)`: downloads from HF hub via `datasets` library
- Follows pattern of `gsm8k.py` and `mtbench.py`
- Supports fallback keys: `prompt`, `question`, `input`, `text`, `instruction`

### `sp_samp/__init__.py`
- Added `load_livecodebench` to imports and `__all__`

### `benchmarks/bench_speculative.py`
- Added `livecodebench` import
- Extended `--eval-task` choices to `["mtbench", "gsm8k", "livecodebench"]`
- Added `elif args.eval_task == "livecodebench":` branch for prompt loading (throughput-only, no reference_answer)

### `sp_samp/cli.py`
- Extended `--eval-task` choices to include `"livecodebench"`

### `scripts/report_yandex_style.py` (new)
- Generates tables in AutoJudge / Yandex Research format:
  `| method | threshold | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |`
- For GSM8K: accuracy = exact-match mean * 100
- For LiveCodeBench: accuracy = "-" (throughput-only)
- Outputs `.md`, `.csv`, `.json`

### `scripts/run_local_7b_1p5b_eval.sh` (new)
- Orchestrates full local evaluation:
  1. Downloads GSM8K and LiveCodeBench datasets if missing
  2. Runs GSM8K sweep: baseline, speculative, AutoJudge thresholds, Top-K grid
  3. Runs LiveCodeBench sweep: same method set
  4. Generates Yandex-style reports for both datasets

### `Makefile`
- Added `local-eval` target and supporting variables

### `requirements.txt`
- Added `datasets>=2.18.0` (needed for LiveCodeBench HF hub download)

### `tests/test_livecodebench.py` (new)
- Tests: basic loading, max_samples, empty file, missing prompt skip, JSON array, alternative keys

### Documentation
- `README.MD`: Added local model workflow and LiveCodeBench sections
- `CLAUDE.md`: Updated architecture, configs, scripts, tests, and constraints sections
- `CODEX.MD`: Added 2026-03-06 change section and new file entries

## AutoJudge Checkpoint Note

The existing checkpoint (`autojudge_qwen25_0p5b_to_3b.pt`) is for the old 0.5B→3B pair. A new checkpoint will be automatically trained for 1.5B→7B on first run (hidden dims: 1536+3584=5120 features).
