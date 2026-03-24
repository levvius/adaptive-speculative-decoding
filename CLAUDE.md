# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A speculative decoding research playground implementing and benchmarking several LLM inference acceleration methods:
- **Speculative Sampling (SpS)**: exact decoding using a small draft model + large target model
- **AutoJudge**: paper-aligned judge decoding (Algorithm 1 label mining + LogisticRegression classifier)
- **Top-K**: lossy baseline for paper-style comparisons
- **SpecExec**: exact target sampling with draft-branch KV cache prefill and pruning

## Common Commands

```bash
# Install dependencies (CPU)
make setup
# Install with GPU extras (bitsandbytes, accelerate)
make setup-gpu

# Syntax check + config validation
make check

# Run all tests
make test
# Run a single test file
.venv/bin/python -m pytest tests/test_sampling.py -q
# Run a single test by name
.venv/bin/python -m pytest tests/test_autojudge.py::test_gsm8k_parsing -q

# Quick toy benchmark (no HF models, fast)
make bench-toy

# Quick HF smoke run (downloads tiny model)
make smoke-hf

# List all preset experiments/models/methods
make list-presets

# Validate config consistency
make validate-configs

# Validate benchmark JSONL output schema
make validate-results RESULTS=datasets/results.jsonl

# Run paper-style GSM8K sweep and generate reports
make paper-eval

# Run local Qwen2.5 7B/1.5B eval (GSM8K + LiveCodeBench) with Yandex-style reports
make local-eval
```

## Long-Run Operations (24-48h)

Canonical long-run mode in this repo is `tmux + staged AutoJudge` (not `nohup` as default).

### 1) GPU preflight

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
```

If another compute PID is active, stop it first (or wait for script preflight gate).

### 2) Start a persistent tmux session

```bash
tmux new -s aj48h
cd /home/robot/Project/adaptive-speculative-decoding
mkdir -p logs datasets reports
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Detach: `Ctrl-b d`  
Reattach: `tmux attach -t aj48h`

### 3) Stage A: AutoJudge checkpoint bootstrap (separate tmp output)

This stage trains/validates `datasets/autojudge_qwen25_1p5b_to_7b.pt` without mixing records into the final report JSONL.

```bash
.venv/bin/python -m sp_samp.cli bench \
  --config-dir configs \
  --experiment qwen25_7b_local_target_1p5b_local_autojudge_k4 \
  --method autojudge \
  --eval-task gsm8k \
  --gsm8k-eval-mode zero_shot_cot \
  --dataset datasets/gsm8k_test.jsonl \
  --autojudge-train-dataset datasets/gsm8k_train.jsonl \
  --autojudge-train-samples 4000 \
  --autojudge-checkpoint datasets/autojudge_qwen25_1p5b_to_7b.pt \
  --autojudge-threshold 0.005 \
  --runs 1 \
  --max-samples 1 \
  --max-new-tokens 256 \
  --k 4 \
  --require-headless \
  --out datasets/results_autojudge_bootstrap_tmp.jsonl
```

### 4) Stage B: Main AutoJudge + Top-K sweep (final JSONL)

```bash
DATE_TAG="$(date +%F)"
.venv/bin/python scripts/write_run_manifest.py \
  --out "reports/local_7b_1p5b_run_manifest_${DATE_TAG}.json"

OUT_GSM8K=datasets/results_local_7b_1p5b_gsm8k.jsonl \
CHECKPOINT_PATH=datasets/autojudge_qwen25_1p5b_to_7b.pt \
MAX_SAMPLES=100 \
RUNS=3 \
DATE_TAG="${DATE_TAG}" \
bash scripts/run_autojudge_topk_gsm8k_bg.sh \
  | tee -a "logs/aj_topk_${DATE_TAG}.log"
```

### 5) Monitoring

```bash
tail -f logs/aj_topk_$(date +%F).log
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader
wc -l datasets/results_local_7b_1p5b_gsm8k.jsonl
```

### 6) Emergency stop / recovery

```bash
# Stop repo-related long runs first.
pkill -f "run_autojudge_topk_gsm8k_bg.sh|sp_samp\\.cli|bench_speculative"

# If GPU memory is still occupied, kill remaining compute-app PIDs.
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do
  kill -9 "$p" 2>/dev/null || true
done

nvidia-smi
```

### 7) Post-run validation and report generation

```bash
DATE_TAG="$(date +%F)"
.venv/bin/python scripts/validate_results_jsonl.py \
  --path datasets/results_local_7b_1p5b_gsm8k.jsonl \
  --strict

.venv/bin/python scripts/report_yandex_style.py \
  --input datasets/results_local_7b_1p5b_gsm8k.jsonl \
  --eval-task gsm8k \
  --manifest "reports/local_7b_1p5b_run_manifest_${DATE_TAG}.json" \
  --out-prefix "reports/yandex_local_7b_1p5b_${DATE_TAG}-gsm8k"
```

### OOM/Race Conditions

- Run only one GPU-heavy job at a time (single-job rule).
- Reuse the same `OUT_GSM8K` file to benefit from benchmark resume mode (`resume_key` skip).
- Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for long sessions to reduce fragmentation-related OOM risk.
- Optional VRAM reduction for the sweep script (if quantized runtime is available):
  - `QUANT=8bit DRAFT_QUANT=8bit bash scripts/run_autojudge_topk_gsm8k_bg.sh`
  - Increase preflight gate if needed: `MIN_FREE_VRAM_MIB=22000`.

## Architecture

### Core Library (`sp_samp/`)

The library is layered — toy/CPU implementations first, HF-backed implementations second:

- **`models.py`**: Abstract `BaseModel` interface (`next_token_probs`) + toy implementations: `FixedModel`, `BigramModel`, `RandomModel`, `NoisyModel`
- **`sampling.py`**: Pure-Python reference implementations of `sample_baseline` and `speculative_sample` operating on `BaseModel`. Returns `SamplingStats`.
- **`specexec.py`**: CPU SpecExec reference implementation. Returns `SpecExecStats` (branch metrics).
- **`hf_adapter.py`**: `HFModel(BaseModel)` wraps HuggingFace causal LMs with KV cache (`KVCacheState`) and optional bitsandbytes quantization. Handles native-quantized checkpoint edge cases.
- **`hf_sampling.py`**: HF-backed `sample_baseline_hf` and `speculative_sample_hf` using `KVCacheState`.
- **`hf_specexec.py`**: HF SpecExec with KV-cache reuse along prefix-tree edges and depth-wise tree passes.
- **`hf_topk.py`**: HF Top-K lossy verification baseline. Returns `TopKStats`.
- **`autojudge.py`**: Paper-aligned AutoJudge — Algorithm 1 GSM8K label mining, `StandardScaler + LogisticRegression` classifier training with recall-target calibration, and inference at the speculative verification stage. Requires `scikit-learn`.
- **`gsm8k.py`**: GSM8K dataset loading and answer equivalence utilities used by AutoJudge.
- **`mtbench.py`**: MT-Bench dataset loader.
- **`livecodebench.py`**: LiveCodeBench dataset loader (JSONL) + HF hub downloader.
- **`cli.py`**: Unified CLI entrypoint (`python -m sp_samp.cli`). Subcommands: `bench`, `autojudge`, `specexec`, `list-presets`. Loads and applies JSON presets from `configs/`.
- **`__init__.py`**: Lazy/optional imports — HF and AutoJudge exports are skipped gracefully when `torch`/`transformers`/`scikit-learn` are absent.
- **`methods/`**: Method-facing re-exports (including SpecExec).

### Benchmark Runner (`benchmarks/bench_speculative.py`)

Single entry point for all method comparisons: `python -m benchmarks.bench_speculative`. Supports toy (no HF) and HF modes, resume mode (skips completed runs via `resume_key` in JSONL), per-run error persistence, system metadata tagging, GSM8K, MT-Bench, and LiveCodeBench eval modes.

### Configs (`configs/`)

All JSON, no code:
- `models.json` — HF model presets (model name, device, dtype, quantization, tokenizer); includes local model presets (`qwen25_7b_instruct_local`, `qwen25_1p5b_instruct_local`) for offline use
- `methods.json` — method presets (baseline, speculative, autojudge, topk, specexec, all, all_paper)
- `experiments.json` — target/draft pairing presets; current paper default pair is `Qwen2.5-0.5B-Instruct` → `Qwen2.5-3B-Instruct`; local 7B/1.5B presets available for offline experiments
- `method_templates.json` — AutoJudge and SpecExec parameter/metric templates

### Scripts (`scripts/`)

- `validate_configs.py` — cross-file config consistency + tokenizer compatibility checks
- `validate_results_jsonl.py` — strict JSONL schema validation for benchmark output
- `run_autojudge_paper_eval.sh` — orchestrates full paper-style GSM8K sweep
- `run_local_7b_1p5b_eval.sh` — orchestrates local Qwen2.5 7B/1.5B GSM8K + LiveCodeBench eval
- `run_gemma2_9b_2b_eval.sh` — thin wrapper over local eval script with Gemma-2 9B/2B defaults
- `run_mistral3_8b_3b_eval.sh` — thin wrapper over local eval script with Mistral-3 8B/3B defaults
- `report_autojudge_paper.py` — aggregates raw JSONL into `.md/.csv/.json` reports in `reports/`
- `report_yandex_style.py` — generates Yandex-style threshold/accuracy/speedup report tables
- `write_run_manifest.py` — writes environment manifest JSON for reproducibility
- `install_dependencies.sh` — idempotent host bootstrap (never modifies NVIDIA drivers)

### Tests (`tests/`)

Tests live at the top of `tests/` (not under `sp_samp/`). Coverage: `test_sampling.py` (baseline/speculative correctness), `test_autojudge.py` (GSM8K parsing, classifier calibration, mining), `test_specexec.py` (distribution correctness, exactness vs baseline), `test_topk.py` (mismatch accept/reject), `test_livecodebench.py` (JSONL parsing, max_samples, key fallbacks).

## Key Constraints

- **Tokenizer compatibility**: Draft and target models must share an identical vocabulary mapping for speculative, AutoJudge, Top-K, and SpecExec. The `Qwen2.5-0.5B → 7B` legacy pair violates this; use `0.5B → 3B` instead. The local `1.5B → 7B` pair has confirmed identical tokenizers.
- **Local models**: Pre-downloaded models go in `models/` (gitignored). Presets `qwen25_7b_instruct_local` and `qwen25_1p5b_instruct_local` point to `models/qwen2.5-7b-Instruct-model` and `models/qwen2.5-1.5b-Instruct-model`.
- **AutoJudge training**: Only valid with GSM8K-format datasets (`question` + `answer` fields). MT-Bench JSONL will fail fast with an actionable error.
- **AutoJudge checkpoint versioning**: Checkpoint format v2 (`autojudge_version=2`). Loading a v1 checkpoint triggers retraining.
- **Benchmark resume**: Re-running with the same `--out` file skips completed `resume_key` entries automatically.
- **GPU checks**: Use `make docker-gpu-check` / `make docker-gpu-check-image` before long runs. RTX 50xx (Blackwell/sm_120) requires `torch==2.9.1+cu128`.
- **Makefile Python**: Prefers `.venv/bin/python`; falls back to `python3`.

## Evaluation Results (Qwen2.5 7B/1.5B, RTX 5090, 2026-03-10)

### GSM8K (k=4, zero-shot CoT, 100 samples x 3 runs)

| Method | Accuracy | Speed | vs Speculative | vs Baseline |
|--------|----------|-------|---------------|-------------|
| Baseline (7B only) | 58.1% | 78.6 tok/s | 1.67x | 1.00x |
| Speculative | 56.9% | 47.2 tok/s | 1.00x | 0.60x |
| AutoJudge t=0.09 (best balanced) | 61.7% | 55.9 tok/s | 1.18x | 0.71x |
| AutoJudge t=1.0 (fastest AJ) | 52.3% | 63.4 tok/s | 1.34x | 0.81x |
| Top-K rank=4 (fastest overall) | 54.3% | 71.5 tok/s | 1.52x | 0.91x |

### LiveCodeBench (k=4, throughput-only)

| Method | Speed | vs Speculative |
|--------|-------|---------------|
| Baseline | 76.4 tok/s | 1.85x |
| Speculative | 41.4 tok/s | 1.00x |
| AutoJudge (all thresholds) | 28-37 tok/s | 0.68-0.90x |
| Top-K rank=all (best) | 44.0 tok/s | 1.06x |

### Known Limitations

- **Speculative slower than baseline**: Expected with small target/draft ratio (7B/1.5B = 4.7x) on single GPU. Paper uses 70B/8B+ ratios with multi-GPU. Draft model overhead not offset by acceptance gains at k=4.
- **AutoJudge slower on LiveCodeBench**: Classifier trained on GSM8K data doesn't generalize to code tasks. Paper trains separate classifiers per task (Section 4.2).
- **AutoJudge accuracy improvement (+3.5% over baseline on GSM8K)**: The judge correctly identifies important tokens, improving answer quality by selectively falling back to the target model.

### Planned Improvements

1. Increase draft window (k=8, k=16) — paper uses W=64
2. GPU-resident classifier (eliminate CPU roundtrip per mismatch)
3. Distributional features (entropy, KL divergence) for better classifier accuracy
4. Confidence-based early accept to skip judge on high-confidence drafts
5. Task-specific training for LiveCodeBench

## Paper Alignment

Source papers are in `papers/`. Audit summary (2026-02-25); full record in `file_changes/2026-02-25-paper-alignment.md`.

### Confirmed correct vs. papers
- AutoJudge C-grid: `_default_c_grid()` → `range(-7, 1)` = {10⁻⁷…10⁰} (8 values, matches Section 3.2). Previously `range(-7, 3)` included out-of-paper 10¹ and 10² — **fixed**.
- Label convention: 0=unimportant / 1=important matches Algorithm 1.
- `_threshold_for_recall` returns highest threshold where recall ≥ target (optimal).
- Final model retrained on full dataset after C-grid search.

### Known intentional deviations (documented; no code change)
- **AutoJudge mining — initial response from TARGET not DRAFT**: Algorithm 1 pseudocode says draft model; code uses target model to match the paper's mathematical definition of I(x). Documented with inline comment.
- **AutoJudge — greedy decoding**: Paper Appendix A describes Gumbel-max stochastic sampling; implementation uses argmax. Valid deterministic variant.
- **SpecExec — BFS vs. SSSP**: Paper uses modified Dijkstra with priority queue and log-prob budget K; code uses BFS level-by-level with `parallel_branches` / `branch_prune_threshold`. Correct distribution preserved; known simplification.
