# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

A speculative decoding research playground implementing and benchmarking several LLM inference acceleration methods:
- **Speculative Sampling (SpS)**: exact decoding using a small draft model + large target model
- **AutoJudge**: paper-aligned judge decoding (Algorithm 1 label mining + LogisticRegression classifier)
- **Top-K**: lossy baseline for paper-style comparisons
- **SpecExec**: exact target sampling with draft-branch KV cache prefill and pruning
- **JointAdaSpec**: thesis-focused joint MDP control over draft length and fuzzy verification threshold

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

# Build the defense deck in 3 formats (PPTX + PDF + HTML) into papers/dist/
make slides

# Validate config consistency
make validate-configs

# Validate benchmark JSONL output schema
make validate-results RESULTS=datasets/results.jsonl

# Run paper-style GSM8K sweep and generate reports
make paper-eval

# Run local Qwen2.5 7B/1.5B eval (GSM8K + LiveCodeBench) with Yandex-style reports
make local-eval

# Run local Llama-3 8B/3B eval (GSM8K + LiveCodeBench)
bash scripts/run_llama3_8b_3b_eval.sh

# Run JointAdaSpec staged pipeline
make jointadaspec-full MODEL_PAIR=qwen7b_1p5b

# Validate Qwen 7B/1.5B Step 2 before launching dependent full Step 3
bash scripts/run_step3_after_step2.sh
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
export PYTORCH_ALLOC_CONF=expandable_segments:True
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
- Keep `PYTORCH_ALLOC_CONF=expandable_segments:True` for long sessions to reduce fragmentation-related OOM risk.
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
- `01_collect_traces.py` — JointAdaSpec trace collection (Hydra pipeline)
- `02_solve_mdp.py` — JointAdaSpec joint + cascade policy solve stage
- `03_benchmark.py` — JointAdaSpec seeded benchmark runner with resume-safe JSONL + legacy prompt logs
- `04_verify_conditions.py` — empirical C1-C4 / N1-N2 verification + diagnostic plots
- `05_write_manifest.py` — writes JointAdaSpec reproducibility manifests
- `run_autojudge_paper_eval.sh` — orchestrates full paper-style GSM8K sweep
- `run_local_7b_1p5b_eval.sh` — orchestrates local Qwen2.5 7B/1.5B GSM8K + LiveCodeBench eval
- `run_gemma2_9b_2b_eval.sh` — thin wrapper over local eval script with Gemma-2 9B/2B defaults
- `run_mistral3_8b_3b_eval.sh` — thin wrapper over local eval script with Mistral-3 8B/3B defaults
- `run_llama3_8b_3b_eval.sh` — thin wrapper over local eval script with Llama-3 8B/3B defaults
- `report_autojudge_paper.py` — aggregates raw JSONL into `.md/.csv/.json` reports in `reports/`
- `report_yandex_style.py` — generates Yandex-style threshold/accuracy/speedup report tables
- `write_run_manifest.py` — writes environment manifest JSON for reproducibility
- `install_dependencies.sh` — idempotent host bootstrap (never modifies NVIDIA drivers)

### Tests (`tests/`)

Tests live at the top of `tests/` (not under `sp_samp/`). Coverage: `test_sampling.py` (baseline/speculative correctness), `test_autojudge.py` (GSM8K parsing, classifier calibration, mining), `test_specexec.py` (distribution correctness, exactness vs baseline), `test_topk.py` (mismatch accept/reject), `test_livecodebench.py` (JSONL parsing, max_samples, key fallbacks).

## JointAdaSpec

`jointadaspec/` is the thesis-specific stack for Joint Adaptive Speculative Decoding. It learns a joint policy over draft length and fuzzy verification threshold on a discretised `(H, K, k)` state space, benchmarks that policy against fixed and cascade baselines, and checks whether the empirical traces satisfy the monotonicity and supermodularity assumptions used in the thesis theorems.

Pipeline stages:
- `scripts/01_collect_traces.py` — collect trace parquet from held-out prompts.
- `scripts/02_solve_mdp.py` — estimate MDP parameters, solve the joint policy, and save both cascade baselines.
- `scripts/03_benchmark.py` — run seeded benchmarks and write `results.jsonl`, legacy `run.jsonl`, CSV summaries, and manifests.
- `scripts/04_verify_conditions.py` — compute C1-C4 / N1-N2 diagnostics and save six plots.
- `scripts/run_step3_after_step2.sh` — watcher that validates Qwen `7B -> 1.5B` Step 2 benchmark/report artifacts before launching the dependent Qwen `14B -> 0.5B` full Step 3 run.
- `reports/templates/*.py` — generate Pareto plots, threshold surfaces, and ablation charts from the saved artifacts.

JointAdaSpec constraints:
- State grid defaults to `N_H=20`, `N_K=20`, `gamma_max=8`; the condition checker reports over the first `min(gamma_max + 1, 4)` `k` slices for the supermodularity plots.
- Draft/target tokenizer compatibility is mandatory. Local Qwen `14B/0.5B` and `7B/1.5B` pairs are configured to share tokenizer files.
- Seed policy is deterministic by default: benchmark runs use `[42, 43, 44, ...]`, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and enable `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Confidence intervals are additive only: the seeded benchmark writes bootstrap 95% CIs for speed, acceptance rate, and GSM8K exact match without breaking the old prompt-level JSONL readers.
- Reproducibility manifests live under `reports/manifests/` and include git SHA, dirty flag, resolved config YAML, seed list, and SHA256 hashes for trace/policy artifacts.
- Smoke-sized runs, including the `2026-04-21` one-prompt smoke and the `2026-04-28` five-prompt Qwen `14B -> 0.5B` smoke, are tracked for pipeline validation only. Do not present them as final best-of-run evidence.

JointAdaSpec limitations:
- The Qwen `14B/0.5B` local config assumes the full local checkpoint shards are present under `models/qwen2.5-14b-Instruct-model`; incomplete local weights will block the smoke/full run until those shards exist.
- Policy transfer between model pairs is untested; solve a fresh policy per target/draft pair.
- Condition failures in `04_verify_conditions.py` are treated as empirical results, not runtime bugs; the script exits `0` and reports them for thesis discussion.
- The Qwen `7B -> 1.5B` full run on `2026-04-28` collected `500` traces and saved policy artifacts, but its benchmark stage failed with a HuggingFace SSL EOF while loading `Qwen/Qwen2.5-7B-Instruct`. The watcher correctly refused to launch the dependent full Step 3 because the expected `results.jsonl`, Pareto, ablation, and threshold-surface artifacts were missing.

## Determinism Policy

- Benchmark runs seed Python, NumPy, CPU torch, and CUDA torch RNGs per run.
- `torch.use_deterministic_algorithms(True, warn_only=True)` is enabled during seeded benchmarks.
- Some CUDA kernels still warn about CuBLAS/CuDNN reproducibility. Treat the manifests + seeds as the primary reproducibility record; small numeric drift is acceptable when the deterministic warning comes from a vendor kernel.

## Key Constraints

- **Tokenizer compatibility**: Draft and target models must share an identical vocabulary mapping for speculative, AutoJudge, Top-K, and SpecExec. The `Qwen2.5-0.5B → 7B` legacy pair violates this; use `0.5B → 3B` instead. The local `1.5B → 7B` pair has confirmed identical tokenizers.
- **Local models**: Pre-downloaded models go in `models/` (gitignored). Presets `qwen25_7b_instruct_local` and `qwen25_1p5b_instruct_local` point to `models/qwen2.5-7b-Instruct-model` and `models/qwen2.5-1.5b-Instruct-model`.
- **AutoJudge training**: Only valid with GSM8K-format datasets (`question` + `answer` fields). MT-Bench JSONL will fail fast with an actionable error.
- **AutoJudge C-grid policy**: Keep paper-aligned C-grid `1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0` (8 values). Do not use out-of-paper values `1e1` or `1e2` in config overrides.
- **AutoJudge checkpoint versioning**: Checkpoint format v2 (`autojudge_version=2`). Loading a v1 checkpoint triggers retraining.
- **Benchmark resume**: Re-running with the same `--out` file skips completed `resume_key` entries automatically.
- **GPU checks**: Use `make docker-gpu-check` / `make docker-gpu-check-image` before long runs. RTX 50xx (Blackwell/sm_120) requires `torch==2.9.1+cu128`.
- **Makefile Python**: Prefers `.venv/bin/python`; falls back to `python3`.

## Evaluation Results

### JointAdaSpec — primary results (RTX 5090, GSM8K zero-shot CoT)

**Status — THESIS SNAPSHOT LOCKED.** Qwen 14B → 0.5B (Run 1 final sprint, 2026-05-14, **500 prompts × 3 seeds = paired n=1500**, McNemar). These numbers are artifact-backed thesis results for the immutable `v1-defense` snapshot. The `main` branch is the place for future block-v1 reruns with regenerated 9-action policies and stricter artifact metadata.

| Method | EM | tok/s | vs speculative | Paired Δ EM vs target_only | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 52.93% | 14.45 | 2.96× | — | — | — |
| speculative (vanilla) | 53.27% | 4.88 | 1.00× | +0.33% | [-3.00%, 3.73%] | 0.877 |
| cascade_verif_then_length | 57.13% | 10.29 | 2.11× | +4.20% | [0.87%, 7.67%] | 0.0149 ✓ |
| **jointadaspec** | **57.00%** | **10.84** | **2.22×** | **+4.07%** | **[0.67%, 7.47%]** | **0.0203 ✓** |

Within the archived thesis snapshot, the **adaptive-control family** (both joint and cascade) beat `target_only` by ~`+4%` EM at `~2.2×` the throughput of vanilla speculative (`p < 0.05`). The defensible defense snapshot is the MDP formulation, theory, repaired block-v1 implementation, semantic artifact validation, artifact-backed Run 1/Run 2/Experiment E evidence, and reproducible future-rerun protocol.

**Honest caveat — joint ties cascade.** Head-to-head paired, `jointadaspec − cascade_verif_then_length = −0.13%`, `p = 0.96`: the two are statistically indistinguishable. The larger `+8.67%` margin and the joint-over-cascade edge seen in the earlier 100-prompt crosscheck (jointadaspec `+8.67%` p=0.0137, cascade `+7.00%`) were small-`n` noise that washed out at power. Reports: `reports/{pareto,ablation,jointadaspec_quality}_qwen14b_0p5b_lock_2026-05-14.*`, `reports/threshold_surface_qwen14b_0p5b_lock_2026-05-14/`.

**Secondary result — NULL at power.** Qwen 7B → 1.5B (Run 2 final sprint, 2026-05-14, **500 prompts × 3 seeds = paired n=1500**, McNemar, held-out window prompts 100–599):

| Method | EM | tok/s | vs speculative | Paired Δ EM vs target_only | p |
|---|---:|---:|---:|---:|---:|
| target_only | 60.20% | 24.93 | 1.90× | — | — |
| speculative | 60.60% | 13.13 | 1.00× | +0.40% | 0.830 |
| cascade_verif_then_length | 57.93% | 15.76 | 1.20× | −2.27% | 0.144 |
| jointadaspec | 58.73% | 15.77 | 1.20× | −1.47% | 0.369 |

At `n = 1500` on the 7B/1.5B pair, **neither adaptive method shows a quality gain over `target_only`** (jointadaspec `−1.47%`, p=0.37). The earlier `+3.50%` (2026-05-12, n=200, **different** held-out window prompts 1100–1299) did **not** survive at power on a fresh window — it was small-`n`/favorable-slice variance, not a robust effect. Both windows are legitimately held-out (traces are mined on the GSM8K *train* split, benchmarks run on the *test* split).

**Triangulation — SETTLED (2026-05-22).** A 3rd independent held-out window (prompts 600–1099, n=1500, May-5 policy unchanged, k=8) yielded `jointadaspec −2.53% vs target, p=0.094` — actually *slightly negative*, even more so than the lock window. Three-window summary on 7B/1.5B:

| window | n | joint − target | p |
|---|---:|---:|---:|
| start=1100 (orig, 2026-05-12) | 200 | +3.50% | 0.146 |
| start=100 lock (2026-05-14) | 1500 | −1.47% | 0.369 |
| **start=600 triangulation** | **1500** | **−2.53%** | **0.094** |

Verdict: the early small-`n` positive was noise; on two fresh well-powered windows the joint policy is at/below target. Slice-independent null on 7B/1.5B.

**Experiment E — adaptive control vs FIXED fuzzy threshold (14B/0.5B, n=300, 2026-05-27).** Ablation against `fuzzy_sd` (fixed γ=8, fixed T ∈ {1.0, 1.25, 1.5, 2.0}) — the missing baseline. This is an empirical ablation, not a theorem, and is part of the artifact-backed thesis snapshot:

| Method | EM | tok/s | accept | paired vs joint |
|---|---:|---:|---:|---:|
| fuzzy_sd_T=1.0 (best fixed) | 53.67% | 2.85 | 15.2% | joint +4.33%, p=0.275 |
| fuzzy_sd_T=1.25 | 50.33% | 2.95 | 16.3% | joint **+7.67%, p=0.0505** |
| fuzzy_sd_T=1.5 | 51.67% | 3.00 | 16.7% | joint +6.33%, p=0.115 |
| fuzzy_sd_T=2.0 | 53.33% | 3.06 | 17.4% | joint +4.67%, p=0.243 |
| **jointadaspec** | **58.00%** | **11.12** | **55.1%** | — |

Under the archived thesis-snapshot implementation, JointAdaSpec is descriptively **+4 to +8% more accurate AND ~3.7× faster** than any fixed-fuzzy-T baseline. It supports the motivation for adaptive control, but future main-branch claims require rerun with regenerated 9-action artifacts and prompt-clustered statistics. Reports: `outputs/jointadaspec_qwen14b_0p5b_fuzzy_ablation_2026-05-25/`, figure `reports/thesis_figs/fig_E_adaptivity_ablation.pdf`.

**κ-sweep — joint vs cascade across the Lagrange trade-off knob (7B/1.5B, 6 κ × n=300, 2026-05-26).** For each κ ∈ {0, 1, 5, 20, 50, 100} the joint and cascade policies were re-solved on the May-5 traces and benchmarked on a fixed slice. `tok/s` is essentially flat across κ (16.28–16.55); EM varies non-monotonically (joint 55.7–60.3%, cascade 55.0–61.0%); **joint ≈ cascade at every κ** (≤3% gap throughout). The joint=cascade equivalence is robust to the trade-off knob, not just the default value. (Theorem-2.4 convexity is not cleanly demonstrated at n=300/κ — honestly noted.) Figure `reports/thesis_figs/fig_bonus_kappa_sweep.pdf`.

**Theorem D — tight explanation of joint ≈ cascade (2026-05-25).** μ*_J(B) = 0.90 on the C4-violating set on both pairs (C4 violated on 89% of states), so Theorem-C's worst-case bound (~360) is vacuous. But the *cascade-policy advantage* A^πC(s, πJ(s)) on B is ε-small (weighted mean 6e-5 on 14B, 1e-3 on 7B; |adv|>0.01 on 0.1–0.6% of B states) → C4 violation is *benign*. Exact stationary-weighted value gap V_joint − V_cascade = +0.005 (14B), +0.10 (7B). Weak dominance (Theorem 2.3) confirmed *exactly*: V_joint ≥ V_cascade on 100% of states. Reproducible at `scripts/analyze_theorem_c_gap.py` → `reports/theorem_c_gap_analysis.json`. Figure `reports/thesis_figs/fig_D_advantage_on_B.pdf`.

### AutoJudge — comparison baseline (Qwen2.5 7B/1.5B, k=4, 2026-03-10, 100 × 3)

| Method | EM | tok/s | vs Speculative |
|---|---:|---:|---:|
| Baseline (7B) | 58.1% | 78.6 | 1.67× |
| Speculative | 56.9% | 47.2 | 1.00× |
| AutoJudge t=0.09 | 61.7% | 55.9 | 1.18× |
| AutoJudge t=1.0 | 52.3% | 63.4 | 1.34× |
| Top-K rank=4 | 54.3% | 71.5 | 1.52× |

### Theory updates (2026-05-15)

Three new theorems strengthen the dissertation; see `reports/theory_improvements_2026-05-15.md` and `reports/dissertation_review_2026-05-15.md` for full statements and proofs.

- **Theorem A.** Sample-complexity bound on `‖V̂ − V*‖∞` for the trace-based MDP estimator (Hoeffding concentration + Laplace bias).
- **Theorem B.** State-only additive quality-risk is Bellman-invariant. Replaces the multiplicative form (which broke contraction). Code change applied via the `quality_risk_form` flag in `MDPConfig`; existing policies default to `multiplicative` for backward compatibility.
- **Theorem C.** Cascade suboptimality is bounded linearly in the C4-violating stationary occupancy mass μ*ᴊ(B). NOTE (2026-05-25): empirically μ*_J(B) ≈ 0.90 on both pairs (C4 violated almost everywhere), so the worst-case Theorem-C bound is *vacuous*. The earlier "+1.7% to +4.3% joint-vs-cascade EM advantage" was small-`n` noise refuted at power (joint−cascade = −0.13%, p=0.96). Theorem C still holds as an upper bound; the tight explanation is **Theorem D**.
- **Theorem D (new, 2026-05-25).** Exact advantage-weighted value gap: `V_joint − V_cascade = (1/(1−γ)) · E_{μ*_J}[A^{π_C}(s, π_J(s))]`. Empirically the per-state cascade advantage `A^{π_C}(s, π_J(s))` on the C4-violating set is ε-small (mean 6e-5 on 14B, 1e-3 on 7B), so the realized value gap is small *despite* μ*_J(B) ≈ 0.90 — C4 violation is *benign*. Weak dominance (Theorem 2.3) confirmed exactly: `V_joint ≥ V_cascade` on **100%** of states, both pairs.

### Known limitations (honest reporting)

- **Joint = cascade at power; the contribution is the adaptive control *family*.** On locked 14B/0.5B (n=1500) `jointadaspec − cascade_verif_then_length = −0.13%`, p=0.96 — a tie; on 7B/1.5B both are at/below target. *But*: Experiment E (n=300 ablation) gives thesis-snapshot evidence that JointAdaSpec can outperform FIXED fuzzy_sd_T baselines. The defended contribution is therefore (a) the unified MDP framework + corrected theory (A/B/C/D/2.3/2.4), (b) repaired block-v1 implementation and artifact validation, and (c) a reproducible future-rerun path for post-defense benchmark claims.
- **Speed honesty.** Plain target AR is the *fastest* method on both pairs (14B: target 14.45 > joint 10.84 > vanilla-spec 4.88 tok/s). The reported "2.22× vs vanilla speculative" / "3.7× vs fuzzy_sd" speedups are **relative to other speculative variants**, not relative to plain AR. The honest framing is "adaptive speculative decoding *recovers* throughput that vanilla speculative would lose," not "faster than baseline."
- **Quality gain is non-monotonic in acceptance (Theorem G).** Churn rate ~45% is flat across acceptance levels; net Δ EM is **+7.4% at low/mid joint-acceptance, −2.6% at high acceptance** — over-trusting the 0.5B draft degrades quality. The headline +4% is a blend; the controller's "sweet spot" is moderate acceptance.
- **7B/1.5B null is slice-independent.** Three held-out windows: +3.50% (n=200, noise) → −1.47% (n=1500) → −2.53% (n=1500). The method gives no quality benefit on the low-ratio (4.7×) pair.
- **Held-out 7B/1.5B (paired n=3000, 2026-05-11): −2.00% EM, p=0.0667** — distribution-shift sensitivity. Joint loses less than cascade (−2.37%) in the same regime; framed as **graceful degradation**.
- **k=16 evaluation on k=8-trained policy: regression** — out-of-distribution deployment (policy state space cannot represent k > γ_max=8). Methodologically expected, not a method weakness.
- **Speculative slower than target on 7B/1.5B** — expected at low target/draft ratio (4.7×) on a single GPU; literature uses 70B/8B+ ratios with multi-GPU.

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
