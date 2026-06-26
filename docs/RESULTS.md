# Results Overview

## Status: Legacy Until Block-v1 Rerun

The tables below preserve the locked thesis artifacts and historical reports.
After the 2026 audit, these JointAdaSpec numbers are treated as **legacy
results / pre-repair evidence**. The current code path has been repaired toward
`block_verify_v1`, but the locked Run 1 report points to a pre-repair 16-action
policy artifact and the raw lock benchmark CSV is not present in this working
copy for prompt-clustered reanalysis. The numbers remain useful for debugging,
regression tests, and thesis-history traceability, but they must not be used as
final evidence for block speculative-decoding speedup until the benchmark suite
is rerun with regenerated block-v1 artifacts.

Defense audit: `docs/DEFENSE_AUDIT_2026-06-26.md`.

## Defense Snapshot

The defensible snapshot is: MDP formulation, theory, repaired block-v1
implementation and semantic artifact checks, plus a reproducible rerun plan.
The legacy tables below show the historical motivation and observed behavior of
the pre-repair artifacts; they are not final block-v1 benchmark claims.

## Legacy / Pre-Repair Locked Results (2026-05-14 — 2026-05-27)

### Run 1 — Qwen 14B → 0.5B (legacy primary, locked 2026-05-14)

GSM8K zero-shot CoT, RTX 5090, 500 prompts × 3 seeds = **n=1500 paired**, McNemar:

| Method | EM | tok/s | vs speculative | Δ vs target_only | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|
| target_only | 52.93% | 14.45 | 2.96× | — | — | — |
| speculative | 53.27% | 4.88 | 1.00× | +0.33% | [−3.00%, 3.73%] | 0.877 |
| cascade_verif_then_length | 57.13% | 10.29 | 2.11× | **+4.20%** | [0.87%, 7.67%] | **0.015 ✓** |
| **jointadaspec** | **57.00%** | **10.84** | **2.22×** | **+4.07%** | [0.67%, 7.47%] | **0.020 ✓** |

Under the pre-repair artifacts, the adaptive-control family (joint + cascade)
outperformed `target_only` at p < 0.05. Treat this as legacy evidence only, not
as a final block-v1 proof. Joint = cascade head-to-head (Δ = −0.13%, p = 0.96):
predicted by Theorem D. Speed note: `target_only` is the fastest method
(14.45 tok/s); the 2.22× is relative to vanilla speculative.

Reports: `reports/{pareto,ablation,jointadaspec_quality}_qwen14b_0p5b_lock_2026-05-14.*`, `reports/threshold_surface_qwen14b_0p5b_lock_2026-05-14/`.

### Run 2 — Qwen 7B → 1.5B (secondary, null at power, locked 2026-05-14)

| Method | EM | tok/s | vs speculative | Δ vs target_only | p |
|---|---:|---:|---:|---:|---:|
| target_only | 60.20% | 24.93 | 1.90× | — | — |
| speculative | 60.60% | 13.13 | 1.00× | +0.40% | 0.830 |
| cascade_verif_then_length | 57.93% | 15.76 | 1.20× | −2.27% | 0.144 |
| jointadaspec | 58.73% | 15.77 | 1.20× | −1.47% | 0.369 |

### Triangulation — 7B/1.5B null is slice-independent (2026-05-22)

Three independent held-out windows all confirm the null:

| Window | n | joint − target_only | p |
|---|---:|---:|---:|
| start=1100 (2026-05-12, original) | 200 | +3.50% | 0.146 |
| start=100 lock (2026-05-14) | 1500 | −1.47% | 0.369 |
| start=600 triangulation (2026-05-22) | 1500 | **−2.53%** | **0.094** |

The early +3.50% was small-n noise. Low model-power ratio (4.7×) on a single GPU is insufficient. Report: `reports/jointadaspec_quality_qwen7b_1p5b_quality_tri_2026-05-20.md`.

### Experiment E — Adaptive vs fixed fuzzy threshold (14B/0.5B, n=300, 2026-05-27)

| Method | EM | tok/s | accept | paired vs joint |
|---|---:|---:|---:|---:|
| fuzzy_sd_T=1.0 (best fixed) | 53.67% | 2.85 | 15.2% | joint +4.33% |
| fuzzy_sd_T=1.25 | 50.33% | 2.95 | 16.3% | joint +7.67%, p≈0.05 |
| fuzzy_sd_T=1.5 | 51.67% | 3.00 | 16.7% | joint +6.33% |
| fuzzy_sd_T=2.0 | 53.33% | 3.06 | 17.4% | joint +4.67% |
| **jointadaspec** | **58.00%** | **11.12** | **55.1%** | — |

This is an ablation experiment, not a mathematical theorem. The observed EM
differences are descriptive under the pre-repair artifacts and support the
motivation for adaptive control; they require the block-v1 rerun plus
prompt-level clustered statistics before they can be used as final claims.
Figure: `reports/thesis_figs/fig_E_adaptivity_ablation.pdf`.
Output dir: `outputs/jointadaspec_qwen14b_0p5b_fuzzy_ablation_2026-05-25/`.

### Theorem D — Exact value gap joint vs cascade (2026-05-25)

Advantage-weighted value gap: `V_joint − V_cascade = 1/(1−γ) · E_{μ*_J}[A^{πC}(s, πJ(s))]`.

| Pair | μ*_J(B) | mean |A^πC| on B | |adv|>0.01 states | V_joint − V_cascade |
|---|---:|---:|---:|---:|
| 14B/0.5B | 0.90 | **6e-5** | 0.1% | +0.005 |
| 7B/1.5B | 0.90 | **1e-3** | 0.6% | +0.10 |

C4 is violated on 90% of states, but the violation is *benign*: cascade advantage on B is ε-small, so the value gap is nearly zero despite near-universal C4 failure. Weak dominance V_joint ≥ V_cascade holds on 100% of states (both pairs). Script: `scripts/analyze_theorem_c_gap.py`. Figure: `reports/thesis_figs/fig_D_advantage_on_B.pdf`.

### κ-sweep — joint vs cascade across trade-off knob (7B/1.5B, 6 κ × n=300, 2026-05-26)

`tok/s` flat (16.28–16.55 tok/s); EM non-monotonic (joint 55.7–60.3%, cascade 55.0–61.0%); joint ≈ cascade at every κ (gap ≤ 3%). Joint–cascade parity is robust to the Lagrange trade-off knob. Figure: `reports/thesis_figs/fig_bonus_kappa_sweep.pdf`.

---

## Historical Runs (through 2026-05-08)

This page is a compact index of benchmark outcomes and where to find full artifacts.

## Latest Artifacts Through 2026-05-08

The latest tracked `jointadaspec/` artifacts include one substantive throughput snapshot, several smoke or partial runs, a quality-aware Qwen `7B -> 1.5B` rerun, and a held-out validation pass. Treat the smoke rows as regression and pipeline evidence, not as statistically stable best-of-run claims.

| Run | Model pair | Scope | Best observed result | Reliability |
|---|---|---|---|---|
| `2026-04-14` | Qwen2.5 `7B -> 1.5B` | `3000` traces, `100` GSM8K prompts, `100` LiveCodeBench prompts | GSM8K: `jointadaspec` `18.99` tok/s, acceptance `0.951`; LiveCodeBench: `jointadaspec` `10.34` tok/s, acceptance `0.890` | Best substantive snapshot |
| `2026-04-21` | Qwen2.5 `7B -> 1.5B` | seeded/schema smoke, `1` prompt, `1` seed, `8` new tokens | `cascade_verif_then_length` `18.47` tok/s, `1.204x` vs target-only | Complete smoke only |
| `2026-04-28` | Qwen2.5 `14B -> 0.5B` | smoke, `5` prompts, `1` seed, `64` new tokens | `cascade_verif_then_length` `13.15` tok/s, `0.725x`; `jointadaspec` `12.85` tok/s, `0.709x`; `target_only` `18.13` tok/s | Complete smoke only |
| `2026-04-28` | Qwen2.5 `7B -> 1.5B` | full trace/solve attempt, `500` traces | policy and condition artifacts saved; no benchmark summary | Partial, benchmark failed |
| `2026-05-05` | Qwen2.5 `7B -> 1.5B` local quality-aware | `500` traces, `100` GSM8K prompts, `3` seeds, `256` new tokens | `cascade_verif_then_length` EM `0.6267`, `15.80` tok/s; `jointadaspec` EM `0.6167`, `15.79` tok/s; `target_only` EM `0.5700`, `25.01` tok/s | Complete quality hypothesis run |
| `2026-05-08` | Qwen2.5 `7B -> 1.5B` local quality-aware held-out | `500` held-out GSM8K prompts, `3` seeds, `256` new tokens | `target_only` EM `0.6020`; `speculative` EM `0.6060`; `jointadaspec` EM `0.5873`; `cascade_verif_then_length` EM `0.5793` | Complete held-out validation, negative/neutral result |

Curated roll-up report: `reports/jointadaspec_qwen_runs_through_2026-04-28.md`.

Quality-aware analysis report: `reports/jointadaspec_quality_qwen7b_1p5b_quality_2026-05-05.md`.

Held-out validation report: `reports/jointadaspec_quality_qwen7b_1p5b_quality_heldout_2026-05-08.md`.

The held-out validation did not support the quality-aware cascade hypothesis. The fixed primary decoder `cascade_verif_then_length` had paired EM delta `-2.27` percentage points versus `target_only` with CI `[-5.27%, 0.60%]` and `p=0.1439`. The original `2026-05-05` improvement therefore remains a first-slice hypothesis that did not generalize to held-out GSM8K.

### 2026-04-28 failure note

The Qwen `7B -> 1.5B` full benchmark stage failed while loading `Qwen/Qwen2.5-7B-Instruct` from HuggingFace with an SSL EOF. The watcher script then refused to launch the dependent full Step 3 because it could not find `outputs/jointadaspec_qwen7b_1p5b_2026-04-28/03_bench_gsm8k/results.jsonl`, `reports/pareto_qwen7b_1p5b_2026-04-28.pdf`, `reports/ablation_qwen7b_1p5b_2026-04-28.pdf`, or `reports/threshold_surface_qwen7b_1p5b_2026-04-28/`.

### Condition checks

| Run | Passed | Failed |
|---|---|---|
| Qwen `7B -> 1.5B`, `2026-04-21` | `c3`, `n1`, `n2` | `c1`, `c2`, `c4` |
| Qwen `7B -> 1.5B`, `2026-04-28` | `n1`, `n2` | `c1`, `c2`, `c3`, `c4` |
| Qwen `14B -> 0.5B`, `2026-04-28` | `c2`, `c3`, `n1`, `n2` | `c1`, `c4` |
| Qwen `7B -> 1.5B` quality-aware, `2026-05-05` | `n1`, `n2` | `c1`, `c2`, `c3`, `c4` |

## Quality-aware Qwen 7B / 1.5B Rerun

Run tag: `2026-05-05`

- Output artifacts: `outputs/jointadaspec_qwen7b_1p5b_quality_2026-05-05/`
- Analysis: `reports/jointadaspec_quality_qwen7b_1p5b_quality_2026-05-05.md`
- Plots: `reports/pareto_qwen7b_1p5b_quality_2026-05-05.pdf`, `reports/ablation_qwen7b_1p5b_quality_2026-05-05.pdf`

| Method | Speed (tok/s) | Acceptance | GSM8K EM | vs Target Speed |
|---|---:|---:|---:|---:|
| target_only | 25.01 | 0.000 | 0.570 | 1.000 |
| speculative | 13.23 | 0.719 | 0.620 | 0.529 |
| jointadaspec | 15.79 | 0.917 | 0.617 | 0.631 |
| cascade_verif_then_length | 15.80 | 0.919 | 0.627 | 0.632 |

Paired EM difference for the fixed primary method `cascade_verif_then_length` versus `target_only` was `+5.67` percentage points on `300` paired prompt-seed rows, with bootstrap CI crossing zero. This is a quality hypothesis, not yet a final statistically stable claim.

## Held-out Quality Validation (Qwen 7B / 1.5B)

Run tag: `2026-05-08`

- Output artifacts: `outputs/jointadaspec_qwen7b_1p5b_quality_heldout_2026-05-08/03_bench_gsm8k/`
- Analysis: `reports/jointadaspec_quality_qwen7b_1p5b_quality_heldout_2026-05-08.md`
- Plots: `reports/pareto_qwen7b_1p5b_quality_heldout_2026-05-08.pdf`, `reports/ablation_qwen7b_1p5b_quality_heldout_2026-05-08.pdf`

| Method | Speed (tok/s) | Acceptance | GSM8K EM | vs Target Speed |
|---|---:|---:|---:|---:|
| target_only | 25.48 | 0.000 | 0.602 | 1.000 |
| speculative | 13.13 | 0.715 | 0.606 | 0.515 |
| jointadaspec | 15.79 | 0.917 | 0.587 | 0.620 |
| cascade_verif_then_length | 15.78 | 0.921 | 0.579 | 0.619 |

Primary success: `False`. Strong success: `False`. The selected quality-aware cascade did not improve held-out GSM8K exact match; it reduced EM by `2.27` percentage points relative to `target_only`, with uncertainty crossing zero.

Partial runtime evidence from the queue is tracked only as incomplete evidence. The Qwen `14B -> 0.5B` cross-check and the extended Qwen `7B -> 1.5B` fallback produced manifests and partial `run.jsonl` files, but no `results.jsonl`; they are not used for benchmark claims.

## Best Substantive JointAdaSpec Snapshot (Qwen 7B / 1.5B)

Run tag: `2026-04-14`

- Report: `reports/jointadaspec_qwen_7b_1p5b_2026-04-14.md`
- Trace artifacts: `outputs/jointadaspec_qwen_2026-04-14/01_traces_gsm8k/`
- Policy artifacts: `outputs/jointadaspec_qwen_2026-04-14/02_solve/`
- GSM8K benchmark: `outputs/jointadaspec_qwen_2026-04-14/03_bench_gsm8k/`
- LiveCodeBench benchmark: `outputs/jointadaspec_qwen_2026-04-14/04_bench_livecodebench/`

### Trace and solve snapshot

| Item | Value |
|---|---:|
| Model pair | `Qwen2.5-7B-Instruct -> Qwen2.5-1.5B-Instruct` |
| Trace count | 3000 |
| Recorded one-step transitions | 479880 |
| `kappa` sweep | `0.0, 0.5, 1.0, 2.0, 5.0` |
| VI iterations (`kappa=0.0`) | 834 |
| VI iterations (`kappa=0.5`) | 830 |
| VI iterations (`kappa=1.0`) | 827 |
| VI iterations (`kappa=2.0`) | 825 |
| VI iterations (`kappa=5.0`) | 824 |

### GSM8K throughput snapshot

This older run predates the seeded benchmark summary path that records task-level GSM8K exact match. It remains the best substantive throughput/acceptance snapshot because it used `100` prompts rather than a smoke-sized slice.

| Method | Speed (tok/s) | Acceptance | vs Vanilla |
|---|---:|---:|---:|
| vanilla_ar | 29.38 | 0.000 | 1.000 |
| fixed_sd | 17.26 | 0.801 | 0.588 |
| fuzzy_sd_T4 | 19.09 | 0.915 | 0.650 |
| jointadaspec | 18.99 | 0.951 | 0.646 |
| specdecpp | 17.81 | 0.835 | 0.606 |

### LiveCodeBench throughput snapshot

| Method | Speed (tok/s) | Acceptance | vs Vanilla |
|---|---:|---:|---:|
| vanilla_ar | 14.36 | 0.000 | 1.000 |
| fixed_sd | 8.05 | 0.659 | 0.560 |
| fuzzy_sd_T4 | 9.62 | 0.843 | 0.670 |
| jointadaspec | 10.34 | 0.890 | 0.720 |
| specdecpp | 9.04 | 0.763 | 0.629 |

### Repro command

```bash
bash scripts/run_jointadaspec_qwen_longrun.sh
```

## Latest Historical `sp_samp` Full Run (Llama 8B / 3B)

Run tag: `2026-03-28-llama-48h-cgrid8`

- Manifest: `reports/llama3_8b_3b_run_manifest_2026-03-28-llama-48h-cgrid8.json`
- GSM8K report: `reports/yandex_llama3_8b_3b_2026-03-28-llama-48h-cgrid8-gsm8k.md`
- LiveCodeBench report: `reports/yandex_llama3_8b_3b_2026-03-28-llama-48h-cgrid8-livecodebench.md`
- Raw outputs:
  - `datasets/results_llama3_8b_3b_gsm8k_2026-03-28-llama-48h-cgrid8.jsonl`
  - `datasets/results_llama3_8b_3b_lcb_2026-03-28-llama-48h-cgrid8.jsonl`

### GSM8K snapshot

| Method | Parameter | Accuracy (%) | Speed (tok/s) |
|---|---:|---:|---:|
| baseline | - | 70.89 | 72.68 |
| speculative | - | 71.89 | 40.68 |
| autojudge | 0.140 | 78.67 | 45.98 |
| topk | all | 75.67 | 59.29 |

### LiveCodeBench snapshot

| Method | Parameter | Speed (tok/s) |
|---|---:|---:|
| baseline | - | 71.52 |
| speculative | - | 34.80 |
| autojudge | 1.000 | 29.27 |
| topk | all | 36.53 |

## Historical Runs (Tracked)

- Qwen local 7B/1.5B:
  - `reports/yandex_local_7b_1p5b_2026-03-10-gsm8k.md`
  - `reports/yandex_local_7b_1p5b_2026-03-10-livecodebench.md`
- JointAdaSpec Qwen 7B/1.5B:
  - `reports/jointadaspec_qwen_7b_1p5b_2026-04-14.md`
- Mistral 8B/3B:
  - `reports/yandex_mistral3_8b_3b_2026-03-20-mistral-gsm8k.md`
  - `reports/yandex_mistral3_8b_3b_2026-03-20-mistral-livecodebench.md`
- Gemma-2 9B/2B:
  - `reports/yandex_gemma2_9b_2b_2026-03-16-gemma-gsm8k.md`
  - `reports/yandex_gemma2_9b_2b_2026-03-16-gemma-livecodebench.md`

## Repro Command (Latest Historical Llama Profile)

```bash
DATE_TAG="$(date +%F)-llama-48h-cgrid8"
CHECKPOINT_PATH="datasets/autojudge_llama3_3b_to_8b_${DATE_TAG}.pt" \
OUT_GSM8K="datasets/results_llama3_8b_3b_gsm8k_${DATE_TAG}.jsonl" \
OUT_LCB="datasets/results_llama3_8b_3b_lcb_${DATE_TAG}.jsonl" \
REPORT_PREFIX="reports/yandex_llama3_8b_3b_${DATE_TAG}" \
MANIFEST_PATH="reports/llama3_8b_3b_run_manifest_${DATE_TAG}.json" \
bash scripts/run_llama3_8b_3b_eval.sh
```

## Validation Commands

```bash
.venv/bin/python scripts/validate_results_jsonl.py --path datasets/results_llama3_8b_3b_gsm8k_2026-03-28-llama-48h-cgrid8.jsonl --strict
.venv/bin/python scripts/validate_results_jsonl.py --path datasets/results_llama3_8b_3b_lcb_2026-03-28-llama-48h-cgrid8.jsonl --strict
```
