# Paper Alignment Review — 2026-02-25

Full correctness audit of the codebase against the three source papers:
- Yandex Research: **AutoJudge** ("Judge Decoding Without Manual Annotation")
- Yandex Research: **SpecExec** ("Massively Parallel Speculative Decoding")
- DeepMind: **Speculative Sampling**

---

## Changes Made

### `sp_samp/autojudge.py` — C-regularisation grid fix

**File**: `sp_samp/autojudge.py`, line 37 (`_default_c_grid`)

**Bug**: `range(-7, 3)` produced 10 values including `10^1` and `10^2` which are not
in the paper's grid.

**Paper (Section 3.2)**: C ∈ {10^0, 10^-1, …, 10^-7} — exactly 8 values.

**Fix**: Changed `range(-7, 3)` → `range(-7, 1)` so the grid is `{10^-7, …, 10^0}`.

Including out-of-paper high-C values (10^1, 10^2) risks under-regularisation on small
datasets such as the 4000-sample GSM8K training set used in the paper.

### `sp_samp/autojudge.py` — paper reference comments

Added `# Paper: ...` comments at key algorithm sites:

| Location | Comment |
|---|---|
| `_default_c_grid` | `# Paper: Section 3.2` |
| `_generate_greedy` signature | `# Paper: Appendix A — greedy vs stochastic` |
| `mine_important_tokens_gsm8k` | `# Paper: Algorithm 1` |
| initial response generation | Note that target model is used (not draft as in pseudocode) — intentional, matches math formula |
| label assignment | `# Paper: Algorithm 1, line 9 — 0=unimportant, 1=important` |
| `train_autojudge_logreg` | `# Paper: Section 3.2` |
| judge call site in inference | `# Paper: Section 3 — judge on mismatches only` |

### `tests/test_autojudge.py` — new test

Added `test_default_c_grid_matches_paper()`:
- Asserts exactly 8 values in `_default_c_grid()`
- Asserts values match `{10^-7, …, 10^0}`
- Asserts `10^1` and `10^2` are absent
- Asserts `10^0 = 1.0` is the maximum value

---

## Findings: Confirmed Correct (no changes needed)

### AutoJudge

| Item | Verdict |
|---|---|
| `max_train_samples` check inside while loop | Correct ✓ |
| `_threshold_for_recall` returns highest threshold where recall ≥ target | Correct ✓ |
| Label convention: 0=unimportant / 1=important | Matches paper ✓ |
| Final model retrained on full dataset after grid search | Correct ✓ |
| StandardScaler fit on train, applied to val; new scaler for final model | Correct ✓ |
| Judge called only on mismatches | Correct ✓ |
| Extra token emitted when full block accepted | Correct ✓ |

### AutoJudge — Intentional Deviations (documented, no code change)

**Initial response from TARGET not DRAFT**
- Paper Algorithm 1 line 1: `ỹ ← GENERATE(x, θ_draft)` (draft model)
- Code: uses `target_model`
- Rationale: matches the paper's mathematical formula for I(x), which is defined relative
  to the target response `y`. No correctness impact; documented with inline comment.

**Greedy decoding instead of Gumbel-max stochastic sampling**
- Paper Appendix A describes Gumbel-max reparameterisation
- Code: argmax throughout — valid deterministic variant; documented with inline comment.

**Feature computation: training vs inference index alignment**
- Both extract hidden state after the draft token position in a causally masked model
- Minor difference: inference context includes future draft tokens (no attention bleed) → consistent.

### SpecExec — Known Architectural Simplification (documented, no code change)

- Paper uses **SSSP / modified Dijkstra** with priority queue and cumulative log-prob budget K
- Code uses **BFS level-by-level** with `parallel_branches` top-N per node and
  `branch_prune_threshold`
- Correct distribution preserved; lower compute complexity per step
- Known trade-off: documented in CLAUDE.md and README.MD

---

## Files Changed

```
sp_samp/autojudge.py         — C-grid fix + paper reference comments
tests/test_autojudge.py      — new test_default_c_grid_matches_paper
CLAUDE.md                    — Paper Alignment section added
README.MD                    — paper reference note added
file_changes/2026-02-25-paper-alignment.md  — this file (new)
```
