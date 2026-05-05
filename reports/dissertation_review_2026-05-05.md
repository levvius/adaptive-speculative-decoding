# Dissertation Review and Draft Update — 2026-05-05

Reviewed source: `papers/ИС61_fpm_КозинАА_2026.docx`.

Produced copy: `papers/ИС61_fpm_КозинАА_2026_исправлено_практика.docx`.

## Scope

- The title page, abstract, and table of contents were preserved byte-for-byte at the DOCX XML paragraph level.
- The corrected copy updates theory, proof wording, figure/table placeholders, references, draft practical chapters, conclusion, and appendices.
- The original DOCX was not modified.

## Main Theory Fixes

- Reframed the MDP state as `s = (H_i, K_prev, k)` to match the inference-time delay of `KL(q || p)`.
- Corrected the action notation to `a = (a_length, T)` and aligned `T_levels` with the implemented geometric grid.
- Replaced dense value-iteration complexity claims with the actual sparse CSR implementation.
- Fixed Theorem 2.3: joint optimization weakly dominates cascade policies by inclusion of policy classes; strict dominance now requires reachable divergence plus strict Bellman advantage.
- Fixed Theorem 2.4 and Appendix A.3: Pareto reasoning now uses scalarization and occupancy-measure convexity instead of the invalid `kappa = D` shortcut.
- Removed unsupported claims that current JointAdaSpec results already cover SpecBench/HumanEval/MT-Bench.

## Practical Draft Added

- Chapter 3 now describes the repository architecture, trace collection, MDP estimation, sparse value iteration, inference policy, baselines, manifests, CI, and runtime constraints.
- Chapter 4 now records the actual JointAdaSpec result status:
  - `2026-04-14` Qwen `7B -> 1.5B`: best substantive snapshot.
  - `2026-04-21` Qwen `7B -> 1.5B`: seeded/schema smoke only.
  - `2026-04-28` Qwen `14B -> 0.5B`: smoke only.
  - `2026-04-28` Qwen `7B -> 1.5B`: partial trace/solve run, benchmark failed on HuggingFace SSL EOF.
- The next experiment is documented as a local quality-aware Qwen `7B -> 1.5B` rerun.

## Code Alignment

- Added conservative quality-aware reward shaping:
  - `quality_risk_K`
  - `quality_risk_k`
- The default values are `0.0`, so existing policies and reports remain backward compatible.
- Added local Qwen `7B -> 1.5B` quality experiment config to avoid repeat HuggingFace SSL failures.

## Verification Targets

- `make check`
- `make test`
- DOCX protected-section hash check
- LibreOffice headless conversion smoke
- Staged diff review before commit
