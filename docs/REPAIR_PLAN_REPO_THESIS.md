# Repository And Thesis Repair Plan

## Summary

Goal: align the code, experiments, README/docs, thesis, and defense materials around one correct version of the method: real block speculative decoding, a matching MDP formulation, honest statistical reporting, reproducible artifacts, and GitHub publication.

Current pre-repair Git state must be checked before every commit. At the time this plan was created, the worktree already contained a modified `papers/ИС61_fpm_КозинАА_2026.docx` and an untracked `AGENTS.md`. Those changes must not be mixed into code commits unless they are explicitly staged as part of a documentation commit.

## Key Changes

1. Mark old JointAdaSpec results as legacy until they are rerun with block verification.
2. Rewrite the decoding core so the draft model forms a block and the target model verifies the whole block in one verification phase.
3. Move `fixed_sd`, `fuzzy_sd`, cascade baselines, and `JointAdaSpec` onto the shared block-verification layer.
4. Reframe the MDP around `continue` and `verify@T`, using delayed `K_prev` and no target call during draft-action selection.
5. Compute reward at the draft-verify cycle level: emitted tokens minus wall-clock cost minus quality penalty.
6. Update trace collection, solver compatibility, JSONL/report metadata, and prompt-level statistics.
7. Rerun experiments after the implementation is fixed.
8. Update README, docs/results, thesis generator, presentation notes, and internal reports.
9. Finish through logical commits, push to GitHub on a non-main branch, and open a PR.

## Commit Plan

Use a separate branch:

```bash
git switch -c codex/repair-repo-thesis
```

Logical commits:

1. `docs: add repository and thesis repair plan`
2. `docs: mark legacy results and update project guidance`
3. `feat: add block speculative verification core`
4. `refactor: migrate jointadaspec decoders to block verification`
5. `refactor: align mdp actions traces and rewards with block decoding`
6. `test: add schema statistics and decoder regression coverage`
7. `docs: revise thesis claims and defense materials`
8. `chore: add reproducibility release artifacts`

## GitHub Finish

Before push:

```bash
git status --short
git diff --stat
make check
make test
```

Push only the repair branch:

```bash
git push -u origin codex/repair-repo-thesis
```

Then open a PR from `codex/repair-repo-thesis` into `main`. Do not push directly to `main`.

## Documentation Updates

- README: describe the legacy/new result boundary.
- `docs/RESULTS.md`: separate old sequential-verifier results from future block-SD results.
- `papers/build_thesis_docx.py`: edit the thesis source generator, not only the final `.docx`.
- `papers/pres.md`: rename "Theorem E" to an experiment/hypothesis.
- `reports/thesis_final_summary_2026-05-20.md`: replace outdated strong claims.
- `reports/theory_improvements_2026-05-15.md`: correct the Theorem B statement.
- JSONL/reporting docs: add prompt-level clustered statistics.
- Reproducibility docs: require clean release commit/tag, exact model and dataset revisions, environment lock, and exact commands.

## Test Plan

- Unit: the target model is called once per verification block.
- Unit: exact speculative decoding at `T=1` preserves the target distribution on toy models.
- Unit: fuzzy verification increases acceptance as `T` grows.
- Unit: `ActionSpace` no longer contains duplicated stop actions.
- Integration: `JointAdaSpecDecoder` does not call the target model while deciding draft continuation.
- Smoke: `make check`, `make test`, `make bench-toy`, and tiny HF smoke where available.
- Report validation: strict JSONL validation and clustered confidence interval checks.

## Post-PR Audit Additions

These items must be handled before treating the repair branch as merge-ready. They cover compatibility, validation, and documentation risks discovered after the first repair commits.

- Environment gate: split checks into MacBook-safe, CPU/CI, and RTX-only groups. Do not claim "tests pass" until a real `.venv` or CI environment has `pytest`, `numpy`, `torch`, and the project dependencies installed.
- Action-space migration: version policy and trace artifacts with `action_space_version=2` and `decoder_semantics=block_verify_v1`. Old 16-action policy `.npz` artifacts must not be reused with the new `continue + verify@T` action space; loaders should reject them or mark them explicitly as legacy.
- Block verification correctness: compare batched block logits with sequential next-token logits on toy or tiny HF models. Cover off-by-one indexing, empty-prefix/BOS fallback, common-vocabulary truncation, and the measured number of target verification passes.
- Target-pass metadata: replace any global `target_pass_mode="batched_block"` assumption with per-method values such as `target_only`, `batched_block`, `sequential_fallback`, and `legacy_sequential`.
- MDP semantics audit: document how `K_prev` is updated after a verify block. The default should be mean block KL unless a later ablation justifies `last_K` or `max_K`. Also verify that the target model is not called during draft-action selection.
- Reward/statistics wording: ensure code comments, tests, reports, and thesis text describe scalar reward as a surrogate for the draft-verify cycle, not as exact throughput optimization unless the implementation moves to an explicit average-reward or SMDP formulation.
- Remaining claim cleanup: add an `rg` gate for stale phrases and fix remaining conflicts in `papers/ВКР_тезисы_и_структура.md`, `reports/dissertation_review_2026-05-15.md`, and `tests/test_mdp_solver.py`, especially old Bellman-invariance wording around Theorem B.
- Schema/report compatibility: document schema v3, ensure legacy readers and report templates either read v3 or fail with a clear legacy message, and add paired prompt-clustered comparisons with Holm correction for multiple comparisons.
- Generated artifacts policy: do not commit the modified thesis `.docx`, PPTX/PDF, or other binary generated artifacts unless explicitly requested. Source generators and Markdown remain the source of truth; release builds should record checksums in a manifest.
- PR checklist: require green CPU tests/CI, review of action-space migration, an explicit invalidation note for old artifacts, and a "GPU benchmarks not run on MacBook" note before merge.

### Post-PR Validation Split

- MacBook-safe: `git diff --check`, `python3 -m py_compile` for changed Python files, `rg` checks for banned or legacy claims, and schema validation on a small synthetic JSONL.
- CPU/CI: `make check`, `make test`, targeted tests for block verification, action-space loading, JSONL schema v3, and report compatibility.
- RTX-only: tiny HF smoke first, then real experiment reruns and regenerated reports. Do not run these on the MacBook.

## Assumptions

- Work happens on `codex/repair-repo-thesis`.
- GitHub remote is `origin`.
- Push goes to the repair branch, not directly to `main`.
- Old results remain preserved as legacy evidence but are not final evidence for block-SD speedup.
- New final thesis claims depend on rerunning the benchmark after block verification is implemented.
