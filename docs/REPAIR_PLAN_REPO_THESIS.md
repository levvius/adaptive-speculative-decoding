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

## Assumptions

- Work happens on `codex/repair-repo-thesis`.
- GitHub remote is `origin`.
- Push goes to the repair branch, not directly to `main`.
- Old results remain preserved as legacy evidence but are not final evidence for block-SD speedup.
- New final thesis claims depend on rerunning the benchmark after block verification is implemented.
