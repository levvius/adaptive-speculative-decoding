# Contributing

Thanks for contributing.

This project is a research-style codebase, so reproducibility and clear experiment hygiene matter more than volume of code changes.

## Local Setup

```bash
make setup
make check
make test
```

If you work on GPU paths, install GPU extras:

```bash
make setup-gpu
```

## Development Workflow

1. Create a branch from `main`.
2. Keep changes scoped to one concern (feature, fix, docs, or results).
3. Run checks before opening a PR.
4. Include rationale in commit messages and PR description.

## Required Checks Before PR

```bash
make check
make test
```

For benchmark-related changes, also run at least one smoke benchmark:

```bash
make bench-toy OUT=/tmp/bench_toy.jsonl
.venv/bin/python scripts/validate_results_jsonl.py --path /tmp/bench_toy.jsonl --strict
```

## Benchmark and Artifact Policy

- `datasets/` is for local data and run outputs (gitignored by default).
- `reports/` contains tracked summary artifacts intended for sharing.
- `logs/` is local runtime noise and is ignored.
- Prefer new output filenames for long runs to avoid mixing old and new records.

## AutoJudge Notes

- Keep paper-aligned C-grid: `1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0`.
- Do not introduce `1e1` or `1e2` into preset defaults.
- If behavior changes, update tests in `tests/test_autojudge.py`.

## Code Style

- Follow existing style in touched files.
- Keep interfaces explicit; avoid hidden behavior changes.
- Add comments only for non-obvious logic.

## Pull Request Checklist

- [ ] Scope is clear and focused
- [ ] `make check` passes
- [ ] `make test` passes
- [ ] Relevant docs/configs updated
- [ ] Added or updated tests for logic changes
- [ ] Benchmark outputs validated when applicable

## Reporting Bugs

Please use the bug report template and include:
- exact command used,
- dataset/model preset,
- stack trace or log snippet,
- expected vs actual behavior.
