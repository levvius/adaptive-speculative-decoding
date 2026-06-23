---
name: core-feature-or-refactor-with-tests
description: Workflow command scaffold for core-feature-or-refactor-with-tests in adaptive-speculative-decoding.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /core-feature-or-refactor-with-tests

Use this workflow when working on **core-feature-or-refactor-with-tests** in `adaptive-speculative-decoding`.

## Goal

Add or refactor core algorithmic code and update corresponding tests to ensure correctness.

## Common Files

- `jointadaspec/utils/probs.py`
- `sp_samp/hf_adapter.py`
- `jointadaspec/baselines/fixed_sd.py`
- `jointadaspec/core/sd_base.py`
- `jointadaspec/inference/jointadaspec.py`
- `jointadaspec/analysis/conditions.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Modify or add core implementation files in jointadaspec/ or sp_samp/
- Update or add corresponding test files in tests/
- Commit changes with feat: or refactor: prefix

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.