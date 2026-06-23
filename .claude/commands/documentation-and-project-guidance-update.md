---
name: documentation-and-project-guidance-update
description: Workflow command scaffold for documentation-and-project-guidance-update in adaptive-speculative-decoding.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /documentation-and-project-guidance-update

Use this workflow when working on **documentation-and-project-guidance-update** in `adaptive-speculative-decoding`.

## Goal

Update documentation files and project guidance materials, including READMEs and thesis-related docs.

## Common Files

- `README.MD`
- `CLAUDE.md`
- `docs/RESULTS.md`
- `docs/ROADMAP.md`
- `docs/REPAIR_PLAN_REPO_THESIS.md`
- `papers/build_newera_docx.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or add documentation files in docs/, papers/, or reports/
- Update README.MD or CLAUDE.md as needed
- Commit changes with a docs: prefix

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.