# GitHub Repository Setup

Use this checklist to make the repository easier to review.

## 1) About section

Set in repository settings:

- Description:
  `Research playground for speculative decoding: Baseline, SpS, AutoJudge, Top-K, and SpecExec with reproducible benchmarks.`
- Website (optional): link to project page or benchmark article.
- Topics:
  `llm`, `inference`, `speculative-decoding`, `benchmarking`, `pytorch`, `huggingface`, `autojudge`, `research-project`.

## 2) Social preview

Upload a simple banner (for example 1280x640) with:
- project name,
- one-line value proposition,
- method names.

## 3) Labels

Recommended labels:
- `bug`
- `enhancement`
- `docs`
- `benchmark`
- `good first issue`
- `question`

## 4) Optional GitHub CLI commands

If `gh` is installed and authenticated:

```bash
gh repo edit levvius/adaptive-speculative-decoding \
  --description "Research playground for speculative decoding with reproducible benchmarks" \
  --add-topic llm --add-topic inference --add-topic speculative-decoding --add-topic benchmarking
```
