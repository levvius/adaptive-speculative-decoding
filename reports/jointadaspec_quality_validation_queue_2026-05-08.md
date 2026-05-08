# JointAdaSpec Quality Validation Queue

Date: `2026-05-08`

## Objective

Validate whether the `2026-05-05` quality-aware Qwen `7B -> 1.5B` policy improves GSM8K exact match on held-out prompts, or whether the apparent improvement was noise or a generic speculative-decoding effect.

## Fixed Primary Claim

- Primary method: `cascade_verif_then_length`
- Baseline: `target_only`
- Main controls: `speculative`, `jointadaspec`
- Main dataset slice: GSM8K test split with `test_start_index=100`
- Main budget: `500` prompts, `3` seeds, `256` max new tokens
- Success criterion: positive paired EM delta for `cascade_verif_then_length` against `target_only`
- Strong success criterion: paired bootstrap CI lower bound above `0`

## Queue

1. Run Qwen `7B -> 1.5B` held-out validation using the fixed policy from `outputs/jointadaspec_qwen7b_1p5b_quality_2026-05-05/02_solve/policy.npz`.
2. If the main run finishes before the 36-hour mark, run a secondary Qwen `14B -> 0.5B` cross-check on `200` held-out GSM8K prompts.
3. If the secondary cross-check fails early, extend the Qwen `7B -> 1.5B` held-out validation to `800` prompts when at least four hours remain.

## Entrypoint

```bash
bash scripts/run_jointadaspec_quality_validation_queue.sh
```

Expected log:

```bash
logs/jointadaspec_quality_validation_queue_2026-05-08.log
```
