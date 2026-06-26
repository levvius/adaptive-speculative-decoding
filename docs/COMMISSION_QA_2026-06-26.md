# Commission Q&A — JointAdaSpec Defense Snapshot

This note lists the short answers to repository-driven questions a committee
member may ask after following the defense QR code. The QR points to the
immutable `v1-defense` snapshot.

## Why are the results tied to the defense snapshot?

The defense uses the immutable `v1-defense` snapshot. Run 1, Run 2, Experiment
E, the κ-sweep, and the Theorem D analysis are artifact-backed thesis results
for that snapshot. The `main` branch continues post-defense research, including
fresh block-v1 reruns and stricter artifact metadata.

## Can I defend `+4.07 pp, p=0.0203`?

Yes, as the Run 1 result of the archived thesis snapshot. The precise wording is
that the defense snapshot shows `+4.07 pp` EM for JointAdaSpec versus
`target_only` under the locked Run 1 protocol. New post-defense block-v1 claims
belong to future `main`-branch reruns.

## Is the artifact/version boundary a bug in the method?

No. It is an engineering/versioning boundary between the archived thesis
snapshot and the newer research code path. It does not contradict the MDP
formulation, the theory, or the locked thesis-snapshot results.

## Why is target-only faster than JointAdaSpec?

On a single GPU and for these model ratios, speculative decoding can lose
throughput to extra draft/verification overhead. The honest speed claim is that
JointAdaSpec recovers throughput relative to vanilla speculative decoding, not
that it beats plain target autoregressive generation.

## Why is joint almost equal to cascade?

That is an expected and useful result. Theorem D explains the small value gap by
the advantage-weighted difference between the joint and cascade policies on the
C4-violating states. Empirically that advantage is tiny, so joint and cascade are
statistically indistinguishable at power.

## Why not call Run 1 final anyway?

Run 1 is final for the archived thesis snapshot. What it is not is a new
post-defense rerun under the latest `main`-branch block-v1 artifact protocol.
Keeping those two scopes separate avoids mixing thesis results with future
research claims.

## Why does the QR point to a tag instead of `main`?

The tag freezes the exact defense snapshot. It prevents accidental later commits
on `main` from changing what the committee sees during questions.

---

## Answers to automated code-review findings

These pre-empt the questions an LLM code-review tool (e.g. Claude Code) typically
raises when pointed at the repository. None of them is a correctness bug.

### "How do I run this? The pipeline fails on my machine."

The full pipeline is GPU-bound and download-heavy by design: it needs a high-end
GPU (RTX 5090, 24–32 GB VRAM), tens of GB of model weights (Qwen2.5-14B ≈ 28 GB
plus a draft model) and the GSM8K dataset. On a laptop/CPU it will not run — this
is expected, not a defect. The CPU-only checks that *do* run in minutes without
downloads are listed at the top of `README.MD` (`make test` — 94 tests,
`make bench-toy`, `make check`, `make list-presets`, `make validate-configs`).

### "Why is the locked run's raw `benchmark.csv` / `results.jsonl` not in the repo?"

Large per-prompt artifacts under `outputs/jointadaspec_*_20*/` are excluded by
`.gitignore`, so they are not in the cloned snapshot. The aggregated reports that
carry the headline numbers (`reports/jointadaspec_quality_qwen14b_0p5b_lock_2026-05-14.{md,json}`)
*are* committed. This is the standard "commit the summary, not the multi-MB raw
log" hygiene; the rerun plan to regenerate raw artifacts lives on `main`.

### "`JointAction.is_stop` and `is_verify` have identical bodies — is that a bug?"

No. In this MDP formulation `verify` and `stop` are the *same* terminal action:
verifying the accumulated draft block ends (stops) the drafting round. Both names
are intentional aliases used by `estimation.py` and `test_mdp_solver.py`; the
behaviour is covered by the passing test suite.

### "SpecExec reports `acceptance_rate == 1.0` — isn't that wrong?"

No. SpecExec is an *exact* sampler that emits exact target draws, so by
construction every emitted token is both proposed and accepted. The interesting
speculative dynamics (branch expansion/pruning) are tracked in the separate
`SpecExecStats` branch fields, not in the token-level acceptance ratio. SpecExec
is a comparison baseline, not the thesis method.

### "Theorem B is labelled 'corrected/wrong' — is the theory broken?"

No — that is a deliberate correction documented as a result. A naïve state-only
additive quality-risk penalty is *not* policy-invariant in a general MDP; the
thesis flags this and uses the valid form. Catching and correcting it is part of
the theoretical contribution, not a defect.

### "The Run 1 policy uses 16 actions but the code defines 9 — mismatch?"

That is the archived-vs-current artifact boundary. The locked Run 1 policy is a
pre-repair 16-action artifact (no `action_space_version` / `decoder_semantics`
metadata); the current `block_verify_v1` code path uses 9 actions and validates
artifact metadata so old and new policies cannot be mixed implicitly. The locked
numbers are defended as the archived thesis snapshot; fresh 9-action reruns are a
`main`-branch task. See `docs/DEFENSE_AUDIT_2026-06-26.md`.
