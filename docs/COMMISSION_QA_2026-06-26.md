# Commission Q&A — JointAdaSpec Defense Snapshot

This note lists the short answers to repository-driven questions a committee
member may ask after following the defense QR code. The QR points to the
immutable `v1-defense` snapshot.

## Why does the README say legacy / pre-repair?

The defense audit found a mismatch between the old policy artifact used by the
large Run 1 report and the current repaired block-v1 semantics. The current code
path uses semantic metadata and rejects legacy policies, but the historical Run 1
numbers were produced before that boundary existed. Therefore Run 1 is preserved
as historical evidence, not as final block-v1 proof.

## Can I defend `+4.07 pp, p=0.0205`?

Only as a legacy / pre-repair observation. The defended result is the MDP
formulation, the theory, the repaired block-v1 implementation, artifact
validation, and the reproducible rerun protocol. A final block-v1 benchmark claim
requires a fresh solve and benchmark with regenerated 9-action artifacts.

## Is this a bug in the method?

It is an artifact-versioning problem in the old experiment, not a contradiction
of the method formulation. The repaired loaders now reject policies without
`action_space_version`, `decoder_semantics`, and `config_hash`, so old artifacts
cannot silently be reused as new block-v1 results.

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

Because the Run 1 policy artifact carries legacy action-space evidence, while
the current block-v1 implementation expects a 9-action policy with semantic
metadata. Calling it final would mix two incompatible semantic versions.

## Why does the QR point to a tag instead of `main`?

The tag freezes the exact defense snapshot. It prevents accidental later commits
from changing what the committee sees during questions.

