# Commission Q&A — JointAdaSpec Defense Snapshot

This note lists the short answers to repository-driven questions a committee
member may ask after following the defense QR code. The QR points to the
immutable `v1-defense` snapshot.

## Why are the results tied to the defense snapshot?

The defense uses the immutable `v1-defense` snapshot. Run 1, Run 2, Experiment
E, the κ-sweep, and the Theorem D analysis are artifact-backed thesis results
for that snapshot. The `main` branch continues post-defense research, including
fresh block-v1 reruns and stricter artifact metadata.

## Can I defend `+4.07 pp, p=0.0205`?

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
