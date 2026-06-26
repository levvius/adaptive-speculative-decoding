# Defense Audit Snapshot (2026-06-26)

This note records the repository facts used for the final defense narrative.
It is intentionally conservative: the defense should not rely on any claim that
the code and released artifacts cannot support when the repository is opened
through the QR code.

## Facts Checked

1. **Current decoder semantics.** The repaired code path is block-oriented:
   `JointAdaSpecDecoder` calls `block_next_token_probs_tensor`, and
   `HFModel.next_token_probs_block` scores `context + continuation` in one
   Hugging Face forward pass. The current `ActionSpace` has 9 actions:
   `continue + verify@T` for the 8 configured thresholds.

2. **Run 1 policy artifact.** The locked Qwen 14B -> 0.5B Run 1 report points
   to `outputs/jointadaspec_qwen14b_0p5b_2026-04-28/02_solve/policy.npz`.
   That artifact predates the semantic metadata update: it has no
   `action_space_version` or `decoder_semantics` metadata, and its `pi_star`
   contains action indices up to 15. It is therefore an archived 16-action policy
   artifact in the thesis snapshot; future `main` reruns use current
   `block_verify_v1` 9-action policies.

3. **Raw lock benchmark availability.** The report
   `reports/jointadaspec_quality_qwen14b_0p5b_lock_2026-05-14.json` references
   `outputs/jointadaspec_qwen14b_0p5b_lock_2026-05-14/03_bench_gsm8k/benchmark.csv`,
   but that raw benchmark file is not present in this working copy. Prompt-level
   clustered statistics for the locked run cannot be recomputed here unless the
   raw artifact is restored from the run machine or backup.

4. **Local checks.** `make check` passes in this working copy. Targeted pytest
   collection requires the project runtime dependencies (`torch`, `pandas`,
   etc.); without the project virtual environment those tests fail at import
   time, not at an asserted behavior.

## Defense Narrative Decision

The codebase can honestly claim that the defense uses an immutable thesis
snapshot and that the block-verification implementation has since been guarded
by semantic metadata. The large-scale numbers remain artifact-backed thesis
results for the archived snapshot. New post-defense block-v1 benchmark claims
should be made from `main` only after a fresh rerun with regenerated artifacts
and prompt-level clustered statistics.

The final defense should therefore emphasize:

- the MDP formulation of joint control over draft length and verification
  threshold;
- the theory results and the joint-vs-cascade explanation;
- the repaired block-v1 implementation and artifact invalidation safeguards;
- the locked thesis-snapshot results as artifact-backed empirical evidence;
- the reproducible `main`-branch rerun plan needed for future block-v1
  benchmark claims.
