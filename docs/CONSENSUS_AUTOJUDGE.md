# Consensus AutoJudge

## Positioning

`consensus_autojudge` is an approximate two-draft decoding method.

- `D1`: fast / weak draft
- `D2`: stronger / slower draft
- `T`: target model

The method is inspired by speculative decoding and staged / multi-candidate drafting,
but it is not presented as a first ensemble-decoding method. The concrete contribution
in this repository is a disagreement-aware, learned gate over two separate draft models.

## Hypothesis

Agreement topology between `D1` and `D2` is a useful reliability signal.

If the drafts agree with high confidence, `D1` can often be accepted cheaply.
If they disagree but `D2` is clearly more confident, escalating to `D2` can avoid a
target call without taking the full quality hit of trusting `D1`.
If both drafts look unstable, the decoder should fall back to the target.

## Runtime Policy

At each token decision inside a draft block:

1. `D1` proposes `y1`.
2. `D2` proposes `y2` and scores `y1`.
3. Ensemble features are computed from `D1` and `D2` only.
4. A 3-way gate predicts one of:
   - `ACCEPT_D1`
   - `ESCALATE_TO_D2`
   - `FALLBACK_TO_TARGET`

This method is approximate. It does not preserve the exact target distribution.

## Feature Families

The first implementation uses lightweight ensemble-aware features:

- top-1 agreement
- rank/probability of each draft's token under the other draft
- top-1 margins and entropies for both drafts
- truncated top-m Jensen-Shannon divergence
- confidence gap between drafts
- agreement streak history
- position within the current draft block

## Training Labels

The first implementation mines a token-level oracle from target greedy generations.

- `ACCEPT_D1`: `D1`'s top-1 token matches the target token
- `ESCALATE_TO_D2`: `D1` misses but `D2` matches the target token
- `FALLBACK_TO_TARGET`: neither draft matches the target token

This choice keeps the first version computationally practical while still testing the
core hypothesis that cross-draft agreement helps separate the three regimes.

## Falsifiable Outcome

The method is only useful if the learned gate or rule gate improves the quality /
latency tradeoff against the current approximate baselines, especially AutoJudge.

Primary metrics:

- GSM8K exact match
- tokens/sec
- target calls per token
- fallback rate
- action distribution (`ACCEPT_D1`, `ESCALATE_TO_D2`, `FALLBACK_TO_TARGET`)
