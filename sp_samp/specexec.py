from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from .models import BaseModel
from .sampling import _weighted_choice


@dataclass
class SpecExecStats:
    proposed: int = 0
    accepted: int = 0
    steps: int = 0
    target_tokens: int = 0
    rejections: int = 0

    branches_total: int = 0
    branches_kept: int = 0
    branches_pruned: int = 0

    target_calls: int = 0
    draft_calls: int = 0
    target_prefills: int = 0
    draft_prefills: int = 0
    max_active_branches: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def avg_tokens_per_step(self) -> float:
        return self.target_tokens / self.steps if self.steps else 0.0

    @property
    def branch_prune_rate(self) -> float:
        total = self.branches_kept + self.branches_pruned
        return self.branches_pruned / total if total else 0.0

    @property
    def effective_parallelism(self) -> float:
        return self.branches_total / self.steps if self.steps else 0.0

    @property
    def target_calls_per_token(self) -> float:
        return self.target_calls / self.target_tokens if self.target_tokens else 0.0

    @property
    def draft_calls_per_token(self) -> float:
        return self.draft_calls / self.target_tokens if self.target_tokens else 0.0


@dataclass
class _Branch:
    tokens: List[int]
    draft_probs: List[List[float]]
    score: float


def _filter_candidates(
    probs: Sequence[float], prune_threshold: float
) -> Tuple[List[int], List[float], int]:
    if not probs:
        raise ValueError("Distribution is empty.")
    max_prob = max(float(p) for p in probs)
    if prune_threshold <= 0.0:
        indices = list(range(len(probs)))
    else:
        cutoff = max_prob * prune_threshold
        indices = [i for i, p in enumerate(probs) if float(p) >= cutoff]
        if not indices:
            best = max(range(len(probs)), key=lambda idx: float(probs[idx]))
            indices = [best]
    weights = [float(probs[idx]) for idx in indices]
    return indices, weights, len(probs) - len(indices)


def _sample_from_candidates(
    rng: random.Random,
    indices: Sequence[int],
    weights: Sequence[float],
) -> int:
    total = float(sum(weights))
    chosen = _weighted_choice(rng, weights, total=total)
    return int(indices[chosen])


def _build_branches(
    draft_model: BaseModel,
    prompt_tokens: Sequence[int],
    generated: Sequence[int],
    block_size: int,
    parallel_branches: int,
    prune_threshold: float,
    rng: random.Random,
    stats: SpecExecStats,
    eos_id: Optional[int],
    eps: float,
) -> List[_Branch]:
    prefix = list(prompt_tokens) + list(generated)
    root_q = draft_model.next_token_probs(prefix)
    stats.draft_calls += 1
    root_indices, _, pruned = _filter_candidates(root_q, prune_threshold)
    stats.branches_kept += len(root_indices)
    stats.branches_pruned += pruned

    first_tokens = sorted(root_indices, key=lambda idx: root_q[idx], reverse=True)[
        :parallel_branches
    ]
    if not first_tokens:
        first_tokens = [max(range(len(root_q)), key=lambda idx: root_q[idx])]

    branches: List[_Branch] = []
    for first in first_tokens:
        tokens = [int(first)]
        draft_probs: List[List[float]] = [list(root_q)]
        score = math.log(max(float(root_q[first]), eps))
        if eos_id is not None and first == eos_id:
            branches.append(_Branch(tokens=tokens, draft_probs=draft_probs, score=score))
            continue

        for _ in range(1, block_size):
            context = prefix + tokens
            q = draft_model.next_token_probs(context)
            stats.draft_calls += 1
            indices, weights, pruned = _filter_candidates(q, prune_threshold)
            stats.branches_kept += len(indices)
            stats.branches_pruned += pruned
            token = _sample_from_candidates(rng, indices, weights)
            tokens.append(token)
            draft_probs.append(list(q))
            score += math.log(max(float(q[token]), eps))
            if eos_id is not None and token == eos_id:
                break
        branches.append(_Branch(tokens=tokens, draft_probs=draft_probs, score=score))

    stats.branches_total += len(branches)
    stats.max_active_branches = max(stats.max_active_branches, len(branches))
    return branches


def _evaluate_branch(
    target_model: BaseModel,
    prompt_tokens: Sequence[int],
    generated: Sequence[int],
    branch: _Branch,
    rng: random.Random,
    stats: SpecExecStats,
    eps: float,
) -> int:
    accepted = 0
    for i, token in enumerate(branch.tokens):
        context = list(prompt_tokens) + list(generated) + branch.tokens[:i]
        p = target_model.next_token_probs(context)
        stats.target_calls += 1
        q = branch.draft_probs[i]
        alpha = min(1.0, float(p[token]) / max(float(q[token]), eps))
        if rng.random() <= alpha:
            accepted += 1
            continue
        break
    return accepted


def specexec_sample(
    target_model: BaseModel,
    draft_model: BaseModel,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    k: int,
    parallel_branches: int = 8,
    branch_prune_threshold: float = 0.0,
    rng: Optional[random.Random] = None,
    eos_id: Optional[int] = None,
    eps: float = 1e-12,
    return_stats: bool = False,
) -> Union[List[int], Tuple[List[int], SpecExecStats]]:
    """Simplified SpecExec-style decoding with draft branch execution."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if parallel_branches <= 0:
        raise ValueError("parallel_branches must be positive.")
    if not (0.0 <= branch_prune_threshold <= 1.0):
        raise ValueError("branch_prune_threshold must be in [0, 1].")
    if target_model.vocab_size != draft_model.vocab_size:
        raise ValueError("target and draft vocab sizes must match.")
    if rng is None:
        rng = random.Random()

    generated: List[int] = []
    stats = SpecExecStats()

    while len(generated) < max_new_tokens:
        stats.steps += 1
        remaining = max_new_tokens - len(generated)
        block_size = min(k, remaining)

        branches = _build_branches(
            draft_model=draft_model,
            prompt_tokens=prompt_tokens,
            generated=generated,
            block_size=block_size,
            parallel_branches=parallel_branches,
            prune_threshold=branch_prune_threshold,
            rng=rng,
            stats=stats,
            eos_id=eos_id,
            eps=eps,
        )
        if not branches:
            raise RuntimeError("SpecExec failed to build candidate branches.")

        accepted_by_branch = [
            _evaluate_branch(
                target_model=target_model,
                prompt_tokens=prompt_tokens,
                generated=generated,
                branch=branch,
                rng=rng,
                stats=stats,
                eps=eps,
            )
            for branch in branches
        ]
        best_idx = max(
            range(len(branches)),
            key=lambda idx: (accepted_by_branch[idx], branches[idx].score),
        )
        best_branch = branches[best_idx]
        best_accepted = accepted_by_branch[best_idx]

        stats.proposed += len(best_branch.tokens)
        stats.accepted += best_accepted

        for token in best_branch.tokens[:best_accepted]:
            generated.append(token)
            stats.target_tokens += 1
            if eos_id is not None and token == eos_id:
                if return_stats:
                    return generated, stats
                return generated
            if len(generated) >= max_new_tokens:
                if return_stats:
                    return generated, stats
                return generated

        context = list(prompt_tokens) + generated
        p = target_model.next_token_probs(context)
        stats.target_calls += 1

        if best_accepted < len(best_branch.tokens):
            stats.rejections += 1
            q_rejected = best_branch.draft_probs[best_accepted]
            residual = [max(float(pi) - float(qi), 0.0) for pi, qi in zip(p, q_rejected)]
            residual_total = float(sum(residual))
            if residual_total <= 0.0:
                residual = list(p)
                residual_total = float(sum(residual))
            token = _weighted_choice(rng, residual, total=residual_total)
        else:
            token = _weighted_choice(rng, p)

        generated.append(token)
        stats.target_tokens += 1
        if eos_id is not None and token == eos_id:
            if return_stats:
                return generated, stats
            return generated

    if return_stats:
        return generated, stats
    return generated
