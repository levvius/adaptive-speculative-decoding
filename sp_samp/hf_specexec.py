from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

from .hf_adapter import HFModel
from .specexec import SpecExecStats


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _sample_from_probs(probs: torch.Tensor) -> int:
    token = torch.multinomial(probs, num_samples=1)
    return int(token.item())


def _filter_indices(probs: torch.Tensor, prune_threshold: float) -> Tuple[torch.Tensor, int]:
    if probs.numel() == 0:
        raise ValueError("Distribution is empty.")
    max_prob = float(torch.max(probs).item())
    if prune_threshold <= 0.0:
        indices = torch.arange(probs.shape[0], device=probs.device)
    else:
        cutoff = max_prob * prune_threshold
        indices = torch.nonzero(probs >= cutoff, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            indices = torch.argmax(probs).reshape(1)
    pruned = int(probs.shape[0] - indices.shape[0])
    return indices, pruned


def _sample_from_candidates(probs: torch.Tensor, indices: torch.Tensor) -> int:
    candidate_probs = probs.index_select(0, indices)
    total = float(candidate_probs.sum().item())
    if total <= 0.0:
        candidate_probs = torch.ones_like(candidate_probs) / float(candidate_probs.numel())
    else:
        candidate_probs = candidate_probs / total
    sampled_pos = _sample_from_probs(candidate_probs)
    return int(indices[sampled_pos].item())


@dataclass
class _HFBranch:
    tokens: List[int]
    draft_probs: List[torch.Tensor]
    score: float


def _build_branches_hf(
    draft_model: HFModel,
    prompt_tokens: Sequence[int],
    generated: Sequence[int],
    block_size: int,
    parallel_branches: int,
    prune_threshold: float,
    stats: SpecExecStats,
    eos_id: Optional[int],
    eps: float,
) -> List[_HFBranch]:
    prefix = list(prompt_tokens) + list(generated)
    root_state = draft_model.prefill(prefix)
    stats.draft_prefills += 1
    stats.draft_calls += 1
    root_q = _softmax(root_state.logits).squeeze(0)
    root_indices, pruned = _filter_indices(root_q, prune_threshold)
    stats.branches_kept += int(root_indices.shape[0])
    stats.branches_pruned += pruned

    scores = root_q.index_select(0, root_indices)
    sorted_pos = torch.argsort(scores, descending=True)
    ordered = root_indices.index_select(0, sorted_pos)
    first_tokens = ordered[:parallel_branches].tolist()
    if not first_tokens:
        first_tokens = [int(torch.argmax(root_q).item())]

    branches: List[_HFBranch] = []
    for first in first_tokens:
        tokens = [int(first)]
        draft_probs = [root_q]
        score = math.log(max(float(root_q[first].item()), eps))
        if eos_id is not None and first == eos_id:
            branches.append(_HFBranch(tokens=tokens, draft_probs=draft_probs, score=score))
            continue

        if block_size <= 1:
            branches.append(_HFBranch(tokens=tokens, draft_probs=draft_probs, score=score))
            continue

        branch_state = draft_model.step([first], root_state)
        stats.draft_calls += 1
        for _ in range(1, block_size):
            q = _softmax(branch_state.logits).squeeze(0)
            indices, pruned = _filter_indices(q, prune_threshold)
            stats.branches_kept += int(indices.shape[0])
            stats.branches_pruned += pruned
            token = _sample_from_candidates(q, indices)
            tokens.append(token)
            draft_probs.append(q)
            score += math.log(max(float(q[token].item()), eps))
            if eos_id is not None and token == eos_id:
                break
            branch_state = draft_model.step([token], branch_state)
            stats.draft_calls += 1
        branches.append(_HFBranch(tokens=tokens, draft_probs=draft_probs, score=score))

    stats.branches_total += len(branches)
    stats.max_active_branches = max(stats.max_active_branches, len(branches))
    return branches


def _evaluate_branch_hf(
    target_model: HFModel,
    prompt_tokens: Sequence[int],
    generated: Sequence[int],
    branch: _HFBranch,
    stats: SpecExecStats,
    eps: float,
) -> int:
    prefix = list(prompt_tokens) + list(generated)
    target_state = target_model.prefill(prefix)
    stats.target_prefills += 1
    stats.target_calls += 1
    accepted = 0
    for i, token in enumerate(branch.tokens):
        p = _softmax(target_state.logits).squeeze(0)
        q = branch.draft_probs[i]
        alpha = min(1.0, float(p[token].item()) / max(float(q[token].item()), eps))
        if torch.rand(1).item() <= alpha:
            accepted += 1
            if i + 1 >= len(branch.tokens):
                break
            target_state = target_model.step([token], target_state)
            stats.target_calls += 1
            continue
        break
    return accepted


def specexec_sample_hf(
    target_model: HFModel,
    draft_model: HFModel,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    k: int,
    parallel_branches: int = 8,
    branch_prune_threshold: float = 0.0,
    eos_id: Optional[int] = None,
    seed: Optional[int] = None,
    eps: float = 1e-12,
    return_stats: bool = False,
) -> Union[List[int], Tuple[List[int], SpecExecStats]]:
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
    if seed is not None:
        torch.manual_seed(seed)

    generated: List[int] = []
    stats = SpecExecStats()

    with torch.no_grad():
        while len(generated) < max_new_tokens:
            stats.steps += 1
            remaining = max_new_tokens - len(generated)
            block_size = min(k, remaining)

            branches = _build_branches_hf(
                draft_model=draft_model,
                prompt_tokens=prompt_tokens,
                generated=generated,
                block_size=block_size,
                parallel_branches=parallel_branches,
                prune_threshold=branch_prune_threshold,
                stats=stats,
                eos_id=eos_id,
                eps=eps,
            )
            if not branches:
                raise RuntimeError("SpecExec failed to build candidate branches.")

            accepted_by_branch = [
                _evaluate_branch_hf(
                    target_model=target_model,
                    prompt_tokens=prompt_tokens,
                    generated=generated,
                    branch=branch,
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

            prefix = list(prompt_tokens) + generated
            target_state = target_model.prefill(prefix)
            stats.target_prefills += 1
            stats.target_calls += 1
            p = _softmax(target_state.logits).squeeze(0)

            if best_accepted < len(best_branch.tokens):
                stats.rejections += 1
                q_rejected = best_branch.draft_probs[best_accepted].to(p.device)
                residual = torch.clamp(p - q_rejected, min=0.0)
                total = float(residual.sum().item())
                if total <= 0.0:
                    residual = p
                    total = float(residual.sum().item())
                residual = residual / total
                token = _sample_from_probs(residual)
            else:
                token = _sample_from_probs(p)

            generated.append(token)
            stats.target_tokens += 1
            if eos_id is not None and token == eos_id:
                if return_stats:
                    return generated, stats
                return generated

    if return_stats:
        return generated, stats
    return generated
