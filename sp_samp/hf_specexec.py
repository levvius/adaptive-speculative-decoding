from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from .hf_adapter import HFModel, KVCacheState
from .specexec import SpecExecError, SpecExecStats

_Prefix = Tuple[int, ...]
_StateCache = Dict[_Prefix, KVCacheState]
_ProbCache = Dict[_Prefix, torch.Tensor]


def _short_prefix(prefix: Sequence[int], tail: int = 8) -> str:
    if len(prefix) <= tail:
        return str(list(prefix))
    return f"...{list(prefix[-tail:])}"


def _validate_probs(
    probs: torch.Tensor,
    vocab_size: int,
    *,
    stage: str,
    prefix: Sequence[int],
) -> torch.Tensor:
    if probs.ndim != 1:
        raise SpecExecError(
            f"{stage}: expected 1D distribution, got shape={tuple(probs.shape)} "
            f"(prefix_len={len(prefix)}, prefix_tail={_short_prefix(prefix)})"
        )
    if int(probs.shape[0]) != vocab_size:
        raise SpecExecError(
            f"{stage}: expected distribution length={vocab_size}, got {int(probs.shape[0])} "
            f"(prefix_len={len(prefix)}, prefix_tail={_short_prefix(prefix)})"
        )
    if not torch.isfinite(probs).all():
        raise SpecExecError(
            f"{stage}: non-finite probability values detected "
            f"(prefix_len={len(prefix)}, prefix_tail={_short_prefix(prefix)})"
        )
    if float(torch.min(probs).item()) < 0.0:
        raise SpecExecError(
            f"{stage}: negative probability values detected "
            f"(prefix_len={len(prefix)}, prefix_tail={_short_prefix(prefix)})"
        )
    total = float(torch.sum(probs).item())
    if total <= 0.0:
        raise SpecExecError(
            f"{stage}: probability sum must be positive, got {total} "
            f"(prefix_len={len(prefix)}, prefix_tail={_short_prefix(prefix)})"
        )
    return probs / total


def _probs_from_state(
    state: KVCacheState,
    *,
    vocab_size: int,
    stage: str,
    prefix: Sequence[int],
) -> torch.Tensor:
    probs = torch.softmax(state.logits, dim=-1).squeeze(0)
    return _validate_probs(probs, vocab_size, stage=stage, prefix=prefix)


def _select_candidate_tokens(
    probs: torch.Tensor,
    prune_threshold: float,
    max_tokens: int,
) -> Tuple[List[int], int]:
    max_prob = float(torch.max(probs).item())
    if prune_threshold <= 0.0:
        candidates = torch.arange(probs.shape[0], device=probs.device)
    else:
        cutoff = max_prob * prune_threshold
        candidates = torch.nonzero(probs >= cutoff, as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            candidates = torch.argmax(probs).reshape(1)

    scores = probs.index_select(0, candidates)
    ordered = torch.argsort(scores, descending=True)
    selected = candidates.index_select(0, ordered)[:max_tokens]
    pruned = int(probs.shape[0] - selected.shape[0])
    return [int(token.item()) for token in selected], pruned


def _build_draft_tree_with_kv(
    draft_model: HFModel,
    root_prefix: _Prefix,
    *,
    max_depth: int,
    parallel_branches: int,
    branch_prune_threshold: float,
    eos_id: Optional[int],
    eps: float,
    stats: SpecExecStats,
) -> Tuple[Set[_Prefix], _StateCache, _ProbCache]:
    tree_nodes: Set[_Prefix] = {root_prefix}

    # One prefill for root, then only token-level KV steps for selected children.
    root_state = draft_model.prefill(root_prefix)
    stats.draft_prefills += 1
    stats.draft_calls += 1
    state_cache: _StateCache = {root_prefix: root_state}
    prob_cache: _ProbCache = {
        root_prefix: _probs_from_state(
            root_state,
            vocab_size=draft_model.vocab_size,
            stage="draft_tree_root",
            prefix=root_prefix,
        )
    }

    frontier: List[Tuple[_Prefix, float]] = [(root_prefix, 0.0)]
    stats.max_active_branches = max(stats.max_active_branches, len(frontier))

    for depth in range(max_depth):
        candidates: List[Tuple[float, _Prefix, _Prefix, int]] = []

        for prefix, score in frontier:
            if eos_id is not None and prefix and prefix[-1] == eos_id:
                continue
            q = prob_cache[prefix]
            token_ids, pruned = _select_candidate_tokens(
                q,
                prune_threshold=branch_prune_threshold,
                max_tokens=parallel_branches,
            )
            stats.branches_kept += len(token_ids)
            stats.branches_pruned += pruned
            for token_id in token_ids:
                child_prefix = prefix + (token_id,)
                child_score = score + math.log(max(float(q[token_id].item()), eps))
                candidates.append((child_score, child_prefix, prefix, token_id))

        if not candidates:
            break

        candidates.sort(key=lambda item: (-item[0], item[1]))
        next_frontier_meta: List[Tuple[float, _Prefix, _Prefix, int]] = []
        seen: Set[_Prefix] = set()
        for candidate in candidates:
            _, child_prefix, _, _ = candidate
            if child_prefix in seen:
                continue
            next_frontier_meta.append(candidate)
            seen.add(child_prefix)
            if len(next_frontier_meta) >= parallel_branches:
                break

        if not next_frontier_meta:
            break

        next_frontier: List[Tuple[_Prefix, float]] = []
        # Level-order expansion over selected tree edges.
        for child_score, child_prefix, parent_prefix, token_id in next_frontier_meta:
            parent_state = state_cache[parent_prefix]
            child_state = draft_model.step([token_id], parent_state)
            stats.draft_calls += 1
            state_cache[child_prefix] = child_state
            prob_cache[child_prefix] = _probs_from_state(
                child_state,
                vocab_size=draft_model.vocab_size,
                stage=f"draft_tree_depth_{depth + 1}",
                prefix=child_prefix,
            )
            tree_nodes.add(child_prefix)
            next_frontier.append((child_prefix, child_score))

        frontier = next_frontier
        stats.max_active_branches = max(stats.max_active_branches, len(frontier))
        stats.branches_total += len(frontier)

    return tree_nodes, state_cache, prob_cache


def _fill_target_cache_with_kv(
    target_model: HFModel,
    root_prefix: _Prefix,
    node_prefixes: Set[_Prefix],
    *,
    stats: SpecExecStats,
) -> _ProbCache:
    if root_prefix not in node_prefixes:
        raise SpecExecError(
            "target_cache_fill: root prefix is missing from draft tree "
            f"(prefix_len={len(root_prefix)}, prefix_tail={_short_prefix(root_prefix)})"
        )

    children_by_parent: Dict[_Prefix, List[int]] = {}
    for prefix in node_prefixes:
        if prefix == root_prefix:
            continue
        parent = prefix[:-1]
        if parent not in node_prefixes:
            raise SpecExecError(
                "target_cache_fill: tree node parent is missing "
                f"(node_len={len(prefix)}, node_tail={_short_prefix(prefix)})"
            )
        children_by_parent.setdefault(parent, []).append(int(prefix[-1]))

    for parent in children_by_parent:
        children_by_parent[parent].sort()

    root_state = target_model.prefill(root_prefix)
    stats.target_prefills += 1
    stats.target_calls += 1
    target_state_cache: _StateCache = {root_prefix: root_state}
    target_prob_cache: _ProbCache = {
        root_prefix: _probs_from_state(
            root_state,
            vocab_size=target_model.vocab_size,
            stage="target_cache_root",
            prefix=root_prefix,
        )
    }

    frontier: List[_Prefix] = [root_prefix]
    visited: Set[_Prefix] = {root_prefix}

    # Depth-wise pass through the draft tree with KV reuse on every edge.
    while frontier:
        next_frontier: List[_Prefix] = []
        for parent_prefix in frontier:
            parent_state = target_state_cache[parent_prefix]
            for token_id in children_by_parent.get(parent_prefix, []):
                child_prefix = parent_prefix + (token_id,)
                child_state = target_model.step([token_id], parent_state)
                stats.target_calls += 1
                target_state_cache[child_prefix] = child_state
                target_prob_cache[child_prefix] = _probs_from_state(
                    child_state,
                    vocab_size=target_model.vocab_size,
                    stage="target_cache_fill",
                    prefix=child_prefix,
                )
                next_frontier.append(child_prefix)
                visited.add(child_prefix)
        frontier = next_frontier

    if len(visited) != len(node_prefixes):
        missing = sorted(node_prefixes - visited, key=lambda p: (len(p), p))
        example = missing[0]
        raise SpecExecError(
            "target_cache_fill: some tree nodes were not visited "
            f"(missing_len={len(example)}, missing_tail={_short_prefix(example)})"
        )
    return target_prob_cache


def _prefill_cache(
    target_model: HFModel,
    draft_model: HFModel,
    current_prefix: _Prefix,
    *,
    k: int,
    parallel_branches: int,
    branch_prune_threshold: float,
    eos_id: Optional[int],
    eps: float,
    stats: SpecExecStats,
) -> _ProbCache:
    stats.steps += 1

    tree_nodes, _, _ = _build_draft_tree_with_kv(
        draft_model=draft_model,
        root_prefix=current_prefix,
        max_depth=k,
        parallel_branches=parallel_branches,
        branch_prune_threshold=branch_prune_threshold,
        eos_id=eos_id,
        eps=eps,
        stats=stats,
    )
    if current_prefix not in tree_nodes:
        tree_nodes.add(current_prefix)
    return _fill_target_cache_with_kv(
        target_model=target_model,
        root_prefix=current_prefix,
        node_prefixes=tree_nodes,
        stats=stats,
    )


def _sample_token(probs: torch.Tensor) -> int:
    token = torch.multinomial(probs, num_samples=1)
    return int(token.item())


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
    cache: _ProbCache = {}
    stats = SpecExecStats()
    prompt_prefix = tuple(int(t) for t in prompt_tokens)

    while len(generated) < max_new_tokens:
        current_prefix = prompt_prefix + tuple(generated)
        if current_prefix not in cache:
            stats.cache_misses += 1
            try:
                cache = _prefill_cache(
                    target_model=target_model,
                    draft_model=draft_model,
                    current_prefix=current_prefix,
                    k=k,
                    parallel_branches=parallel_branches,
                    branch_prune_threshold=branch_prune_threshold,
                    eos_id=eos_id,
                    eps=eps,
                    stats=stats,
                )
            except SpecExecError:
                raise
            except Exception as exc:  # pragma: no cover
                raise SpecExecError(
                    f"spec_exec_prefill_failed: {exc} "
                    f"(generated={len(generated)}, prefix_len={len(current_prefix)}, "
                    f"prefix_tail={_short_prefix(current_prefix)})"
                ) from exc
            if current_prefix not in cache:
                raise SpecExecError(
                    "spec_exec_prefill_missing_current_prefix "
                    f"(prefix_len={len(current_prefix)}, prefix_tail={_short_prefix(current_prefix)})"
                )
        else:
            stats.cache_hits += 1

        token = _sample_token(cache[current_prefix])
        generated.append(token)
        stats.target_tokens += 1
        stats.proposed += 1
        stats.accepted += 1
        if eos_id is not None and token == eos_id:
            break

    if return_stats:
        return generated, stats
    return generated

