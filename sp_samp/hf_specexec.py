from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from .hf_adapter import HFModel
from .specexec import SpecExecError, SpecExecStats

_Prefix = Tuple[int, ...]


def _short_prefix(prefix: Sequence[int], tail: int = 8) -> str:
    if len(prefix) <= tail:
        return str(list(prefix))
    return f"...{list(prefix[-tail:])}"


def _validate_probs(
    probs: Sequence[float],
    vocab_size: int,
    *,
    stage: str,
    prefix: Sequence[int],
) -> List[float]:
    if len(probs) != vocab_size:
        raise SpecExecError(
            f"{stage}: expected distribution length={vocab_size}, got {len(probs)} "
            f"(prefix_len={len(prefix)}, prefix_tail={_short_prefix(prefix)})"
        )
    normalized: List[float] = []
    total = 0.0
    for idx, value in enumerate(probs):
        p = float(value)
        if not math.isfinite(p):
            raise SpecExecError(
                f"{stage}: non-finite probability at token={idx}: {p} "
                f"(prefix_len={len(prefix)}, prefix_tail={_short_prefix(prefix)})"
            )
        if p < 0.0:
            raise SpecExecError(
                f"{stage}: negative probability at token={idx}: {p} "
                f"(prefix_len={len(prefix)}, prefix_tail={_short_prefix(prefix)})"
            )
        normalized.append(p)
        total += p
    if total <= 0.0:
        raise SpecExecError(
            f"{stage}: probability sum must be positive, got {total} "
            f"(prefix_len={len(prefix)}, prefix_tail={_short_prefix(prefix)})"
        )
    return [p / total for p in normalized]


def _select_candidate_tokens(
    probs: Sequence[float],
    prune_threshold: float,
    max_tokens: int,
) -> Tuple[List[int], int]:
    max_prob = max(float(p) for p in probs)
    if prune_threshold <= 0.0:
        candidates = list(range(len(probs)))
    else:
        cutoff = max_prob * prune_threshold
        candidates = [idx for idx, p in enumerate(probs) if float(p) >= cutoff]
        if not candidates:
            best = max(range(len(probs)), key=lambda idx: float(probs[idx]))
            candidates = [best]

    candidates.sort(key=lambda idx: (-float(probs[idx]), idx))
    if len(candidates) > max_tokens:
        candidates = candidates[:max_tokens]

    pruned = len(probs) - len(candidates)
    return candidates, pruned


def _build_draft_tree(
    draft_model: HFModel,
    root_prefix: _Prefix,
    *,
    max_depth: int,
    parallel_branches: int,
    branch_prune_threshold: float,
    eos_id: Optional[int],
    eps: float,
    stats: SpecExecStats,
) -> Set[_Prefix]:
    tree_nodes: Set[_Prefix] = {root_prefix}
    frontier: List[Tuple[_Prefix, float]] = [(root_prefix, 0.0)]
    stats.max_active_branches = max(stats.max_active_branches, len(frontier))

    for depth in range(max_depth):
        children: List[Tuple[_Prefix, float]] = []
        for prefix, score in frontier:
            if eos_id is not None and prefix and prefix[-1] == eos_id:
                continue
            q_raw = draft_model.next_token_probs(prefix)
            stats.draft_calls += 1
            q = _validate_probs(
                q_raw,
                draft_model.vocab_size,
                stage=f"draft_tree_depth_{depth}",
                prefix=prefix,
            )
            token_ids, pruned = _select_candidate_tokens(
                q,
                prune_threshold=branch_prune_threshold,
                max_tokens=parallel_branches,
            )
            stats.branches_kept += len(token_ids)
            stats.branches_pruned += pruned
            for token_id in token_ids:
                child_prefix = prefix + (int(token_id),)
                child_score = score + math.log(max(float(q[token_id]), eps))
                children.append((child_prefix, child_score))

        if not children:
            break

        children.sort(key=lambda item: (-item[1], item[0]))
        next_frontier: List[Tuple[_Prefix, float]] = []
        seen: Set[_Prefix] = set()
        for candidate in children:
            prefix, _ = candidate
            if prefix in seen:
                continue
            next_frontier.append(candidate)
            seen.add(prefix)
            if len(next_frontier) >= parallel_branches:
                break

        if not next_frontier:
            break

        frontier = next_frontier
        stats.max_active_branches = max(stats.max_active_branches, len(frontier))
        stats.branches_total += len(frontier)
        for prefix, _ in frontier:
            tree_nodes.add(prefix)

    return tree_nodes


def _fill_target_cache(
    target_model: HFModel,
    node_prefixes: Set[_Prefix],
    *,
    stats: SpecExecStats,
) -> Dict[_Prefix, List[float]]:
    cache: Dict[_Prefix, List[float]] = {}
    ordered = sorted(node_prefixes, key=lambda prefix: (len(prefix), prefix))
    for prefix in ordered:
        p_raw = target_model.next_token_probs(prefix)
        stats.target_calls += 1
        cache[prefix] = _validate_probs(
            p_raw,
            target_model.vocab_size,
            stage="target_cache_fill",
            prefix=prefix,
        )
    return cache


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
) -> Dict[_Prefix, List[float]]:
    stats.steps += 1
    stats.target_prefills += 1
    stats.draft_prefills += 1

    tree_nodes = _build_draft_tree(
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
    return _fill_target_cache(target_model, tree_nodes, stats=stats)


def _sample_token(probs: Sequence[float], device: str) -> int:
    probs_t = torch.tensor(probs, dtype=torch.float32, device=device)
    token = torch.multinomial(probs_t, num_samples=1)
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
    cache: Dict[_Prefix, List[float]] = {}
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

        token = _sample_token(cache[current_prefix], target_model.device)
        generated.append(token)
        stats.target_tokens += 1
        stats.proposed += 1
        stats.accepted += 1
        if eos_id is not None and token == eos_id:
            break

    if return_stats:
        return generated, stats
    return generated

