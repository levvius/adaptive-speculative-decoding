from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .hf_adapter import HFModel


@dataclass
class TopKStats:
    proposed: int = 0
    accepted: int = 0
    steps: int = 0
    target_tokens: int = 0
    rejections: int = 0

    topk_mismatches: int = 0
    topk_accepted_mismatches: int = 0
    topk_rank_effective: int = 0

    target_calls: int = 0
    draft_prefills: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def avg_tokens_per_step(self) -> float:
        return self.target_tokens / self.steps if self.steps else 0.0

    @property
    def topk_accept_rate(self) -> float:
        if self.topk_mismatches == 0:
            return 0.0
        return self.topk_accepted_mismatches / self.topk_mismatches

    @property
    def target_calls_per_token(self) -> float:
        if self.target_tokens == 0:
            return 0.0
        return self.target_calls / self.target_tokens


def _argmax_token_from_logits(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def _tokenizer_vocab_size(model: HFModel) -> int:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return int(model.vocab_size)
    try:
        return int(len(tokenizer))
    except Exception:
        pass
    try:
        return int(len(tokenizer.get_vocab()))
    except Exception:
        return int(model.vocab_size)


def _common_vocab_size(target_model: HFModel, draft_model: HFModel) -> int:
    sizes = (
        int(target_model.vocab_size),
        int(draft_model.vocab_size),
        _tokenizer_vocab_size(target_model),
        _tokenizer_vocab_size(draft_model),
    )
    common = min(sizes)
    if common <= 0:
        raise ValueError(f"common vocab size must be positive, got {common} from {sizes}.")
    return common


def topk_sample_hf(
    target_model: HFModel,
    draft_model: HFModel,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    k: int,
    topk_rank: Optional[int] = 4,
    eos_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], TopKStats]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if topk_rank is not None and topk_rank <= 0:
        raise ValueError("topk_rank must be positive or None (all).")
    common_vocab_n = _common_vocab_size(target_model, draft_model)

    if seed is not None:
        torch.manual_seed(seed)

    generated: List[int] = []
    stats = TopKStats()
    base_prompt = target_model.ensure_prefix(prompt_tokens)

    while len(generated) < max_new_tokens:
        stats.steps += 1
        remaining = max_new_tokens - len(generated)
        block = min(k, remaining)

        prefix = list(base_prompt) + generated
        draft_state = draft_model.prefill(prefix)
        stats.draft_prefills += 1

        draft_tokens: List[int] = []
        for _ in range(block):
            token = _argmax_token_from_logits(draft_state.logits.squeeze(0)[:common_vocab_n])
            draft_tokens.append(token)
            if eos_id is not None and token == eos_id:
                break
            draft_state = draft_model.step([token], draft_state)

        if not draft_tokens:
            break

        stats.proposed += len(draft_tokens)

        full_seq = prefix + draft_tokens
        target_logits, _ = target_model.logits_and_last_hidden(full_seq)
        stats.target_calls += 1

        accepted_tokens: List[int] = []
        rejected = False
        start = len(prefix) - 1
        vocab_size = common_vocab_n
        effective_topk = vocab_size if topk_rank is None else min(int(topk_rank), vocab_size)
        stats.topk_rank_effective = effective_topk

        for i, draft_token in enumerate(draft_tokens):
            logits_row = target_logits[0, start + i, :vocab_size]
            target_token = int(torch.argmax(logits_row, dim=-1).item())

            if draft_token == target_token:
                accepted_tokens.append(draft_token)
                stats.accepted += 1
                continue

            stats.topk_mismatches += 1
            if effective_topk >= vocab_size:
                in_topk = True
            else:
                topk_ids = torch.topk(logits_row, k=effective_topk, dim=-1).indices
                in_topk = bool((topk_ids == int(draft_token)).any().item())

            if in_topk:
                accepted_tokens.append(draft_token)
                stats.accepted += 1
                stats.topk_accepted_mismatches += 1
                continue

            stats.rejections += 1
            accepted_tokens.append(target_token)
            rejected = True
            break

        generated.extend(accepted_tokens)
        if len(generated) > max_new_tokens:
            generated = generated[:max_new_tokens]
        stats.target_tokens = len(generated)

        if generated and eos_id is not None and generated[-1] == eos_id:
            return generated, stats
        if len(generated) >= max_new_tokens:
            return generated, stats

        # Keep one extra target token when full block is accepted.
        if not rejected and len(accepted_tokens) == len(draft_tokens):
            extra_logits = target_logits[0, len(prefix) + len(draft_tokens) - 1, :vocab_size]
            extra_token = int(torch.argmax(extra_logits, dim=-1).item())
            generated.append(extra_token)
            if len(generated) > max_new_tokens:
                generated = generated[:max_new_tokens]
            stats.target_tokens = len(generated)
            if eos_id is not None and generated[-1] == eos_id:
                return generated, stats

    return generated, stats
