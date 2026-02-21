from __future__ import annotations

from dataclasses import dataclass

import torch

from sp_samp.hf_topk import topk_sample_hf


@dataclass
class _FakeState:
    context: list[int]
    logits: torch.Tensor


class _FakeDraftModel:
    def __init__(self, draft_token: int, vocab_size: int = 5) -> None:
        self.vocab_size = vocab_size
        self._draft_token = int(draft_token)

    def ensure_prefix(self, tokens):
        if tokens:
            return list(tokens)
        return [0]

    def _one_hot(self, token: int) -> torch.Tensor:
        logits = torch.full((1, self.vocab_size), -1000.0)
        logits[0, int(token)] = 1000.0
        return logits

    def prefill(self, tokens):
        context = self.ensure_prefix(tokens)
        return _FakeState(context=context, logits=self._one_hot(self._draft_token))

    def step(self, new_tokens, state: _FakeState):
        context = list(state.context) + list(new_tokens)
        return _FakeState(context=context, logits=self._one_hot(self._draft_token))


class _FakeTargetModel:
    def __init__(self, vocab_size: int = 5) -> None:
        self.vocab_size = vocab_size

    def ensure_prefix(self, tokens):
        if tokens:
            return list(tokens)
        return [0]

    def logits_and_last_hidden(self, tokens):
        seq = self.ensure_prefix(tokens)
        logits = torch.full((1, len(seq), self.vocab_size), -1000.0)
        for i in range(len(seq)):
            # For the first predicted token: rank 1 -> id=2, rank 2 -> id=1.
            if tuple(seq[: i + 1]) == (0,):
                logits[0, i, 2] = 10.0
                logits[0, i, 1] = 9.0
            else:
                logits[0, i, 0] = 10.0
        hidden = torch.zeros((1, len(seq), 1))
        return logits, hidden


def test_topk_accepts_mismatch_when_inside_topk():
    target = _FakeTargetModel(vocab_size=5)
    draft = _FakeDraftModel(draft_token=1, vocab_size=5)
    out, stats = topk_sample_hf(
        target_model=target,
        draft_model=draft,
        prompt_tokens=[0],
        max_new_tokens=1,
        k=1,
        topk_rank=2,
        eos_id=None,
        seed=0,
    )
    assert out == [1]
    assert stats.topk_mismatches == 1
    assert stats.topk_accepted_mismatches == 1
    assert stats.rejections == 0


def test_topk_rejects_mismatch_when_outside_topk():
    target = _FakeTargetModel(vocab_size=5)
    draft = _FakeDraftModel(draft_token=1, vocab_size=5)
    out, stats = topk_sample_hf(
        target_model=target,
        draft_model=draft,
        prompt_tokens=[0],
        max_new_tokens=1,
        k=1,
        topk_rank=1,
        eos_id=None,
        seed=0,
    )
    assert out == [2]
    assert stats.topk_mismatches == 1
    assert stats.topk_accepted_mismatches == 0
    assert stats.rejections == 1
