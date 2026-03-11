from __future__ import annotations

from dataclasses import dataclass

import torch

from sp_samp.hf_sampling import speculative_sample_hf


@dataclass
class _FakeState:
    context: list[int]
    logits: torch.Tensor


class _FakeTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)

    def __len__(self) -> int:
        return self.vocab_size


class _FakeHFModel:
    def __init__(
        self,
        vocab_size: int,
        tokenizer_vocab_size: int,
        logits_by_context: dict[tuple[int, ...], list[float]],
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.tokenizer = _FakeTokenizer(tokenizer_vocab_size)
        self._logits_by_context = {
            tuple(ctx): torch.tensor(values, dtype=torch.float32)
            for ctx, values in logits_by_context.items()
        }

    def ensure_prefix(self, tokens):
        if tokens:
            return list(tokens)
        return [0]

    def _state_for_context(self, context: list[int]) -> _FakeState:
        key = tuple(context)
        if key not in self._logits_by_context:
            raise KeyError(f"No logits defined for context={key}")
        logits = self._logits_by_context[key].view(1, -1)
        return _FakeState(context=context, logits=logits)

    def prefill(self, tokens):
        return self._state_for_context(self.ensure_prefix(tokens))

    def step(self, new_tokens, state: _FakeState):
        for token in new_tokens:
            token = int(token)
            if token < 0 or token >= self.vocab_size:
                raise ValueError(
                    f"Token {token} is outside model vocab [0, {self.vocab_size - 1}]"
                )
        context = list(state.context) + [int(token) for token in new_tokens]
        return self._state_for_context(context)


def _build_logits(vocab_size: int, primary: int, secondary: int) -> list[float]:
    logits = [-1000.0] * vocab_size
    logits[int(primary)] = 1000.0
    logits[int(secondary)] = 900.0
    return logits


def test_speculative_sample_hf_handles_padded_vocab_mismatch():
    # common_vocab_n = min(target=8, draft=6, tokenizer_target=4, tokenizer_draft=4) = 4
    target = _FakeHFModel(
        vocab_size=8,
        tokenizer_vocab_size=4,
        logits_by_context={
            (0,): _build_logits(vocab_size=8, primary=7, secondary=2),
            (0, 2): _build_logits(vocab_size=8, primary=6, secondary=3),
            (0, 2, 3): _build_logits(vocab_size=8, primary=6, secondary=1),
        },
    )
    draft = _FakeHFModel(
        vocab_size=6,
        tokenizer_vocab_size=4,
        logits_by_context={
            (0,): _build_logits(vocab_size=6, primary=5, secondary=2),
            (0, 2): _build_logits(vocab_size=6, primary=5, secondary=3),
            (0, 2, 3): _build_logits(vocab_size=6, primary=5, secondary=1),
        },
    )

    generated, stats = speculative_sample_hf(
        target_model=target,
        draft_model=draft,
        prompt_tokens=[0],
        max_new_tokens=2,
        k=1,
        eos_id=None,
        seed=0,
        return_stats=True,
    )
    assert generated == [2, 3]
    assert max(generated) < 4
    assert stats.rejections == 0
