from __future__ import annotations

import random
from typing import List, Sequence


def _normalize(probs: Sequence[float]) -> List[float]:
    total = float(sum(probs))
    if total <= 0.0:
        raise ValueError("Probability distribution has non-positive total.")
    return [float(p) / total for p in probs]


class BaseModel:
    """Minimal next-token probability interface."""

    def __init__(self, vocab_size: int) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        self.vocab_size = vocab_size

    def next_token_probs(self, context_tokens: Sequence[int]) -> List[float]:
        raise NotImplementedError

    def _validate(self, probs: Sequence[float]) -> List[float]:
        if len(probs) != self.vocab_size:
            raise ValueError(
                f"Expected probs of length {self.vocab_size}, got {len(probs)}."
            )
        return list(probs)


class FixedModel(BaseModel):
    """Context-independent categorical distribution."""

    def __init__(self, probs: Sequence[float]) -> None:
        super().__init__(len(probs))
        self._probs = _normalize(probs)

    def next_token_probs(self, context_tokens: Sequence[int]) -> List[float]:
        return list(self._probs)


class BigramModel(BaseModel):
    """Simple bigram model using a transition matrix."""

    def __init__(self, transitions: Sequence[Sequence[float]]) -> None:
        vocab_size = len(transitions)
        if vocab_size == 0:
            raise ValueError("transitions must be non-empty.")
        for row in transitions:
            if len(row) != vocab_size:
                raise ValueError("transition matrix must be square.")
        super().__init__(vocab_size)
        self._transitions = [_normalize(row) for row in transitions]

    def next_token_probs(self, context_tokens: Sequence[int]) -> List[float]:
        if context_tokens:
            prev = int(context_tokens[-1]) % self.vocab_size
        else:
            prev = 0
        return list(self._transitions[prev])


class RandomModel(BaseModel):
    """Random bigram model (fixed after initialization)."""

    def __init__(self, vocab_size: int, seed: int = 0) -> None:
        super().__init__(vocab_size)
        rng = random.Random(seed)
        transitions = []
        for _ in range(vocab_size):
            row = [rng.random() for _ in range(vocab_size)]
            transitions.append(_normalize(row))
        self._transitions = transitions

    def next_token_probs(self, context_tokens: Sequence[int]) -> List[float]:
        if context_tokens:
            prev = int(context_tokens[-1]) % self.vocab_size
        else:
            prev = 0
        return list(self._transitions[prev])


class NoisyModel(BaseModel):
    """Mixture of a base model and uniform noise."""

    def __init__(self, base_model: BaseModel, noise: float) -> None:
        if not (0.0 <= noise <= 1.0):
            raise ValueError("noise must be between 0 and 1.")
        super().__init__(base_model.vocab_size)
        self._base = base_model
        self._noise = float(noise)

    def next_token_probs(self, context_tokens: Sequence[int]) -> List[float]:
        probs = self._base.next_token_probs(context_tokens)
        if self._noise <= 0.0:
            return list(probs)
        uniform = 1.0 / self.vocab_size
        return [
            (1.0 - self._noise) * p + self._noise * uniform for p in probs
        ]
