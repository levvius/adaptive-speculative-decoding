from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sp_samp.autojudge import (
    AutoJudgeTrainConfig,
    _default_c_grid,
    autojudge_sample_hf,
    mine_important_tokens_gsm8k,
    parse_c_grid,
    train_autojudge_logreg,
)
from sp_samp.gsm8k import answers_equivalent, extract_final_answer


@dataclass
class _FakeState:
    context: list[int]
    logits: torch.Tensor


class _FakeTokenizer:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        if self.mode == "same":
            return "The final answer is 10"
        lead = int(tokens[0]) if tokens else -1
        if lead == 1:
            return "The final answer is 10"
        return "The final answer is 20"


class _FakeHFModel:
    def __init__(
        self,
        transition: dict[tuple[int, ...], int],
        tokenizer: _FakeTokenizer,
        vocab_size: int = 6,
        hidden_offset: float = 0.0,
    ) -> None:
        self.transition = transition
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.hidden_offset = hidden_offset
        self.eos_token_id = None

    def ensure_prefix(self, tokens):
        if tokens:
            return list(tokens)
        return [0]

    def _next(self, context: list[int]) -> int:
        return int(self.transition.get(tuple(context), 0))

    def _one_hot_logits(self, token: int) -> torch.Tensor:
        logits = torch.full((1, self.vocab_size), -1000.0)
        logits[0, token] = 1000.0
        return logits

    def prefill(self, tokens):
        context = self.ensure_prefix(tokens)
        token = self._next(context)
        return _FakeState(context=context, logits=self._one_hot_logits(token))

    def step(self, new_tokens, state: _FakeState):
        context = list(state.context) + list(new_tokens)
        token = self._next(context)
        return _FakeState(context=context, logits=self._one_hot_logits(token))

    def logits_and_last_hidden(self, tokens):
        seq = self.ensure_prefix(tokens)
        logits = torch.full((1, len(seq), self.vocab_size), -1000.0)
        hidden = torch.zeros((1, len(seq), 2), dtype=torch.float32)
        for i in range(len(seq)):
            next_token = self._next(seq[: i + 1])
            logits[0, i, next_token] = 1000.0
            hidden[0, i, 0] = float(seq[i]) + self.hidden_offset
            hidden[0, i, 1] = float(i) + self.hidden_offset
        return logits, hidden


class _ConstantJudge:
    def __init__(self, probability: float, threshold: float = 0.5) -> None:
        self._p = float(probability)
        self.threshold = float(threshold)

    def predict_important_prob(self, x: np.ndarray) -> np.ndarray:
        return np.full((x.shape[0],), self._p, dtype=np.float64)


def test_parse_c_grid():
    assert parse_c_grid(None)[0] == 1e-7
    assert parse_c_grid("1e-4, 1e-2,1") == (1e-4, 1e-2, 1.0)


def test_default_c_grid_matches_paper():
    # Paper: Section 3.2 — C ∈ {10^0, 10^-1, …, 10^-7} (exactly 8 values).
    grid = _default_c_grid()
    assert len(grid) == 8, f"Expected 8 C values, got {len(grid)}: {grid}"
    expected = tuple(10.0**p for p in range(-7, 1))
    assert grid == expected, f"C grid mismatch: {grid} != {expected}"
    # Must NOT contain 10^1 or 10^2 (not in paper).
    assert 10.0 not in grid, "10^1 should not be in the C grid"
    assert 100.0 not in grid, "10^2 should not be in the C grid"
    # Must contain 10^0 = 1.0 as the largest value.
    assert 1.0 in grid, "10^0 = 1.0 should be in the C grid"


def test_gsm8k_extraction_and_equivalence():
    assert extract_final_answer("Reasoning... The final answer is 3/2") == "3/2"
    assert answers_equivalent("1.5", "3/2")


def test_train_autojudge_logreg_recall_target():
    torch.manual_seed(0)
    x = torch.randn(600, 8)
    y = ((x[:, 0] + x[:, 1]) > 0).float().unsqueeze(-1)
    cfg = AutoJudgeTrainConfig(
        recall_target=0.9,
        train_split=0.9,
        c_grid=(1e-2, 1e-1, 1.0, 10.0),
        seed=0,
    )
    clf, auc = train_autojudge_logreg(x=x, y=y, cfg=cfg)
    probs = clf.predict_important_prob(x.numpy())
    preds = (probs >= clf.threshold).astype(np.int64)
    y_true = y.numpy().reshape(-1).astype(np.int64)
    positives = y_true == 1
    recall = float((preds[positives] == 1).mean())
    assert recall >= 0.9
    assert auc >= 0.5


def _make_training_models(mode: str):
    tokenizer = _FakeTokenizer(mode)
    target_transition = {
        (0,): 1,
        (0, 1): 3,
        (0, 2): 3,
    }
    draft_transition = {
        (0,): 2,
        (0, 1): 3,
        (0, 2): 3,
    }
    target = _FakeHFModel(target_transition, tokenizer=tokenizer, hidden_offset=10.0)
    draft = _FakeHFModel(draft_transition, tokenizer=tokenizer, hidden_offset=20.0)
    return target, draft


def test_mine_important_tokens_marks_unimportant():
    target, draft = _make_training_models(mode="same")
    cfg = AutoJudgeTrainConfig(max_train_samples=2, max_new_tokens=2, seed=0)
    x, y = mine_important_tokens_gsm8k(
        target_model=target,
        draft_model=draft,
        prompts=[[0]],
        cfg=cfg,
        eos_id=None,
    )
    assert x.shape[0] >= 1
    assert int(y[0].item()) == 0


def test_mine_important_tokens_marks_important():
    target, draft = _make_training_models(mode="different")
    cfg = AutoJudgeTrainConfig(max_train_samples=2, max_new_tokens=2, seed=0)
    x, y = mine_important_tokens_gsm8k(
        target_model=target,
        draft_model=draft,
        prompts=[[0]],
        cfg=cfg,
        eos_id=None,
    )
    assert x.shape[0] >= 1
    assert int(y[0].item()) == 1


def test_autojudge_sample_hf_accepts_unimportant_and_continues():
    tokenizer = _FakeTokenizer("same")
    target_transition = {
        (0,): 2,
        (0, 1): 2,
        (0, 1, 2): 2,
    }
    draft_transition = {
        (0,): 1,
        (0, 1): 2,
        (0, 1, 2): 2,
    }
    target = _FakeHFModel(target_transition, tokenizer=tokenizer, hidden_offset=10.0)
    draft = _FakeHFModel(draft_transition, tokenizer=tokenizer, hidden_offset=20.0)

    judge = _ConstantJudge(probability=0.0, threshold=0.5)
    out, stats = autojudge_sample_hf(
        target_model=target,
        draft_model=draft,
        judge_model=judge,
        prompt_tokens=[0],
        max_new_tokens=3,
        k=2,
        threshold=None,
        eos_id=None,
        seed=0,
    )
    assert out[:2] == [1, 2]
    assert stats.judge_total == 1
    assert stats.judge_accepted == 1
    assert stats.judge_rejected == 0


def test_autojudge_sample_hf_rejects_important_mismatch():
    tokenizer = _FakeTokenizer("same")
    target_transition = {
        (0,): 2,
        (0, 2): 2,
    }
    draft_transition = {
        (0,): 1,
        (0, 2): 2,
    }
    target = _FakeHFModel(target_transition, tokenizer=tokenizer, hidden_offset=10.0)
    draft = _FakeHFModel(draft_transition, tokenizer=tokenizer, hidden_offset=20.0)

    judge = _ConstantJudge(probability=1.0, threshold=0.5)
    out, stats = autojudge_sample_hf(
        target_model=target,
        draft_model=draft,
        judge_model=judge,
        prompt_tokens=[0],
        max_new_tokens=2,
        k=2,
        threshold=None,
        eos_id=None,
        seed=0,
    )
    assert out[0] == 2
    assert stats.judge_rejected >= 1
    assert stats.target_fallbacks >= 1
