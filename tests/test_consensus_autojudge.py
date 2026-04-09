from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from benchmarks.bench_speculative import _all_tokenizers_compatible
from sp_samp.consensus_autojudge import (
    ACTION_ACCEPT_D1,
    ACTION_ESCALATE_TO_D2,
    ACTION_FALLBACK_TO_TARGET,
    ConsensusAutoJudgeTrainConfig,
    _extract_ensemble_features,
    build_consensus_gate_classifier,
    consensus_autojudge_sample_hf,
    mine_consensus_training_examples_gsm8k,
    train_consensus_gate_classifier,
)


@dataclass
class _FakeState:
    context: list[int]
    logits: torch.Tensor


class _FakeTokenizer:
    def __init__(self, vocab_size: int, prefix: str = "tok") -> None:
        self.vocab_size = int(vocab_size)
        self._vocab = {f"{prefix}_{idx}": idx for idx in range(self.vocab_size)}

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


class _FakeHFModel:
    def __init__(
        self,
        logits_by_context: dict[tuple[int, ...], list[float]],
        tokenizer: _FakeTokenizer,
        vocab_size: int,
    ) -> None:
        self._logits_by_context = {
            tuple(ctx): torch.tensor(values, dtype=torch.float32)
            for ctx, values in logits_by_context.items()
        }
        self.tokenizer = tokenizer
        self.vocab_size = int(vocab_size)
        self.eos_token_id = None

    def ensure_prefix(self, tokens):
        if tokens:
            return list(tokens)
        return [0]

    def _state_for_context(self, context: list[int]) -> _FakeState:
        key = tuple(context)
        if key not in self._logits_by_context:
            raise KeyError(f"No logits defined for context={key}")
        return _FakeState(context=context, logits=self._logits_by_context[key].view(1, -1))

    def prefill(self, tokens):
        return self._state_for_context(self.ensure_prefix(tokens))

    def step(self, new_tokens, state: _FakeState):
        context = list(state.context) + [int(tok) for tok in new_tokens]
        return self._state_for_context(context)

    def logits_and_last_hidden(self, tokens):
        seq = self.ensure_prefix(tokens)
        logits = []
        for i in range(len(seq)):
            logits.append(self._state_for_context(seq[: i + 1]).logits.squeeze(0))
        stacked = torch.stack(logits, dim=0).unsqueeze(0)
        hidden = torch.zeros((1, len(seq), 2), dtype=torch.float32)
        return stacked, hidden


class _ConstantConsensusGate:
    def __init__(self, probs: list[float], feature_mode: str = "ensemble", top_m: int = 8) -> None:
        self._probs = np.array([probs], dtype=np.float64)
        self.feature_mode = feature_mode
        self.top_m = int(top_m)
        self.fallback_threshold = 0.5

    def predict_action_probs(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self._probs, repeats=x.shape[0], axis=0)


def _build_logits(vocab_size: int, primary: int, secondary: int, *, high: float = 6.0, low: float = 5.0) -> list[float]:
    logits = [-6.0] * int(vocab_size)
    logits[int(primary)] = float(high)
    logits[int(secondary)] = float(low)
    return logits


def test_extract_ensemble_features_shape_and_finite():
    d1 = torch.randn(7)
    d2 = torch.randn(7)
    feat = _extract_ensemble_features(
        d1_logits_row=d1,
        d2_logits_row=d2,
        agreement_streak=2,
        block_position=1,
        block_size=4,
        top_m=4,
        feature_mode="ensemble",
    )
    assert feat.shape == (17,)
    assert torch.all(torch.isfinite(feat))

    feat_d1 = _extract_ensemble_features(
        d1_logits_row=d1,
        d2_logits_row=d2,
        agreement_streak=2,
        block_position=1,
        block_size=4,
        top_m=4,
        feature_mode="d1_only",
    )
    assert feat_d1.shape == (5,)
    assert torch.all(torch.isfinite(feat_d1))


def test_train_consensus_gate_classifier_outputs_three_way_probs():
    x = torch.tensor(
        [
            [-3.0, -3.0],
            [-2.0, -2.5],
            [0.0, 3.0],
            [0.5, 2.5],
            [3.0, 0.0],
            [2.5, 0.5],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([ACTION_ACCEPT_D1, ACTION_ACCEPT_D1, ACTION_ESCALATE_TO_D2, ACTION_ESCALATE_TO_D2, ACTION_FALLBACK_TO_TARGET, ACTION_FALLBACK_TO_TARGET], dtype=torch.int64)
    clf, _, _ = train_consensus_gate_classifier(x=x, y=y)
    probs = clf.predict_action_probs(x[:3].numpy())
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    preds = clf.predict_action(x[:3].numpy())
    assert set(preds.tolist()).issubset({0, 1, 2})


def test_mine_consensus_training_examples_three_way_labels():
    tokenizer = _FakeTokenizer(vocab_size=6)
    target = _FakeHFModel(
        logits_by_context={
            (0,): _build_logits(6, 1, 2),
            (0, 1): _build_logits(6, 2, 1),
            (0, 1, 2): _build_logits(6, 3, 2),
            (0, 1, 2, 3): _build_logits(6, 3, 2),
        },
        tokenizer=tokenizer,
        vocab_size=6,
    )
    draft = _FakeHFModel(
        logits_by_context={
            (0,): _build_logits(6, 1, 2),
            (0, 1): _build_logits(6, 4, 2),
            (0, 1, 2): _build_logits(6, 4, 5),
            (0, 1, 2, 3): _build_logits(6, 4, 5),
        },
        tokenizer=tokenizer,
        vocab_size=6,
    )
    draft2 = _FakeHFModel(
        logits_by_context={
            (0,): _build_logits(6, 2, 1),
            (0, 1): _build_logits(6, 2, 4),
            (0, 1, 2): _build_logits(6, 5, 4),
            (0, 1, 2, 3): _build_logits(6, 5, 4),
        },
        tokenizer=tokenizer,
        vocab_size=6,
    )
    cfg = ConsensusAutoJudgeTrainConfig(max_train_samples=3, max_new_tokens=3, k=2, top_m=4)
    x, y = mine_consensus_training_examples_gsm8k(
        target_model=target,
        draft_model=draft,
        draft2_model=draft2,
        prompts=[[0]],
        cfg=cfg,
        eos_id=None,
    )
    assert x.shape[0] == 3
    assert y.tolist() == [ACTION_ACCEPT_D1, ACTION_ESCALATE_TO_D2, ACTION_FALLBACK_TO_TARGET]


def test_consensus_autojudge_rule_accepts_agreement():
    tokenizer = _FakeTokenizer(vocab_size=5)
    target = _FakeHFModel({(0,): _build_logits(5, 1, 2)}, tokenizer=tokenizer, vocab_size=5)
    draft = _FakeHFModel({(0,): _build_logits(5, 1, 2), (0, 1): _build_logits(5, 1, 2)}, tokenizer=tokenizer, vocab_size=5)
    draft2 = _FakeHFModel({(0,): _build_logits(5, 1, 2), (0, 1): _build_logits(5, 1, 2)}, tokenizer=tokenizer, vocab_size=5)
    out, stats = consensus_autojudge_sample_hf(
        target_model=target,
        draft_model=draft,
        draft2_model=draft2,
        prompt_tokens=[0],
        max_new_tokens=1,
        k=1,
        gate_mode="rule",
        eos_id=None,
        seed=0,
    )
    assert out == [1]
    assert stats.gate_accept_d1 == 1
    assert stats.accepted_d1 == 1
    assert stats.target_calls == 0


def test_consensus_autojudge_rule_escalates_to_d2():
    tokenizer = _FakeTokenizer(vocab_size=5)
    target = _FakeHFModel({(0,): _build_logits(5, 2, 1)}, tokenizer=tokenizer, vocab_size=5)
    draft = _FakeHFModel({(0,): _build_logits(5, 1, 2, high=6.0, low=5.9), (0, 2): _build_logits(5, 2, 1)}, tokenizer=tokenizer, vocab_size=5)
    draft2 = _FakeHFModel({(0,): _build_logits(5, 2, 1, high=6.0, low=5.5), (0, 2): _build_logits(5, 2, 1)}, tokenizer=tokenizer, vocab_size=5)
    out, stats = consensus_autojudge_sample_hf(
        target_model=target,
        draft_model=draft,
        draft2_model=draft2,
        prompt_tokens=[0],
        max_new_tokens=1,
        k=1,
        gate_mode="rule",
        eos_id=None,
        seed=0,
    )
    assert out == [2]
    assert stats.gate_escalate_d2 == 1
    assert stats.accepted_d2 == 1
    assert stats.target_calls == 0


def test_consensus_autojudge_rule_falls_back_to_target():
    tokenizer = _FakeTokenizer(vocab_size=5)
    target = _FakeHFModel({(0,): _build_logits(5, 2, 1), (0, 2): _build_logits(5, 2, 1)}, tokenizer=tokenizer, vocab_size=5)
    draft = _FakeHFModel({(0,): _build_logits(5, 1, 2), (0, 2): _build_logits(5, 1, 2)}, tokenizer=tokenizer, vocab_size=5)
    draft2 = _FakeHFModel({(0,): _build_logits(5, 3, 2), (0, 2): _build_logits(5, 3, 2)}, tokenizer=tokenizer, vocab_size=5)
    out, stats = consensus_autojudge_sample_hf(
        target_model=target,
        draft_model=draft,
        draft2_model=draft2,
        prompt_tokens=[0],
        max_new_tokens=1,
        k=1,
        gate_mode="rule",
        eos_id=None,
        seed=0,
    )
    assert out == [2]
    assert stats.gate_fallback == 1
    assert stats.target_calls == 1
    assert stats.target_fallbacks == 1
    assert stats.rejections == 1


def test_consensus_autojudge_handles_padded_vocab_mismatch():
    tokenizer = _FakeTokenizer(vocab_size=4)
    target = _FakeHFModel({(0,): _build_logits(7, 2, 1), (0, 2): _build_logits(7, 2, 1)}, tokenizer=tokenizer, vocab_size=7)
    draft = _FakeHFModel({(0,): _build_logits(6, 5, 1), (0, 2): _build_logits(6, 5, 1)}, tokenizer=tokenizer, vocab_size=6)
    draft2 = _FakeHFModel({(0,): _build_logits(5, 4, 2), (0, 2): _build_logits(5, 4, 2)}, tokenizer=tokenizer, vocab_size=5)
    gate = _ConstantConsensusGate([0.0, 0.0, 1.0])
    out, stats = consensus_autojudge_sample_hf(
        target_model=target,
        draft_model=draft,
        draft2_model=draft2,
        gate_model=gate,
        gate_mode="learned",
        prompt_tokens=[0],
        max_new_tokens=1,
        k=1,
        eos_id=None,
        seed=0,
    )
    assert out == [2]
    assert max(out) < 4
    assert stats.target_calls == 1


def test_consensus_autojudge_is_deterministic_with_seed():
    tokenizer = _FakeTokenizer(vocab_size=5)
    target = _FakeHFModel({(0,): _build_logits(5, 1, 2), (0, 1): _build_logits(5, 1, 2)}, tokenizer=tokenizer, vocab_size=5)
    draft = _FakeHFModel({(0,): _build_logits(5, 1, 2), (0, 1): _build_logits(5, 1, 2)}, tokenizer=tokenizer, vocab_size=5)
    draft2 = _FakeHFModel({(0,): _build_logits(5, 1, 2), (0, 1): _build_logits(5, 1, 2)}, tokenizer=tokenizer, vocab_size=5)
    gate = _ConstantConsensusGate([1.0, 0.0, 0.0])
    out1, stats1 = consensus_autojudge_sample_hf(
        target_model=target,
        draft_model=draft,
        draft2_model=draft2,
        gate_model=gate,
        gate_mode="learned",
        prompt_tokens=[0],
        max_new_tokens=1,
        k=1,
        eos_id=None,
        seed=123,
    )
    out2, stats2 = consensus_autojudge_sample_hf(
        target_model=target,
        draft_model=draft,
        draft2_model=draft2,
        gate_model=gate,
        gate_mode="learned",
        prompt_tokens=[0],
        max_new_tokens=1,
        k=1,
        eos_id=None,
        seed=123,
    )
    assert out1 == out2
    assert stats1.accepted == stats2.accepted
    assert stats1.target_calls == stats2.target_calls


def test_all_tokenizers_compatible_rejects_mismatch():
    tok_a = _FakeTokenizer(vocab_size=4, prefix="a")
    tok_b = _FakeTokenizer(vocab_size=4, prefix="a")
    tok_c = _FakeTokenizer(vocab_size=4, prefix="c")
    assert _all_tokenizers_compatible(tok_a, tok_b)
    assert not _all_tokenizers_compatible(tok_a, tok_b, tok_c)


def test_build_consensus_gate_classifier_smoke():
    tokenizer = _FakeTokenizer(vocab_size=6)
    target = _FakeHFModel(
        logits_by_context={
            (0,): _build_logits(6, 1, 2),
            (0, 1): _build_logits(6, 2, 1),
            (0, 1, 2): _build_logits(6, 3, 2),
            (0, 1, 2, 3): _build_logits(6, 3, 2),
        },
        tokenizer=tokenizer,
        vocab_size=6,
    )
    draft = _FakeHFModel(
        logits_by_context={
            (0,): _build_logits(6, 1, 2),
            (0, 1): _build_logits(6, 4, 2),
            (0, 1, 2): _build_logits(6, 4, 5),
            (0, 1, 2, 3): _build_logits(6, 4, 5),
        },
        tokenizer=tokenizer,
        vocab_size=6,
    )
    draft2 = _FakeHFModel(
        logits_by_context={
            (0,): _build_logits(6, 2, 1),
            (0, 1): _build_logits(6, 2, 4),
            (0, 1, 2): _build_logits(6, 5, 4),
            (0, 1, 2, 3): _build_logits(6, 5, 4),
        },
        tokenizer=tokenizer,
        vocab_size=6,
    )
    cfg = ConsensusAutoJudgeTrainConfig(max_train_samples=3, max_new_tokens=3, k=2, top_m=4)
    clf, train_samples, val_accuracy, val_macro_f1 = build_consensus_gate_classifier(
        target_model=target,
        draft_model=draft,
        draft2_model=draft2,
        prompts=[[0]],
        cfg=cfg,
        eos_id=None,
    )
    assert train_samples == 3
    assert 0.0 <= val_accuracy <= 1.0
    assert 0.0 <= val_macro_f1 <= 1.0

    out, stats = consensus_autojudge_sample_hf(
        target_model=target,
        draft_model=draft,
        draft2_model=draft2,
        gate_model=clf,
        gate_mode="learned",
        prompt_tokens=[0],
        max_new_tokens=1,
        k=1,
        eos_id=None,
        seed=0,
    )
    assert len(out) == 1
    assert stats.gate_total == 1
