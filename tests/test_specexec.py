import random

import pytest

from sp_samp.models import FixedModel
from sp_samp.specexec import specexec_sample


class SequenceRNG:
    def __init__(self, values):
        self._values = list(values)

    def random(self):
        if not self._values:
            return 0.0
        return self._values.pop(0)


def test_specexec_generates_requested_tokens_and_stats():
    target = FixedModel([0.5, 0.5])
    draft = FixedModel([0.5, 0.5])
    out, stats = specexec_sample(
        target_model=target,
        draft_model=draft,
        prompt_tokens=[],
        max_new_tokens=5,
        k=3,
        parallel_branches=3,
        rng=random.Random(0),
        return_stats=True,
    )
    assert len(out) == 5
    assert stats.steps >= 1
    assert stats.proposed >= stats.accepted
    assert stats.branches_total >= stats.steps
    assert 0.0 <= stats.branch_prune_rate <= 1.0


def test_specexec_rejection_samples_from_residual():
    target = FixedModel([0.8, 0.1, 0.1])
    draft = FixedModel([0.1, 0.8, 0.1])
    rng = SequenceRNG([0.9, 0.2])
    out, stats = specexec_sample(
        target_model=target,
        draft_model=draft,
        prompt_tokens=[],
        max_new_tokens=1,
        k=1,
        parallel_branches=1,
        rng=rng,
        return_stats=True,
    )
    assert out == [0]
    assert stats.rejections == 1
    assert stats.accepted == 0


def test_specexec_stops_on_eos():
    eos_id = 1
    target = FixedModel([0.0, 1.0])
    draft = FixedModel([0.0, 1.0])
    out = specexec_sample(
        target_model=target,
        draft_model=draft,
        prompt_tokens=[],
        max_new_tokens=6,
        k=4,
        eos_id=eos_id,
        rng=random.Random(1),
    )
    assert out == [eos_id]


def test_specexec_validates_parallel_branches():
    target = FixedModel([0.7, 0.3])
    draft = FixedModel([0.7, 0.3])
    with pytest.raises(ValueError):
        specexec_sample(
            target_model=target,
            draft_model=draft,
            prompt_tokens=[],
            max_new_tokens=1,
            k=1,
            parallel_branches=0,
        )
