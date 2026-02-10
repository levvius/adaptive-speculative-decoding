import random

from sp_samp.models import FixedModel
from sp_samp.sampling import sample_baseline, speculative_sample


class SequenceRNG:
    def __init__(self, values):
        self._values = list(values)

    def random(self):
        if not self._values:
            return 0.0
        return self._values.pop(0)


def test_rejection_samples_from_residual():
    target = FixedModel([0.8, 0.1, 0.1])
    draft = FixedModel([0.1, 0.8, 0.1])
    rng = SequenceRNG([0.5, 0.9, 0.2])
    out = speculative_sample(
        target, draft, prompt_tokens=[], max_new_tokens=1, k=1, rng=rng
    )
    assert out == [0]


def test_all_accepts_adds_extra_token():
    target = FixedModel([0.5, 0.5])
    draft = FixedModel([0.5, 0.5])
    rng = SequenceRNG([0.1, 0.1, 0.1, 0.1, 0.1])
    out = speculative_sample(
        target, draft, prompt_tokens=[], max_new_tokens=3, k=2, rng=rng
    )
    assert len(out) == 3


def test_empirical_distribution_matches_target():
    target = FixedModel([0.7, 0.2, 0.1])
    draft = FixedModel([0.2, 0.7, 0.1])
    rng = random.Random(0)
    counts = [0, 0, 0]
    trials = 20000
    for _ in range(trials):
        out = speculative_sample(
            target, draft, prompt_tokens=[], max_new_tokens=1, k=3, rng=rng
        )
        counts[out[0]] += 1
    freqs = [c / trials for c in counts]
    assert abs(freqs[0] - 0.7) < 0.02
    assert abs(freqs[1] - 0.2) < 0.02
    assert abs(freqs[2] - 0.1) < 0.02


def test_eos_stops_generation():
    eos_id = 1
    target = FixedModel([0.0, 1.0])
    draft = FixedModel([0.0, 1.0])
    rng = SequenceRNG([0.1, 0.1])
    out = speculative_sample(
        target,
        draft,
        prompt_tokens=[],
        max_new_tokens=5,
        k=3,
        rng=rng,
        eos_id=eos_id,
    )
    assert out == [eos_id]


def test_return_stats_for_baseline():
    target = FixedModel([0.6, 0.4])
    rng = SequenceRNG([0.2, 0.8, 0.1])
    out, stats = sample_baseline(
        target, prompt_tokens=[], max_new_tokens=3, rng=rng, return_stats=True
    )
    assert len(out) == stats.target_tokens
    assert stats.acceptance_rate == 1.0
