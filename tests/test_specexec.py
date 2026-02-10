import random

import pytest

from sp_samp.models import BaseModel, FixedModel, NoisyModel, RandomModel
from sp_samp.sampling import sample_baseline
from sp_samp.specexec import SpecExecError, specexec_sample


class BadShapeModel(BaseModel):
    def __init__(self) -> None:
        super().__init__(vocab_size=3)

    def next_token_probs(self, context_tokens):
        # Intentionally wrong size for error-path coverage.
        return [0.5, 0.5]


def test_specexec_generates_requested_tokens_and_stats():
    target = FixedModel([0.5, 0.5])
    draft = FixedModel([0.5, 0.5])
    out, stats = specexec_sample(
        target_model=target,
        draft_model=draft,
        prompt_tokens=[],
        max_new_tokens=6,
        k=3,
        parallel_branches=4,
        rng=random.Random(0),
        return_stats=True,
    )
    assert len(out) == 6
    assert stats.proposed == 6
    assert stats.accepted == 6
    assert stats.acceptance_rate == 1.0
    assert stats.cache_misses >= 1
    assert 0.0 <= stats.cache_hit_rate <= 1.0


def test_specexec_empirical_distribution_matches_target():
    target = FixedModel([0.7, 0.2, 0.1])
    draft = FixedModel([0.2, 0.7, 0.1])
    rng = random.Random(0)
    counts = [0, 0, 0]
    trials = 20000
    for _ in range(trials):
        out = specexec_sample(
            target_model=target,
            draft_model=draft,
            prompt_tokens=[],
            max_new_tokens=1,
            k=4,
            parallel_branches=8,
            branch_prune_threshold=0.2,
            rng=rng,
        )
        counts[out[0]] += 1
    freqs = [c / trials for c in counts]
    assert abs(freqs[0] - 0.7) < 0.02
    assert abs(freqs[1] - 0.2) < 0.02
    assert abs(freqs[2] - 0.1) < 0.02


def test_specexec_matches_baseline_for_same_seed():
    target = RandomModel(vocab_size=64, seed=11)
    draft = NoisyModel(target, noise=0.4)

    for seed in range(10):
        rng_base = random.Random(seed)
        rng_spec = random.Random(seed)
        baseline = sample_baseline(
            target_model=target,
            prompt_tokens=[1, 5, 7],
            max_new_tokens=20,
            rng=rng_base,
        )
        specexec = specexec_sample(
            target_model=target,
            draft_model=draft,
            prompt_tokens=[1, 5, 7],
            max_new_tokens=20,
            k=4,
            parallel_branches=6,
            branch_prune_threshold=0.1,
            rng=rng_spec,
        )
        assert specexec == baseline


def test_specexec_stops_on_eos():
    eos_id = 1
    target = FixedModel([0.0, 1.0])
    draft = FixedModel([0.0, 1.0])
    out = specexec_sample(
        target_model=target,
        draft_model=draft,
        prompt_tokens=[],
        max_new_tokens=8,
        k=4,
        eos_id=eos_id,
        rng=random.Random(1),
    )
    assert out == [eos_id]


def test_specexec_errors_include_context():
    target = BadShapeModel()
    draft = FixedModel([0.2, 0.3, 0.5])
    with pytest.raises(SpecExecError) as exc:
        specexec_sample(
            target_model=target,
            draft_model=draft,
            prompt_tokens=[42],
            max_new_tokens=1,
            k=2,
            parallel_branches=2,
            rng=random.Random(0),
        )
    msg = str(exc.value)
    assert "target_cache_fill" in msg
    assert "prefix_len=" in msg
    assert "prefix_tail=" in msg


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
