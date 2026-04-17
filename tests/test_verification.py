from __future__ import annotations

import torch

from jointadaspec.core.verification import (
    fuzzy_verification,
    modified_rejection_sampling,
    verify_draft_chain,
)


def test_mrs_preserves_distribution() -> None:
    p = torch.tensor([0.7, 0.2, 0.1])
    q = torch.tensor([0.2, 0.7, 0.1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    counts = torch.zeros_like(p)

    for _ in range(12000):
        draft_token = int(torch.multinomial(q, num_samples=1, generator=generator).item())
        accepted, corrective_fn = modified_rejection_sampling(
            p=p,
            q=q,
            draft_token=draft_token,
            generator=generator,
        )
        emitted = draft_token if accepted else corrective_fn(generator)
        counts[emitted] += 1

    freqs = counts / counts.sum()
    assert torch.max(torch.abs(freqs - p)).item() < 0.025


def test_fuzzy_verification_accept_rate() -> None:
    p = torch.tensor([0.2, 0.8])
    q = torch.tensor([0.8, 0.2])

    accept_t1 = 0
    accept_t2 = 0
    generator_1 = torch.Generator(device="cpu")
    generator_2 = torch.Generator(device="cpu")
    generator_1.manual_seed(123)
    generator_2.manual_seed(123)

    for _ in range(4000):
        accepted_1, _ = fuzzy_verification(p, q, draft_token=0, T=1.0, generator=generator_1)
        accepted_2, _ = fuzzy_verification(p, q, draft_token=0, T=2.0, generator=generator_2)
        accept_t1 += int(accepted_1)
        accept_t2 += int(accepted_2)

    assert accept_t2 > accept_t1


def test_verify_draft_chain_stops_at_first_reject() -> None:
    p_list = [torch.tensor([0.1, 0.9]), torch.tensor([0.9, 0.1])]
    q_list = [torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9])]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    n_accepted, corrective = verify_draft_chain(
        p_list=p_list,
        q_list=q_list,
        draft_tokens=[0, 1],
        T=1.0,
        generator=generator,
        p_bonus=torch.tensor([0.5, 0.5]),
    )

    assert n_accepted == 0
    assert corrective in {0, 1}
