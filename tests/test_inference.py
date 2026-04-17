from __future__ import annotations

import numpy as np
import torch

from jointadaspec.inference import JointAdaSpecDecoder, JointAdaSpecPolicy
from jointadaspec.mdp import MDPConfig
from jointadaspec.mdp.spaces import ActionSpace, StateSpace
from sp_samp.models import FixedModel


class ToyModelAdapter:
    def __init__(self, model: FixedModel) -> None:
        self.model = model
        self.vocab_size = model.vocab_size
        self.device = "cpu"

    def next_token_probs(self, context_tokens):
        return self.model.next_token_probs(context_tokens)


def _build_policy(config: MDPConfig) -> JointAdaSpecPolicy:
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    pi_star = np.zeros(config.num_states, dtype=np.int32)
    continue_idx = action_space.encode("continue", 1.0)
    stop_idx = action_space.encode("stop", 1.0)
    for state_idx in range(config.num_states):
        _, _, k = state_space.decode(state_idx)
        pi_star[state_idx] = stop_idx if k >= config.gamma_max else continue_idx
    return JointAdaSpecPolicy(config=config, pi_star=pi_star, V_star=np.zeros(config.num_states))


def test_policy_save_load_roundtrip(tmp_path) -> None:
    config = MDPConfig(N_H=2, N_K=2, gamma_max=2, T_levels=(1.0, 2.0))
    policy = _build_policy(config)
    path = tmp_path / "policy.npz"
    policy.save(path)
    loaded = JointAdaSpecPolicy.load(path)

    assert np.array_equal(policy.pi_star, loaded.pi_star)
    assert loaded.config == policy.config


def test_get_action_valid() -> None:
    config = MDPConfig(N_H=2, N_K=2, gamma_max=2, T_levels=(1.0, 2.0))
    policy = _build_policy(config)
    action_length, threshold = policy.get_action(H=0.5, K=0.5, k=0)
    assert action_length in {"stop", "continue"}
    assert threshold in config.T_levels


def test_jointadaspec_decoder_toy_run() -> None:
    config = MDPConfig(N_H=2, N_K=2, gamma_max=2, T_levels=(1.0, 2.0))
    policy = _build_policy(config)
    target = ToyModelAdapter(FixedModel([0.8, 0.2]))
    draft = ToyModelAdapter(FixedModel([0.8, 0.2]))
    decoder = JointAdaSpecDecoder(target_model=target, draft_model=draft, policy=policy)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    result = decoder.generate(prompt_ids=[0], max_new_tokens=5, generator=generator)

    assert result.n_tokens_generated == 5
    assert result.acceptance_rate > 0.0
    assert result.n_target_calls == result.n_draft_calls
