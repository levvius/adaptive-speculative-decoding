from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from jointadaspec.baselines import SpecDecPPDecoder
from jointadaspec.inference import JointAdaSpecDecoder, JointAdaSpecPolicy
from jointadaspec.mdp import MDPConfig
from jointadaspec.mdp.spaces import ActionSpace, StateSpace
from jointadaspec.semantics import BLOCK_DECODER_SEMANTICS, semantic_metadata
from sp_samp.models import FixedModel


class ToyModelAdapter:
    def __init__(self, model: FixedModel) -> None:
        self.model = model
        self.vocab_size = model.vocab_size
        self.device = "cpu"

    def next_token_probs(self, context_tokens):
        return self.model.next_token_probs(context_tokens)


class CountingBlockModel(ToyModelAdapter):
    def __init__(self, model: FixedModel) -> None:
        super().__init__(model)
        self.next_calls = 0
        self.block_calls = 0

    def next_token_probs(self, context_tokens):
        self.next_calls += 1
        return super().next_token_probs(context_tokens)

    def next_token_probs_block(self, context_tokens, continuation_tokens):
        self.block_calls += 1
        return [
            self.model.next_token_probs(list(context_tokens) + list(continuation_tokens[:idx]))
            for idx in range(len(continuation_tokens) + 1)
        ]


def _build_policy(config: MDPConfig) -> JointAdaSpecPolicy:
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    pi_star = np.zeros(config.num_states, dtype=np.int32)
    continue_idx = action_space.encode("continue", 1.0)
    stop_idx = action_space.encode("verify", 1.0)
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


def test_policy_loader_rejects_legacy_npz_without_semantic_metadata(tmp_path) -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=1, T_levels=(1.0,))
    path = tmp_path / "legacy_policy.npz"
    np.savez_compressed(path, pi_star=np.zeros(config.num_states, dtype=np.int32))

    with pytest.raises(ValueError, match="metadata_json"):
        JointAdaSpecPolicy.load(path)


def test_policy_loader_rejects_mismatched_action_space_metadata(tmp_path) -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=1, T_levels=(1.0,))
    metadata = semantic_metadata(
        config=config,
        policy_kind="jointadaspec",
        num_actions=config.num_actions + 1,
    )
    path = tmp_path / "wrong_actions_policy.npz"
    np.savez_compressed(
        path,
        pi_star=np.zeros(config.num_states, dtype=np.int32),
        metadata_json=np.array(json.dumps(metadata), dtype=np.str_),
    )

    with pytest.raises(ValueError, match="num_actions"):
        JointAdaSpecPolicy.load(path)


def test_policy_loader_rejects_non_integer_semantic_metadata(tmp_path) -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=1, T_levels=(1.0,))
    metadata = semantic_metadata(
        config=config,
        policy_kind="jointadaspec",
        num_actions=config.num_actions,
    )
    metadata["action_space_version"] = "two"
    path = tmp_path / "bad_version_policy.npz"
    np.savez_compressed(
        path,
        pi_star=np.zeros(config.num_states, dtype=np.int32),
        metadata_json=np.array(json.dumps(metadata), dtype=np.str_),
    )

    with pytest.raises(ValueError, match="non-integer action_space_version"):
        JointAdaSpecPolicy.load(path)


def test_policy_loader_rejects_non_integer_num_actions_metadata(tmp_path) -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=1, T_levels=(1.0,))
    metadata = semantic_metadata(
        config=config,
        policy_kind="jointadaspec",
        num_actions=config.num_actions,
    )
    metadata["num_actions"] = "nine"
    path = tmp_path / "bad_num_actions_policy.npz"
    np.savez_compressed(
        path,
        pi_star=np.zeros(config.num_states, dtype=np.int32),
        metadata_json=np.array(json.dumps(metadata), dtype=np.str_),
    )

    with pytest.raises(ValueError, match="non-integer num_actions"):
        JointAdaSpecPolicy.load(path)


def test_get_action_valid() -> None:
    config = MDPConfig(N_H=2, N_K=2, gamma_max=2, T_levels=(1.0, 2.0))
    policy = _build_policy(config)
    action_length, threshold = policy.get_action(H=0.5, K=0.5, k=0)
    assert action_length in {"verify", "continue"}
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
    assert result.n_target_calls < result.n_draft_calls
    assert result.decoder_semantics == BLOCK_DECODER_SEMANTICS


def test_jointadaspec_decoder_verifies_block_with_one_target_call() -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=3, T_levels=(1.0,))
    policy = _build_policy(config)
    target = CountingBlockModel(FixedModel([0.8, 0.2]))
    draft = CountingBlockModel(FixedModel([0.8, 0.2]))
    decoder = JointAdaSpecDecoder(target_model=target, draft_model=draft, policy=policy)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    result = decoder.generate(prompt_ids=[0], max_new_tokens=4, generator=generator)

    assert result.n_target_calls == 1
    assert target.block_calls == 1
    assert target.next_calls == 0
    assert result.n_target_verified_positions == 4


def test_jointadaspec_decoder_does_not_draft_past_remaining_tokens() -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=4, T_levels=(1.0,))
    policy = _build_policy(config)
    target = CountingBlockModel(FixedModel([0.8, 0.2]))
    draft = CountingBlockModel(FixedModel([0.8, 0.2]))
    decoder = JointAdaSpecDecoder(target_model=target, draft_model=draft, policy=policy)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    result = decoder.generate(prompt_ids=[0], max_new_tokens=1, generator=generator)

    assert result.n_tokens_generated == 1
    assert result.n_draft_calls == 2
    assert result.n_target_calls == 1
    assert result.n_target_verified_positions == 2
    assert target.block_calls == 1


def test_jointadaspec_decoder_metrics_do_not_emit_stop_action() -> None:
    config = MDPConfig(N_H=1, N_K=1, gamma_max=0, T_levels=(1.0,))
    policy = _build_policy(config)
    target = ToyModelAdapter(FixedModel([0.8, 0.2]))
    draft = ToyModelAdapter(FixedModel([0.8, 0.2]))
    decoder = JointAdaSpecDecoder(target_model=target, draft_model=draft, policy=policy)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    result = decoder.generate(prompt_ids=[0], max_new_tokens=2, generator=generator)

    assert result.per_step_metrics
    assert {metric["action_length"] for metric in result.per_step_metrics} == {"verify"}


def test_specdecpp_forced_target_metrics_use_verify_not_stop() -> None:
    target = ToyModelAdapter(FixedModel([0.5, 0.5]))
    draft = ToyModelAdapter(FixedModel([0.5, 0.5]))
    decoder = SpecDecPPDecoder(
        target_model=target,
        draft_model=draft,
        gamma_max=3,
        entropy_threshold=0.0,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    result = decoder.generate(prompt_ids=[0], max_new_tokens=2, generator=generator)

    assert result.per_step_metrics
    assert {metric["action_length"] for metric in result.per_step_metrics} == {"verify"}
