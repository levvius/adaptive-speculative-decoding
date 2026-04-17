from __future__ import annotations

import torch

from jointadaspec.baselines import FixedSDDecoder, VanillaARDecoder
from jointadaspec.inference import JointAdaSpecDecoder, JointAdaSpecPolicy
from jointadaspec.mdp import MDPConfig, collect_traces, estimate_mdp_parameters, solve_mdp
from sp_samp.models import FixedModel


class ToyModelAdapter:
    def __init__(self, model: FixedModel) -> None:
        self.model = model
        self.vocab_size = model.vocab_size
        self.device = "cpu"

    def next_token_probs(self, context_tokens):
        return self.model.next_token_probs(context_tokens)


def test_end_to_end_toy_pipeline(tmp_path) -> None:
    target = ToyModelAdapter(FixedModel([0.7, 0.2, 0.1]))
    draft = ToyModelAdapter(FixedModel([0.6, 0.3, 0.1]))
    config = MDPConfig(N_H=4, N_K=4, gamma_max=2, T_levels=(1.0, 2.0), nu_min=1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    prompts = [[0], [1], [2], [0, 1], [1, 2]]

    traces_path = collect_traces(
        target_model=target,
        draft_model=draft,
        prompts=prompts,
        n_traces=40,
        output_path=tmp_path / "traces.parquet",
        config=config,
        generator=generator,
    )
    estimate = estimate_mdp_parameters(traces_path=traces_path, config=config)
    V_star, pi_star, _ = solve_mdp(estimate.transitions, estimate.rewards, config)
    policy = JointAdaSpecPolicy(config=config, pi_star=pi_star, V_star=V_star)

    joint_decoder = JointAdaSpecDecoder(target_model=target, draft_model=draft, policy=policy)
    vanilla_decoder = VanillaARDecoder(target_model=target)
    fixed_decoder = FixedSDDecoder(target_model=target, draft_model=draft, gamma=2)

    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(11)
    joint_result = joint_decoder.generate(prompt_ids=[0], max_new_tokens=6, generator=eval_generator)
    eval_generator.manual_seed(11)
    vanilla_result = vanilla_decoder.generate(prompt_ids=[0], max_new_tokens=6, generator=eval_generator)
    eval_generator.manual_seed(11)
    fixed_result = fixed_decoder.generate(prompt_ids=[0], max_new_tokens=6, generator=eval_generator)

    assert traces_path.exists()
    assert (tmp_path / "traces_meta.json").exists()
    assert estimate.transitions.shape == (config.num_states * config.num_actions, config.num_states)
    assert joint_result.n_tokens_generated == 6
    assert vanilla_result.n_tokens_generated == 6
    assert fixed_result.n_tokens_generated == 6
