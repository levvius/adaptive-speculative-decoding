from __future__ import annotations

import json

import pandas as pd
import torch

from jointadaspec.mdp import MDPConfig, collect_traces
from sp_samp.models import FixedModel


class CountingBlockAdapter:
    def __init__(self, probs: list[float]) -> None:
        self.model = FixedModel(probs)
        self.vocab_size = self.model.vocab_size
        self.device = "cpu"
        self.block_calls = 0

    def next_token_probs(self, context_tokens):
        return self.model.next_token_probs(context_tokens)

    def next_token_probs_block(self, context_tokens, continuation_tokens):
        self.block_calls += 1
        return [
            self.model.next_token_probs(list(context_tokens) + list(continuation_tokens[:idx]))
            for idx in range(len(continuation_tokens) + 1)
        ]


def _generator(seed: int = 0) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def test_trace_collection_resumes_from_jsonl_checkpoints(tmp_path) -> None:
    target = CountingBlockAdapter([0.7, 0.2, 0.1])
    draft = CountingBlockAdapter([0.6, 0.3, 0.1])
    config = MDPConfig(N_H=2, N_K=2, gamma_max=1, T_levels=(1.0, 2.0), nu_min=1)
    output_path = tmp_path / "traces.parquet"

    collect_traces(
        target_model=target,
        draft_model=draft,
        prompts=[[0], [1], [2]],
        n_traces=2,
        output_path=output_path,
        config=config,
        generator=_generator(123),
        resume=True,
        checkpoint_every=1,
        progress_every=0,
    )
    output_path.unlink()
    output_path.with_name("traces_meta.json").unlink()

    collect_traces(
        target_model=target,
        draft_model=draft,
        prompts=[[0], [1], [2]],
        n_traces=5,
        output_path=output_path,
        config=config,
        generator=_generator(123),
        resume=True,
        checkpoint_every=1,
        progress_every=0,
    )

    frame = pd.read_parquet(output_path)
    assert sorted(frame["trace_idx"].unique().tolist()) == [0, 1, 2, 3, 4]
    assert not frame.duplicated(["trace_idx", "rollout_step", "action_idx"]).any()

    meta = json.loads(output_path.with_name("traces_meta.json").read_text(encoding="utf-8"))
    assert meta["n_traces"] == 5
    assert meta["num_records"] == len(frame)
    checkpoint_files = sorted((tmp_path / "traces_checkpoint").glob("trace_*.jsonl"))
    assert len(checkpoint_files) == 5


def test_trace_collection_reuses_target_block_per_state(tmp_path) -> None:
    target = CountingBlockAdapter([0.7, 0.2, 0.1])
    draft = CountingBlockAdapter([0.6, 0.3, 0.1])
    config = MDPConfig(
        N_H=2,
        N_K=2,
        gamma_max=1,
        T_levels=(1.0, 1.5, 2.0, 2.5),
        nu_min=1,
    )

    collect_traces(
        target_model=target,
        draft_model=draft,
        prompts=[[0]],
        n_traces=1,
        output_path=tmp_path / "traces.parquet",
        config=config,
        generator=_generator(0),
        resume=False,
        checkpoint_every=1,
        progress_every=0,
    )

    # gamma_max=1 gives max(2, gamma_max + 2) == 3 rollout states. Each state has
    # several verify-threshold actions, but the target block should be computed
    # once per state and reused across those thresholds.
    assert target.block_calls == 3
