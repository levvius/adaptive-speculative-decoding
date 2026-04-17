"""Trace collection for estimating the JointAdaSpec tabular MDP."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Sequence

import pandas as pd
import torch

from jointadaspec.core.features import entropy, kl_divergence
from jointadaspec.core.verification import fuzzy_verification, tv_distance_step
from jointadaspec.mdp.spaces import ActionSpace, JointAction, MDPConfig, StateSpace
from jointadaspec.utils.probs import common_vocab_size, next_token_probs_tensor


def _git_commit_or_none() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _ensure_prompt_tokens(model: Any, prompt: str | Sequence[int]) -> list[int]:
    if isinstance(prompt, str):
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            return [int(token) for token in encoded] or [0]
        vocab_size = int(getattr(model, "vocab_size", 256))
        tokens = [ord(ch) % max(vocab_size, 1) for ch in prompt[:64]]
        return tokens or [0]
    return [int(token) for token in prompt]


def _sample_token(probs: torch.Tensor, generator: torch.Generator) -> int:
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def _transition_from_action(
    *,
    target_model: Any,
    draft_model: Any,
    context_tokens: list[int],
    k: int,
    action: JointAction,
    p_probs: torch.Tensor,
    q_probs: torch.Tensor,
    state_space: StateSpace,
    generator: torch.Generator,
    config: MDPConfig,
    common_vocab_n: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    accepted = False
    proposed = False

    if action.is_stop:
        emitted_token = _sample_token(p_probs, generator)
        next_k = 0
        d_step = 0.0
    else:
        proposed = True
        draft_token = _sample_token(q_probs, generator)
        accepted, corrective_fn = fuzzy_verification(
            p=p_probs,
            q=q_probs,
            draft_token=draft_token,
            T=action.threshold,
            generator=generator,
        )
        emitted_token = draft_token if accepted else corrective_fn(generator)
        next_k = min(k + 1, config.gamma_max) if accepted else 0
        d_step = tv_distance_step(p_probs, q_probs, action.threshold)

    next_context = list(context_tokens) + [int(emitted_token)]
    p_next = next_token_probs_tensor(target_model, next_context, common_vocab_n)
    q_next = next_token_probs_tensor(draft_model, next_context, common_vocab_n)
    next_H = entropy(q_next)
    next_K = kl_divergence(q_next, p_next)
    next_state_idx = state_space.encode(next_H, next_K, next_k)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    reward = float((1 if accepted else 0) - config.c_time * elapsed_ms - config.kappa * d_step)

    return {
        "next_context": next_context,
        "next_k": next_k,
        "next_state_idx": next_state_idx,
        "next_H": next_H,
        "next_K": next_K,
        "reward": reward,
        "accepted": int(accepted),
        "proposed": int(proposed),
        "emitted_token": int(emitted_token),
        "step_time_ms": elapsed_ms,
        "d_step": d_step,
    }


def collect_traces(
    target_model: Any,
    draft_model: Any,
    prompts: Sequence[str | Sequence[int]],
    n_traces: int,
    output_path: Path,
    config: MDPConfig,
    generator: torch.Generator,
) -> Path:
    """Collect exploratory one-step transitions and save them as Parquet."""
    if n_traces <= 0:
        raise ValueError("n_traces must be positive.")
    if not prompts:
        raise ValueError("prompts must be non-empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    common_vocab_n = common_vocab_size(target_model, draft_model)
    records: list[dict[str, Any]] = []

    for trace_idx in range(n_traces):
        raw_prompt = prompts[trace_idx % len(prompts)]
        context_tokens = _ensure_prompt_tokens(target_model, raw_prompt)
        k = 0

        for rollout_step in range(max(2, config.gamma_max + 2)):
            p_probs = next_token_probs_tensor(target_model, context_tokens, common_vocab_n)
            q_probs = next_token_probs_tensor(draft_model, context_tokens, common_vocab_n)
            H = entropy(q_probs)
            K = kl_divergence(q_probs, p_probs)
            state_idx = state_space.encode(H, K, k)
            valid_action_indices = action_space.valid_action_indices(k)
            results_by_action: dict[int, dict[str, Any]] = {}

            for action_idx in valid_action_indices:
                action = action_space.decode(action_idx)
                result = _transition_from_action(
                    target_model=target_model,
                    draft_model=draft_model,
                    context_tokens=context_tokens,
                    k=k,
                    action=action,
                    p_probs=p_probs,
                    q_probs=q_probs,
                    state_space=state_space,
                    generator=generator,
                    config=config,
                    common_vocab_n=common_vocab_n,
                )
                results_by_action[action_idx] = result
                records.append(
                    {
                        "trace_idx": trace_idx,
                        "rollout_step": rollout_step,
                        "state_idx": state_idx,
                        "action_idx": action_idx,
                        "action_length": action.length_action,
                        "threshold": action.threshold,
                        "reward": result["reward"],
                        "next_state_idx": result["next_state_idx"],
                        "accepted": result["accepted"],
                        "proposed": result["proposed"],
                        "emitted_token": result["emitted_token"],
                        "step_time_ms": result["step_time_ms"],
                        "d_step": result["d_step"],
                        "H": H,
                        "K": K,
                        "k": k,
                        "next_H": result["next_H"],
                        "next_K": result["next_K"],
                        "next_k": result["next_k"],
                    }
                )

            chosen_offset = int(
                torch.randint(len(valid_action_indices), size=(1,), generator=generator).item()
            )
            chosen_action_idx = valid_action_indices[chosen_offset]
            chosen_result = results_by_action[chosen_action_idx]
            context_tokens = chosen_result["next_context"]
            k = int(chosen_result["next_k"])

    pd.DataFrame.from_records(records).to_parquet(output_path, index=False)

    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit_or_none(),
        "n_traces": n_traces,
        "num_records": len(records),
        "config": asdict(config),
    }
    meta_path = output_path.with_name(f"{output_path.stem}_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
