"""Trace collection for estimating the JointAdaSpec tabular MDP."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Sequence

import pandas as pd
import torch

from jointadaspec.core.features import entropy, kl_divergence
from jointadaspec.core.verification import tv_distance_step, verify_draft_chain
from jointadaspec.mdp.spaces import (
    ActionSpace,
    JointAction,
    MDPConfig,
    StateSpace,
    quality_risk_penalty,
    quality_risk_weight,
)
from jointadaspec.semantics import TRACE_SCHEMA_VERSION, semantic_metadata
from jointadaspec.utils.probs import (
    block_next_token_probs_tensor,
    common_vocab_size,
    next_token_probs_tensor,
)


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
    draft_tokens: list[int],
    q_list: list[torch.Tensor],
    k: int,
    action: JointAction,
    q_probs: torch.Tensor,
    K_prev: float,
    state_space: StateSpace,
    generator: torch.Generator,
    config: MDPConfig,
    common_vocab_n: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    accepted = 0
    proposed = 0
    emitted_count = 0
    d_step = 0.0

    if action.length_action == "continue" and k < config.gamma_max:
        draft_token = _sample_token(q_probs, generator)
        next_context = list(context_tokens)
        next_draft_tokens = list(draft_tokens) + [int(draft_token)]
        next_q_list = list(q_list) + [q_probs]
        next_k = len(next_draft_tokens)
        next_K = float(K_prev)
        q_next = next_token_probs_tensor(
            draft_model,
            next_context + next_draft_tokens,
            common_vocab_n,
        )
        next_H = entropy(q_next)
        next_state_idx = state_space.encode(next_H, next_K, next_k)
        emitted_token = int(draft_token)
        proposed = 1
    else:
        if not draft_tokens:
            p_block = block_next_token_probs_tensor(target_model, context_tokens, [], common_vocab_n)
            p_probs = p_block[0]
            emitted_token = _sample_token(p_probs, generator)
            emitted_tokens = [int(emitted_token)]
            next_K = kl_divergence(q_probs, p_probs)
            accepted = 0
        else:
            p_block = block_next_token_probs_tensor(
                target_model,
                context_tokens,
                draft_tokens,
                common_vocab_n,
            )
            p_list = p_block[:-1]
            p_bonus = p_block[-1]
            n_accepted, corrective_token = verify_draft_chain(
                p_list=p_list,
                q_list=q_list,
                draft_tokens=draft_tokens,
                T=action.threshold,
                generator=generator,
                p_bonus=p_bonus,
            )
            accepted = int(n_accepted)
            emitted_tokens = list(draft_tokens[:n_accepted]) + [int(corrective_token)]
            K_values = [kl_divergence(q, p) for q, p in zip(q_list, p_list, strict=True)]
            next_K = float(sum(K_values) / len(K_values)) if K_values else float(K_prev)
            d_step = float(
                sum(tv_distance_step(p, q, action.threshold) for p, q in zip(p_list, q_list, strict=True))
                / max(len(q_list), 1)
            )
            emitted_token = int(emitted_tokens[-1])
        emitted_count = len(emitted_tokens)
        next_context = list(context_tokens) + emitted_tokens
        next_draft_tokens = []
        next_q_list = []
        next_k = 0
        q_next = next_token_probs_tensor(draft_model, next_context, common_vocab_n)
        next_H = entropy(q_next)
        next_state_idx = state_space.encode(next_H, next_K, next_k)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    reward = float(emitted_count - config.c_time * elapsed_ms - config.kappa * d_step)

    return {
        "next_context": next_context,
        "next_draft_tokens": next_draft_tokens,
        "next_q_list": next_q_list,
        "next_k": next_k,
        "next_state_idx": next_state_idx,
        "next_H": next_H,
        "next_K": next_K,
        "reward": reward,
        "accepted": int(accepted),
        "proposed": int(proposed),
        "emitted_token": int(emitted_token),
        "emitted_count": int(emitted_count),
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
        draft_tokens: list[int] = []
        q_list: list[torch.Tensor] = []
        K_prev = float(config.K_init)
        k = 0

        for rollout_step in range(max(2, config.gamma_max + 2)):
            q_probs = next_token_probs_tensor(
                draft_model,
                context_tokens + draft_tokens,
                common_vocab_n,
            )
            H = entropy(q_probs)
            K = float(K_prev)
            state_idx = state_space.encode(H, K, k)
            _, i_K, _ = state_space.decode(state_idx)
            additive_form = config.quality_risk_form == "additive"
            if additive_form:
                state_penalty = quality_risk_penalty(config, i_K=i_K, k=k)
                risk_weight = 1.0
            else:
                state_penalty = 0.0
                risk_weight = quality_risk_weight(config, i_K=i_K, k=k)
            valid_action_indices = action_space.valid_action_indices(k)
            results_by_action: dict[int, dict[str, Any]] = {}

            for action_idx in valid_action_indices:
                action = action_space.decode(action_idx)
                result = _transition_from_action(
                    target_model=target_model,
                    draft_model=draft_model,
                    context_tokens=context_tokens,
                    draft_tokens=draft_tokens,
                    q_list=q_list,
                    k=k,
                    action=action,
                    q_probs=q_probs,
                    K_prev=K_prev,
                    state_space=state_space,
                    generator=generator,
                    config=config,
                    common_vocab_n=common_vocab_n,
                )
                if additive_form:
                    reward = (
                        float(result["emitted_count"])
                        - config.c_time * float(result["step_time_ms"])
                        - config.kappa * float(result["d_step"])
                        - state_penalty
                    )
                else:
                    reward = (
                        float(result["emitted_count"])
                        - config.c_time * float(result["step_time_ms"])
                        - config.kappa * risk_weight * float(result["d_step"])
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
                        "reward": reward,
                        "next_state_idx": result["next_state_idx"],
                        "accepted": result["accepted"],
                        "proposed": result["proposed"],
                        "emitted_token": result["emitted_token"],
                        "emitted_count": result["emitted_count"],
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
            draft_tokens = chosen_result["next_draft_tokens"]
            q_list = chosen_result["next_q_list"]
            K_prev = float(chosen_result["next_K"])
            k = int(chosen_result["next_k"])

    pd.DataFrame.from_records(records).to_parquet(output_path, index=False)

    meta = semantic_metadata(
        config=config,
        num_actions=action_space.num_actions,
    )
    meta.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit_or_none(),
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "n_traces": n_traces,
            "num_records": len(records),
            "artifact_kind": "traces",
        }
    )
    meta_path = output_path.with_name(f"{output_path.stem}_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
