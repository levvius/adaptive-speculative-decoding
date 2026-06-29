"""Trace collection for estimating the JointAdaSpec tabular MDP."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import pandas as pd
import torch

from jointadaspec.core.features import draft_confidence, entropy, kl_divergence
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
    verify_p_block: list[torch.Tensor] | None = None,
    verify_p_block_time_ms: float = 0.0,
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
        next_C = draft_confidence(q_next, config.draft_conf_feature)
        next_state_idx = state_space.encode(next_H, next_K, next_k, C=next_C)
        emitted_token = int(draft_token)
        proposed = 1
    else:
        if not draft_tokens:
            p_block = verify_p_block
            if p_block is None:
                p_block = block_next_token_probs_tensor(target_model, context_tokens, [], common_vocab_n)
            p_probs = p_block[0]
            emitted_token = _sample_token(p_probs, generator)
            emitted_tokens = [int(emitted_token)]
            next_K = kl_divergence(q_probs, p_probs)
            accepted = 0
        else:
            p_block = verify_p_block
            if p_block is None:
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
        next_C = draft_confidence(q_next, config.draft_conf_feature)
        next_state_idx = state_space.encode(next_H, next_K, next_k, C=next_C)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if action.is_verify and verify_p_block is not None:
        elapsed_ms += float(verify_p_block_time_ms)
    reward = float(emitted_count - config.c_time * elapsed_ms - config.kappa * d_step)

    return {
        "next_context": next_context,
        "next_draft_tokens": next_draft_tokens,
        "next_q_list": next_q_list,
        "next_k": next_k,
        "next_state_idx": next_state_idx,
        "next_H": next_H,
        "next_K": next_K,
        "next_C": next_C,
        "reward": reward,
        "accepted": int(accepted),
        "proposed": int(proposed),
        "emitted_token": int(emitted_token),
        "emitted_count": int(emitted_count),
        "step_time_ms": elapsed_ms,
        "d_step": d_step,
    }


def _checkpoint_dir_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_checkpoint")


def _checkpoint_path(checkpoint_dir: Path, start_trace: int, end_trace: int) -> Path:
    if start_trace == end_trace:
        return checkpoint_dir / f"trace_{start_trace:06d}.jsonl"
    return checkpoint_dir / f"trace_{start_trace:06d}_{end_trace:06d}.jsonl"


def _write_checkpoint_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _read_checkpoint_records(checkpoint_dir: Path) -> tuple[list[dict[str, Any]], set[int]]:
    if not checkpoint_dir.is_dir():
        return [], set()
    records: list[dict[str, Any]] = []
    completed: set[int] = set()
    for path in sorted(checkpoint_dir.glob("trace_*.jsonl")):
        if path.name.endswith(".tmp"):
            continue
        trace_indices: set[int] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append(record)
                trace_indices.add(int(record["trace_idx"]))
        completed.update(trace_indices)
    return records, completed


def _format_eta(seconds: float) -> str:
    if seconds == float("inf"):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def collect_traces(
    target_model: Any,
    draft_model: Any,
    prompts: Sequence[str | Sequence[int]],
    n_traces: int,
    output_path: Path,
    config: MDPConfig,
    generator: torch.Generator,
    *,
    resume: bool = True,
    checkpoint_every: int = 1,
    progress_every: int = 5,
) -> Path:
    """Collect exploratory one-step transitions and save them as Parquet."""
    if n_traces <= 0:
        raise ValueError("n_traces must be positive.")
    if not prompts:
        raise ValueError("prompts must be non-empty.")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive.")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = _checkpoint_dir_for(output_path)
    action_space = ActionSpace(config)
    state_space = StateSpace(config)
    common_vocab_n = common_vocab_size(target_model, draft_model)
    records, completed_trace_indices = (
        _read_checkpoint_records(checkpoint_dir) if resume else ([], set())
    )
    completed_trace_indices = {idx for idx in completed_trace_indices if 0 <= idx < n_traces}
    if completed_trace_indices:
        print(
            f"[trace-checkpoint] loaded {len(completed_trace_indices)}/{n_traces} "
            f"completed traces from {checkpoint_dir}",
            flush=True,
        )

    started_at = time.perf_counter()
    base_seed = int(generator.initial_seed())
    pending_records: list[dict[str, Any]] = []
    pending_start_trace: int | None = None
    newly_completed = 0

    for trace_idx in range(n_traces):
        if trace_idx in completed_trace_indices:
            continue
        trace_generator = torch.Generator(device="cpu")
        trace_generator.manual_seed((base_seed + int(trace_idx)) % (2**63 - 1))
        raw_prompt = prompts[trace_idx % len(prompts)]
        context_tokens = _ensure_prompt_tokens(target_model, raw_prompt)
        draft_tokens: list[int] = []
        q_list: list[torch.Tensor] = []
        K_prev = float(config.K_init)
        k = 0
        trace_records: list[dict[str, Any]] = []

        for rollout_step in range(max(2, config.gamma_max + 2)):
            q_probs = next_token_probs_tensor(
                draft_model,
                context_tokens + draft_tokens,
                common_vocab_n,
            )
            H = entropy(q_probs)
            K = float(K_prev)
            C = draft_confidence(q_probs, config.draft_conf_feature)
            state_idx = state_space.encode(H, K, k, C=C)
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
            verify_p_block: list[torch.Tensor] | None = None
            verify_p_block_time_ms = 0.0
            if any(action_space.decode(action_idx).is_verify for action_idx in valid_action_indices):
                verify_started = time.perf_counter()
                verify_p_block = block_next_token_probs_tensor(
                    target_model,
                    context_tokens,
                    draft_tokens,
                    common_vocab_n,
                )
                verify_p_block_time_ms = (time.perf_counter() - verify_started) * 1000.0

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
                    generator=trace_generator,
                    config=config,
                    common_vocab_n=common_vocab_n,
                    verify_p_block=verify_p_block if action.is_verify else None,
                    verify_p_block_time_ms=verify_p_block_time_ms if action.is_verify else 0.0,
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
                trace_records.append(
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
                        "C": C,
                        "k": k,
                        "next_H": result["next_H"],
                        "next_K": result["next_K"],
                        "next_C": result["next_C"],
                        "next_k": result["next_k"],
                    }
                )

            chosen_offset = int(
                torch.randint(len(valid_action_indices), size=(1,), generator=trace_generator).item()
            )
            chosen_action_idx = valid_action_indices[chosen_offset]
            chosen_result = results_by_action[chosen_action_idx]
            context_tokens = chosen_result["next_context"]
            draft_tokens = chosen_result["next_draft_tokens"]
            q_list = chosen_result["next_q_list"]
            K_prev = float(chosen_result["next_K"])
            k = int(chosen_result["next_k"])

        records.extend(trace_records)
        pending_records.extend(trace_records)
        pending_start_trace = trace_idx if pending_start_trace is None else pending_start_trace
        newly_completed += 1
        completed_trace_indices.add(trace_idx)
        should_checkpoint = newly_completed % checkpoint_every == 0 or trace_idx == n_traces - 1
        if should_checkpoint and pending_records:
            checkpoint_path = _checkpoint_path(checkpoint_dir, pending_start_trace, trace_idx)
            _write_checkpoint_records(checkpoint_path, pending_records)
            pending_records = []
            pending_start_trace = None
        if progress_every and (
            newly_completed % progress_every == 0 or len(completed_trace_indices) == n_traces
        ):
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            rate = newly_completed / elapsed
            remaining = n_traces - len(completed_trace_indices)
            eta = float("inf") if rate <= 0 else remaining / rate
            print(
                f"[trace-progress] completed={len(completed_trace_indices)}/{n_traces} "
                f"new={newly_completed} rate={rate:.3f} traces/s eta={_format_eta(eta)}",
                flush=True,
            )

    records = [record for record in records if int(record["trace_idx"]) < n_traces]
    records.sort(
        key=lambda record: (
            int(record["trace_idx"]),
            int(record["rollout_step"]),
            int(record["action_idx"]),
        )
    )
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
