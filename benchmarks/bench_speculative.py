from __future__ import annotations

import argparse
import hashlib
import json
import random
import socket
import statistics
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

try:
    import torch
except ModuleNotFoundError:
    torch = None

from sp_samp.models import NoisyModel, RandomModel
from sp_samp.gsm8k import (
    answers_equivalent,
    extract_final_answer,
    extract_reference_answer,
    load_gsm8k,
)
from sp_samp.livecodebench import load_livecodebench
from sp_samp.mtbench import load_mtbench
from sp_samp.sampling import SamplingStats, sample_baseline, speculative_sample
from sp_samp.specexec import SpecExecStats, specexec_sample

HFModel = None
AutoJudgeClassifier = None
AutoJudgeTrainConfig = None
build_autojudge_classifier = None
parse_c_grid = None
warn_deprecated_autojudge_args = None
AutoJudgeStats = None
autojudge_sample_hf = None
ConsensusGateClassifier = None
ConsensusAutoJudgeTrainConfig = None
ConsensusAutoJudgeStats = None
build_consensus_gate_classifier = None
consensus_autojudge_sample_hf = None
sample_baseline_hf = None
speculative_sample_hf = None
specexec_sample_hf = None
topk_sample_hf = None
TopKStats = None

if torch is not None:
    from sp_samp.autojudge import (
        AutoJudgeClassifier,
        AutoJudgeStats,
        AutoJudgeTrainConfig,
        autojudge_sample_hf,
        build_autojudge_classifier,
        parse_c_grid,
        warn_deprecated_autojudge_args,
    )
    from sp_samp.consensus_autojudge import (
        ConsensusAutoJudgeStats,
        ConsensusAutoJudgeTrainConfig,
        ConsensusGateClassifier,
        build_consensus_gate_classifier,
        consensus_autojudge_sample_hf,
    )
    from sp_samp.hf_adapter import HFModel
    from sp_samp.hf_sampling import sample_baseline_hf, speculative_sample_hf
    from sp_samp.hf_specexec import specexec_sample_hf
    from sp_samp.hf_topk import TopKStats, topk_sample_hf


@dataclass
class EvalSample:
    prompt: str
    reference_answer: Optional[str] = None


class HashTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        for word in text.split():
            digest = hashlib.sha1(word.encode("utf-8")).hexdigest()
            token_id = int(digest, 16) % self.vocab_size
            tokens.append(token_id)
        return tokens


def _default_prompts() -> List[str]:
    return [
        "Explain speculative decoding in simple terms.",
        "Write a short story about a robot and a lighthouse.",
        "Summarize the causes of the French Revolution.",
        "Give three ideas for healthy office snacks.",
        "Describe the trade-offs of breadth-first vs depth-first search.",
    ]


def _build_gsm8k_prompt(question: str, mode: str) -> str:
    q = question.strip()
    if mode == "plain":
        return f"Question: {q}\nAnswer:"
    return (
        "Solve the following grade school math problem. "
        "Show your reasoning, then end with: The final answer is <number>.\n\n"
        f"Question: {q}\nAnswer:"
    )


def _parse_topk_rank(spec: str) -> Optional[int]:
    raw = str(spec).strip().lower()
    if raw == "all":
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("topk_rank must be a positive integer or 'all'.")
    return value


def _accumulate_stats(total, stats) -> None:
    base_fields = ["proposed", "accepted", "steps", "target_tokens", "rejections"]
    for field in base_fields:
        if hasattr(total, field) and hasattr(stats, field):
            setattr(total, field, getattr(total, field) + getattr(stats, field))
    # This metric represents a peak and must be aggregated with max, not sum.
    if hasattr(total, "max_active_branches") and hasattr(stats, "max_active_branches"):
        total.max_active_branches = max(total.max_active_branches, stats.max_active_branches)
    extra_fields = [
        "gate_total",
        "gate_accept_d1",
        "gate_escalate_d2",
        "gate_fallback",
        "accepted_d1",
        "accepted_d2",
        "judge_total",
        "judge_accepted",
        "judge_rejected",
        "topk_mismatches",
        "topk_accepted_mismatches",
        "target_calls",
        "target_fallbacks",
        "draft_calls",
        "draft_prefills",
        "draft2_calls",
        "draft2_prefills",
        "target_prefills",
        "branches_total",
        "branches_kept",
        "branches_pruned",
        "cache_hits",
        "cache_misses",
    ]
    for field in extra_fields:
        if hasattr(total, field) and hasattr(stats, field):
            setattr(total, field, getattr(total, field) + getattr(stats, field))


def _run_once(
    prompts: Iterable[EvalSample],
    encode_fn: Callable[[str], List[int]],
    decode_fn: Optional[Callable[[List[int]], str]],
    target_model,
    draft_model,
    draft2_model,
    method: str,
    eval_task: str,
    max_new_tokens: int,
    k: int,
    seed: int,
    autojudge_model=None,
    autojudge_threshold: Optional[float] = None,
    consensus_gate_model=None,
    consensus_gate_mode: str = "learned",
    consensus_fallback_threshold: Optional[float] = None,
    consensus_top_m: int = 8,
    consensus_feature_mode: str = "ensemble",
    consensus_disable_escalation: bool = False,
    topk_rank: Optional[int] = 4,
    specexec_parallel_branches: int = 8,
    specexec_branch_prune_threshold: float = 0.0,
) -> Tuple[float, float, int, object, int, int, int]:
    rng = random.Random(seed)
    if torch is not None:
        torch.manual_seed(seed)
    total_tokens = 0
    total_prompt_tokens = 0
    gsm8k_correct = 0
    gsm8k_total = 0
    if method == "autojudge":
        if AutoJudgeStats is None:
            raise RuntimeError("AutoJudge requires torch dependencies.")
        total_stats = AutoJudgeStats()
    elif method == "consensus_autojudge":
        if ConsensusAutoJudgeStats is None:
            raise RuntimeError("Consensus AutoJudge requires torch dependencies.")
        total_stats = ConsensusAutoJudgeStats()
    elif method == "topk":
        if TopKStats is None:
            raise RuntimeError("Top-k HF method requires torch dependencies.")
        total_stats = TopKStats()
    elif method == "specexec":
        total_stats = SpecExecStats()
    else:
        total_stats = SamplingStats()
    start = time.perf_counter()
    for prompt_idx, sample in enumerate(prompts):
        prompt = sample.prompt
        reference_answer = sample.reference_answer
        prompt_tokens = encode_fn(prompt)
        total_prompt_tokens += len(prompt_tokens)
        try:
            if HFModel is not None and isinstance(target_model, HFModel):
                if method == "baseline":
                    generated, stats = sample_baseline_hf(
                        target_model,
                        prompt_tokens,
                        max_new_tokens,
                        eos_id=target_model.eos_token_id,
                        return_stats=True,
                    )
                elif method == "speculative":
                    generated, stats = speculative_sample_hf(
                        target_model,
                        draft_model,
                        prompt_tokens,
                        max_new_tokens,
                        k=k,
                        eos_id=target_model.eos_token_id,
                        return_stats=True,
                    )
                elif method == "autojudge":
                    if autojudge_model is None:
                        raise ValueError("AutoJudge model is required for autojudge method.")
                    generated, stats = autojudge_sample_hf(
                        target_model=target_model,
                        draft_model=draft_model,
                        judge_model=autojudge_model,
                        prompt_tokens=prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        k=k,
                        threshold=autojudge_threshold,
                        eos_id=target_model.eos_token_id,
                        seed=seed,
                    )
                elif method == "consensus_autojudge":
                    if consensus_autojudge_sample_hf is None:
                        raise RuntimeError("Consensus AutoJudge HF implementation is unavailable.")
                    if draft2_model is None:
                        raise ValueError("Second draft model is required for consensus_autojudge.")
                    generated, stats = consensus_autojudge_sample_hf(
                        target_model=target_model,
                        draft_model=draft_model,
                        draft2_model=draft2_model,
                        gate_model=consensus_gate_model,
                        gate_mode=consensus_gate_mode,
                        prompt_tokens=prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        k=k,
                        fallback_threshold=consensus_fallback_threshold,
                        eos_id=target_model.eos_token_id,
                        seed=seed,
                        top_m=consensus_top_m,
                        feature_mode=consensus_feature_mode,
                        disable_escalation=consensus_disable_escalation,
                    )
                elif method == "topk":
                    if topk_sample_hf is None:
                        raise RuntimeError("Top-k HF implementation is unavailable.")
                    generated, stats = topk_sample_hf(
                        target_model=target_model,
                        draft_model=draft_model,
                        prompt_tokens=prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        k=k,
                        topk_rank=topk_rank,
                        eos_id=target_model.eos_token_id,
                        seed=seed,
                    )
                elif method == "specexec":
                    if specexec_sample_hf is None:
                        raise RuntimeError("SpecExec HF implementation is unavailable.")
                    generated, stats = specexec_sample_hf(
                        target_model=target_model,
                        draft_model=draft_model,
                        prompt_tokens=prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        k=k,
                        parallel_branches=specexec_parallel_branches,
                        branch_prune_threshold=specexec_branch_prune_threshold,
                        eos_id=target_model.eos_token_id,
                        seed=seed,
                        return_stats=True,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
            else:
                if method == "baseline":
                    generated, stats = sample_baseline(
                        target_model,
                        prompt_tokens,
                        max_new_tokens,
                        rng=rng,
                        return_stats=True,
                    )
                elif method == "speculative":
                    generated, stats = speculative_sample(
                        target_model,
                        draft_model,
                        prompt_tokens,
                        max_new_tokens,
                        k=k,
                        rng=rng,
                        return_stats=True,
                    )
                elif method == "autojudge":
                    raise ValueError("AutoJudge is currently supported only for HF models.")
                elif method == "consensus_autojudge":
                    raise ValueError("consensus_autojudge is currently supported only for HF models.")
                elif method == "topk":
                    raise ValueError("Top-k is currently supported only for HF models.")
                elif method == "specexec":
                    generated, stats = specexec_sample(
                        target_model,
                        draft_model,
                        prompt_tokens,
                        max_new_tokens,
                        k=k,
                        parallel_branches=specexec_parallel_branches,
                        branch_prune_threshold=specexec_branch_prune_threshold,
                        rng=rng,
                        return_stats=True,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
        except Exception as exc:
            raise RuntimeError(
                f"method={method} failed on prompt_index={prompt_idx} "
                f"(prompt_tokens={len(prompt_tokens)}, max_new_tokens={max_new_tokens}, "
                f"k={k}, seed={seed}): {exc}"
            ) from exc
        total_tokens += len(generated)
        if eval_task == "gsm8k" and reference_answer is not None:
            if decode_fn is None:
                raise RuntimeError("decode_fn is required for GSM8K quality evaluation.")
            generated_text = decode_fn(generated)
            predicted_answer = extract_final_answer(generated_text)
            if answers_equivalent(predicted_answer, reference_answer):
                gsm8k_correct += 1
            gsm8k_total += 1
        _accumulate_stats(total_stats, stats)
    duration = time.perf_counter() - start
    tokens_per_sec = total_tokens / duration if duration > 0 else 0.0
    return (
        tokens_per_sec,
        duration,
        total_tokens,
        total_stats,
        total_prompt_tokens,
        gsm8k_correct,
        gsm8k_total,
    )


def _resolve_out_path(path: Optional[str]) -> Path:
    if path is None:
        return Path("datasets") / "results.jsonl"
    out_path = Path(path)
    if out_path.is_dir():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return out_path / f"results_{timestamp}.jsonl"
    return out_path


def _append_result(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None
    return out.strip()


def _torch_load_checkpoint(path: Path):
    if torch is None:
        raise RuntimeError("torch is required to load checkpoints.")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _collect_system_metadata(require_headless: bool = False) -> Dict[str, Optional[str]]:
    git_sha = _run_cmd(["git", "rev-parse", "HEAD"])
    hostname = socket.gethostname()

    torch_version = None
    cuda_runtime = None
    if torch is not None:
        torch_version = getattr(torch, "__version__", None)
        cuda_runtime = getattr(torch.version, "cuda", None)

    transformers_version = None
    try:
        import transformers  # type: ignore

        transformers_version = getattr(transformers, "__version__", None)
    except Exception:
        transformers_version = None

    gpu_name = None
    gpu_driver = None
    display_active = None
    nvidia_out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,display_active",
            "--format=csv,noheader",
        ]
    )
    if nvidia_out:
        rows = [row.strip() for row in nvidia_out.splitlines() if row.strip()]
        if rows:
            names: List[str] = []
            drivers: List[str] = []
            displays: List[str] = []
            for row in rows:
                parts = [part.strip() for part in row.split(",")]
                if len(parts) >= 3:
                    names.append(parts[0])
                    drivers.append(parts[1])
                    displays.append(parts[2].lower())
            gpu_name = "; ".join(names) if names else None
            gpu_driver = "; ".join(drivers) if drivers else None
            if displays:
                active = any(value in {"enabled", "on", "active"} for value in displays)
                display_active = "enabled" if active else "disabled"
                if active and require_headless:
                    raise SystemExit(
                        "Headless mode required, but nvidia-smi reports display-active GPU. "
                        "Stop GUI/Xorg/Wayland sessions or rerun without --require-headless."
                    )
                if active and not require_headless:
                    print(
                        "[WARN] GPU display is active; for long runs prefer headless mode "
                        "(pass --require-headless to enforce)."
                    )

    return {
        "git_sha": git_sha,
        "hostname": hostname,
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "cuda_runtime": cuda_runtime,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "display_active": display_active,
    }


def _base_record_fields(
    *,
    method: str,
    backend: str,
    resolved_target_model: str,
    resolved_draft_model: str,
    resolved_draft2_model: Optional[str],
    resolved_target_tokenizer: str,
    resolved_draft_tokenizer: str,
    resolved_draft2_tokenizer: Optional[str],
    args: argparse.Namespace,
    resolved_draft_device: str,
    resolved_draft_dtype: str,
    resolved_draft_quant: Optional[str],
    resolved_draft_bnb_compute_dtype: Optional[str],
    resolved_draft2_device: Optional[str],
    resolved_draft2_dtype: Optional[str],
    resolved_draft2_quant: Optional[str],
    resolved_draft2_bnb_compute_dtype: Optional[str],
    resolved_autojudge_threshold_used: float,
    resolved_autojudge_threshold_calibrated: float,
    autojudge_train_samples: int,
    autojudge_val_auc: float,
    autojudge_val_recall: float,
    resolved_consensus_fallback_threshold_used: Optional[float],
    consensus_train_samples: int,
    consensus_val_accuracy: float,
    consensus_val_macro_f1: float,
) -> Dict[str, object]:
    return {
        "method": method,
        "backend": backend,
        "target_model": resolved_target_model,
        "draft_model": resolved_draft_model,
        "draft2_model": resolved_draft2_model,
        "tokenizer": resolved_target_tokenizer,
        "draft_tokenizer": resolved_draft_tokenizer,
        "draft2_tokenizer": resolved_draft2_tokenizer,
        "device": args.device,
        "dtype": args.dtype,
        "quant": args.quant,
        "bnb_compute_dtype": args.bnb_compute_dtype,
        "draft_device": resolved_draft_device,
        "draft_dtype": resolved_draft_dtype,
        "draft_quant": resolved_draft_quant,
        "draft_bnb_compute_dtype": resolved_draft_bnb_compute_dtype,
        "draft2_device": resolved_draft2_device,
        "draft2_dtype": resolved_draft2_dtype,
        "draft2_quant": resolved_draft2_quant,
        "draft2_bnb_compute_dtype": resolved_draft2_bnb_compute_dtype,
        "use_chat_template": args.use_chat_template,
        "system_prompt": args.system_prompt,
        "k": args.k,
        "max_new_tokens": args.max_new_tokens,
        "max_samples": args.max_samples,
        "turn_index": args.turn_index,
        "dataset": args.dataset,
        "eval_task": args.eval_task,
        "gsm8k_eval_mode": args.gsm8k_eval_mode,
        "topk_rank": args.topk_rank,
        "topk_grid": args.topk_grid,
        "autojudge_threshold": resolved_autojudge_threshold_used,
        "autojudge_threshold_used": resolved_autojudge_threshold_used,
        "autojudge_threshold_calibrated": resolved_autojudge_threshold_calibrated,
        "autojudge_task": args.autojudge_task,
        "autojudge_classifier": args.autojudge_classifier,
        "autojudge_features": getattr(args, "autojudge_features", "hidden"),
        "autojudge_train_dataset": args.autojudge_train_dataset,
        "autojudge_recall_target": args.autojudge_recall_target,
        "autojudge_train_split": args.autojudge_train_split,
        "autojudge_c_grid": args.autojudge_c_grid,
        "autojudge_train_samples": autojudge_train_samples,
        "autojudge_train_loss": 0.0,
        "autojudge_val_auc": autojudge_val_auc,
        "autojudge_val_recall": autojudge_val_recall,
        # Legacy alias kept for backward compatibility.
        "autojudge_threshold_selected": resolved_autojudge_threshold_calibrated,
        "autojudge_checkpoint": args.autojudge_checkpoint,
        "consensus_gate": getattr(args, "consensus_gate", "learned"),
        "consensus_features": getattr(args, "consensus_features", "ensemble"),
        "consensus_train_dataset": getattr(args, "consensus_train_dataset", None),
        "consensus_train_samples": consensus_train_samples,
        "consensus_train_split": getattr(args, "consensus_train_split", None),
        "consensus_fallback_threshold": resolved_consensus_fallback_threshold_used,
        "consensus_checkpoint": getattr(args, "consensus_checkpoint", None),
        "consensus_top_m": getattr(args, "consensus_top_m", None),
        "consensus_disable_escalation": bool(
            getattr(args, "consensus_disable_escalation", False)
        ),
        "consensus_val_accuracy": consensus_val_accuracy,
        "consensus_val_macro_f1": consensus_val_macro_f1,
        "parallel_branches": args.parallel_branches,
        "branch_prune_threshold": args.branch_prune_threshold,
    }


def _resume_identity(
    *,
    run: int,
    base_fields: Dict[str, object],
) -> Dict[str, object]:
    identity = dict(base_fields)
    identity["run"] = run
    identity["summary"] = False
    identity["status"] = "ok"
    return identity


def _make_resume_key(identity: Dict[str, object]) -> str:
    return json.dumps(identity, ensure_ascii=False, sort_keys=True)


def _record_resume_key(record: Dict[str, object]) -> Optional[str]:
    existing = record.get("resume_key")
    if isinstance(existing, str):
        return existing
    if record.get("summary"):
        return None
    if record.get("status") not in {None, "ok"}:
        return None
    required = [
        "method",
        "backend",
        "target_model",
        "draft_model",
        "draft2_model",
        "tokenizer",
        "draft_tokenizer",
        "draft2_tokenizer",
        "device",
        "dtype",
        "quant",
        "bnb_compute_dtype",
        "draft_device",
        "draft_dtype",
        "draft_quant",
        "draft_bnb_compute_dtype",
        "draft2_device",
        "draft2_dtype",
        "draft2_quant",
        "draft2_bnb_compute_dtype",
        "use_chat_template",
        "system_prompt",
        "k",
        "max_new_tokens",
        "max_samples",
        "turn_index",
        "dataset",
        "eval_task",
        "gsm8k_eval_mode",
        "topk_rank",
        "topk_grid",
        "autojudge_threshold",
        "autojudge_threshold_used",
        "autojudge_threshold_calibrated",
        "autojudge_task",
        "autojudge_classifier",
        "autojudge_features",
        "autojudge_train_dataset",
        "autojudge_recall_target",
        "autojudge_train_split",
        "autojudge_c_grid",
        "autojudge_train_samples",
        "autojudge_train_loss",
        "autojudge_val_auc",
        "autojudge_val_recall",
        "autojudge_threshold_selected",
        "autojudge_checkpoint",
        "consensus_gate",
        "consensus_features",
        "consensus_train_dataset",
        "consensus_train_samples",
        "consensus_train_split",
        "consensus_fallback_threshold",
        "consensus_checkpoint",
        "consensus_top_m",
        "consensus_disable_escalation",
        "consensus_val_accuracy",
        "consensus_val_macro_f1",
        "parallel_branches",
        "branch_prune_threshold",
        "run",
    ]
    missing = [key for key in required if key not in record]
    if missing:
        return None
    identity = {key: record.get(key) for key in required}
    identity["summary"] = False
    identity["status"] = "ok"
    return _make_resume_key(identity)


def _load_completed_run_keys(path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            key = _record_resume_key(record)
            if key:
                keys.add(key)
    return keys


def _resolve_methods(method: str) -> List[str]:
    if method == "both":
        return ["baseline", "speculative"]
    if method == "all":
        return ["baseline", "speculative", "autojudge", "topk", "specexec"]
    if method == "all_paper":
        return ["baseline", "speculative", "autojudge", "topk"]
    return [method]


def _tokenizers_compatible(target_tokenizer, draft_tokenizer) -> bool:
    if target_tokenizer is draft_tokenizer:
        return True
    try:
        target_vocab = target_tokenizer.get_vocab()
        draft_vocab = draft_tokenizer.get_vocab()
    except Exception:
        return False
    return target_vocab == draft_vocab


def _all_tokenizers_compatible(*tokenizers) -> bool:
    filtered = [tokenizer for tokenizer in tokenizers if tokenizer is not None]
    if len(filtered) <= 1:
        return True
    base = filtered[0]
    for tokenizer in filtered[1:]:
        if not _tokenizers_compatible(base, tokenizer):
            return False
    return True


def _uses_native_quantized_checkpoint(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    name = model_name.lower()
    return "gpt-oss" in name or "gpt_oss" in name


def _is_cuda_device(device: Optional[str]) -> bool:
    return bool(device) and str(device).startswith("cuda")


def _current_cuda_arch_tag() -> Optional[str]:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"sm_{major}{minor}"


def _torch_cuda_arch_list() -> List[str]:
    if torch is None:
        return []
    get_arch_list = getattr(torch.cuda, "get_arch_list", None)
    if get_arch_list is None:
        return []
    try:
        arch_list = get_arch_list()
    except Exception:
        return []
    return [arch for arch in arch_list if isinstance(arch, str)]


def _validate_torch_cuda_arch_support(device: Optional[str]) -> None:
    if torch is None or not _is_cuda_device(device) or not torch.cuda.is_available():
        return
    current_arch = _current_cuda_arch_tag()
    supported_arches = _torch_cuda_arch_list()
    if not current_arch or not supported_arches:
        return
    if current_arch in supported_arches:
        return
    gpu_name = "unknown"
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    raise SystemExit(
        f"Installed torch build does not support this GPU architecture: {gpu_name} ({current_arch}). "
        f"Supported arches in current torch: {', '.join(supported_arches)}. "
        "For RTX 50xx / Blackwell use torch with sm_120 support (recommended torch>=2.7 with cu128+). "
        "Docker example: "
        "`make docker-build-gpu-safe CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 "
        "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 TORCH_VERSION=2.9.1`."
    )


def _stats_record_fields(stats) -> dict:
    record = {
        "acceptance_rate": stats.acceptance_rate,
        "avg_tokens_per_step": stats.avg_tokens_per_step,
        "proposed": stats.proposed,
        "accepted": stats.accepted,
        "steps": stats.steps,
        "rejections": stats.rejections,
    }
    if hasattr(stats, "judge_accept_rate"):
        record["judge_accept_rate"] = stats.judge_accept_rate
    if hasattr(stats, "gate_accept_d1_rate"):
        record["gate_accept_d1_rate"] = stats.gate_accept_d1_rate
    if hasattr(stats, "gate_escalate_rate"):
        record["gate_escalate_rate"] = stats.gate_escalate_rate
    if hasattr(stats, "gate_fallback_rate"):
        record["gate_fallback_rate"] = stats.gate_fallback_rate
    if hasattr(stats, "topk_accept_rate"):
        record["topk_accept_rate"] = stats.topk_accept_rate
    if hasattr(stats, "target_fallback_rate"):
        record["target_fallback_rate"] = stats.target_fallback_rate
    if hasattr(stats, "target_calls_per_token"):
        record["target_calls_per_token"] = stats.target_calls_per_token
    if hasattr(stats, "draft_calls_per_token"):
        record["draft_calls_per_token"] = stats.draft_calls_per_token
    if hasattr(stats, "draft2_calls_per_token"):
        record["draft2_calls_per_token"] = stats.draft2_calls_per_token
    if hasattr(stats, "branch_prune_rate"):
        record["branch_prune_rate"] = stats.branch_prune_rate
    if hasattr(stats, "effective_parallelism"):
        record["effective_parallelism"] = stats.effective_parallelism
    if hasattr(stats, "cache_hit_rate"):
        record["cache_hit_rate"] = stats.cache_hit_rate
    for key in [
        "judge_total",
        "judge_accepted",
        "judge_rejected",
        "gate_total",
        "gate_accept_d1",
        "gate_escalate_d2",
        "gate_fallback",
        "accepted_d1",
        "accepted_d2",
        "target_calls",
        "target_fallbacks",
        "draft_calls",
        "draft_prefills",
        "draft2_calls",
        "draft2_prefills",
        "target_prefills",
        "branches_total",
        "branches_kept",
        "branches_pruned",
        "max_active_branches",
        "cache_hits",
        "cache_misses",
        "train_samples",
        "train_loss",
        "val_auc",
        "val_recall",
        "val_accuracy",
        "val_macro_f1",
        "threshold_selected",
        "fallback_threshold_selected",
        "topk_rank_effective",
        "topk_mismatches",
        "topk_accepted_mismatches",
    ]:
        if hasattr(stats, key):
            record[key] = getattr(stats, key)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speculative Sampling benchmark.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to MT-Bench JSON/JSONL.")
    parser.add_argument("--max-samples", type=int, default=50, help="Max prompts to use.")
    parser.add_argument("--turn-index", type=int, default=0, help="Which MT-Bench turn to use.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--k", type=int, default=4, help="Draft length per step.")
    parser.add_argument("--runs", type=int, default=5, help="Number of repeated runs.")
    parser.add_argument(
        "--method",
        type=str,
        default="both",
        choices=[
            "baseline",
            "speculative",
            "autojudge",
            "consensus_autojudge",
            "topk",
            "specexec",
            "both",
            "all",
            "all_paper",
        ],
    )
    parser.add_argument(
        "--eval-task",
        type=str,
        default="mtbench",
        choices=["mtbench", "gsm8k", "livecodebench"],
        help="Evaluation task dataset mode.",
    )
    parser.add_argument(
        "--gsm8k-eval-mode",
        type=str,
        default="zero_shot_cot",
        choices=["zero_shot_cot", "plain"],
        help="Prompt style for GSM8K evaluation.",
    )
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--draft-noise", type=float, default=0.2, help="Noise for draft model.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hf-model", type=str, default=None, help="HF model name/path for target.")
    parser.add_argument("--hf-draft-model", type=str, default=None, help="HF model name/path for draft.")
    parser.add_argument("--hf-draft2-model", type=str, default=None, help="HF model name/path for second draft.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name/path.")
    parser.add_argument("--draft-tokenizer", type=str, default=None, help="Tokenizer name/path for draft model.")
    parser.add_argument("--draft2-tokenizer", type=str, default=None, help="Tokenizer name/path for second draft model.")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu/cuda/mps.")
    parser.add_argument("--draft-device", type=str, default=None, help="Device override for draft model.")
    parser.add_argument("--draft2-device", type=str, default=None, help="Device override for second draft model.")
    parser.add_argument("--dtype", type=str, default="auto", help="Torch dtype: auto/float16/bfloat16/float32.")
    parser.add_argument("--draft-dtype", type=str, default=None, help="Torch dtype override for draft model.")
    parser.add_argument("--draft2-dtype", type=str, default=None, help="Torch dtype override for second draft model.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code for HF models.")
    parser.add_argument("--add-special-tokens", action="store_true", help="Use tokenizer special tokens for prompts.")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply tokenizer chat template to prompts.")
    parser.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt for chat template.")
    parser.add_argument("--quant", type=str, default=None, choices=["8bit", "4bit"], help="Quantization for HF models.")
    parser.add_argument("--draft-quant", type=str, default=None, choices=["8bit", "4bit"], help="Quantization override for draft model.")
    parser.add_argument("--draft2-quant", type=str, default=None, choices=["8bit", "4bit"], help="Quantization override for second draft model.")
    parser.add_argument("--bnb-compute-dtype", type=str, default="bfloat16", help="Compute dtype for 4-bit quantization.")
    parser.add_argument("--draft-bnb-compute-dtype", type=str, default=None, help="Compute dtype override for draft 4-bit quantization.")
    parser.add_argument("--draft2-bnb-compute-dtype", type=str, default=None, help="Compute dtype override for second draft 4-bit quantization.")
    parser.add_argument("--autojudge-threshold", type=float, default=None, help="Decision threshold override for judge acceptance. If unset, calibrated threshold is used.")
    parser.add_argument("--autojudge-task", type=str, default="gsm8k", choices=["gsm8k"], help="AutoJudge training/equivalence task mode.")
    parser.add_argument(
        "--autojudge-classifier",
        type=str,
        default="logreg",
        choices=["logreg", "hgbt"],
        help="AutoJudge classifier backend.",
    )
    parser.add_argument("--autojudge-train-dataset", type=str, default=None, help="Optional dataset path for AutoJudge training. Defaults to --dataset.")
    parser.add_argument("--autojudge-train-samples", type=int, default=4000, help="Max mined mismatches for AutoJudge training.")
    parser.add_argument("--autojudge-recall-target", type=float, default=0.9, help="Validation recall target for threshold calibration.")
    parser.add_argument("--autojudge-train-split", type=float, default=0.9, help="Train split fraction for AutoJudge classifier calibration.")
    parser.add_argument("--autojudge-c-grid", type=str, default=None, help="Comma-separated C values for LogisticRegression grid search (e.g. 1e-7,1e-6,...,1e0).")
    parser.add_argument("--autojudge-train-steps", type=int, default=None, help="Deprecated and ignored.")
    parser.add_argument("--autojudge-train-batch-size", type=int, default=None, help="Deprecated and ignored.")
    parser.add_argument("--autojudge-train-lr", type=float, default=None, help="Deprecated and ignored.")
    parser.add_argument("--autojudge-audit-ratio", type=float, default=None, help="Deprecated and ignored.")
    parser.add_argument("--autojudge-checkpoint", type=str, default=None, help="Path to save/load judge checkpoint (.pt).")
    parser.add_argument(
        "--consensus-gate",
        type=str,
        default="learned",
        choices=["learned", "rule"],
        help="Consensus AutoJudge gate mode.",
    )
    parser.add_argument(
        "--consensus-features",
        type=str,
        default="ensemble",
        choices=["ensemble", "d1_only"],
        help="Consensus AutoJudge feature set.",
    )
    parser.add_argument("--consensus-train-dataset", type=str, default=None, help="Optional dataset path for consensus gate training. Defaults to --dataset.")
    parser.add_argument("--consensus-train-samples", type=int, default=4000, help="Max mined token decisions for consensus gate training.")
    parser.add_argument("--consensus-train-split", type=float, default=0.9, help="Train split fraction for consensus gate validation.")
    parser.add_argument("--consensus-fallback-threshold", type=float, default=0.5, help="Fallback probability threshold for learned consensus gate.")
    parser.add_argument("--consensus-checkpoint", type=str, default=None, help="Path to save/load consensus gate checkpoint (.pt).")
    parser.add_argument("--consensus-top-m", type=int, default=8, help="Top-m support used by consensus features and divergence estimates.")
    parser.add_argument("--consensus-disable-escalation", action="store_true", help="Disable ACCEPT_D2 / escalation path and force a 2-way accept-or-fallback policy at runtime.")
    parser.add_argument(
        "--topk-rank",
        type=str,
        default="4",
        help="Top-k rank for topk method. Use integer or 'all'.",
    )
    parser.add_argument(
        "--topk-grid",
        type=str,
        default="2,4,8,16,32,all",
        help="Comma-separated top-k sweep values used by orchestration scripts.",
    )
    parser.add_argument("--parallel-branches", type=int, default=8, help="Number of draft branches for SpecExec.")
    parser.add_argument("--branch-prune-threshold", type=float, default=0.0, help="Draft-probability pruning threshold in [0,1] for SpecExec.")
    parser.add_argument("--require-headless", action="store_true", help="Fail fast if GPU is display-active (recommended for long runs).")
    parser.add_argument("--out", type=str, default=None, help="Path to JSONL metrics file.")
    return parser


def run_with_args(args: argparse.Namespace) -> None:
    methods = _resolve_methods(args.method)
    needs_draft = any(
        m in {"speculative", "autojudge", "consensus_autojudge", "topk", "specexec"}
        for m in methods
    )
    needs_draft2 = "consensus_autojudge" in methods
    try:
        resolved_topk_rank = _parse_topk_rank(args.topk_rank)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if "autojudge" in methods and not args.hf_model:
        if args.method == "all":
            raise SystemExit(
                "Method 'all' includes AutoJudge, which requires HF models. "
                "Set --hf-model/--hf-draft-model or choose a method without AutoJudge."
            )
        raise SystemExit("AutoJudge requires HF models. Set --hf-model and --hf-draft-model.")

    if "topk" in methods and not args.hf_model:
        raise SystemExit("Top-k method currently requires HF models.")

    if "consensus_autojudge" in methods:
        if not args.hf_model:
            raise SystemExit(
                "consensus_autojudge requires HF models. Set --hf-model, --hf-draft-model, and --hf-draft2-model."
            )
        if not args.hf_draft_model or not args.hf_draft2_model:
            raise SystemExit(
                "consensus_autojudge requires both --hf-draft-model and --hf-draft2-model."
            )

    if args.eval_task == "gsm8k":
        if not args.hf_model:
            raise SystemExit("--eval-task gsm8k currently requires HF models.")
        if not args.dataset:
            raise SystemExit("--eval-task gsm8k requires --dataset path.")
        gsm_samples = load_gsm8k(path=args.dataset, max_samples=args.max_samples)
        if not gsm_samples:
            raise SystemExit("No GSM8K samples loaded from dataset.")
        prompts: List[EvalSample] = [
            EvalSample(
                prompt=_build_gsm8k_prompt(sample.question, mode=args.gsm8k_eval_mode),
                reference_answer=extract_reference_answer(sample.answer),
            )
            for sample in gsm_samples
        ]
    elif args.eval_task == "livecodebench":
        if not args.hf_model:
            raise SystemExit("--eval-task livecodebench currently requires HF models.")
        if not args.dataset:
            raise SystemExit("--eval-task livecodebench requires --dataset path.")
        lcb_prompts = load_livecodebench(path=args.dataset, max_samples=args.max_samples)
        if not lcb_prompts:
            raise SystemExit("No LiveCodeBench prompts loaded from dataset.")
        prompts = [EvalSample(prompt=p) for p in lcb_prompts]
    else:
        if args.dataset:
            mt_prompts = load_mtbench(
                args.dataset, turn_index=args.turn_index, max_samples=args.max_samples
            )
            if not mt_prompts:
                raise SystemExit("No prompts loaded from dataset.")
            prompts = [EvalSample(prompt=p) for p in mt_prompts]
        else:
            prompts = [EvalSample(prompt=p) for p in _default_prompts()[: args.max_samples]]

    target_has_native_quant = _uses_native_quantized_checkpoint(args.hf_model)
    draft_has_native_quant = _uses_native_quantized_checkpoint(args.hf_draft_model)
    draft2_has_native_quant = _uses_native_quantized_checkpoint(args.hf_draft2_model)
    draft_disable_inherited_quant = False
    draft2_disable_inherited_quant = False

    if target_has_native_quant and args.quant in {"4bit", "8bit"}:
        print(
            "[WARN] Target model provides native quantization config. "
            "Ignoring --quant override to avoid config conflicts."
        )
        args.quant = None

    if draft_has_native_quant and args.draft_quant in {"4bit", "8bit"}:
        print(
            "[WARN] Draft model provides native quantization config. "
            "Ignoring --draft-quant override to avoid config conflicts."
        )
        args.draft_quant = None
    if draft_has_native_quant and args.draft_quant is None and args.quant in {"4bit", "8bit"}:
        print(
            "[WARN] Draft model provides native quantization config. "
            "Draft will not inherit --quant override."
        )
        draft_disable_inherited_quant = True

    if draft2_has_native_quant and args.draft2_quant in {"4bit", "8bit"}:
        print(
            "[WARN] Second draft model provides native quantization config. "
            "Ignoring --draft2-quant override to avoid config conflicts."
        )
        args.draft2_quant = None
    if draft2_has_native_quant and args.draft2_quant is None and args.quant in {"4bit", "8bit"}:
        print(
            "[WARN] Second draft model provides native quantization config. "
            "Second draft will not inherit --quant override."
        )
        draft2_disable_inherited_quant = True

    resolved_target_tokenizer = args.tokenizer
    resolved_draft_tokenizer = args.draft_tokenizer
    resolved_draft2_tokenizer = args.draft2_tokenizer
    resolved_target_model = args.hf_model or "toy_random"
    resolved_draft_model = args.hf_draft_model or args.hf_model or "toy_noisy"
    resolved_draft2_model = args.hf_draft2_model or args.hf_draft_model or args.hf_model
    resolved_draft_device = args.draft_device or args.device
    resolved_draft_dtype = args.draft_dtype or args.dtype
    resolved_draft2_device = args.draft2_device or args.device
    resolved_draft2_dtype = args.draft2_dtype or args.dtype
    decode_fn: Optional[Callable[[List[int]], str]] = None
    if draft_disable_inherited_quant:
        resolved_draft_quant = args.draft_quant
    else:
        resolved_draft_quant = args.draft_quant if args.draft_quant is not None else args.quant
    resolved_draft_bnb_compute_dtype = (
        args.draft_bnb_compute_dtype
        if args.draft_bnb_compute_dtype is not None
        else args.bnb_compute_dtype
    )
    if draft2_disable_inherited_quant:
        resolved_draft2_quant = args.draft2_quant
    else:
        resolved_draft2_quant = args.draft2_quant if args.draft2_quant is not None else args.quant
    resolved_draft2_bnb_compute_dtype = (
        args.draft2_bnb_compute_dtype
        if args.draft2_bnb_compute_dtype is not None
        else args.bnb_compute_dtype
    )

    if args.hf_model:
        if torch is None or HFModel is None:
            raise SystemExit(
                "HF benchmark requested but torch/transformers dependencies are missing."
            )
        if _is_cuda_device(args.device) and not torch.cuda.is_available():
            raise SystemExit(
                "CUDA device requested but unavailable in this runtime. "
                "For Docker, run `make docker-gpu-check` and `make docker-gpu-check-image`. "
                "If checks fail, configure nvidia-container-toolkit and restart Docker."
            )
        if _is_cuda_device(args.draft_device) and not torch.cuda.is_available():
            raise SystemExit(
                "Draft CUDA device requested but unavailable in this runtime. "
                "Check Docker GPU runtime and nvidia-container-toolkit setup."
            )
        if _is_cuda_device(args.draft2_device) and not torch.cuda.is_available():
            raise SystemExit(
                "Second draft CUDA device requested but unavailable in this runtime. "
                "Check Docker GPU runtime and nvidia-container-toolkit setup."
            )
        _validate_torch_cuda_arch_support(args.device)
        _validate_torch_cuda_arch_support(args.draft_device)
        _validate_torch_cuda_arch_support(args.draft2_device)
        target_tokenizer_name = args.tokenizer or args.hf_model
        resolved_target_tokenizer = target_tokenizer_name
        target_model = HFModel(
            args.hf_model,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            tokenizer_name=target_tokenizer_name,
            quantization=args.quant,
            bnb_compute_dtype=args.bnb_compute_dtype,
        )

        if needs_draft:
            draft_name = args.hf_draft_model or args.hf_model
            resolved_draft_model = draft_name
            draft_tokenizer_name = (
                args.draft_tokenizer or args.tokenizer or args.hf_draft_model or args.hf_model
            )
            resolved_draft_tokenizer = draft_tokenizer_name
            draft_model = HFModel(
                draft_name,
                device=resolved_draft_device,
                dtype=resolved_draft_dtype,
                trust_remote_code=args.trust_remote_code,
                tokenizer_name=draft_tokenizer_name,
                quantization=resolved_draft_quant,
                bnb_compute_dtype=resolved_draft_bnb_compute_dtype,
            )
            if not _tokenizers_compatible(target_model.tokenizer, draft_model.tokenizer):
                raise SystemExit(
                    "Target and draft tokenizers are incompatible. "
                    "Use models sharing identical token-id mapping."
                )
            if target_model.vocab_size != draft_model.vocab_size:
                print(
                    f"[WARN] Target and draft config vocab sizes differ "
                    f"({target_model.vocab_size} vs {draft_model.vocab_size}). "
                    f"Tokenizers are compatible; difference is padding. "
                    f"Using min vocab_size={min(target_model.vocab_size, draft_model.vocab_size)}."
                )
        else:
            draft_model = target_model
            resolved_draft_model = args.hf_model
            resolved_draft_tokenizer = target_tokenizer_name

        if needs_draft2:
            draft2_name = args.hf_draft2_model or args.hf_draft_model or args.hf_model
            resolved_draft2_model = draft2_name
            draft2_tokenizer_name = (
                args.draft2_tokenizer
                or args.tokenizer
                or args.hf_draft2_model
                or args.hf_draft_model
                or args.hf_model
            )
            resolved_draft2_tokenizer = draft2_tokenizer_name
            draft2_model = HFModel(
                draft2_name,
                device=resolved_draft2_device,
                dtype=resolved_draft2_dtype,
                trust_remote_code=args.trust_remote_code,
                tokenizer_name=draft2_tokenizer_name,
                quantization=resolved_draft2_quant,
                bnb_compute_dtype=resolved_draft2_bnb_compute_dtype,
            )
            if not _all_tokenizers_compatible(
                target_model.tokenizer,
                draft_model.tokenizer,
                draft2_model.tokenizer,
            ):
                raise SystemExit(
                    "Target, draft, and second draft tokenizers are incompatible. "
                    "Use models sharing identical token-id mapping."
                )
            vocab_sizes = (target_model.vocab_size, draft_model.vocab_size, draft2_model.vocab_size)
            if len(set(vocab_sizes)) > 1:
                print(
                    "[WARN] Target/draft/draft2 config vocab sizes differ "
                    f"{vocab_sizes}. Tokenizers are compatible; difference is padding. "
                    f"Using min vocab_size={min(vocab_sizes)}."
                )
        else:
            draft2_model = draft_model
            resolved_draft2_model = resolved_draft_model
            resolved_draft2_tokenizer = resolved_draft_tokenizer
            resolved_draft2_device = resolved_draft_device
            resolved_draft2_dtype = resolved_draft_dtype
            resolved_draft2_quant = resolved_draft_quant
            resolved_draft2_bnb_compute_dtype = resolved_draft_bnb_compute_dtype

        hf_tokenizer = target_model.tokenizer

        def encode_fn(text: str) -> List[int]:
            if args.use_chat_template:
                messages = []
                if args.system_prompt:
                    messages.append({"role": "system", "content": args.system_prompt})
                messages.append({"role": "user", "content": text})
                tokenized = hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if hasattr(tokenized, "keys") and "input_ids" in tokenized:
                    tokenized = tokenized["input_ids"]
                if isinstance(tokenized, dict):
                    tokenized = tokenized.get("input_ids", tokenized)
                if hasattr(tokenized, "tolist"):
                    tokenized = tokenized.tolist()
                if (
                    isinstance(tokenized, list)
                    and tokenized
                    and isinstance(tokenized[0], list)
                ):
                    tokenized = tokenized[0]
                return [int(tok) for tok in tokenized]
            return hf_tokenizer.encode(text, add_special_tokens=args.add_special_tokens)

        def decode_fn(token_ids: List[int]) -> str:
            return hf_tokenizer.decode(token_ids, skip_special_tokens=True)

    else:
        tokenizer = HashTokenizer(args.vocab_size)

        def encode_fn(text: str) -> List[int]:
            return tokenizer.encode(text)

        target_model = RandomModel(args.vocab_size, seed=args.seed)
        draft_model = (
            NoisyModel(target_model, noise=args.draft_noise) if needs_draft else target_model
        )
        draft2_model = draft_model
        resolved_draft_model = "toy_noisy" if needs_draft else "toy_random"
        resolved_draft2_model = resolved_draft_model
        resolved_target_tokenizer = f"hash:{args.vocab_size}"
        resolved_draft_tokenizer = f"hash:{args.vocab_size}"
        resolved_draft2_tokenizer = f"hash:{args.vocab_size}"
        resolved_draft2_device = args.device
        resolved_draft2_dtype = args.dtype
        resolved_draft2_quant = args.quant
        resolved_draft2_bnb_compute_dtype = args.bnb_compute_dtype

    if "autojudge" in methods and (HFModel is None or not isinstance(target_model, HFModel)):
        raise SystemExit("AutoJudge requires HF models. Set --hf-model and --hf-draft-model.")
    if "consensus_autojudge" in methods and (
        HFModel is None or not isinstance(target_model, HFModel)
    ):
        raise SystemExit(
            "consensus_autojudge requires HF models. Set --hf-model, --hf-draft-model, and --hf-draft2-model."
        )

    autojudge_model = None
    autojudge_train_samples = 0
    autojudge_val_auc = 0.0
    autojudge_val_recall = 0.0
    autojudge_threshold_calibrated = 0.0
    if "autojudge" in methods:
        if (
            AutoJudgeClassifier is None
            or AutoJudgeTrainConfig is None
            or build_autojudge_classifier is None
            or parse_c_grid is None
            or warn_deprecated_autojudge_args is None
            or torch is None
        ):
            raise SystemExit(
                "AutoJudge dependencies are missing (torch/transformers/scikit-learn)."
            )

        if (
            args.autojudge_train_steps is not None
            or args.autojudge_train_batch_size is not None
            or args.autojudge_train_lr is not None
            or args.autojudge_audit_ratio is not None
        ):
            warn_deprecated_autojudge_args()

        if not (0.0 < float(args.autojudge_train_split) < 1.0):
            raise SystemExit("--autojudge-train-split must be in (0,1).")
        if not (0.0 < float(args.autojudge_recall_target) <= 1.0):
            raise SystemExit("--autojudge-recall-target must be in (0,1].")
        requested_autojudge_classifier = str(
            getattr(args, "autojudge_classifier", "logreg")
        ).strip().lower()
        if requested_autojudge_classifier not in {"logreg", "hgbt"}:
            raise SystemExit(
                "--autojudge-classifier must be one of: logreg, hgbt."
            )
        args.autojudge_classifier = requested_autojudge_classifier

        requested_autojudge_features = str(
            getattr(args, "autojudge_features", "hidden") or "hidden"
        ).strip().lower()
        if requested_autojudge_features not in {"hidden", "dist"}:
            raise SystemExit(
                "--autojudge-features must be one of: hidden, dist."
            )
        args.autojudge_features = requested_autojudge_features

        should_train = True
        checkpoint_path = Path(args.autojudge_checkpoint) if args.autojudge_checkpoint else None
        if checkpoint_path is not None and checkpoint_path.exists():
            payload = _torch_load_checkpoint(checkpoint_path)
            classifier = payload.get("classifier") if isinstance(payload, dict) else None
            version = payload.get("autojudge_version") if isinstance(payload, dict) else None
            payload_classifier_backend = "logreg"
            if isinstance(payload, dict):
                payload_classifier_backend = str(
                    payload.get(
                        "classifier_backend",
                        getattr(classifier, "classifier_backend", "logreg"),
                    )
                ).strip().lower()
            payload_features = "hidden"
            if isinstance(payload, dict):
                payload_features = str(
                    payload.get("autojudge_features", "hidden")
                ).strip().lower()
            if (
                version == 2
                and classifier is not None
                and hasattr(classifier, "predict_important_prob")
                and hasattr(classifier, "threshold")
                and payload_classifier_backend == requested_autojudge_classifier
                and payload_features == requested_autojudge_features
            ):
                autojudge_model = classifier
                autojudge_train_samples = int(payload.get("train_samples", 0))
                autojudge_val_auc = float(payload.get("val_auc", 0.0))
                autojudge_val_recall = float(payload.get("val_recall", 0.0))
                autojudge_threshold_calibrated = float(
                    payload.get("threshold_selected", classifier.threshold)
                )
                should_train = False
            else:
                reason = "legacy/unsupported checkpoint"
                if version == 2 and classifier is not None:
                    if payload_classifier_backend != requested_autojudge_classifier:
                        reason = (
                            f"classifier mismatch ckpt={payload_classifier_backend} "
                            f"requested={requested_autojudge_classifier}"
                        )
                    elif payload_features != requested_autojudge_features:
                        reason = (
                            f"features mismatch ckpt={payload_features} "
                            f"requested={requested_autojudge_features}"
                        )
                print(
                    f"[WARN] {reason} at {checkpoint_path}; retraining AutoJudge head."
                )

        if should_train:
            train_dataset = args.autojudge_train_dataset or args.dataset
            if not train_dataset:
                raise SystemExit(
                    "AutoJudge training requires a GSM8K train dataset when checkpoint is absent.\n"
                    "Provide --autojudge-train-dataset <gsm8k_train.jsonl> (or set --dataset to GSM8K).\n"
                    "Example:\n"
                    "  python -m sp_samp.cli autojudge --config-dir configs "
                    "--experiment qwen25_7b_target_qwen25_0p5b_autojudge_k4 "
                    "--dataset datasets/mt_bench.jsonl "
                    "--autojudge-train-dataset datasets/gsm8k_train.jsonl "
                    "--out datasets/results_autojudge.jsonl"
                )
            train_dataset_path = Path(train_dataset)
            if not train_dataset_path.exists():
                raise SystemExit(
                    f"AutoJudge train dataset path not found: {train_dataset_path}. "
                    "Provide a valid GSM8K JSON/JSONL file with 'question' and 'answer' fields."
                )

            try:
                gsm_samples = load_gsm8k(
                    path=str(train_dataset_path),
                    max_samples=args.autojudge_train_samples,
                )
            except Exception as exc:
                raise SystemExit(
                    f"Failed to load AutoJudge train dataset from {train_dataset_path}: {exc}"
                ) from exc
            if not gsm_samples:
                raise SystemExit(
                    "AutoJudge training dataset is not GSM8K-compatible. "
                    "Expected records with 'question' and 'answer' fields. "
                    "MT-Bench files are not suitable for AutoJudge training."
                )

            train_cfg = AutoJudgeTrainConfig(
                task=args.autojudge_task,
                max_train_samples=args.autojudge_train_samples,
                max_new_tokens=args.max_new_tokens,
                k=args.k,
                train_split=args.autojudge_train_split,
                recall_target=args.autojudge_recall_target,
                c_grid=parse_c_grid(args.autojudge_c_grid),
                classifier=requested_autojudge_classifier,
                seed=args.seed,
                use_dist_features=(requested_autojudge_features == "dist"),
            )

            training_prompts = [encode_fn(sample.question) for sample in gsm_samples]
            mining_cache_path = (
                str(checkpoint_path) + ".mining_cache.pt" if checkpoint_path is not None else None
            )
            autojudge_model, autojudge_train_samples, autojudge_val_auc = (
                build_autojudge_classifier(
                    target_model=target_model,
                    draft_model=draft_model,
                    prompts=training_prompts,
                    cfg=train_cfg,
                    eos_id=target_model.eos_token_id,
                    device="cpu",
                    mining_cache_path=mining_cache_path,
                )
            )
            autojudge_threshold_calibrated = float(autojudge_model.threshold)
            autojudge_val_recall = float(getattr(autojudge_model, "val_recall", 0.0))
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "autojudge_version": 2,
                        "classifier": autojudge_model,
                        "train_samples": autojudge_train_samples,
                        "val_auc": autojudge_val_auc,
                        "val_recall": autojudge_val_recall,
                        "threshold_selected": autojudge_threshold_calibrated,
                        "classifier_backend": requested_autojudge_classifier,
                        "autojudge_features": requested_autojudge_features,
                        "classifier_model_label": getattr(
                            autojudge_model, "model_label", requested_autojudge_classifier
                        ),
                        "created_at": datetime.now().isoformat(),
                    },
                    checkpoint_path,
                )

    resolved_autojudge_threshold_used = (
        float(args.autojudge_threshold)
        if args.autojudge_threshold is not None
        else float(autojudge_threshold_calibrated)
    )
    resolved_autojudge_threshold_calibrated = float(autojudge_threshold_calibrated)

    consensus_gate_model = None
    consensus_train_samples = 0
    consensus_val_accuracy = 0.0
    consensus_val_macro_f1 = 0.0
    resolved_consensus_fallback_threshold_used = float(args.consensus_fallback_threshold)
    if "consensus_autojudge" in methods:
        if (
            ConsensusGateClassifier is None
            or ConsensusAutoJudgeTrainConfig is None
            or build_consensus_gate_classifier is None
            or consensus_autojudge_sample_hf is None
            or torch is None
        ):
            raise SystemExit(
                "Consensus AutoJudge dependencies are missing (torch/transformers/scikit-learn)."
            )
        if not (0.0 < float(args.consensus_train_split) < 1.0):
            raise SystemExit("--consensus-train-split must be in (0,1).")
        if int(args.consensus_top_m) <= 0:
            raise SystemExit("--consensus-top-m must be positive.")
        if not (0.0 <= float(args.consensus_fallback_threshold) <= 1.0):
            raise SystemExit("--consensus-fallback-threshold must be in [0,1].")

        requested_consensus_gate = str(getattr(args, "consensus_gate", "learned") or "learned").strip().lower()
        if requested_consensus_gate not in {"learned", "rule"}:
            raise SystemExit("--consensus-gate must be one of: learned, rule.")
        args.consensus_gate = requested_consensus_gate

        requested_consensus_features = str(
            getattr(args, "consensus_features", "ensemble") or "ensemble"
        ).strip().lower()
        if requested_consensus_features not in {"ensemble", "d1_only"}:
            raise SystemExit("--consensus-features must be one of: ensemble, d1_only.")
        args.consensus_features = requested_consensus_features

        if requested_consensus_gate == "learned":
            should_train = True
            checkpoint_path = Path(args.consensus_checkpoint) if args.consensus_checkpoint else None
            if checkpoint_path is not None and checkpoint_path.exists():
                payload = _torch_load_checkpoint(checkpoint_path)
                classifier = payload.get("classifier") if isinstance(payload, dict) else None
                version = payload.get("consensus_autojudge_version") if isinstance(payload, dict) else None
                payload_features = None if not isinstance(payload, dict) else payload.get("consensus_features")
                payload_top_m = None if not isinstance(payload, dict) else payload.get("consensus_top_m")
                if (
                    version == 1
                    and classifier is not None
                    and hasattr(classifier, "predict_action_probs")
                    and str(payload_features) == requested_consensus_features
                    and int(payload_top_m) == int(args.consensus_top_m)
                ):
                    consensus_gate_model = classifier
                    consensus_train_samples = int(payload.get("train_samples", 0))
                    consensus_val_accuracy = float(payload.get("val_accuracy", 0.0))
                    consensus_val_macro_f1 = float(payload.get("val_macro_f1", 0.0))
                    should_train = False
                else:
                    print(
                        f"[WARN] legacy/unsupported checkpoint at {checkpoint_path}; retraining consensus gate."
                    )

            if should_train:
                train_dataset = args.consensus_train_dataset or args.dataset
                if not train_dataset:
                    raise SystemExit(
                        "Consensus AutoJudge training requires a GSM8K train dataset when checkpoint is absent."
                    )
                train_dataset_path = Path(train_dataset)
                if not train_dataset_path.exists():
                    raise SystemExit(
                        f"Consensus AutoJudge train dataset path not found: {train_dataset_path}."
                    )
                try:
                    gsm_train_samples = load_gsm8k(
                        path=str(train_dataset_path),
                        max_samples=args.consensus_train_samples,
                    )
                except Exception as exc:
                    raise SystemExit(
                        f"Failed to load consensus train dataset from {train_dataset_path}: {exc}"
                    ) from exc
                if not gsm_train_samples:
                    raise SystemExit(
                        "Consensus AutoJudge training dataset is not GSM8K-compatible. "
                        "Expected records with 'question' and 'answer' fields."
                    )

                train_cfg = ConsensusAutoJudgeTrainConfig(
                    task="gsm8k",
                    max_train_samples=args.consensus_train_samples,
                    max_new_tokens=args.max_new_tokens,
                    k=args.k,
                    train_split=args.consensus_train_split,
                    seed=args.seed,
                    top_m=args.consensus_top_m,
                    feature_mode=requested_consensus_features,
                    fallback_threshold=float(args.consensus_fallback_threshold),
                )
                training_prompts = [encode_fn(sample.question) for sample in gsm_train_samples]
                (
                    consensus_gate_model,
                    consensus_train_samples,
                    consensus_val_accuracy,
                    consensus_val_macro_f1,
                ) = build_consensus_gate_classifier(
                    target_model=target_model,
                    draft_model=draft_model,
                    draft2_model=draft2_model,
                    prompts=training_prompts,
                    cfg=train_cfg,
                    eos_id=target_model.eos_token_id,
                    device="cpu",
                )
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "consensus_autojudge_version": 1,
                            "classifier": consensus_gate_model,
                            "train_samples": consensus_train_samples,
                            "val_accuracy": consensus_val_accuracy,
                            "val_macro_f1": consensus_val_macro_f1,
                            "fallback_threshold": resolved_consensus_fallback_threshold_used,
                            "consensus_features": requested_consensus_features,
                            "consensus_top_m": int(args.consensus_top_m),
                            "created_at": datetime.now().isoformat(),
                        },
                        checkpoint_path,
                    )
        else:
            resolved_consensus_fallback_threshold_used = float(args.consensus_fallback_threshold)

    system_metadata = _collect_system_metadata(require_headless=args.require_headless)
    out_path = _resolve_out_path(args.out)
    completed_run_keys = _load_completed_run_keys(out_path)
    if completed_run_keys:
        print(
            f"[INFO] Resume mode: found {len(completed_run_keys)} completed run(s) in {out_path}."
        )

    backend = "hf" if args.hf_model else "toy"

    for method in methods:
        base_fields = _base_record_fields(
            method=method,
            backend=backend,
            resolved_target_model=resolved_target_model,
            resolved_draft_model=resolved_draft_model,
            resolved_draft2_model=resolved_draft2_model,
            resolved_target_tokenizer=resolved_target_tokenizer,
            resolved_draft_tokenizer=resolved_draft_tokenizer,
            resolved_draft2_tokenizer=resolved_draft2_tokenizer,
            args=args,
            resolved_draft_device=resolved_draft_device,
            resolved_draft_dtype=resolved_draft_dtype,
            resolved_draft_quant=resolved_draft_quant,
            resolved_draft_bnb_compute_dtype=resolved_draft_bnb_compute_dtype,
            resolved_draft2_device=resolved_draft2_device,
            resolved_draft2_dtype=resolved_draft2_dtype,
            resolved_draft2_quant=resolved_draft2_quant,
            resolved_draft2_bnb_compute_dtype=resolved_draft2_bnb_compute_dtype,
            resolved_autojudge_threshold_used=resolved_autojudge_threshold_used,
            resolved_autojudge_threshold_calibrated=resolved_autojudge_threshold_calibrated,
            autojudge_train_samples=autojudge_train_samples,
            autojudge_val_auc=autojudge_val_auc,
            autojudge_val_recall=autojudge_val_recall,
            resolved_consensus_fallback_threshold_used=resolved_consensus_fallback_threshold_used,
            consensus_train_samples=consensus_train_samples,
            consensus_val_accuracy=consensus_val_accuracy,
            consensus_val_macro_f1=consensus_val_macro_f1,
        )

        results = []
        acceptance_rates = []
        avg_tokens_per_step = []
        judge_accept_rates = []
        gate_accept_d1_rates = []
        gate_escalate_rates = []
        gate_fallback_rates = []
        topk_accept_rates = []
        fallback_rates = []
        cache_hit_rates = []
        gsm8k_exact_match_rates = []
        gsm8k_correct_counts = []
        gsm8k_total_counts = []
        successful_runs = 0
        failed_runs = 0
        skipped_runs = 0

        for run in range(args.runs):
            run_number = run + 1
            identity = _resume_identity(run=run_number, base_fields=base_fields)
            resume_key = _make_resume_key(identity)
            if resume_key in completed_run_keys:
                skipped_runs += 1
                print(f"{method} run {run_number}: skipped (already completed).")
                continue

            try:
                (
                    tps,
                    duration,
                    total_tokens,
                    stats,
                    prompt_tokens,
                    gsm8k_correct,
                    gsm8k_total,
                ) = _run_once(
                    prompts=prompts,
                    encode_fn=encode_fn,
                    decode_fn=decode_fn,
                    target_model=target_model,
                    draft_model=draft_model,
                    draft2_model=draft2_model,
                    method=method,
                    eval_task=args.eval_task,
                    max_new_tokens=args.max_new_tokens,
                    k=args.k,
                    seed=args.seed + run,
                    autojudge_model=autojudge_model,
                    autojudge_threshold=resolved_autojudge_threshold_used,
                    consensus_gate_model=consensus_gate_model,
                    consensus_gate_mode=args.consensus_gate,
                    consensus_fallback_threshold=resolved_consensus_fallback_threshold_used,
                    consensus_top_m=args.consensus_top_m,
                    consensus_feature_mode=args.consensus_features,
                    consensus_disable_escalation=args.consensus_disable_escalation,
                    topk_rank=resolved_topk_rank,
                    specexec_parallel_branches=args.parallel_branches,
                    specexec_branch_prune_threshold=args.branch_prune_threshold,
                )
            except Exception as exc:
                failed_runs += 1
                error_record = {
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "summary": False,
                    "run": run_number,
                    "resume_key": resume_key,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                error_record.update(base_fields)
                error_record.update(system_metadata)
                _append_result(out_path, error_record)
                print(f"[ERROR] {method} run {run_number} failed: {exc}")
                continue

            successful_runs += 1

            if hasattr(stats, "train_samples"):
                if method == "consensus_autojudge":
                    stats.train_samples = consensus_train_samples
                else:
                    stats.train_samples = autojudge_train_samples
            if hasattr(stats, "train_loss"):
                stats.train_loss = 0.0
            if hasattr(stats, "val_auc"):
                stats.val_auc = autojudge_val_auc
            if hasattr(stats, "val_recall"):
                stats.val_recall = autojudge_val_recall
            if hasattr(stats, "val_accuracy"):
                stats.val_accuracy = consensus_val_accuracy
            if hasattr(stats, "val_macro_f1"):
                stats.val_macro_f1 = consensus_val_macro_f1
            if hasattr(stats, "threshold_selected"):
                stats.threshold_selected = resolved_autojudge_threshold_calibrated
            if hasattr(stats, "fallback_threshold_selected"):
                stats.fallback_threshold_selected = resolved_consensus_fallback_threshold_used
            results.append(tps)
            acceptance_rates.append(stats.acceptance_rate)
            avg_tokens_per_step.append(stats.avg_tokens_per_step)
            if hasattr(stats, "judge_accept_rate"):
                judge_accept_rates.append(stats.judge_accept_rate)
            if hasattr(stats, "gate_accept_d1_rate"):
                gate_accept_d1_rates.append(stats.gate_accept_d1_rate)
            if hasattr(stats, "gate_escalate_rate"):
                gate_escalate_rates.append(stats.gate_escalate_rate)
            if hasattr(stats, "gate_fallback_rate"):
                gate_fallback_rates.append(stats.gate_fallback_rate)
            if hasattr(stats, "topk_accept_rate"):
                topk_accept_rates.append(stats.topk_accept_rate)
            if hasattr(stats, "target_fallback_rate"):
                fallback_rates.append(stats.target_fallback_rate)
            if hasattr(stats, "cache_hit_rate"):
                cache_hit_rates.append(stats.cache_hit_rate)
            gsm8k_exact_match = (
                float(gsm8k_correct) / float(gsm8k_total) if gsm8k_total > 0 else 0.0
            )
            if args.eval_task == "gsm8k":
                gsm8k_exact_match_rates.append(gsm8k_exact_match)
                gsm8k_correct_counts.append(gsm8k_correct)
                gsm8k_total_counts.append(gsm8k_total)

            record = {
                "timestamp": datetime.now().isoformat(),
                "status": "ok",
                "summary": False,
                "run": run_number,
                "resume_key": resume_key,
                "total_prompt_tokens": prompt_tokens,
                "total_generated_tokens": total_tokens,
                "duration_sec": duration,
                "tokens_per_sec": tps,
            }
            if args.eval_task == "gsm8k":
                record["gsm8k_exact_match"] = gsm8k_exact_match
                record["gsm8k_correct"] = gsm8k_correct
                record["gsm8k_total"] = gsm8k_total
            record.update(base_fields)
            record.update(system_metadata)
            record.update(_stats_record_fields(stats))
            _append_result(out_path, record)
            completed_run_keys.add(resume_key)

            msg = (
                f"{method} run {run_number}: {tps:.2f} tok/s "
                f"({total_tokens} tokens in {duration:.3f}s), "
                f"accept={stats.acceptance_rate:.3f}, "
                f"avg_tokens/step={stats.avg_tokens_per_step:.3f}"
            )
            if hasattr(stats, "judge_accept_rate"):
                msg += (
                    f", judge_accept={stats.judge_accept_rate:.3f}, "
                    f"fallback={stats.target_fallback_rate:.3f}"
                )
            if hasattr(stats, "gate_accept_d1_rate"):
                msg += (
                    f", gate_accept_d1={stats.gate_accept_d1_rate:.3f}, "
                    f"gate_escalate={stats.gate_escalate_rate:.3f}, "
                    f"gate_fallback={stats.gate_fallback_rate:.3f}"
                )
            if hasattr(stats, "cache_hit_rate"):
                msg += (
                    f", cache_hit={stats.cache_hit_rate:.3f}, "
                    f"target_calls/token={stats.target_calls_per_token:.3f}"
                )
            if hasattr(stats, "topk_accept_rate"):
                msg += f", topk_accept={stats.topk_accept_rate:.3f}"
            if args.eval_task == "gsm8k":
                msg += f", gsm8k_em={gsm8k_exact_match:.3f}"
            print(msg)

        summary_record = {
            "timestamp": datetime.now().isoformat(),
            "summary": True,
            "runs": args.runs,
            "runs_successful": successful_runs,
            "runs_failed": failed_runs,
            "runs_skipped": skipped_runs,
            "status": "ok" if successful_runs > 0 else ("error" if failed_runs > 0 else "skipped"),
        }
        summary_record.update(base_fields)
        summary_record.update(system_metadata)

        if successful_runs > 0:
            median_tps = statistics.median(results)
            median_accept = statistics.median(acceptance_rates)
            median_tps_step = statistics.median(avg_tokens_per_step)
            median_judge_accept = (
                statistics.median(judge_accept_rates) if judge_accept_rates else None
            )
            median_gate_accept_d1 = (
                statistics.median(gate_accept_d1_rates) if gate_accept_d1_rates else None
            )
            median_gate_escalate = (
                statistics.median(gate_escalate_rates) if gate_escalate_rates else None
            )
            median_gate_fallback = (
                statistics.median(gate_fallback_rates) if gate_fallback_rates else None
            )
            median_topk_accept = (
                statistics.median(topk_accept_rates) if topk_accept_rates else None
            )
            median_fallback = statistics.median(fallback_rates) if fallback_rates else None
            median_cache_hit = statistics.median(cache_hit_rates) if cache_hit_rates else None
            summary_record["tokens_per_sec_median"] = median_tps
            summary_record["acceptance_rate_median"] = median_accept
            summary_record["avg_tokens_per_step_median"] = median_tps_step
            if median_judge_accept is not None:
                summary_record["judge_accept_rate_median"] = median_judge_accept
            if median_gate_accept_d1 is not None:
                summary_record["gate_accept_d1_rate_median"] = median_gate_accept_d1
            if median_gate_escalate is not None:
                summary_record["gate_escalate_rate_median"] = median_gate_escalate
            if median_gate_fallback is not None:
                summary_record["gate_fallback_rate_median"] = median_gate_fallback
            if median_topk_accept is not None:
                summary_record["topk_accept_rate_median"] = median_topk_accept
            if median_fallback is not None:
                summary_record["target_fallback_rate_median"] = median_fallback
            if median_cache_hit is not None:
                summary_record["cache_hit_rate_median"] = median_cache_hit
            if args.eval_task == "gsm8k":
                gsm8k_correct_total = int(sum(gsm8k_correct_counts))
                gsm8k_total_total = int(sum(gsm8k_total_counts))
                summary_record["gsm8k_correct"] = gsm8k_correct_total
                summary_record["gsm8k_total"] = gsm8k_total_total
                summary_record["gsm8k_exact_match"] = (
                    float(gsm8k_correct_total) / float(gsm8k_total_total)
                    if gsm8k_total_total > 0
                    else 0.0
                )
            _append_result(out_path, summary_record)

            msg = (
                f"{method} median: {median_tps:.2f} tok/s, "
                f"accept={median_accept:.3f}, "
                f"avg_tokens/step={median_tps_step:.3f}"
            )
            if median_judge_accept is not None and median_fallback is not None:
                msg += (
                    f", judge_accept={median_judge_accept:.3f}, "
                    f"fallback={median_fallback:.3f}"
                )
            if (
                median_gate_accept_d1 is not None
                and median_gate_escalate is not None
                and median_gate_fallback is not None
            ):
                msg += (
                    f", gate_accept_d1={median_gate_accept_d1:.3f}, "
                    f"gate_escalate={median_gate_escalate:.3f}, "
                    f"gate_fallback={median_gate_fallback:.3f}"
                )
            if median_topk_accept is not None:
                msg += f", topk_accept={median_topk_accept:.3f}"
            if median_cache_hit is not None:
                msg += f", cache_hit={median_cache_hit:.3f}"
            if args.eval_task == "gsm8k":
                msg += f", gsm8k_em={summary_record['gsm8k_exact_match']:.3f}"
            print(msg)
        elif failed_runs > 0:
            summary_record["error_message"] = "No successful runs for this method."
            _append_result(out_path, summary_record)
            print(f"[ERROR] {method}: no successful runs. See JSONL error records.")
        else:
            summary_record["info_message"] = "All runs were skipped by resume mode."
            _append_result(out_path, summary_record)
            print(f"{method}: all runs skipped by resume mode.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_with_args(args)


if __name__ == "__main__":
    main()
