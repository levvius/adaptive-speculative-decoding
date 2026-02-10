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
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

try:
    import torch
except ModuleNotFoundError:
    torch = None

from sp_samp.models import NoisyModel, RandomModel
from sp_samp.mtbench import load_mtbench
from sp_samp.sampling import SamplingStats, sample_baseline, speculative_sample
from sp_samp.specexec import SpecExecStats, specexec_sample

HFModel = None
AutoJudgeTrainConfig = None
JudgeMLP = None
build_autojudge_classifier = None
AutoJudgeStats = None
autojudge_sample_hf = None
sample_baseline_hf = None
speculative_sample_hf = None
specexec_sample_hf = None

if torch is not None:
    from sp_samp.autojudge import (
        AutoJudgeStats,
        AutoJudgeTrainConfig,
        JudgeMLP,
        autojudge_sample_hf,
        build_autojudge_classifier,
    )
    from sp_samp.hf_adapter import HFModel
    from sp_samp.hf_sampling import sample_baseline_hf, speculative_sample_hf
    from sp_samp.hf_specexec import specexec_sample_hf


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


def _accumulate_stats(total, stats) -> None:
    base_fields = ["proposed", "accepted", "steps", "target_tokens", "rejections"]
    for field in base_fields:
        if hasattr(total, field) and hasattr(stats, field):
            setattr(total, field, getattr(total, field) + getattr(stats, field))
    extra_fields = [
        "judge_total",
        "judge_accepted",
        "judge_rejected",
        "target_calls",
        "target_fallbacks",
        "draft_calls",
        "draft_prefills",
        "target_prefills",
        "branches_total",
        "branches_kept",
        "branches_pruned",
        "max_active_branches",
        "cache_hits",
        "cache_misses",
        "audit_samples",
        "audit_expected_accept",
    ]
    for field in extra_fields:
        if hasattr(total, field) and hasattr(stats, field):
            setattr(total, field, getattr(total, field) + getattr(stats, field))


def _run_once(
    prompts: Iterable[str],
    encode_fn: Callable[[str], List[int]],
    target_model,
    draft_model,
    method: str,
    max_new_tokens: int,
    k: int,
    seed: int,
    autojudge_model=None,
    autojudge_threshold: float = 0.5,
    autojudge_audit_ratio: float = 0.0,
    specexec_parallel_branches: int = 8,
    specexec_branch_prune_threshold: float = 0.0,
) -> Tuple[float, float, int, object, int]:
    rng = random.Random(seed)
    if torch is not None:
        torch.manual_seed(seed)
    total_tokens = 0
    total_prompt_tokens = 0
    if method == "autojudge":
        if AutoJudgeStats is None:
            raise RuntimeError("AutoJudge requires torch dependencies.")
        total_stats = AutoJudgeStats()
    elif method == "specexec":
        total_stats = SpecExecStats()
    else:
        total_stats = SamplingStats()
    start = time.perf_counter()
    for prompt_idx, prompt in enumerate(prompts):
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
                        audit_ratio=autojudge_audit_ratio,
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
        _accumulate_stats(total_stats, stats)
    duration = time.perf_counter() - start
    tokens_per_sec = total_tokens / duration if duration > 0 else 0.0
    return tokens_per_sec, duration, total_tokens, total_stats, total_prompt_tokens


def _resolve_out_path(path: Optional[str]) -> Path:
    if path is None:
        return Path("benchmarks") / "results.jsonl"
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
    resolved_target_tokenizer: str,
    resolved_draft_tokenizer: str,
    args: argparse.Namespace,
    resolved_draft_device: str,
    resolved_draft_dtype: str,
    resolved_draft_quant: Optional[str],
    resolved_draft_bnb_compute_dtype: Optional[str],
    autojudge_train_samples: int,
    autojudge_train_loss: float,
) -> Dict[str, object]:
    return {
        "method": method,
        "backend": backend,
        "target_model": resolved_target_model,
        "draft_model": resolved_draft_model,
        "tokenizer": resolved_target_tokenizer,
        "draft_tokenizer": resolved_draft_tokenizer,
        "device": args.device,
        "dtype": args.dtype,
        "quant": args.quant,
        "bnb_compute_dtype": args.bnb_compute_dtype,
        "draft_device": resolved_draft_device,
        "draft_dtype": resolved_draft_dtype,
        "draft_quant": resolved_draft_quant,
        "draft_bnb_compute_dtype": resolved_draft_bnb_compute_dtype,
        "use_chat_template": args.use_chat_template,
        "system_prompt": args.system_prompt,
        "k": args.k,
        "max_new_tokens": args.max_new_tokens,
        "max_samples": args.max_samples,
        "turn_index": args.turn_index,
        "dataset": args.dataset,
        "autojudge_threshold": args.autojudge_threshold,
        "autojudge_train_samples": autojudge_train_samples,
        "autojudge_train_loss": autojudge_train_loss,
        "autojudge_checkpoint": args.autojudge_checkpoint,
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
        "tokenizer",
        "draft_tokenizer",
        "device",
        "dtype",
        "quant",
        "bnb_compute_dtype",
        "draft_device",
        "draft_dtype",
        "draft_quant",
        "draft_bnb_compute_dtype",
        "use_chat_template",
        "system_prompt",
        "k",
        "max_new_tokens",
        "max_samples",
        "turn_index",
        "dataset",
        "autojudge_threshold",
        "autojudge_train_samples",
        "autojudge_train_loss",
        "autojudge_checkpoint",
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
        return ["baseline", "speculative", "autojudge", "specexec"]
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
    if hasattr(stats, "target_fallback_rate"):
        record["target_fallback_rate"] = stats.target_fallback_rate
    if hasattr(stats, "target_calls_per_token"):
        record["target_calls_per_token"] = stats.target_calls_per_token
    if hasattr(stats, "draft_calls_per_token"):
        record["draft_calls_per_token"] = stats.draft_calls_per_token
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
        "target_calls",
        "target_fallbacks",
        "draft_calls",
        "draft_prefills",
        "target_prefills",
        "branches_total",
        "branches_kept",
        "branches_pruned",
        "max_active_branches",
        "cache_hits",
        "cache_misses",
        "train_samples",
        "train_loss",
        "audit_samples",
        "audit_expected_accept_mean",
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
        choices=["baseline", "speculative", "autojudge", "specexec", "both", "all"],
    )
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--draft-noise", type=float, default=0.2, help="Noise for draft model.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hf-model", type=str, default=None, help="HF model name/path for target.")
    parser.add_argument("--hf-draft-model", type=str, default=None, help="HF model name/path for draft.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name/path.")
    parser.add_argument("--draft-tokenizer", type=str, default=None, help="Tokenizer name/path for draft model.")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu/cuda/mps.")
    parser.add_argument("--draft-device", type=str, default=None, help="Device override for draft model.")
    parser.add_argument("--dtype", type=str, default="auto", help="Torch dtype: auto/float16/bfloat16/float32.")
    parser.add_argument("--draft-dtype", type=str, default=None, help="Torch dtype override for draft model.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code for HF models.")
    parser.add_argument("--add-special-tokens", action="store_true", help="Use tokenizer special tokens for prompts.")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply tokenizer chat template to prompts.")
    parser.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt for chat template.")
    parser.add_argument("--quant", type=str, default=None, choices=["8bit", "4bit"], help="Quantization for HF models.")
    parser.add_argument("--draft-quant", type=str, default=None, choices=["8bit", "4bit"], help="Quantization override for draft model.")
    parser.add_argument("--bnb-compute-dtype", type=str, default="bfloat16", help="Compute dtype for 4-bit quantization.")
    parser.add_argument("--draft-bnb-compute-dtype", type=str, default=None, help="Compute dtype override for draft 4-bit quantization.")
    parser.add_argument("--autojudge-threshold", type=float, default=0.5, help="Decision threshold for judge acceptance.")
    parser.add_argument("--autojudge-train-samples", type=int, default=4000, help="Number of synthetic judge training samples.")
    parser.add_argument("--autojudge-train-steps", type=int, default=400, help="Judge optimizer steps.")
    parser.add_argument("--autojudge-train-batch-size", type=int, default=128, help="Judge training batch size.")
    parser.add_argument("--autojudge-train-lr", type=float, default=1e-3, help="Judge training learning rate.")
    parser.add_argument("--autojudge-audit-ratio", type=float, default=0.0, help="Fraction of accepted tokens to audit with target model.")
    parser.add_argument("--autojudge-checkpoint", type=str, default=None, help="Path to save/load judge checkpoint (.pt).")
    parser.add_argument("--parallel-branches", type=int, default=8, help="Number of draft branches for SpecExec.")
    parser.add_argument("--branch-prune-threshold", type=float, default=0.0, help="Draft-probability pruning threshold in [0,1] for SpecExec.")
    parser.add_argument("--require-headless", action="store_true", help="Fail fast if GPU is display-active (recommended for long runs).")
    parser.add_argument("--out", type=str, default=None, help="Path to JSONL metrics file.")
    return parser


def run_with_args(args: argparse.Namespace) -> None:
    methods = _resolve_methods(args.method)
    needs_draft = any(m in {"speculative", "autojudge", "specexec"} for m in methods)
    if "autojudge" in methods and not args.hf_model:
        if args.method == "all":
            raise SystemExit(
                "Method 'all' includes AutoJudge, which requires HF models. "
                "Set --hf-model/--hf-draft-model or choose a method without AutoJudge."
            )
        raise SystemExit("AutoJudge requires HF models. Set --hf-model and --hf-draft-model.")

    if args.dataset:
        prompts = load_mtbench(
            args.dataset, turn_index=args.turn_index, max_samples=args.max_samples
        )
        if not prompts:
            raise SystemExit("No prompts loaded from dataset.")
    else:
        prompts = _default_prompts()[: args.max_samples]

    resolved_target_tokenizer = args.tokenizer
    resolved_draft_tokenizer = args.draft_tokenizer
    resolved_target_model = args.hf_model or "toy_random"
    resolved_draft_model = args.hf_draft_model or args.hf_model or "toy_noisy"
    resolved_draft_device = args.draft_device or args.device
    resolved_draft_dtype = args.draft_dtype or args.dtype
    resolved_draft_quant = args.draft_quant if args.draft_quant is not None else args.quant
    resolved_draft_bnb_compute_dtype = (
        args.draft_bnb_compute_dtype
        if args.draft_bnb_compute_dtype is not None
        else args.bnb_compute_dtype
    )

    if args.hf_model:
        if torch is None or HFModel is None:
            raise SystemExit(
                "HF benchmark requested but torch/transformers dependencies are missing."
            )
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
            if target_model.vocab_size != draft_model.vocab_size:
                raise SystemExit("Target and draft vocab sizes differ.")
            if not _tokenizers_compatible(target_model.tokenizer, draft_model.tokenizer):
                raise SystemExit(
                    "Target and draft tokenizers are incompatible. "
                    "Use models sharing identical token-id mapping."
                )
        else:
            draft_model = target_model
            resolved_draft_model = args.hf_model
            resolved_draft_tokenizer = target_tokenizer_name

        hf_tokenizer = target_model.tokenizer

        def encode_fn(text: str) -> List[int]:
            if args.use_chat_template:
                messages = []
                if args.system_prompt:
                    messages.append({"role": "system", "content": args.system_prompt})
                messages.append({"role": "user", "content": text})
                return hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            return hf_tokenizer.encode(text, add_special_tokens=args.add_special_tokens)

    else:
        tokenizer = HashTokenizer(args.vocab_size)

        def encode_fn(text: str) -> List[int]:
            return tokenizer.encode(text)

        target_model = RandomModel(args.vocab_size, seed=args.seed)
        draft_model = (
            NoisyModel(target_model, noise=args.draft_noise) if needs_draft else target_model
        )
        resolved_draft_model = "toy_noisy" if needs_draft else "toy_random"
        resolved_target_tokenizer = f"hash:{args.vocab_size}"
        resolved_draft_tokenizer = f"hash:{args.vocab_size}"

    if "autojudge" in methods and (HFModel is None or not isinstance(target_model, HFModel)):
        raise SystemExit("AutoJudge requires HF models. Set --hf-model and --hf-draft-model.")

    autojudge_model = None
    autojudge_train_samples = 0
    autojudge_train_loss = 0.0
    if "autojudge" in methods:
        if (
            AutoJudgeTrainConfig is None
            or JudgeMLP is None
            or build_autojudge_classifier is None
            or torch is None
        ):
            raise SystemExit("AutoJudge dependencies are missing (torch/transformers).")
        judge_device = draft_model.device if isinstance(draft_model, HFModel) else "cpu"
        checkpoint_path = Path(args.autojudge_checkpoint) if args.autojudge_checkpoint else None
        if checkpoint_path is not None and checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location="cpu")
            in_features = int(payload.get("in_features", 7))
            autojudge_model = JudgeMLP(in_features=in_features)
            autojudge_model.load_state_dict(payload["state_dict"])
            autojudge_model.to(judge_device)
            autojudge_model.eval()
            autojudge_train_samples = int(payload.get("train_samples", 0))
            autojudge_train_loss = float(payload.get("train_loss", 0.0))
        else:
            train_cfg = AutoJudgeTrainConfig(
                max_train_samples=args.autojudge_train_samples,
                max_new_tokens=args.max_new_tokens,
                k=args.k,
                train_steps=args.autojudge_train_steps,
                batch_size=args.autojudge_train_batch_size,
                lr=args.autojudge_train_lr,
                seed=args.seed,
            )
            training_prompts = [encode_fn(p) for p in prompts]
            autojudge_model, autojudge_train_samples, autojudge_train_loss = (
                build_autojudge_classifier(
                    target_model=target_model,
                    draft_model=draft_model,
                    prompts=training_prompts,
                    cfg=train_cfg,
                    eos_id=target_model.eos_token_id,
                    device=judge_device,
                )
            )
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": autojudge_model.state_dict(),
                        "in_features": 7,
                        "train_samples": autojudge_train_samples,
                        "train_loss": autojudge_train_loss,
                        "created_at": datetime.now().isoformat(),
                    },
                    checkpoint_path,
                )

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
            resolved_target_tokenizer=resolved_target_tokenizer,
            resolved_draft_tokenizer=resolved_draft_tokenizer,
            args=args,
            resolved_draft_device=resolved_draft_device,
            resolved_draft_dtype=resolved_draft_dtype,
            resolved_draft_quant=resolved_draft_quant,
            resolved_draft_bnb_compute_dtype=resolved_draft_bnb_compute_dtype,
            autojudge_train_samples=autojudge_train_samples,
            autojudge_train_loss=autojudge_train_loss,
        )

        results = []
        acceptance_rates = []
        avg_tokens_per_step = []
        judge_accept_rates = []
        fallback_rates = []
        cache_hit_rates = []
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
                tps, duration, total_tokens, stats, prompt_tokens = _run_once(
                    prompts=prompts,
                    encode_fn=encode_fn,
                    target_model=target_model,
                    draft_model=draft_model,
                    method=method,
                    max_new_tokens=args.max_new_tokens,
                    k=args.k,
                    seed=args.seed + run,
                    autojudge_model=autojudge_model,
                    autojudge_threshold=args.autojudge_threshold,
                    autojudge_audit_ratio=args.autojudge_audit_ratio,
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
                stats.train_samples = autojudge_train_samples
            if hasattr(stats, "train_loss"):
                stats.train_loss = autojudge_train_loss
            results.append(tps)
            acceptance_rates.append(stats.acceptance_rate)
            avg_tokens_per_step.append(stats.avg_tokens_per_step)
            if hasattr(stats, "judge_accept_rate"):
                judge_accept_rates.append(stats.judge_accept_rate)
            if hasattr(stats, "target_fallback_rate"):
                fallback_rates.append(stats.target_fallback_rate)
            if hasattr(stats, "cache_hit_rate"):
                cache_hit_rates.append(stats.cache_hit_rate)

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
            if hasattr(stats, "cache_hit_rate"):
                msg += (
                    f", cache_hit={stats.cache_hit_rate:.3f}, "
                    f"target_calls/token={stats.target_calls_per_token:.3f}"
                )
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
            median_fallback = statistics.median(fallback_rates) if fallback_rates else None
            median_cache_hit = statistics.median(cache_hit_rates) if cache_hit_rates else None
            summary_record["tokens_per_sec_median"] = median_tps
            summary_record["acceptance_rate_median"] = median_accept
            summary_record["avg_tokens_per_step_median"] = median_tps_step
            if median_judge_accept is not None:
                summary_record["judge_accept_rate_median"] = median_judge_accept
            if median_fallback is not None:
                summary_record["target_fallback_rate_median"] = median_fallback
            if median_cache_hit is not None:
                summary_record["cache_hit_rate_median"] = median_cache_hit
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
            if median_cache_hit is not None:
                msg += f", cache_hit={median_cache_hit:.3f}"
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
