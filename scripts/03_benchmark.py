from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import os
import random
import socket
import subprocess
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from hydra.utils import get_original_cwd
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from jointadaspec.baselines import (
    CascadePolicy,
    FixedSDDecoder,
    FuzzySDDecoder,
    SpecDecPPDecoder,
    VanillaARDecoder,
)
from jointadaspec.inference import JointAdaSpecDecoder, JointAdaSpecPolicy
from jointadaspec.utils.manifest import write_manifest
from jointadaspec.utils import ExperimentLogger, load_model_pair
from sp_samp.gsm8k import answers_equivalent, extract_final_answer, extract_reference_answer, load_gsm8k
from sp_samp.livecodebench import load_livecodebench
from sp_samp.mtbench import load_mtbench

try:
    import transformers
except Exception:  # pragma: no cover - optional in CPU-only contexts
    transformers = None


RESULTS_FILENAME = "results.jsonl"


@dataclass(frozen=True)
class EvalSample:
    prompt: str
    reference_answer: str | None = None


def _experiment_cfg(cfg: DictConfig) -> DictConfig:
    return cfg.experiments if "experiments" in cfg else cfg


def _make_output_dir(cfg: DictConfig) -> Path:
    if cfg.output_dir:
        return Path(get_original_cwd()) / str(cfg.output_dir)
    root = Path(get_original_cwd()) / str(cfg.output_root)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return root / str(cfg.experiment_name) / timestamp


def _config_model_name(model_cfg: DictConfig, *, tokenizer: bool = False) -> str | None:
    key = "tokenizer" if tokenizer else "hf_model"
    value = model_cfg.get(key)
    return None if value in {None, ""} else str(value)


def _seed_list(n_seeds: int) -> list[int]:
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive.")
    return [42 + offset for offset in range(n_seeds)]


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_eval_samples(dataset_cfg: DictConfig) -> list[EvalSample]:
    dataset_name = str(dataset_cfg.name).lower()
    max_samples = int(dataset_cfg.test_max_samples)
    if dataset_name == "gsm8k":
        samples = load_gsm8k(str(dataset_cfg.path), max_samples=max_samples)
        prompt_mode = str(dataset_cfg.get("eval_mode", "zero_shot_cot"))
        prompts = []
        for sample in samples:
            if prompt_mode == "plain":
                prompt = f"Question: {sample.question.strip()}\nAnswer:"
            else:
                prompt = (
                    "Solve the following grade school math problem. "
                    "Show your reasoning, then end with: The final answer is <number>.\n\n"
                    f"Question: {sample.question.strip()}\nAnswer:"
                )
            prompts.append(EvalSample(prompt=prompt, reference_answer=extract_reference_answer(sample.answer)))
        return prompts
    if dataset_name == "livecodebench":
        prompts = load_livecodebench(str(dataset_cfg.path), max_samples=max_samples)
        return [EvalSample(prompt=prompt) for prompt in prompts]
    if dataset_name == "mtbench":
        prompts = load_mtbench(
            str(dataset_cfg.path),
            turn_index=int(dataset_cfg.get("turn_index", 0)),
            max_samples=max_samples,
        )
        return [EvalSample(prompt=prompt) for prompt in prompts]
    raise ValueError(f"Unsupported dataset name '{dataset_name}'.")


def _encode_prompt(target_model, prompt: str) -> list[int]:
    tokenizer = getattr(target_model, "tokenizer", None)
    if tokenizer is not None:
        return [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)] or [0]
    return [ord(ch) % int(getattr(target_model, "vocab_size", 256)) for ch in prompt[:64]] or [0]


def _decode_generated(target_model, generated_ids: list[int]) -> str:
    tokenizer = getattr(target_model, "tokenizer", None)
    if tokenizer is not None:
        return str(tokenizer.decode(generated_ids, skip_special_tokens=True))
    return " ".join(str(token) for token in generated_ids)


def _legacy_run_path(output_dir: Path) -> Path:
    return output_dir / "run.jsonl"


def _results_path(output_dir: Path) -> Path:
    return output_dir / RESULTS_FILENAME


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _read_legacy_prompt_records(run_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not run_path.exists():
        return records
    for payload in _read_jsonl(run_path):
        if "status" in payload or "summary" in payload:
            continue
        if "decoder" not in payload or "prompt_idx" not in payload:
            continue
        records.append(payload)
    return records


def _load_existing_results(results_path: Path) -> tuple[list[dict[str, object]], set[str]]:
    records = _read_jsonl(results_path)
    completed = {
        str(record["resume_key"])
        for record in records
        if not bool(record.get("summary"))
        and record.get("status") == "ok"
        and isinstance(record.get("resume_key"), str)
    }
    return records, completed


def _bootstrap_ci(values: list[float], *, seed: int, n_resamples: int = 1000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        picks = rng.integers(0, array.size, size=array.size)
        samples[idx] = float(np.median(array[picks]))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _prompt_stats_from_result(result, *, decoder_name: str) -> dict[str, float]:
    proposed = 0
    accepted = 0
    rejected = 0
    if decoder_name == "target_only":
        steps = result.n_tokens_generated
    else:
        metrics = result.per_step_metrics or []
        proposed = sum(1 for metric in metrics if metric.get("action_length") == "continue")
        accepted = sum(1 for metric in metrics if metric.get("accepted") is True)
        if proposed and accepted == 0 and result.acceptance_rate > 0.0:
            accepted = int(round(result.acceptance_rate * proposed))
        rejected = max(proposed - accepted, 0)
        steps = max(len(metrics), result.n_tokens_generated)
    avg_tokens_per_step = 0.0 if steps <= 0 else float(result.n_tokens_generated) / float(steps)
    return {
        "proposed": float(proposed),
        "accepted": float(accepted),
        "steps": float(steps),
        "rejections": float(rejected),
        "avg_tokens_per_step": float(avg_tokens_per_step),
    }


def _method_aliases(name: str) -> set[str]:
    aliases = {name}
    if name == "target_only":
        aliases.add("vanilla_ar")
    elif name == "speculative":
        aliases.add("fixed_sd")
    elif name == "adaptive_length":
        aliases.add("specdecpp")
    elif name == "cascade_length_then_verif":
        aliases.add("cascade_len_then_verif")
    return aliases


def _is_enabled(active_names: set[str], method_name: str) -> bool:
    aliases = _method_aliases(method_name)
    if any(alias in active_names for alias in aliases):
        return True
    if method_name.startswith("fuzzy_sd_T") and ("fuzzy_sd" in active_names or method_name in active_names):
        return True
    return False


def _companion_policy_path(policy_path: Path, prefix: str) -> Path:
    stem = policy_path.stem
    suffix = stem[len("policy") :] if stem.startswith("policy") else ""
    return policy_path.with_name(f"{prefix}{suffix}.npz")


def _build_decoders(
    cfg: DictConfig,
    policy: JointAdaSpecPolicy | None,
    target_model,
    draft_model,
) -> dict[str, object]:
    decoders: dict[str, object] = {
        "target_only": VanillaARDecoder(target_model=target_model),
        "speculative": FixedSDDecoder(
            target_model=target_model,
            draft_model=draft_model,
            gamma=int(cfg.fixed_sd_gamma),
        ),
        "adaptive_length": SpecDecPPDecoder(
            target_model=target_model,
            draft_model=draft_model,
            gamma_max=int(cfg.fuzzy_sd_gamma),
            entropy_threshold=float(cfg.specdecpp_entropy_threshold),
        ),
    }
    for threshold in cfg.fuzzy_sd_T_grid:
        key = f"fuzzy_sd_T{float(threshold):g}"
        decoders[key] = FuzzySDDecoder(
            target_model=target_model,
            draft_model=draft_model,
            gamma=int(cfg.fuzzy_sd_gamma),
            threshold=float(threshold),
        )
    if policy is not None:
        decoders["jointadaspec"] = JointAdaSpecDecoder(
            target_model=target_model,
            draft_model=draft_model,
            policy=policy,
        )
    return decoders


def _build_record_base(exp_cfg: DictConfig, *, method: str) -> dict[str, object]:
    target_cfg = exp_cfg.model_pairs.target
    draft_cfg = exp_cfg.model_pairs.draft
    k_value = int(exp_cfg.get("fixed_sd_gamma", exp_cfg.get("gamma_max", 0)))
    return {
        "schema_version": 2,
        "method": method,
        "backend": "jointadaspec_hf",
        "target_model": _config_model_name(target_cfg),
        "draft_model": _config_model_name(draft_cfg),
        "tokenizer": _config_model_name(target_cfg, tokenizer=True),
        "draft_tokenizer": _config_model_name(draft_cfg, tokenizer=True),
        "device": str(target_cfg.get("device", "cpu")),
        "dtype": str(target_cfg.get("dtype", "auto")),
        "quant": target_cfg.get("quant"),
        "bnb_compute_dtype": str(target_cfg.get("bnb_compute_dtype", "bfloat16")),
        "draft_device": str(draft_cfg.get("device", target_cfg.get("device", "cpu"))),
        "draft_dtype": str(draft_cfg.get("dtype", target_cfg.get("dtype", "auto"))),
        "draft_quant": draft_cfg.get("quant"),
        "draft_bnb_compute_dtype": str(
            draft_cfg.get("bnb_compute_dtype", target_cfg.get("bnb_compute_dtype", "bfloat16"))
        ),
        "use_chat_template": bool(target_cfg.get("use_chat_template", False)),
        "system_prompt": None,
        "k": k_value,
        "max_new_tokens": int(exp_cfg.datasets.max_new_tokens),
        "max_samples": int(exp_cfg.datasets.test_max_samples),
        "turn_index": int(exp_cfg.datasets.get("turn_index", 0)),
        "dataset": str(exp_cfg.datasets.path),
        "autojudge_threshold": None,
        "autojudge_train_samples": 0,
        "autojudge_train_loss": 0.0,
        "autojudge_checkpoint": None,
        "parallel_branches": None,
        "branch_prune_threshold": 0.0,
        "eval_task": str(exp_cfg.datasets.name),
    }


def _record_resume_key(exp_cfg: DictConfig, *, method: str, seed: int) -> str:
    target_cfg = exp_cfg.model_pairs.target
    draft_cfg = exp_cfg.model_pairs.draft
    identity = {
        "method": method,
        "seed": int(seed),
        "dataset": str(exp_cfg.datasets.path),
        "eval_task": str(exp_cfg.datasets.name),
        "max_new_tokens": int(exp_cfg.datasets.max_new_tokens),
        "max_samples": int(exp_cfg.datasets.test_max_samples),
        "target_model": _config_model_name(target_cfg),
        "draft_model": _config_model_name(draft_cfg),
        "fixed_sd_gamma": int(exp_cfg.get("fixed_sd_gamma", 0)),
        "fuzzy_sd_gamma": int(exp_cfg.get("fuzzy_sd_gamma", 0)),
        "specdecpp_entropy_threshold": float(exp_cfg.get("specdecpp_entropy_threshold", 0.0)),
        "fuzzy_sd_T_grid": [float(value) for value in exp_cfg.get("fuzzy_sd_T_grid", [])],
        "policy_path": None if not exp_cfg.policy_path else str(exp_cfg.policy_path),
    }
    return json.dumps(identity, sort_keys=True)


def _load_cascade_policies(policy_path: Path) -> dict[str, CascadePolicy]:
    policies: dict[str, CascadePolicy] = {}
    for method_name, prefix in {
        "cascade_length_then_verif": "cascade_length_then_verif",
        "cascade_verif_then_length": "cascade_verif_then_length",
    }.items():
        candidate = _companion_policy_path(policy_path, prefix)
        if candidate.exists():
            policies[method_name] = CascadePolicy.load(candidate)
    return policies


def _decode_gsm8k_exact_match(target_model, generated_ids: list[int], reference_answer: str | None) -> float | None:
    if reference_answer is None:
        return None
    decoded = _decode_generated(target_model, generated_ids)
    answer = extract_final_answer(decoded)
    return 1.0 if answers_equivalent(answer, reference_answer) else 0.0


def _read_legacy_prompt_frame(run_path: Path) -> pd.DataFrame:
    records = _read_legacy_prompt_records(run_path)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    if "decoder" not in frame.columns:
        return pd.DataFrame()
    return frame


def _write_legacy_prompt_artifacts(output_dir: Path) -> tuple[Path, Path]:
    run_path = _legacy_run_path(output_dir)
    frame = _read_legacy_prompt_frame(run_path)
    benchmark_path = output_dir / "benchmark.csv"
    summary_path = output_dir / "summary.csv"
    if frame.empty:
        pd.DataFrame().to_csv(benchmark_path, index=False)
        pd.DataFrame().to_csv(summary_path, index=False)
        return benchmark_path, summary_path

    frame.to_csv(benchmark_path, index=False)
    grouped = frame.groupby("decoder", as_index=False).agg(
        mean_tokens_per_sec=("tokens_per_sec", "mean"),
        mean_acceptance_rate=("acceptance_rate", "mean"),
        runs=("prompt_idx", "count"),
    )
    if "gsm8k_exact_match" in frame.columns:
        gsm_values = frame.groupby("decoder", as_index=False)["gsm8k_exact_match"].mean()
        grouped = grouped.merge(gsm_values, on="decoder", how="left")
    if not grouped.empty and "target_only" in set(grouped["decoder"]):
        baseline_tps = float(grouped.loc[grouped["decoder"] == "target_only", "mean_tokens_per_sec"].iloc[0])
        grouped["speedup_vs_target_only"] = grouped["mean_tokens_per_sec"].map(
            lambda value: 0.0 if baseline_tps <= 0.0 else float(value) / baseline_tps
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(summary_path, index=False)
    return benchmark_path, summary_path


def _system_metadata() -> dict[str, object]:
    def _run(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip() or None
        except Exception:
            return None

    gpu_line = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    gpu_name = None
    gpu_driver = None
    if gpu_line:
        pieces = [piece.strip() for piece in gpu_line.split(",", maxsplit=1)]
        if pieces:
            gpu_name = pieces[0]
        if len(pieces) > 1:
            gpu_driver = pieces[1]
    return {
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "hostname": socket.gethostname(),
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "torch_version": torch.__version__,
        "transformers_version": None if transformers is None else transformers.__version__,
        "display_active": "enabled" if os.environ.get("DISPLAY") else "disabled",
    }


def _summary_records(
    *,
    exp_cfg: DictConfig,
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        method = str(record["method"])
        grouped.setdefault(method, []).append(record)

    summary_records: list[dict[str, object]] = []
    system = _system_metadata()
    for method, method_records in sorted(grouped.items()):
        ok_records = [record for record in method_records if record.get("status") == "ok"]
        first = ok_records[0] if ok_records else method_records[0]
        summary_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": True,
            "runs": len(method_records),
            "runs_successful": len(ok_records),
            "runs_failed": sum(1 for record in method_records if record.get("status") == "error"),
            "runs_skipped": sum(1 for record in method_records if record.get("status") == "skipped"),
            "status": "ok" if ok_records else "error",
            **_build_record_base(exp_cfg, method=method),
            **system,
        }
        if first:
            for key in _build_record_base(exp_cfg, method=method):
                if key in first:
                    summary_record[key] = first[key]
        if ok_records:
            tps_values = [float(record["tokens_per_sec"]) for record in ok_records]
            acc_values = [float(record["acceptance_rate"]) for record in ok_records]
            avg_step_values = [float(record["avg_tokens_per_step"]) for record in ok_records]
            summary_record["tokens_per_sec_median"] = float(np.median(tps_values))
            summary_record["acceptance_rate_median"] = float(np.median(acc_values))
            summary_record["avg_tokens_per_step_median"] = float(np.median(avg_step_values))
            tps_ci = _bootstrap_ci(tps_values, seed=123 + len(method))
            acc_ci = _bootstrap_ci(acc_values, seed=456 + len(method))
            summary_record["tokens_per_sec_ci_low"] = tps_ci[0]
            summary_record["tokens_per_sec_ci_high"] = tps_ci[1]
            summary_record["acceptance_rate_ci_low"] = acc_ci[0]
            summary_record["acceptance_rate_ci_high"] = acc_ci[1]
            if all(record.get("gsm8k_exact_match") is not None for record in ok_records):
                em_values = [float(record["gsm8k_exact_match"]) for record in ok_records]
                total_correct = sum(int(record["gsm8k_correct"]) for record in ok_records)
                total_total = sum(int(record["gsm8k_total"]) for record in ok_records)
                em_ci = _bootstrap_ci(em_values, seed=789 + len(method))
                summary_record["gsm8k_correct"] = total_correct
                summary_record["gsm8k_total"] = total_total
                summary_record["gsm8k_exact_match"] = 0.0 if total_total <= 0 else float(total_correct / total_total)
                summary_record["gsm8k_exact_match_ci_low"] = em_ci[0]
                summary_record["gsm8k_exact_match_ci_high"] = em_ci[1]
            if any(record.get("error_message") for record in method_records):
                errors = [str(record["error_message"]) for record in method_records if record.get("error_message")]
                summary_record["error_message"] = errors[0]
        summary_records.append(summary_record)
    return summary_records


@hydra.main(version_base=None, config_path="../configs", config_name="experiments/default")
def main(cfg: DictConfig) -> None:
    exp_cfg = _experiment_cfg(cfg)
    output_dir = _make_output_dir(exp_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(output_dir=output_dir, config=cfg)
    target_model, draft_model = load_model_pair(exp_cfg.model_pairs)
    samples = _load_eval_samples(exp_cfg.datasets)
    n_seeds = int(exp_cfg.get("n_seeds", 3))
    seeds = _seed_list(n_seeds)
    start_timestamp = datetime.now(UTC).isoformat()
    manifest_name = f"{output_dir.parent.name}_{output_dir.name}.json"
    manifest_path = Path(get_original_cwd()) / "reports" / "manifests" / manifest_name
    write_manifest(
        out_path=manifest_path,
        resolved_config_yaml=OmegaConf.to_yaml(cfg),
        seed_list=seeds,
        traces_path=None if not exp_cfg.traces_path else Path(str(exp_cfg.traces_path)),
        policy_path=None if not exp_cfg.policy_path else Path(str(exp_cfg.policy_path)),
        start_timestamp=start_timestamp,
    )

    joint_policy = JointAdaSpecPolicy.load(Path(exp_cfg.policy_path)) if exp_cfg.policy_path else None
    decoders = _build_decoders(exp_cfg, joint_policy, target_model, draft_model)
    if exp_cfg.policy_path:
        for method_name, cascade_policy in _load_cascade_policies(Path(exp_cfg.policy_path)).items():
            decoders[method_name] = JointAdaSpecDecoder(
                target_model=target_model,
                draft_model=draft_model,
                policy=cascade_policy,
            )

    active_names = set(str(name) for name in exp_cfg.baselines)
    if joint_policy is not None:
        active_names.add("jointadaspec")

    results_path = _results_path(output_dir)
    existing_records, completed_run_keys = _load_existing_results(results_path)
    new_records: list[dict[str, object]] = []

    for method_name, decoder in decoders.items():
        if not _is_enabled(active_names, method_name):
            continue
        for run_index, seed in enumerate(seeds, start=1):
            resume_key = _record_resume_key(exp_cfg, method=method_name, seed=seed)
            if resume_key in completed_run_keys:
                continue
            _set_deterministic_seed(seed)
            total_prompt_tokens = 0
            total_generated_tokens = 0
            total_duration_sec = 0.0
            total_proposed = 0.0
            total_accepted = 0.0
            total_rejections = 0.0
            total_steps = 0.0
            gsm8k_correct = 0
            gsm8k_total = 0
            try:
                for prompt_idx, sample in enumerate(samples):
                    prompt_ids = _encode_prompt(target_model, sample.prompt)
                    prompt_generator = torch.Generator(device="cpu")
                    prompt_generator.manual_seed(seed + prompt_idx)
                    result = decoder.generate(
                        prompt_ids=prompt_ids,
                        max_new_tokens=int(exp_cfg.datasets.max_new_tokens),
                        generator=prompt_generator,
                    )
                    tokens_per_sec = (
                        result.n_tokens_generated / (result.total_time_ms / 1000.0)
                        if result.total_time_ms > 0.0
                        else 0.0
                    )
                    prompt_stats = _prompt_stats_from_result(result, decoder_name=method_name)
                    total_prompt_tokens += len(prompt_ids)
                    total_generated_tokens += result.n_tokens_generated
                    total_duration_sec += result.total_time_ms / 1000.0
                    total_proposed += prompt_stats["proposed"]
                    total_accepted += prompt_stats["accepted"]
                    total_rejections += prompt_stats["rejections"]
                    total_steps += prompt_stats["steps"]
                    gsm8k_exact_match = _decode_gsm8k_exact_match(
                        target_model,
                        result.generated_ids,
                        sample.reference_answer,
                    )
                    if gsm8k_exact_match is not None:
                        gsm8k_correct += int(gsm8k_exact_match)
                        gsm8k_total += 1
                    logger.log(
                        {
                            "decoder": method_name,
                            "run": run_index,
                            "seed": seed,
                            "prompt_idx": prompt_idx,
                            "n_tokens_generated": result.n_tokens_generated,
                            "acceptance_rate": result.acceptance_rate,
                            "total_time_ms": result.total_time_ms,
                            "tokens_per_sec": tokens_per_sec,
                            "n_target_calls": result.n_target_calls,
                            "n_draft_calls": result.n_draft_calls,
                            "gsm8k_exact_match": gsm8k_exact_match,
                        }
                    )

                acceptance_rate = 0.0 if total_proposed <= 0 else float(total_accepted / total_proposed)
                tokens_per_sec = 0.0 if total_duration_sec <= 0.0 else float(total_generated_tokens / total_duration_sec)
                avg_tokens_per_step = 0.0 if total_steps <= 0 else float(total_generated_tokens / total_steps)
                record = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "status": "ok",
                    "summary": False,
                    "run": run_index,
                    "seed": seed,
                    "resume_key": resume_key,
                    **_build_record_base(exp_cfg, method=method_name),
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_generated_tokens": total_generated_tokens,
                    "duration_sec": total_duration_sec,
                    "tokens_per_sec": tokens_per_sec,
                    "acceptance_rate": acceptance_rate,
                    "avg_tokens_per_step": avg_tokens_per_step,
                    "proposed": total_proposed,
                    "accepted": total_accepted,
                    "steps": total_steps,
                    "rejections": total_rejections,
                    "gsm8k_correct": gsm8k_correct,
                    "gsm8k_total": gsm8k_total,
                    "gsm8k_exact_match": None if gsm8k_total <= 0 else float(gsm8k_correct / gsm8k_total),
                }
            except Exception:
                record = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "status": "error",
                    "summary": False,
                    "run": run_index,
                    "seed": seed,
                    "resume_key": resume_key,
                    **_build_record_base(exp_cfg, method=method_name),
                    "error_type": "benchmark_error",
                    "error_message": traceback.format_exc().splitlines()[-1],
                    "traceback": traceback.format_exc(),
                }
            new_records.append(record)
            completed_run_keys.add(resume_key)

    all_non_summary = [record for record in existing_records if not bool(record.get("summary"))] + new_records
    summary_records = _summary_records(exp_cfg=exp_cfg, records=all_non_summary)
    with results_path.open("w", encoding="utf-8") as fh:
        for record in all_non_summary + summary_records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    benchmark_path, summary_path = _write_legacy_prompt_artifacts(output_dir)
    logger.finalize(
        {
            "stage": "benchmark",
            "benchmark_path": str(benchmark_path),
            "summary_path": str(summary_path),
            "results_path": str(results_path),
            "num_records": len(all_non_summary),
            "n_seeds": n_seeds,
            "manifest_path": str(manifest_path),
        }
    )
    print(results_path)


if __name__ == "__main__":
    main()
