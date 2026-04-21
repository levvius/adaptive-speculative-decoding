from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


def pretty_method_name(method: str) -> str:
    mapping = {
        "vanilla_ar": "target_only",
        "target_only": "target_only",
        "fixed_sd": "speculative",
        "speculative": "speculative",
        "specdecpp": "adaptive_length",
        "adaptive_length": "adaptive_length",
        "cascade_length_then_verif": "cascade_len_then_verif",
        "cascade_verif_then_length": "cascade_verif_then_length",
    }
    return mapping.get(method, method)


def infer_dataset_name(record: dict[str, Any], source_path: Path) -> str:
    eval_task = record.get("eval_task")
    if isinstance(eval_task, str) and eval_task:
        return eval_task
    dataset = record.get("dataset")
    if isinstance(dataset, str):
        lowered = dataset.lower()
        if "gsm8k" in lowered:
            return "gsm8k"
        if "livecodebench" in lowered:
            return "livecodebench"
        if "mt_bench" in lowered or "mtbench" in lowered:
            return "mtbench"
    stem = source_path.parent.name.lower()
    if "gsm8k" in stem:
        return "gsm8k"
    if "livecodebench" in stem:
        return "livecodebench"
    if "mtbench" in stem or "mt_bench" in stem:
        return "mtbench"
    return "unknown"


def quality_field(dataset_name: str) -> str:
    return "gsm8k_exact_match" if dataset_name == "gsm8k" else "acceptance_rate"


def quality_label(dataset_name: str) -> str:
    return "Exact Match" if dataset_name == "gsm8k" else "Acceptance Rate"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _bootstrap_ci(values: list[float], *, seed: int = 42, n_resamples: int = 1000) -> tuple[float | None, float | None]:
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


def _aggregate_legacy_records(records: list[dict[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    grouped_prompt: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    dataset_name = "unknown"
    for record in records:
        if "decoder" not in record or "prompt_idx" not in record:
            continue
        dataset_name = infer_dataset_name(record, source_path)
        run_value = int(record.get("run", 1))
        grouped_prompt[(pretty_method_name(str(record["decoder"])), run_value)].append(record)

    per_seed_rows: list[dict[str, Any]] = []
    for (method, run_value), group in grouped_prompt.items():
        total_tokens = sum(float(record.get("n_tokens_generated", 0.0)) for record in group)
        total_seconds = sum(float(record.get("total_time_ms", 0.0)) for record in group) / 1000.0
        tps = 0.0 if total_seconds <= 0.0 else float(total_tokens / total_seconds)
        acceptance = float(np.mean([float(record.get("acceptance_rate", 0.0)) for record in group]))
        em_values = [float(record["gsm8k_exact_match"]) for record in group if record.get("gsm8k_exact_match") is not None]
        per_seed_rows.append(
            {
                "dataset_name": dataset_name,
                "method": method,
                "run": run_value,
                "tokens_per_sec": tps,
                "acceptance_rate": acceptance,
                "gsm8k_exact_match": None if not em_values else float(np.mean(em_values)),
                "source_path": str(source_path),
            }
        )

    grouped_summary: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped_summary[(row["dataset_name"], row["method"])] .append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset_name, method), group in grouped_summary.items():
        tps_values = [float(row["tokens_per_sec"]) for row in group]
        acc_values = [float(row["acceptance_rate"]) for row in group]
        em_values = [float(row["gsm8k_exact_match"]) for row in group if row["gsm8k_exact_match"] is not None]
        tps_ci = _bootstrap_ci(tps_values, seed=11 + len(method))
        acc_ci = _bootstrap_ci(acc_values, seed=23 + len(method))
        em_ci = _bootstrap_ci(em_values, seed=37 + len(method)) if em_values else (None, None)
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "method": method,
                "tokens_per_sec_median": float(np.median(tps_values)),
                "tokens_per_sec_ci_low": tps_ci[0],
                "tokens_per_sec_ci_high": tps_ci[1],
                "acceptance_rate_median": float(np.median(acc_values)),
                "acceptance_rate_ci_low": acc_ci[0],
                "acceptance_rate_ci_high": acc_ci[1],
                "gsm8k_exact_match": None if not em_values else float(np.mean(em_values)),
                "gsm8k_exact_match_ci_low": em_ci[0],
                "gsm8k_exact_match_ci_high": em_ci[1],
                "source_path": str(source_path),
                "manifest_path": None,
            }
        )
    return summary_rows


def load_summary_rows(paths: list[Path], manifest_path: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        records = _read_jsonl(path)
        summary_records = [record for record in records if bool(record.get("summary"))]
        if summary_records:
            for record in summary_records:
                dataset_name = infer_dataset_name(record, path)
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "method": pretty_method_name(str(record["method"])),
                        "tokens_per_sec_median": float(record["tokens_per_sec_median"]),
                        "tokens_per_sec_ci_low": record.get("tokens_per_sec_ci_low"),
                        "tokens_per_sec_ci_high": record.get("tokens_per_sec_ci_high"),
                        "acceptance_rate_median": float(record["acceptance_rate_median"]),
                        "acceptance_rate_ci_low": record.get("acceptance_rate_ci_low"),
                        "acceptance_rate_ci_high": record.get("acceptance_rate_ci_high"),
                        "gsm8k_exact_match": record.get("gsm8k_exact_match"),
                        "gsm8k_exact_match_ci_low": record.get("gsm8k_exact_match_ci_low"),
                        "gsm8k_exact_match_ci_high": record.get("gsm8k_exact_match_ci_high"),
                        "source_path": str(path),
                        "manifest_path": None if manifest_path is None else str(manifest_path),
                    }
                )
            continue
        rows.extend(_aggregate_legacy_records(records, path))
    return rows


def manifest_footnote(manifest_path: Path | None, source_paths: list[Path]) -> str:
    if manifest_path is not None:
        return f"Source manifest: {manifest_path}"
    if len(source_paths) == 1:
        return f"Source: {source_paths[0]}"
    return "Sources: " + ", ".join(str(path) for path in source_paths)
