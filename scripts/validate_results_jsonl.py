from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


METHODS = {"baseline", "speculative", "autojudge", "consensus_autojudge", "topk", "specexec"}
STATUSES = {"ok", "error", "skipped"}

BASE_FIELDS = {
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
}

OPTIONAL_BASE_FIELDS = {
    "draft2_model",
    "draft2_tokenizer",
    "draft2_device",
    "draft2_dtype",
    "draft2_quant",
    "draft2_bnb_compute_dtype",
    "eval_task",
    "gsm8k_eval_mode",
    "topk_rank",
    "topk_grid",
    "autojudge_threshold_used",
    "autojudge_threshold_calibrated",
    "autojudge_task",
    "autojudge_classifier",
    "autojudge_features",
    "autojudge_train_dataset",
    "autojudge_recall_target",
    "autojudge_train_split",
    "autojudge_c_grid",
    "autojudge_val_auc",
    "autojudge_val_recall",
    "autojudge_threshold_selected",
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
}

SYSTEM_FIELDS = {
    "git_sha",
    "hostname",
    "gpu_name",
    "gpu_driver",
    "cuda_runtime",
    "torch_version",
    "transformers_version",
    "display_active",
}

COMMON_OK_FIELDS = {
    "total_prompt_tokens",
    "total_generated_tokens",
    "duration_sec",
    "tokens_per_sec",
    "acceptance_rate",
    "avg_tokens_per_step",
    "proposed",
    "accepted",
    "steps",
    "rejections",
}

AUTOJUDGE_OK_FIELDS = {
    "judge_accept_rate",
    "target_fallback_rate",
    "judge_total",
    "judge_accepted",
    "judge_rejected",
    "target_calls",
    "target_fallbacks",
    "draft_calls",
    "draft_prefills",
    "train_samples",
    "train_loss",
    "val_auc",
    "val_recall",
    "threshold_selected",
}

CONSENSUS_OK_FIELDS = {
    "gate_accept_d1_rate",
    "gate_escalate_rate",
    "gate_fallback_rate",
    "target_fallback_rate",
    "target_calls_per_token",
    "draft_calls_per_token",
    "draft2_calls_per_token",
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
    "train_samples",
    "val_accuracy",
    "val_macro_f1",
    "fallback_threshold_selected",
}

SPECEXEC_OK_FIELDS = {
    "target_calls_per_token",
    "draft_calls_per_token",
    "branch_prune_rate",
    "effective_parallelism",
    "cache_hit_rate",
    "target_calls",
    "draft_calls",
    "draft_prefills",
    "target_prefills",
    "branches_total",
    "branches_kept",
    "branches_pruned",
    "max_active_branches",
    "cache_hits",
    "cache_misses",
}

TOPK_OK_FIELDS = {
    "topk_accept_rate",
    "topk_rank_effective",
    "topk_mismatches",
    "topk_accepted_mismatches",
    "target_calls",
    "target_calls_per_token",
    "draft_prefills",
}

GSM8K_FIELDS = {
    "gsm8k_exact_match",
    "gsm8k_correct",
    "gsm8k_total",
}

SUMMARY_FIELDS = {
    "runs",
    "runs_successful",
    "runs_failed",
    "runs_skipped",
}

SUMMARY_MEDIAN_FIELDS = {
    "tokens_per_sec_median",
    "acceptance_rate_median",
    "avg_tokens_per_step_median",
}

SUMMARY_OPTIONAL_FIELDS = {
    "judge_accept_rate_median",
    "gate_accept_d1_rate_median",
    "gate_escalate_rate_median",
    "gate_fallback_rate_median",
    "topk_accept_rate_median",
    "target_fallback_rate_median",
    "cache_hit_rate_median",
    "error_message",
    "info_message",
}

GENERAL_FIELDS = {
    "timestamp",
    "status",
    "summary",
    "run",
    "resume_key",
    "error_type",
    "error_message",
    "traceback",
}

ALLOWED_FIELDS = (
    GENERAL_FIELDS
    | BASE_FIELDS
    | OPTIONAL_BASE_FIELDS
    | SYSTEM_FIELDS
    | COMMON_OK_FIELDS
    | AUTOJUDGE_OK_FIELDS
    | CONSENSUS_OK_FIELDS
    | TOPK_OK_FIELDS
    | SPECEXEC_OK_FIELDS
    | GSM8K_FIELDS
    | SUMMARY_FIELDS
    | SUMMARY_MEDIAN_FIELDS
    | SUMMARY_OPTIONAL_FIELDS
)


def _kind(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _require_keys(
    record: Dict[str, Any],
    required: Iterable[str],
    errors: List[str],
    ctx: str,
) -> None:
    for key in required:
        if key not in record:
            errors.append(f"{ctx}: missing required field '{key}'.")


def _check_type(
    record: Dict[str, Any],
    key: str,
    expected: str,
    errors: List[str],
    ctx: str,
    allow_none: bool = False,
) -> None:
    if key not in record:
        return
    value = record[key]
    kind = _kind(value)
    if allow_none and kind == "none":
        return
    if kind != expected:
        errors.append(
            f"{ctx}: field '{key}' must be {expected}, got {kind}."
        )


def _validate_record(
    record: Dict[str, Any],
    line_no: int,
    errors: List[str],
    warnings: List[str],
    strict: bool,
) -> None:
    ctx = f"line {line_no}"
    _require_keys(record, {"timestamp", "status", "summary"} | BASE_FIELDS, errors, ctx)

    _check_type(record, "timestamp", "string", errors, ctx)
    _check_type(record, "status", "string", errors, ctx)
    _check_type(record, "summary", "bool", errors, ctx)
    _check_type(record, "method", "string", errors, ctx)
    _check_type(record, "backend", "string", errors, ctx)
    _check_type(record, "use_chat_template", "bool", errors, ctx)
    _check_type(record, "k", "number", errors, ctx)
    _check_type(record, "max_new_tokens", "number", errors, ctx)
    _check_type(record, "max_samples", "number", errors, ctx)
    _check_type(record, "turn_index", "number", errors, ctx)

    status = record.get("status")
    if status not in STATUSES:
        errors.append(f"{ctx}: unsupported status '{status}'.")

    method = record.get("method")
    if method not in METHODS:
        errors.append(f"{ctx}: unsupported method '{method}'.")

    for key in SYSTEM_FIELDS:
        if key in record:
            _check_type(record, key, "string", errors, ctx, allow_none=True)

    for key in BASE_FIELDS:
        if key == "use_chat_template":
            _check_type(record, key, "bool", errors, ctx)
        elif key in {
            "k",
            "max_new_tokens",
            "max_samples",
            "turn_index",
            "autojudge_threshold",
            "autojudge_train_samples",
            "autojudge_train_loss",
            "autojudge_recall_target",
            "autojudge_train_split",
            "autojudge_val_auc",
            "autojudge_val_recall",
            "autojudge_threshold_selected",
            "parallel_branches",
            "branch_prune_threshold",
        }:
            _check_type(record, key, "number", errors, ctx, allow_none=True)
        elif key in {"quant", "draft_quant"}:
            _check_type(record, key, "string", errors, ctx, allow_none=True)
        else:
            _check_type(record, key, "string", errors, ctx, allow_none=True)

    for key in OPTIONAL_BASE_FIELDS:
        if key in {
            "eval_task",
            "gsm8k_eval_mode",
            "topk_rank",
            "topk_grid",
            "autojudge_classifier",
            "autojudge_features",
            "draft2_model",
            "draft2_tokenizer",
            "draft2_device",
            "draft2_dtype",
            "draft2_quant",
            "draft2_bnb_compute_dtype",
            "consensus_gate",
            "consensus_features",
            "consensus_train_dataset",
            "consensus_checkpoint",
        }:
            _check_type(record, key, "string", errors, ctx, allow_none=True)
        elif key in {
            "autojudge_threshold_used",
            "autojudge_threshold_calibrated",
            "autojudge_recall_target",
            "autojudge_train_split",
            "autojudge_val_auc",
            "autojudge_val_recall",
            "autojudge_threshold_selected",
            "consensus_train_samples",
            "consensus_train_split",
            "consensus_fallback_threshold",
            "consensus_top_m",
            "consensus_val_accuracy",
            "consensus_val_macro_f1",
        }:
            _check_type(record, key, "number", errors, ctx, allow_none=True)
        elif key == "consensus_disable_escalation":
            _check_type(record, key, "bool", errors, ctx, allow_none=True)
        else:
            _check_type(record, key, "string", errors, ctx, allow_none=True)

    is_summary = bool(record.get("summary"))
    if is_summary:
        _require_keys(record, SUMMARY_FIELDS, errors, ctx)
        for key in SUMMARY_FIELDS:
            _check_type(record, key, "number", errors, ctx)
        if record.get("runs_successful", 0) > 0:
            _require_keys(record, SUMMARY_MEDIAN_FIELDS, errors, ctx)
        for key in SUMMARY_MEDIAN_FIELDS | SUMMARY_OPTIONAL_FIELDS:
            if key in record:
                _check_type(record, key, "number" if key.endswith("_median") else "string", errors, ctx)
        for key in GSM8K_FIELDS:
            if key in record:
                _check_type(
                    record,
                    key,
                    "number",
                    errors,
                    ctx,
                    allow_none=(key == "gsm8k_exact_match"),
                )
        return

    _require_keys(record, {"run"}, errors, ctx)
    _check_type(record, "run", "number", errors, ctx)

    if status == "ok":
        _require_keys(record, {"resume_key"} | COMMON_OK_FIELDS, errors, ctx)
        _check_type(record, "resume_key", "string", errors, ctx)
        for key in COMMON_OK_FIELDS:
            _check_type(record, key, "number", errors, ctx)
        if method == "autojudge":
            _require_keys(record, AUTOJUDGE_OK_FIELDS, errors, ctx)
            for key in AUTOJUDGE_OK_FIELDS:
                _check_type(record, key, "number", errors, ctx)
        if method == "consensus_autojudge":
            _require_keys(record, CONSENSUS_OK_FIELDS, errors, ctx)
            for key in CONSENSUS_OK_FIELDS:
                _check_type(record, key, "number", errors, ctx)
        if method == "topk":
            _require_keys(record, TOPK_OK_FIELDS, errors, ctx)
            for key in TOPK_OK_FIELDS:
                _check_type(record, key, "number", errors, ctx)
        if method == "specexec":
            _require_keys(record, SPECEXEC_OK_FIELDS, errors, ctx)
            for key in SPECEXEC_OK_FIELDS:
                _check_type(record, key, "number", errors, ctx)
        for key in GSM8K_FIELDS:
            if key in record:
                _check_type(
                    record,
                    key,
                    "number",
                    errors,
                    ctx,
                    allow_none=(key == "gsm8k_exact_match"),
                )
    elif status == "error":
        _require_keys(record, {"resume_key", "error_type", "error_message", "traceback"}, errors, ctx)
        _check_type(record, "resume_key", "string", errors, ctx)
        _check_type(record, "error_type", "string", errors, ctx)
        _check_type(record, "error_message", "string", errors, ctx)
        _check_type(record, "traceback", "string", errors, ctx)

    unknown = sorted(set(record.keys()) - ALLOWED_FIELDS)
    if unknown:
        message = f"{ctx}: unknown field(s): {', '.join(unknown)}"
        if strict:
            errors.append(message)
        else:
            warnings.append(message)


def validate_results(path: Path, strict: bool = False) -> int:
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return 2

    errors: List[str] = []
    warnings: List[str] = []
    type_map: Dict[str, Set[str]] = {}
    total_records = 0

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            total_records += 1
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"line {line_no}: expected JSON object.")
                continue
            _validate_record(payload, line_no, errors, warnings, strict)

            for key, value in payload.items():
                kind = _kind(value)
                current = type_map.setdefault(key, set())
                current.add(kind)

    for key, kinds in sorted(type_map.items()):
        non_none = {k for k in kinds if k != "none"}
        if len(non_none) > 1:
            errors.append(
                f"type mismatch: field '{key}' has inconsistent types: {sorted(non_none)}"
            )

    print(f"Validated records: {total_records}")
    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[ERROR] {error}", file=sys.stderr)

    if errors:
        print(f"Validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print("Validation passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark JSONL result records.")
    parser.add_argument("--path", type=str, required=True, help="Path to JSONL results file.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat unknown fields as errors.",
    )
    args = parser.parse_args()
    return validate_results(Path(args.path), strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
