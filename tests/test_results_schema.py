from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from jointadaspec.semantics import (
    ACTION_SPACE_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    TARGET_ONLY_DECODER_SEMANTICS,
    TARGET_PASS_BATCHED_BLOCK,
    TARGET_PASS_TARGET_ONLY,
)


def _load_validator_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_results_jsonl.py"
    spec = importlib.util.spec_from_file_location("validate_results_jsonl_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _minimal_schema_v3_record() -> dict[str, object]:
    return {
        "timestamp": "2026-06-23T00:00:00+00:00",
        "status": "ok",
        "summary": False,
        "run": 1,
        "resume_key": "unit",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "stat_unit": "prompt",
        "action_space_version": ACTION_SPACE_VERSION,
        "target_pass_mode": TARGET_PASS_TARGET_ONLY,
        "decoder_semantics": TARGET_ONLY_DECODER_SEMANTICS,
        "method": "target_only",
        "backend": "jointadaspec_hf",
        "target_model": "target",
        "draft_model": "draft",
        "tokenizer": "tok",
        "draft_tokenizer": "tok",
        "device": "cpu",
        "dtype": "float32",
        "quant": None,
        "bnb_compute_dtype": "float32",
        "draft_device": "cpu",
        "draft_dtype": "float32",
        "draft_quant": None,
        "draft_bnb_compute_dtype": "float32",
        "use_chat_template": False,
        "system_prompt": None,
        "k": 4,
        "max_new_tokens": 8,
        "max_samples": 1,
        "turn_index": 0,
        "dataset": "dataset.jsonl",
        "autojudge_threshold": None,
        "autojudge_train_samples": 0,
        "autojudge_train_loss": 0.0,
        "autojudge_checkpoint": None,
        "parallel_branches": None,
        "branch_prune_threshold": 0.0,
        "total_prompt_tokens": 3,
        "total_generated_tokens": 8,
        "duration_sec": 1.0,
        "tokens_per_sec": 8.0,
        "acceptance_rate": 0.0,
        "avg_tokens_per_step": 1.0,
        "proposed": 0.0,
        "accepted": 0.0,
        "steps": 8.0,
        "rejections": 0.0,
    }


def _write_jsonl(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def test_validate_results_accepts_schema_v3_semantic_fields(tmp_path) -> None:
    module = _load_validator_module()
    path = tmp_path / "results.jsonl"
    _write_jsonl(path, _minimal_schema_v3_record())

    assert module.validate_results(path, strict=True) == 0


def test_validate_results_rejects_missing_action_space_version(tmp_path) -> None:
    module = _load_validator_module()
    record = _minimal_schema_v3_record()
    record.pop("action_space_version")
    path = tmp_path / "results_missing_action_space.jsonl"
    _write_jsonl(path, record)

    assert module.validate_results(path, strict=True) == 1


def test_validate_results_rejects_wrong_target_pass_for_target_only(tmp_path) -> None:
    module = _load_validator_module()
    record = _minimal_schema_v3_record()
    record["target_pass_mode"] = TARGET_PASS_BATCHED_BLOCK
    path = tmp_path / "results_wrong_target_pass.jsonl"
    _write_jsonl(path, record)

    assert module.validate_results(path, strict=True) == 1
