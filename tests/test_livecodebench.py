from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sp_samp.livecodebench import load_livecodebench


def _write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_load_livecodebench_basic():
    records = [
        {"prompt": "Write a function that adds two numbers.", "question_id": "q1"},
        {"prompt": "Implement binary search.", "question_id": "q2"},
        {"prompt": "Sort a list of integers.", "question_id": "q3"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "lcb.jsonl"
        _write_jsonl(path, records)
        prompts = load_livecodebench(str(path))
        assert len(prompts) == 3
        assert prompts[0] == "Write a function that adds two numbers."
        assert prompts[2] == "Sort a list of integers."


def test_load_livecodebench_max_samples():
    records = [
        {"prompt": f"Problem {i}"} for i in range(10)
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "lcb.jsonl"
        _write_jsonl(path, records)
        prompts = load_livecodebench(str(path), max_samples=3)
        assert len(prompts) == 3
        assert prompts[0] == "Problem 0"


def test_load_livecodebench_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "lcb.jsonl"
        path.write_text("", encoding="utf-8")
        prompts = load_livecodebench(str(path))
        assert prompts == []


def test_load_livecodebench_skips_missing_prompt():
    records = [
        {"question_id": "q1"},
        {"prompt": "Valid prompt.", "question_id": "q2"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "lcb.jsonl"
        _write_jsonl(path, records)
        prompts = load_livecodebench(str(path))
        assert len(prompts) == 1
        assert prompts[0] == "Valid prompt."


def test_load_livecodebench_json_array():
    records = [
        {"prompt": "Array problem 1"},
        {"prompt": "Array problem 2"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "lcb.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(records, fh)
        prompts = load_livecodebench(str(path))
        assert len(prompts) == 2


def test_load_livecodebench_alternative_keys():
    """Supports 'question', 'input', 'text', 'instruction' as fallback keys."""
    records = [
        {"question": "Q1"},
        {"input": "I1"},
        {"text": "T1"},
        {"instruction": "Inst1"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "lcb.jsonl"
        _write_jsonl(path, records)
        prompts = load_livecodebench(str(path))
        assert len(prompts) == 4
        assert prompts == ["Q1", "I1", "T1", "Inst1"]
