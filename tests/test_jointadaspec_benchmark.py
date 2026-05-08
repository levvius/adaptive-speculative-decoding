from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from omegaconf import OmegaConf

from sp_samp.gsm8k import GSM8KSample


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "03_benchmark.py"
    spec = importlib.util.spec_from_file_location("jointadaspec_benchmark_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_eval_samples_applies_gsm8k_start_index(monkeypatch) -> None:
    module = _load_benchmark_module()

    def fake_load_gsm8k(path: str, max_samples: int | None = None):
        assert path == "unused.jsonl"
        assert max_samples == 7
        return [
            GSM8KSample(question=f"question {idx}", answer=f"#### {idx}")
            for idx in range(max_samples or 0)
        ]

    monkeypatch.setattr(module, "load_gsm8k", fake_load_gsm8k)
    cfg = OmegaConf.create(
        {
            "name": "gsm8k",
            "path": "unused.jsonl",
            "test_start_index": 2,
            "test_max_samples": 5,
            "eval_mode": "plain",
        }
    )

    samples = module._load_eval_samples(cfg)

    assert len(samples) == 5
    assert "question 2" in samples[0].prompt
    assert "question 1" not in samples[0].prompt
    assert samples[0].reference_answer == "2"
    assert "question 6" in samples[-1].prompt


def test_load_eval_samples_default_start_index_is_zero(monkeypatch) -> None:
    module = _load_benchmark_module()

    def fake_load_gsm8k(path: str, max_samples: int | None = None):
        assert max_samples == 3
        return [
            GSM8KSample(question=f"question {idx}", answer=f"#### {idx}")
            for idx in range(max_samples or 0)
        ]

    monkeypatch.setattr(module, "load_gsm8k", fake_load_gsm8k)
    cfg = OmegaConf.create(
        {
            "name": "gsm8k",
            "path": "unused.jsonl",
            "test_max_samples": 3,
            "eval_mode": "plain",
        }
    )

    samples = module._load_eval_samples(cfg)

    assert len(samples) == 3
    assert "question 0" in samples[0].prompt
