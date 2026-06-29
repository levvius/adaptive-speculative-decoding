"""Tests for the Qwen3.5 (model_type 'qwen3_5') loader compatibility guard.

Qwen3.5 checkpoints are vision-language models that older transformers builds do
not register. The loader must surface an actionable error instead of a cryptic
KeyError. This is validated with a tiny synthetic config.json so the test does not
depend on the gitignored multi-GB weights.
"""

from __future__ import annotations

import json

import pytest

from sp_samp.hf_adapter import _load_model_config_with_compat


def _qwen35_registered() -> bool:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    return "qwen3_5" in CONFIG_MAPPING_NAMES


def test_qwen35_config_raises_actionable_error_when_unsupported(tmp_path) -> None:
    if _qwen35_registered():
        pytest.skip("Installed transformers registers qwen3_5; the upgrade guard is a no-op.")
    config_dir = tmp_path / "Qwen3.5-tiny"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="qwen3_5"):
        _load_model_config_with_compat(str(config_dir), trust_remote_code=False)


def test_non_qwen35_config_is_unaffected_by_guard(tmp_path) -> None:
    # A bogus model_type should NOT trip the qwen3_5 guard; it falls through to the
    # normal AutoConfig path (which raises its own error, not the qwen3_5 message).
    config_dir = tmp_path / "bogus"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"model_type": "not_a_real_model_type"}), encoding="utf-8"
    )
    with pytest.raises(Exception) as exc_info:
        _load_model_config_with_compat(str(config_dir), trust_remote_code=False)
    assert "qwen3_5" not in str(exc_info.value)
