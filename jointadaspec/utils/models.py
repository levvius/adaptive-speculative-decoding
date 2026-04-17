"""Model-loading helpers backed by ``sp_samp.hf_adapter.HFModel``."""

from __future__ import annotations

from typing import Any, Mapping

from sp_samp.hf_adapter import HFModel


def _to_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    items = getattr(config, "items", None)
    if callable(items):
        return dict(items())
    raise TypeError(f"Expected a mapping-like config, got {type(config)!r}.")


def _load_single_model(config: Mapping[str, Any]) -> HFModel:
    hf_model = config.get("hf_model")
    if not hf_model:
        raise ValueError("Model config is missing 'hf_model'.")
    return HFModel(
        model_name=str(hf_model),
        device=str(config.get("device", "cpu")),
        dtype=str(config.get("dtype", "auto")),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        tokenizer_name=config.get("tokenizer"),
        use_fast_tokenizer=bool(config.get("use_fast_tokenizer", True)),
        quantization=config.get("quant"),
        bnb_compute_dtype=str(config.get("bnb_compute_dtype", "bfloat16")),
    )


def load_model_pair(config: Any) -> tuple[HFModel, HFModel]:
    """Load target and draft HF models from a config object."""
    mapping = _to_mapping(config)
    if "target" not in mapping or "draft" not in mapping:
        raise ValueError("Config must contain top-level 'target' and 'draft' sections.")
    target = _load_single_model(_to_mapping(mapping["target"]))
    draft = _load_single_model(_to_mapping(mapping["draft"]))
    return target, draft
