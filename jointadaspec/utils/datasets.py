"""Dataset-loading wrappers around ``sp_samp`` loaders."""

from __future__ import annotations

from typing import Any, Mapping

from sp_samp.gsm8k import load_gsm8k
from sp_samp.livecodebench import load_livecodebench
from sp_samp.mtbench import load_mtbench


def _to_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    items = getattr(config, "items", None)
    if callable(items):
        return dict(items())
    raise TypeError(f"Expected a mapping-like config, got {type(config)!r}.")


def load_dataset(config: Any, split: str) -> list[str]:
    """Load prompts for trace collection or benchmarking."""
    mapping = _to_mapping(config)
    split = str(split)
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")

    name = str(mapping.get("name", "")).lower()
    if not name:
        raise ValueError("Dataset config is missing 'name'.")

    path_key = "train_path" if split == "train" and mapping.get("train_path") else "path"
    path = mapping.get(path_key)
    if not path:
        raise ValueError(f"Dataset path for split '{split}' is missing.")
    max_samples_key = "train_max_samples" if split == "train" else "test_max_samples"
    max_samples = mapping.get(max_samples_key)

    if name == "gsm8k":
        return [sample.question for sample in load_gsm8k(str(path), max_samples=max_samples)]
    if name == "livecodebench":
        return load_livecodebench(str(path), max_samples=max_samples)
    if name == "mtbench":
        turn_index = int(mapping.get("turn_index", 0))
        return load_mtbench(str(path), turn_index=turn_index, max_samples=max_samples)
    raise ValueError(f"Unsupported dataset name '{name}'.")
