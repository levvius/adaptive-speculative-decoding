from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional


def _extract_prompt(item: dict, turn_index: int) -> Optional[str]:
    if "turns" in item and isinstance(item["turns"], list):
        turns = item["turns"]
        if 0 <= turn_index < len(turns):
            return turns[turn_index]
        return None
    for key in ("question", "prompt", "text", "instruction"):
        if key in item and isinstance(item[key], str):
            return item[key]
    return None


def _iter_json_items(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if isinstance(item, dict):
                        yield item
            else:
                yield data


def load_mtbench(
    path: str, turn_index: int = 0, max_samples: Optional[int] = None
) -> List[str]:
    """Load MT-Bench prompts from a JSON/JSONL file."""
    prompts: List[str] = []
    for item in _iter_json_items(Path(path)):
        prompt = _extract_prompt(item, turn_index)
        if prompt is None:
            continue
        prompts.append(prompt)
        if max_samples is not None and len(prompts) >= max_samples:
            break
    return prompts
