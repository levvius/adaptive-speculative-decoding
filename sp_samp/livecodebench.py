from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional


def _iter_json_items(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload
        return

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            for item in data["data"]:
                if isinstance(item, dict):
                    yield item
            return
        yield data


def _extract_prompt(item: dict) -> Optional[str]:
    for key in ("prompt", "question", "input", "text", "instruction"):
        if key in item and isinstance(item[key], str):
            return item[key]
    return None


def load_livecodebench(
    path: str, max_samples: Optional[int] = None
) -> List[str]:
    """Load LiveCodeBench prompts from a JSON/JSONL file."""
    prompts: List[str] = []
    for item in _iter_json_items(Path(path)):
        prompt = _extract_prompt(item)
        if prompt is None:
            continue
        prompts.append(prompt)
        if max_samples is not None and len(prompts) >= max_samples:
            break
    return prompts


_LCB_VERSION_FILES: dict = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}


def download_livecodebench(
    output_path: str,
    version_tag: str = "release_v5",
) -> str:
    """Download LiveCodeBench dataset from HF hub and save as JSONL.

    Downloads raw JSONL files directly via ``huggingface_hub`` (no loading
    script required). Requires ``pip install huggingface_hub``.
    Returns the output path.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'huggingface_hub' package is required for LiveCodeBench download. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    files = _LCB_VERSION_FILES.get(version_tag)
    if files is None:
        raise ValueError(
            f"Unknown version_tag '{version_tag}'. "
            f"Valid options: {sorted(_LCB_VERSION_FILES)}"
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fh:
        for fname in files:
            local = hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                filename=fname,
                repo_type="dataset",
            )
            with open(local, "r", encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    record = {
                        "prompt": row.get("question_content") or row.get("prompt", ""),
                        "question_id": row.get("question_id", ""),
                        "contest_date": row.get("contest_date", ""),
                        "difficulty": row.get("difficulty", ""),
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return str(out)
