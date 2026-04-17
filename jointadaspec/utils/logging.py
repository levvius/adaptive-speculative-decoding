"""Structured JSON experiment logging."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional at import time
    OmegaConf = None


def _git_commit_or_none() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")


class ExperimentLogger:
    def __init__(self, output_dir: Path, config: Any) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_path = self.output_dir / "run.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.config_path = self.output_dir / "config.yaml"
        self.git_commit_hash = _git_commit_or_none()
        self.timestamp = datetime.now(UTC).isoformat()
        self.config = config

    def log(self, data: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_commit_hash": self.git_commit_hash,
            **data,
        }
        with self.run_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")

    def finalize(self, summary: dict[str, Any]) -> None:
        payload = {
            "timestamp": self.timestamp,
            "git_commit_hash": self.git_commit_hash,
            **summary,
        }
        self.summary_path.write_text(
            json.dumps(payload, indent=2, default=_json_default, sort_keys=True),
            encoding="utf-8",
        )
        if OmegaConf is not None:
            self.config_path.write_text(OmegaConf.to_yaml(self.config), encoding="utf-8")
        else:
            self.config_path.write_text(str(self.config), encoding="utf-8")
