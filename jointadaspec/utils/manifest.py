"""Run manifest helpers for reproducible JointAdaSpec experiments."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip() or None
    except Exception:
        return None


def _git_dirty() -> bool:
    status = _run(["git", "status", "--porcelain"])
    return bool(status)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_payload(
    *,
    resolved_config_yaml: str,
    seed_list: list[int],
    traces_path: Path | None = None,
    policy_path: Path | None = None,
    start_timestamp: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "start_timestamp": start_timestamp or datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": _run(["python3", "--version"]),
        "kernel": _run(["uname", "-a"]),
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": _git_dirty(),
        "hostname": socket.gethostname(),
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ]
        ),
        "hydra_config_yaml": resolved_config_yaml,
        "seed_list": [int(seed) for seed in seed_list],
        "trace_path": None if traces_path is None else str(traces_path),
        "policy_path": None if policy_path is None else str(policy_path),
        "trace_parquet_sha256": _sha256(traces_path),
        "policy_npz_sha256": _sha256(policy_path),
    }

    try:
        import numpy

        payload["numpy_version"] = numpy.__version__
    except Exception:
        payload["numpy_version"] = None
    try:
        import scipy

        payload["scipy_version"] = scipy.__version__
    except Exception:
        payload["scipy_version"] = None
    try:
        import hydra

        payload["hydra_version"] = hydra.__version__
    except Exception:
        payload["hydra_version"] = None
    try:
        import transformers

        payload["transformers_version"] = transformers.__version__
    except Exception:
        payload["transformers_version"] = None
    try:
        import torch

        payload["torch_version"] = torch.__version__
        payload["cuda_runtime"] = getattr(torch.version, "cuda", None)
        payload["cuda_available"] = torch.cuda.is_available()
        payload["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        payload["torch_version"] = None
        payload["cuda_runtime"] = None
        payload["cuda_available"] = None
        payload["cuda_device_name"] = None
    return payload


def write_manifest(
    *,
    out_path: Path,
    resolved_config_yaml: str,
    seed_list: list[int],
    traces_path: Path | None = None,
    policy_path: Path | None = None,
    start_timestamp: str | None = None,
) -> Path:
    payload = build_manifest_payload(
        resolved_config_yaml=resolved_config_yaml,
        seed_list=seed_list,
        traces_path=traces_path,
        policy_path=policy_path,
        start_timestamp=start_timestamp,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
