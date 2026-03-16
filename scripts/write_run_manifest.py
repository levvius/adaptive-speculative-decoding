from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


def _run(cmd: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Write environment manifest for benchmark runs.")
    parser.add_argument("--out", required=True, help="Output manifest path.")
    args = parser.parse_args()

    payload = {
        "generated_at": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": _run(["python3", "--version"]),
        "kernel": _run(["uname", "-a"]),
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ]
        ),
    }

    try:
        import torch

        payload["torch_version"] = torch.__version__
        payload["cuda_runtime"] = getattr(torch.version, "cuda", None)
        payload["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)
        else:
            payload["cuda_device_name"] = None
    except ImportError:
        payload["torch_version"] = None
        payload["cuda_runtime"] = None
        payload["cuda_available"] = None
        payload["cuda_device_name"] = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
