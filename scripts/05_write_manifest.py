from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jointadaspec.utils.manifest import write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a JointAdaSpec reproducibility manifest.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config-file", default=None, help="Optional path to resolved config YAML.")
    parser.add_argument("--config-yaml", default=None, help="Optional inline config YAML.")
    parser.add_argument("--seed-list", default="42,43,44")
    parser.add_argument("--traces", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--start-timestamp", default=None)
    args = parser.parse_args()

    config_yaml = ""
    if args.config_file:
        config_yaml = Path(args.config_file).read_text(encoding="utf-8")
    elif args.config_yaml:
        config_yaml = str(args.config_yaml)

    write_manifest(
        out_path=Path(args.out),
        resolved_config_yaml=config_yaml,
        seed_list=[int(seed.strip()) for seed in str(args.seed_list).split(",") if seed.strip()],
        traces_path=None if args.traces is None else Path(args.traces),
        policy_path=None if args.policy is None else Path(args.policy),
        start_timestamp=args.start_timestamp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
