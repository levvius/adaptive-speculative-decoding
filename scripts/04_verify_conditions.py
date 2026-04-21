from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jointadaspec.analysis import evaluate_conditions
from jointadaspec.utils.manifest import write_manifest


def _status_tag(passed: bool) -> str:
    return "OK" if passed else "WARN"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify empirical JointAdaSpec MDP conditions.")
    parser.add_argument("--traces", required=True, help="Trace parquet path.")
    parser.add_argument("--policy", required=True, help="Solved JointAdaSpec policy NPZ path.")
    parser.add_argument("--out", default=None, help="Output JSON report path.")
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path("reports") / f"conditions_{datetime.now(UTC).date().isoformat()}.json"
    manifest_path = Path("reports") / "manifests" / f"{out_path.stem}.json"
    write_manifest(
        out_path=manifest_path,
        resolved_config_yaml=json.dumps(
            {
                "traces": args.traces,
                "policy": args.policy,
                "out": str(out_path),
                "bootstrap_resamples": int(args.bootstrap_resamples),
                "seed": int(args.seed),
            },
            indent=2,
        ),
        seed_list=[int(args.seed)],
        traces_path=Path(args.traces),
        policy_path=Path(args.policy),
        start_timestamp=datetime.now(UTC).isoformat(),
    )
    try:
        report = evaluate_conditions(
            traces_path=Path(args.traces),
            policy_path=Path(args.policy),
            out_path=out_path,
            n_resamples=int(args.bootstrap_resamples),
            seed=int(args.seed),
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    checks = report["checks"]
    for key in ["c1", "c2", "c3", "c4", "n1", "n2"]:
        print(f"[{_status_tag(bool(checks[key]['passed']))}] {key.upper()}: {checks[key]}")

    if checks["c1"]["passed"] is False or checks["c2"]["passed"] is False:
        print("[WARN] Monotonicity conditions are only partially supported empirically.", file=sys.stderr)
    if checks["c3"]["fraction_nonnegative"] < 0.85 or checks["c4"]["fraction_nonnegative"] < 0.85:
        print("[WARN] Supermodularity fraction fell below 0.85; treat this as a thesis limitation, not a runtime error.", file=sys.stderr)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
