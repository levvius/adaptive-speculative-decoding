from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from reports.templates._common import load_summary_rows, manifest_footnote, pretty_method_name


DISPLAY_NAMES = {
    "target_only": "target_only",
    "speculative": "speculative",
    "fuzzy_sd_T1.5": "fuzzy_sd",
    "adaptive_length": "adaptive_length",
    "cascade_length_then_verif": "cascade_1",
    "cascade_verif_then_length": "cascade_2",
    "jointadaspec": "jointadaspec",
}


def _speedup_ci(row: dict[str, object], baseline_row: dict[str, object]) -> tuple[float | None, float | None]:
    low = row.get("tokens_per_sec_ci_low")
    high = row.get("tokens_per_sec_ci_high")
    base_low = baseline_row.get("tokens_per_sec_ci_low")
    base_high = baseline_row.get("tokens_per_sec_ci_high")
    if None in {low, high, base_low, base_high}:
        return None, None
    if float(base_high) <= 0.0 or float(base_low) <= 0.0:
        return None, None
    return float(low) / float(base_high), float(high) / float(base_low)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot JointAdaSpec ablation speedups.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.input]
    manifest_path = None if args.manifest is None else Path(args.manifest)
    rows = load_summary_rows(input_paths, manifest_path=manifest_path)
    datasets = sorted({row["dataset_name"] for row in rows})
    methods = [
        "target_only",
        "speculative",
        "fuzzy_sd_T1.5",
        "adaptive_length",
        "cascade_length_then_verif",
        "cascade_verif_then_length",
        "jointadaspec",
    ]
    x = np.arange(len(methods))
    width = 0.8 / max(len(datasets), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for dataset_index, dataset_name in enumerate(datasets):
        dataset_rows = {pretty_method_name(str(row["method"])): row for row in rows if row["dataset_name"] == dataset_name}
        baseline_row = dataset_rows.get("target_only")
        if baseline_row is None:
            continue
        values = []
        errors_low = []
        errors_high = []
        for method in methods:
            row = dataset_rows.get(method)
            if row is None:
                values.append(np.nan)
                errors_low.append(0.0)
                errors_high.append(0.0)
                continue
            speedup = float(row["tokens_per_sec_median"]) / float(baseline_row["tokens_per_sec_median"])
            values.append(speedup)
            low, high = _speedup_ci(row, baseline_row)
            if low is None or high is None:
                errors_low.append(0.0)
                errors_high.append(0.0)
            else:
                errors_low.append(speedup - low)
                errors_high.append(high - speedup)
        positions = x + (dataset_index - (len(datasets) - 1) / 2.0) * width
        ax.bar(positions, values, width=width, label=dataset_name, yerr=np.array([errors_low, errors_high]), capsize=3)

    ax.set_title("Ablation Speedup over target_only")
    ax.set_xlabel("Method")
    ax.set_ylabel("Speedup")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[method] for method in methods], rotation=25, ha="right")
    ax.legend()
    fig.text(0.01, 0.01, manifest_footnote(manifest_path, input_paths), fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
