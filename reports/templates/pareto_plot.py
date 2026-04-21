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

from jointadaspec.metrics.pareto import dominates
from reports.templates._common import (
    load_summary_rows,
    manifest_footnote,
    quality_field,
    quality_label,
)


def _output_paths(base_out: Path, dataset_name: str, multi_dataset: bool) -> tuple[Path, Path]:
    stem = base_out.with_suffix("")
    if multi_dataset:
        stem = stem.with_name(f"{stem.name}_{dataset_name}")
    return stem.with_suffix(".pdf"), stem.with_suffix(".png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot JointAdaSpec Pareto frontiers.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="Output PDF path or prefix.")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.input]
    manifest_path = None if args.manifest is None else Path(args.manifest)
    rows = load_summary_rows(input_paths, manifest_path=manifest_path)
    datasets = sorted({row["dataset_name"] for row in rows})
    for dataset_name in datasets:
        dataset_rows = [row for row in rows if row["dataset_name"] == dataset_name]
        q_field = quality_field(dataset_name)
        points = []
        for row in dataset_rows:
            quality = row[q_field] if q_field == "gsm8k_exact_match" else row["acceptance_rate_median"]
            if quality is None:
                continue
            points.append((row, float(row["tokens_per_sec_median"]), float(quality)))
        frontier_methods = set()
        for row, speed, quality in points:
            point = (speed, quality)
            if not any(dominates((other_speed, other_quality), point) for other_row, other_speed, other_quality in points if other_row is not row):
                frontier_methods.add(row["method"])

        fig, ax = plt.subplots(figsize=(9, 6))
        for row, speed, quality in points:
            xerr = None
            if row["tokens_per_sec_ci_low"] is not None and row["tokens_per_sec_ci_high"] is not None:
                xerr = np.array([[speed - float(row["tokens_per_sec_ci_low"])], [float(row["tokens_per_sec_ci_high"]) - speed]])
            y_low = row.get(f"{q_field}_ci_low") if q_field == "gsm8k_exact_match" else row.get("acceptance_rate_ci_low")
            y_high = row.get(f"{q_field}_ci_high") if q_field == "gsm8k_exact_match" else row.get("acceptance_rate_ci_high")
            yerr = None
            if y_low is not None and y_high is not None:
                yerr = np.array([[quality - float(y_low)], [float(y_high) - quality]])
            marker = "o" if row["method"] in frontier_methods else "x"
            ax.errorbar(speed, quality, xerr=xerr, yerr=yerr, fmt=marker, markersize=8, capsize=3, label=row["method"])

        ax.set_title(f"Pareto Frontier: {dataset_name}")
        ax.set_xlabel("Speed (tokens/s)")
        ax.set_ylabel(quality_label(dataset_name))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=8)
        fig.text(0.01, 0.01, manifest_footnote(manifest_path, input_paths), fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        pdf_path, png_path = _output_paths(Path(args.out), dataset_name, len(datasets) > 1)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
