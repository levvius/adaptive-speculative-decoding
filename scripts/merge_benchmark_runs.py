"""Merge JointAdaSpec benchmark CSVs from separate policy runs into one frame.

Each ``scripts/03_benchmark.py`` run benchmarks a single joint policy, so the
3-D baseline policy and the draft-confidence policy land in different
``benchmark.csv`` files. When both runs use the same seeds and held-out prompt
window, their rows are paired by ``(seed, prompt_idx)``. This helper concatenates
a base run with one or more variant runs — relabelling each variant's joint
decoder so the prompt-clustered analyzer in ``analyze_jointadaspec_quality.py``
can compare ``jointadaspec_conf`` against ``jointadaspec`` (3-D), ``cascade``,
and ``target_only`` in a single paired/clustered table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_KEY_COLUMNS = ("seed", "prompt_idx")


def merge_runs(
    *,
    base_path: Path,
    variants: list[tuple[Path, str, str]],
) -> pd.DataFrame:
    """Concatenate a base benchmark CSV with relabelled variant decoders.

    Parameters
    ----------
    base_path:
        Benchmark CSV kept verbatim (all decoders).
    variants:
        ``(csv_path, source_decoder, new_label)`` triples. From each variant CSV
        only ``source_decoder`` rows are kept and renamed to ``new_label``.
    """
    base = pd.read_csv(base_path)
    for column in _KEY_COLUMNS + ("decoder",):
        if column not in base.columns:
            raise ValueError(f"Base benchmark {base_path} is missing column '{column}'.")
    base_keys = set(map(tuple, base[list(_KEY_COLUMNS)].drop_duplicates().to_numpy()))

    frames = [base]
    for csv_path, source_decoder, new_label in variants:
        frame = pd.read_csv(csv_path)
        if "decoder" not in frame.columns:
            raise ValueError(f"Variant benchmark {csv_path} is missing column 'decoder'.")
        subset = frame[frame["decoder"] == source_decoder].copy()
        if subset.empty:
            raise ValueError(
                f"Variant benchmark {csv_path} has no rows for decoder '{source_decoder}'."
            )
        subset["decoder"] = new_label
        variant_keys = set(map(tuple, subset[list(_KEY_COLUMNS)].drop_duplicates().to_numpy()))
        missing = base_keys - variant_keys
        if missing:
            raise ValueError(
                f"Variant '{new_label}' from {csv_path} is missing {len(missing)} "
                f"(seed, prompt_idx) pairs present in the base run; runs are not paired."
            )
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def _parse_variant(value: str) -> tuple[Path, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "variant must be 'csv_path:source_decoder:new_label'."
        )
    return Path(parts[0]), parts[1], parts[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        type=_parse_variant,
        metavar="CSV:SRC_DECODER:NEW_LABEL",
        help="Repeatable. Keep SRC_DECODER rows from CSV, relabelled to NEW_LABEL.",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    merged = merge_runs(base_path=args.base, variants=args.variant)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
