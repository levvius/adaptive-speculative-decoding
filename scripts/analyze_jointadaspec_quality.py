from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover - scipy is a project dependency, keep fallback for reports
    binomtest = None


def _bootstrap_ci(values: np.ndarray, *, seed: int, n_resamples: int = 10000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        picks = rng.integers(0, values.size, size=values.size)
        samples[idx] = float(values[picks].mean())
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _mcnemar_pvalue(wins: int, losses: int) -> float | None:
    total = wins + losses
    if total == 0:
        return 1.0
    if binomtest is None:
        return None
    return float(binomtest(min(wins, losses), total, 0.5).pvalue)


def _method_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, group in sorted(frame.groupby("decoder")):
        rows.append(
            {
                "method": str(method),
                "runs": int(len(group)),
                "tokens_per_sec": float(group["tokens_per_sec"].mean()),
                "acceptance_rate": float(group["acceptance_rate"].mean()),
                "gsm8k_exact_match": float(group["gsm8k_exact_match"].mean()),
            }
        )
    return rows


def _paired_comparisons(
    frame: pd.DataFrame,
    *,
    baseline: str,
    methods: Iterable[str],
) -> list[dict[str, object]]:
    wide = frame.pivot_table(
        index=["seed", "prompt_idx"],
        columns="decoder",
        values="gsm8k_exact_match",
        aggfunc="first",
    )
    rows: list[dict[str, object]] = []
    for method in methods:
        if method == baseline or method not in wide.columns or baseline not in wide.columns:
            continue
        paired = wide[[baseline, method]].dropna()
        if paired.empty:
            continue
        diff = (paired[method] - paired[baseline]).to_numpy(dtype=np.float64)
        wins = int(((paired[method] == 1.0) & (paired[baseline] == 0.0)).sum())
        losses = int(((paired[method] == 0.0) & (paired[baseline] == 1.0)).sum())
        ties = int((paired[method] == paired[baseline]).sum())
        low, high = _bootstrap_ci(diff, seed=20260508 + len(method))
        rows.append(
            {
                "method": method,
                "baseline": baseline,
                "paired_n": int(len(paired)),
                "mean_diff": float(diff.mean()),
                "ci_low": low,
                "ci_high": high,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "mcnemar_p": _mcnemar_pvalue(wins, losses),
            }
        )
    return rows


def _fmt_pct(value: object) -> str:
    if value is None:
        return "-"
    number = float(value)
    if np.isnan(number):
        return "-"
    return f"{100.0 * number:.2f}%"


def _fmt_float(value: object) -> str:
    if value is None:
        return "-"
    number = float(value)
    if np.isnan(number):
        return "-"
    return f"{number:.4f}"


def write_report(
    *,
    benchmark_path: Path,
    out_path: Path,
    baseline: str,
    primary: str,
    controls: list[str],
) -> dict[str, object]:
    frame = pd.read_csv(benchmark_path)
    required = {"decoder", "seed", "prompt_idx", "tokens_per_sec", "acceptance_rate", "gsm8k_exact_match"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Benchmark CSV missing required columns: {', '.join(missing)}")

    methods = list(dict.fromkeys([primary, *controls]))
    summary = _method_summary(frame)
    comparisons = _paired_comparisons(frame, baseline=baseline, methods=methods)
    primary_cmp = next((row for row in comparisons if row["method"] == primary), None)
    success = bool(primary_cmp and float(primary_cmp["mean_diff"]) > 0.0)
    strong_success = bool(primary_cmp and float(primary_cmp["ci_low"]) > 0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# JointAdaSpec Quality Analysis",
        "",
        f"- Benchmark: `{benchmark_path}`",
        f"- Primary method: `{primary}`",
        f"- Baseline: `{baseline}`",
        f"- Primary success: `{success}`",
        f"- Strong success: `{strong_success}`",
        "",
        "## Method Summary",
        "",
        "| Method | Runs | tok/s mean | Acceptance | GSM8K EM |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {method} | {runs} | {tps} | {acc} | {em} |".format(
                method=row["method"],
                runs=row["runs"],
                tps=_fmt_float(row["tokens_per_sec"]),
                acc=_fmt_pct(row["acceptance_rate"]),
                em=_fmt_pct(row["gsm8k_exact_match"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired EM Comparisons",
            "",
            "| Method | Baseline | Paired n | EM diff | 95% CI | Wins | Losses | Ties | p-value |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        pvalue = row["mcnemar_p"]
        pvalue_text = "-" if pvalue is None else f"{float(pvalue):.4f}"
        lines.append(
            "| {method} | {baseline} | {paired_n} | {diff} | [{low}, {high}] | {wins} | {losses} | {ties} | {pvalue} |".format(
                method=row["method"],
                baseline=row["baseline"],
                paired_n=row["paired_n"],
                diff=_fmt_pct(row["mean_diff"]),
                low=_fmt_pct(row["ci_low"]),
                high=_fmt_pct(row["ci_high"]),
                wins=row["wins"],
                losses=row["losses"],
                ties=row["ties"],
                pvalue=pvalue_text,
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "benchmark_path": str(benchmark_path),
        "report_path": str(out_path),
        "baseline": baseline,
        "primary": primary,
        "primary_success": success,
        "primary_strong_success": strong_success,
        "summary": summary,
        "comparisons": comparisons,
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline", default="target_only")
    parser.add_argument("--primary", default="cascade_verif_then_length")
    parser.add_argument(
        "--controls",
        nargs="*",
        default=["target_only", "speculative", "jointadaspec"],
    )
    args = parser.parse_args()
    write_report(
        benchmark_path=args.benchmark,
        out_path=args.out,
        baseline=args.baseline,
        primary=args.primary,
        controls=args.controls,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
