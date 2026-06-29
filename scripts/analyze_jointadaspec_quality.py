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


def _cluster_bootstrap_ci(
    prompt_values: list[tuple[int, float]],
    *,
    seed: int,
    n_resamples: int = 10000,
) -> tuple[float, float]:
    if not prompt_values:
        return float("nan"), float("nan")
    clusters: dict[int, list[float]] = {}
    for prompt_idx, value in prompt_values:
        clusters.setdefault(int(prompt_idx), []).append(float(value))
    cluster_means = np.asarray(
        [float(np.mean(values)) for _, values in sorted(clusters.items())],
        dtype=np.float64,
    )
    return _bootstrap_ci(cluster_means, seed=seed, n_resamples=n_resamples)


def _cluster_bootstrap_pvalue(
    prompt_values: list[tuple[int, float]],
    *,
    seed: int,
    n_resamples: int = 10000,
) -> float | None:
    """Two-sided prompt-clustered bootstrap p-value for H0: mean diff == 0.

    Seeds of the same prompt are *not* independent, so the per-row McNemar test
    over-counts evidence. This test resamples whole prompt clusters (mean over the
    seeds of each prompt), shifts the bootstrap distribution to the null, and reports
    the two-sided tail probability. It is the significance companion to the cluster
    CI already produced by :func:`_cluster_bootstrap_ci`.
    """
    clusters: dict[int, list[float]] = {}
    for prompt_idx, value in prompt_values:
        clusters.setdefault(int(prompt_idx), []).append(float(value))
    cluster_means = np.asarray(
        [float(np.mean(values)) for _, values in sorted(clusters.items())],
        dtype=np.float64,
    )
    if cluster_means.size < 2:
        return None
    observed = float(cluster_means.mean())
    rng = np.random.default_rng(seed)
    n = cluster_means.size
    extreme = 0
    for _ in range(n_resamples):
        picks = rng.integers(0, n, size=n)
        centered = float(cluster_means[picks].mean()) - observed  # shift to null
        if abs(centered) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (n_resamples + 1))


def _mcnemar_pvalue(wins: int, losses: int) -> float | None:
    total = wins + losses
    if total == 0:
        return 1.0
    if binomtest is None:
        return None
    return float(binomtest(min(wins, losses), total, 0.5).pvalue)


def _with_holm_adjustment(
    rows: list[dict[str, object]],
    *,
    p_key: str = "mcnemar_p",
    holm_key: str = "holm_p",
    sig_key: str = "holm_significant_0p05",
) -> list[dict[str, object]]:
    indexed_pvalues = [
        (idx, float(row[p_key]))
        for idx, row in enumerate(rows)
        if row.get(p_key) is not None
    ]
    indexed_pvalues.sort(key=lambda item: item[1])
    m = len(indexed_pvalues)
    running_max = 0.0
    for rank, (idx, pvalue) in enumerate(indexed_pvalues, start=1):
        adjusted = min(1.0, (m - rank + 1) * pvalue)
        running_max = max(running_max, adjusted)
        rows[idx][holm_key] = running_max
        rows[idx][sig_key] = running_max <= 0.05
    for row in rows:
        row.setdefault(holm_key, None)
        row.setdefault(sig_key, False)
    return rows


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
        prompt_values = []
        for (_seed, prompt_idx), row_values in paired.iterrows():
            prompt_values.append(
                (int(prompt_idx), float(row_values[method] - row_values[baseline]))
            )
        wins = int(((paired[method] == 1.0) & (paired[baseline] == 0.0)).sum())
        losses = int(((paired[method] == 0.0) & (paired[baseline] == 1.0)).sum())
        ties = int((paired[method] == paired[baseline]).sum())
        low, high = _cluster_bootstrap_ci(prompt_values, seed=20260508 + len(method))
        cluster_p = _cluster_bootstrap_pvalue(prompt_values, seed=20260508 + len(method))
        rows.append(
            {
                "method": method,
                "baseline": baseline,
                "paired_n": int(len(paired)),
                "cluster_n": int(len({prompt_idx for _seed, prompt_idx in paired.index})),
                "mean_diff": float(diff.mean()),
                "ci_low": low,
                "ci_high": high,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "mcnemar_p": _mcnemar_pvalue(wins, losses),
                "cluster_p": cluster_p,
            }
        )
    _with_holm_adjustment(rows)
    _with_holm_adjustment(
        rows,
        p_key="cluster_p",
        holm_key="cluster_holm_p",
        sig_key="cluster_holm_significant_0p05",
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
            "| Method | Baseline | Paired n | Clusters | EM diff | Cluster 95% CI | Wins | Losses | Ties | McNemar p | Cluster p | Cluster Holm p |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        pvalue = row["mcnemar_p"]
        pvalue_text = "-" if pvalue is None else f"{float(pvalue):.4f}"
        cluster_p = row["cluster_p"]
        cluster_p_text = "-" if cluster_p is None else f"{float(cluster_p):.4f}"
        cluster_holm = row["cluster_holm_p"]
        cluster_holm_text = "-" if cluster_holm is None else f"{float(cluster_holm):.4f}"
        lines.append(
            "| {method} | {baseline} | {paired_n} | {cluster_n} | {diff} | [{low}, {high}] | {wins} | {losses} | {ties} | {pvalue} | {cluster_p} | {cluster_holm_p} |".format(
                method=row["method"],
                baseline=row["baseline"],
                paired_n=row["paired_n"],
                cluster_n=row["cluster_n"],
                diff=_fmt_pct(row["mean_diff"]),
                low=_fmt_pct(row["ci_low"]),
                high=_fmt_pct(row["ci_high"]),
                wins=row["wins"],
                losses=row["losses"],
                ties=row["ties"],
                pvalue=pvalue_text,
                cluster_p=cluster_p_text,
                cluster_holm_p=cluster_holm_text,
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
