from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_quality_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_jointadaspec_quality.py"
    spec = importlib.util.spec_from_file_location("analyze_jointadaspec_quality_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quality_report_uses_prompt_clusters_and_holm_adjustment(tmp_path) -> None:
    module = _load_quality_module()
    rows: list[dict[str, object]] = []
    for seed in (42, 43):
        for prompt_idx, baseline_em, joint_em, speculative_em in [
            (0, 0.0, 1.0, 0.0),
            (1, 1.0, 1.0, 0.0),
        ]:
            rows.extend(
                [
                    {
                        "decoder": "target_only",
                        "seed": seed,
                        "prompt_idx": prompt_idx,
                        "tokens_per_sec": 10.0,
                        "acceptance_rate": 0.0,
                        "gsm8k_exact_match": baseline_em,
                    },
                    {
                        "decoder": "jointadaspec",
                        "seed": seed,
                        "prompt_idx": prompt_idx,
                        "tokens_per_sec": 8.0,
                        "acceptance_rate": 0.5,
                        "gsm8k_exact_match": joint_em,
                    },
                    {
                        "decoder": "speculative",
                        "seed": seed,
                        "prompt_idx": prompt_idx,
                        "tokens_per_sec": 7.0,
                        "acceptance_rate": 0.4,
                        "gsm8k_exact_match": speculative_em,
                    },
                ]
            )
    benchmark_path = tmp_path / "benchmark.csv"
    pd.DataFrame.from_records(rows).to_csv(benchmark_path, index=False)

    payload = module.write_report(
        benchmark_path=benchmark_path,
        out_path=tmp_path / "report.md",
        baseline="target_only",
        primary="jointadaspec",
        controls=["target_only", "speculative"],
    )

    comparisons = {row["method"]: row for row in payload["comparisons"]}
    assert comparisons["jointadaspec"]["paired_n"] == 4
    assert comparisons["jointadaspec"]["cluster_n"] == 2
    assert "holm_p" in comparisons["jointadaspec"]
    assert "holm_significant_0p05" in comparisons["jointadaspec"]
    # Prompt-clustered p-value (and its Holm adjustment) must be reported alongside
    # the per-row McNemar p-value, with a valid probability range.
    cluster_p = comparisons["jointadaspec"]["cluster_p"]
    assert cluster_p is not None and 0.0 <= float(cluster_p) <= 1.0
    assert "cluster_holm_p" in comparisons["jointadaspec"]
    assert "cluster_holm_significant_0p05" in comparisons["jointadaspec"]


def test_cluster_bootstrap_pvalue_separates_clear_effect_from_null() -> None:
    module = _load_quality_module()
    # Two seeds per prompt; a clean positive effect on every prompt cluster.
    positive = [(p, 1.0) for p in range(20)] + [(p, 1.0) for p in range(20)]
    p_pos = module._cluster_bootstrap_pvalue(positive, seed=0)
    assert p_pos is not None and p_pos < 0.05
    # A symmetric, zero-mean signal across clusters should not be significant.
    null = [(p, 1.0) for p in range(10)] + [(p, -1.0) for p in range(10, 20)]
    p_null = module._cluster_bootstrap_pvalue(null, seed=0)
    assert p_null is not None and p_null > 0.05
    # Fewer than two clusters cannot be bootstrapped.
    assert module._cluster_bootstrap_pvalue([(0, 1.0)], seed=0) is None
