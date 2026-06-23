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
