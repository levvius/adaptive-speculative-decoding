from __future__ import annotations

import json
from pathlib import Path

from reports.templates._common import load_summary_rows, pretty_method_name


def _summary_record(method: str) -> dict[str, object]:
    return {
        "summary": True,
        "method": method,
        "eval_task": "gsm8k",
        "tokens_per_sec_median": 10.0,
        "tokens_per_sec_ci_low": 9.0,
        "tokens_per_sec_ci_high": 11.0,
        "acceptance_rate_median": 0.5,
        "acceptance_rate_ci_low": 0.4,
        "acceptance_rate_ci_high": 0.6,
        "gsm8k_exact_match": 0.55,
        "gsm8k_exact_match_cluster_ci_low": 0.50,
        "gsm8k_exact_match_cluster_ci_high": 0.60,
    }


def test_pretty_method_name_preserves_canonical_cascade_length_name() -> None:
    assert pretty_method_name("cascade_length_then_verif") == "cascade_length_then_verif"
    assert pretty_method_name("cascade_len_then_verif") == "cascade_length_then_verif"


def test_load_summary_rows_preserves_cascade_length_for_ablation(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    records = [
        _summary_record("target_only"),
        _summary_record("cascade_length_then_verif"),
        _summary_record("cascade_verif_then_length"),
    ]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    rows = load_summary_rows([path])

    methods = {row["method"] for row in rows}
    assert "cascade_length_then_verif" in methods
    assert "cascade_verif_then_length" in methods
