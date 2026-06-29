"""Tests for merging separate JointAdaSpec benchmark runs into one paired frame."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_merge_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "merge_benchmark_runs.py"
    spec = importlib.util.spec_from_file_location("merge_benchmark_runs_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(decoders: list[str], prompts: int = 3, seeds: tuple[int, ...] = (42, 43)) -> pd.DataFrame:
    rows = []
    for decoder in decoders:
        for seed in seeds:
            for prompt_idx in range(prompts):
                rows.append(
                    {
                        "decoder": decoder,
                        "seed": seed,
                        "prompt_idx": prompt_idx,
                        "gsm8k_exact_match": 1.0 if decoder == "jointadaspec" else 0.0,
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_merge_relabels_variant_and_keeps_base(tmp_path) -> None:
    module = _load_merge_module()
    base = tmp_path / "base.csv"
    variant = tmp_path / "variant.csv"
    _frame(["target_only", "cascade_verif_then_length", "jointadaspec"]).to_csv(base, index=False)
    _frame(["target_only", "jointadaspec"]).to_csv(variant, index=False)

    merged = module.merge_runs(
        base_path=base,
        variants=[(variant, "jointadaspec", "jointadaspec_conf")],
    )
    decoders = set(merged["decoder"].unique())
    assert decoders == {
        "target_only",
        "cascade_verif_then_length",
        "jointadaspec",
        "jointadaspec_conf",
    }
    # The base 'jointadaspec' is preserved, and exactly one relabelled copy is added.
    assert int((merged["decoder"] == "jointadaspec").sum()) == 6
    assert int((merged["decoder"] == "jointadaspec_conf").sum()) == 6


def test_merge_rejects_unpaired_variant(tmp_path) -> None:
    module = _load_merge_module()
    base = tmp_path / "base.csv"
    variant = tmp_path / "variant.csv"
    _frame(["target_only", "jointadaspec"], prompts=4).to_csv(base, index=False)
    _frame(["jointadaspec"], prompts=2).to_csv(variant, index=False)  # fewer prompts

    with pytest.raises(ValueError, match="not paired"):
        module.merge_runs(
            base_path=base,
            variants=[(variant, "jointadaspec", "jointadaspec_conf")],
        )


def test_merge_rejects_missing_source_decoder(tmp_path) -> None:
    module = _load_merge_module()
    base = tmp_path / "base.csv"
    variant = tmp_path / "variant.csv"
    _frame(["target_only", "jointadaspec"]).to_csv(base, index=False)
    _frame(["target_only"]).to_csv(variant, index=False)

    with pytest.raises(ValueError, match="no rows for decoder"):
        module.merge_runs(
            base_path=base,
            variants=[(variant, "jointadaspec", "jointadaspec_conf")],
        )
