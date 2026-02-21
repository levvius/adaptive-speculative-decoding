from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class GroupStats:
    method: str
    setting: str
    eval_task: Optional[str]
    gsm8k_eval_mode: Optional[str]
    autojudge_threshold_used: Optional[float]
    topk_rank: Optional[str]
    runs: int
    tokens_per_sec_median: float
    avg_tokens_per_step_median: float
    gsm8k_exact_match_mean: Optional[float]
    speedup_vs_speculative: Optional[float]
    accuracy_delta_vs_speculative: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "setting": self.setting,
            "eval_task": self.eval_task,
            "gsm8k_eval_mode": self.gsm8k_eval_mode,
            "autojudge_threshold_used": self.autojudge_threshold_used,
            "topk_rank": self.topk_rank,
            "runs": self.runs,
            "tokens_per_sec_median": self.tokens_per_sec_median,
            "avg_tokens_per_step_median": self.avg_tokens_per_step_median,
            "gsm8k_exact_match_mean": self.gsm8k_exact_match_mean,
            "speedup_vs_speculative": self.speedup_vs_speculative,
            "accuracy_delta_vs_speculative": self.accuracy_delta_vs_speculative,
        }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "none", "null"}:
            return None
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _read_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    records.append(payload)
    return records


def _setting_key(record: Dict[str, Any]) -> Tuple[str, str]:
    method = str(record.get("method"))
    if method == "autojudge":
        threshold = _safe_float(
            record.get("autojudge_threshold_used", record.get("autojudge_threshold"))
        )
        if threshold is None:
            return method, "threshold=calibrated"
        return method, f"threshold={threshold:g}"
    if method == "topk":
        rank = record.get("topk_rank", "4")
        return method, f"topk_rank={rank}"
    return method, "default"


def _pareto_frontier(rows: List[GroupStats]) -> List[GroupStats]:
    candidates = [r for r in rows if r.gsm8k_exact_match_mean is not None]
    frontier: List[GroupStats] = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            better_or_equal_speed = other.tokens_per_sec_median >= row.tokens_per_sec_median
            better_or_equal_acc = (
                other.gsm8k_exact_match_mean is not None
                and row.gsm8k_exact_match_mean is not None
                and other.gsm8k_exact_match_mean >= row.gsm8k_exact_match_mean
            )
            strictly_better = (
                other.tokens_per_sec_median > row.tokens_per_sec_median
                or (
                    other.gsm8k_exact_match_mean is not None
                    and row.gsm8k_exact_match_mean is not None
                    and other.gsm8k_exact_match_mean > row.gsm8k_exact_match_mean
                )
            )
            if better_or_equal_speed and better_or_equal_acc and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(
        key=lambda r: (
            -(r.gsm8k_exact_match_mean if r.gsm8k_exact_match_mean is not None else -1.0),
            -r.tokens_per_sec_median,
        )
    )
    return frontier


def _to_markdown_table(rows: List[GroupStats]) -> str:
    header = (
        "| Method | Setting | Runs | tok/s median | avg tok/step median | GSM8K EM mean | "
        "Speedup vs speculative | Accuracy delta vs speculative |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        [
            "| "
            + " | ".join(
                [
                    row.method,
                    row.setting,
                    str(row.runs),
                    _format_float(row.tokens_per_sec_median, digits=3),
                    _format_float(row.avg_tokens_per_step_median, digits=3),
                    _format_float(row.gsm8k_exact_match_mean, digits=4),
                    _format_float(row.speedup_vs_speculative, digits=3),
                    _format_float(row.accuracy_delta_vs_speculative, digits=4),
                ]
            )
            + " |\n"
            for row in rows
        ]
    )
    return header + body


def build_report(
    records: List[Dict[str, Any]],
) -> Tuple[List[GroupStats], Dict[str, Any]]:
    ok_runs: List[Dict[str, Any]] = []
    dedup: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if rec.get("summary"):
            continue
        if rec.get("status") != "ok":
            continue
        resume_key = rec.get("resume_key")
        if isinstance(resume_key, str):
            dedup[resume_key] = rec
        else:
            ok_runs.append(rec)
    ok_runs.extend(dedup.values())

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in ok_runs:
        grouped[_setting_key(rec)].append(rec)

    grouped_stats: List[GroupStats] = []
    speculative_ref_speed: Optional[float] = None
    speculative_ref_acc: Optional[float] = None
    for (method, setting), rows in grouped.items():
        tps_values = [_safe_float(r.get("tokens_per_sec")) for r in rows]
        tps_values = [x for x in tps_values if x is not None]
        if not tps_values:
            continue
        avg_step_values = [_safe_float(r.get("avg_tokens_per_step")) for r in rows]
        avg_step_values = [x for x in avg_step_values if x is not None]
        em_values = [_safe_float(r.get("gsm8k_exact_match")) for r in rows]
        em_values = [x for x in em_values if x is not None]
        entry = GroupStats(
            method=method,
            setting=setting,
            eval_task=rows[0].get("eval_task"),
            gsm8k_eval_mode=rows[0].get("gsm8k_eval_mode"),
            autojudge_threshold_used=_safe_float(
                rows[0].get("autojudge_threshold_used", rows[0].get("autojudge_threshold"))
            ),
            topk_rank=str(rows[0].get("topk_rank")) if rows[0].get("topk_rank") is not None else None,
            runs=len(rows),
            tokens_per_sec_median=statistics.median(tps_values),
            avg_tokens_per_step_median=(
                statistics.median(avg_step_values) if avg_step_values else 0.0
            ),
            gsm8k_exact_match_mean=(statistics.mean(em_values) if em_values else None),
            speedup_vs_speculative=None,
            accuracy_delta_vs_speculative=None,
        )
        grouped_stats.append(entry)
        if method == "speculative":
            speculative_ref_speed = entry.tokens_per_sec_median
            speculative_ref_acc = entry.gsm8k_exact_match_mean

    for row in grouped_stats:
        if speculative_ref_speed and speculative_ref_speed > 0:
            row.speedup_vs_speculative = row.tokens_per_sec_median / speculative_ref_speed
        if row.gsm8k_exact_match_mean is not None and speculative_ref_acc is not None:
            row.accuracy_delta_vs_speculative = row.gsm8k_exact_match_mean - speculative_ref_acc

    grouped_stats.sort(
        key=lambda r: (
            {"baseline": 0, "speculative": 1, "autojudge": 2, "topk": 3, "specexec": 4}.get(
                r.method, 99
            ),
            r.setting,
        )
    )

    derived = {
        "total_ok_runs": len(ok_runs),
        "pareto_frontier": [row.to_dict() for row in _pareto_frontier(grouped_stats)],
    }
    return grouped_stats, derived


def _write_outputs(
    out_prefix: Path,
    groups: List[GroupStats],
    derived: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
    inputs: List[str],
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "inputs": inputs,
        "manifest": manifest,
        "groups": [g.to_dict() for g in groups],
        "derived": derived,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "setting",
                "eval_task",
                "gsm8k_eval_mode",
                "autojudge_threshold_used",
                "topk_rank",
                "runs",
                "tokens_per_sec_median",
                "avg_tokens_per_step_median",
                "gsm8k_exact_match_mean",
                "speedup_vs_speculative",
                "accuracy_delta_vs_speculative",
            ],
        )
        writer.writeheader()
        for group in groups:
            writer.writerow(group.to_dict())

    lines: List[str] = []
    lines.append("# AutoJudge Paper-Style Report")
    lines.append("")
    lines.append("## Environment snapshot")
    lines.append("")
    if manifest:
        lines.append("```json")
        lines.append(json.dumps(manifest, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("Manifest not provided.")
    lines.append("")
    lines.append("## Aggregated metrics")
    lines.append("")
    lines.append(_to_markdown_table(groups))
    lines.append("")

    autojudge_rows = [row for row in groups if row.method == "autojudge"]
    topk_rows = [row for row in groups if row.method == "topk"]
    if autojudge_rows:
        lines.append("## AutoJudge threshold sweep")
        lines.append("")
        lines.append(_to_markdown_table(autojudge_rows))
        lines.append("")
    if topk_rows:
        lines.append("## Top-K sweep")
        lines.append("")
        lines.append(_to_markdown_table(topk_rows))
        lines.append("")

    lines.append("## Pareto-like shortlist")
    lines.append("")
    pareto_rows = [GroupStats(**row) for row in derived.get("pareto_frontier", [])]
    if pareto_rows:
        lines.append(_to_markdown_table(pareto_rows))
    else:
        lines.append("No Pareto candidates (missing GSM8K exact-match data).")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `speedup_vs_speculative` is computed from median tokens/sec.")
    lines.append(
        "- `accuracy_delta_vs_speculative` is computed from mean GSM8K exact-match over run records."
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate paper-style AutoJudge benchmark runs.")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL result files.")
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output path prefix without extension (writes .md/.csv/.json).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON manifest file with environment metadata.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    for path in input_paths:
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")

    manifest_payload: Optional[Dict[str, Any]] = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    records = _read_jsonl(input_paths)
    groups, derived = build_report(records)
    _write_outputs(
        out_prefix=Path(args.out_prefix),
        groups=groups,
        derived=derived,
        manifest=manifest_payload,
        inputs=[str(p) for p in input_paths],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
