#!/usr/bin/env python3
"""Generate Yandex/AutoJudge-style report tables from benchmark JSONL.

Produces tables in the format:
  | threshold | accuracy, % | speed, tokens/s | speculative decoding | speedup (ours) |

Outputs .md, .csv, .json (same triple pattern as report_autojudge_paper.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class YandexRow:
    method: str
    setting: str
    threshold: Optional[float]
    accuracy_pct: Optional[float]
    speed_tps: float
    speculative_tps: Optional[float]
    speedup_ours: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "setting": self.setting,
            "threshold": self.threshold,
            "accuracy_pct": self.accuracy_pct,
            "speed_tps": self.speed_tps,
            "speculative_tps": self.speculative_tps,
            "speedup_ours": self.speedup_ours,
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


def _fmt(value: Optional[float], digits: int = 3) -> str:
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
    if method == "consensus_autojudge":
        gate = str(record.get("consensus_gate", "learned"))
        threshold = _safe_float(record.get("consensus_fallback_threshold"))
        features = str(record.get("consensus_features", "ensemble"))
        escalation = "off" if bool(record.get("consensus_disable_escalation")) else "on"
        if gate == "rule":
            return method, f"gate=rule,features={features},escalate={escalation}"
        if threshold is None:
            return method, f"gate=learned,features={features},escalate={escalation}"
        return method, (
            f"gate=learned,fallback={threshold:g},features={features},escalate={escalation}"
        )
    if method == "topk":
        rank = record.get("topk_rank", "4")
        return method, f"topk_rank={rank}"
    return method, "default"


def build_yandex_report(
    records: List[Dict[str, Any]],
    eval_task: Optional[str] = None,
) -> List[YandexRow]:
    ok_runs: List[Dict[str, Any]] = []
    dedup: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if rec.get("summary"):
            continue
        if rec.get("status") != "ok":
            continue
        if eval_task and rec.get("eval_task") != eval_task:
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

    speculative_ref_speed: Optional[float] = None
    rows: List[YandexRow] = []

    for (method, setting), group in grouped.items():
        tps_values = [_safe_float(r.get("tokens_per_sec")) for r in group]
        tps_values = [x for x in tps_values if x is not None]
        if not tps_values:
            continue

        speed = statistics.median(tps_values)

        em_values = [_safe_float(r.get("gsm8k_exact_match")) for r in group]
        em_values = [x for x in em_values if x is not None]
        accuracy_pct = (statistics.mean(em_values) * 100.0) if em_values else None

        threshold: Optional[float] = None
        if method == "autojudge":
            threshold = _safe_float(
                group[0].get("autojudge_threshold_used", group[0].get("autojudge_threshold"))
            )
        if method == "consensus_autojudge":
            threshold = _safe_float(group[0].get("consensus_fallback_threshold"))

        row = YandexRow(
            method=method,
            setting=setting,
            threshold=threshold,
            accuracy_pct=accuracy_pct,
            speed_tps=speed,
            speculative_tps=None,
            speedup_ours=None,
        )
        rows.append(row)
        if method == "speculative":
            speculative_ref_speed = speed

    for row in rows:
        row.speculative_tps = speculative_ref_speed
        if speculative_ref_speed and speculative_ref_speed > 0:
            row.speedup_ours = row.speed_tps / speculative_ref_speed

    rows.sort(
        key=lambda r: (
            {"baseline": 0, "speculative": 1, "autojudge": 2, "consensus_autojudge": 3, "topk": 4, "specexec": 5}.get(
                r.method, 99
            ),
            r.setting,
        )
    )

    return rows


def _param_display(row: YandexRow) -> str:
    """Display the method-specific parameter (threshold for AJ, rank for Top-K)."""
    if row.method == "autojudge" and row.threshold is not None:
        return _fmt(row.threshold)
    if row.method == "consensus_autojudge":
        if row.setting.startswith("gate=rule"):
            return "rule"
        suffixes: List[str] = []
        if "features=d1_only" in row.setting:
            suffixes.append("d1_only")
        if "escalate=off" in row.setting:
            suffixes.append("no_esc")
        base = _fmt(row.threshold) if row.threshold is not None else "learned"
        if suffixes:
            return f"{base}/{'/'.join(suffixes)}"
        return base
    if row.method == "topk" and row.setting.startswith("topk_rank="):
        return row.setting.split("=", 1)[1]
    return "-"


def _to_markdown_table(rows: List[YandexRow], title: str = "") -> str:
    header = (
        "| method | parameter | accuracy, % | speed, tokens/s | "
        "speculative decoding | speedup (ours) |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        [
            "| "
            + " | ".join(
                [
                    row.method,
                    _param_display(row),
                    _fmt(row.accuracy_pct, digits=2),
                    _fmt(row.speed_tps, digits=2),
                    _fmt(row.speculative_tps, digits=2),
                    _fmt(row.speedup_ours, digits=3),
                ]
            )
            + " |\n"
            for row in rows
        ]
    )
    result = ""
    if title:
        result += f"### {title}\n\n"
    return result + header + body


def write_outputs(
    out_prefix: Path,
    rows: List[YandexRow],
    manifest: Optional[Dict[str, Any]],
    inputs: List[str],
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "inputs": inputs,
        "manifest": manifest,
        "rows": [r.to_dict() for r in rows],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "setting",
                "threshold",
                "accuracy_pct",
                "speed_tps",
                "speculative_tps",
                "speedup_ours",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())

    lines: List[str] = []
    lines.append("# Yandex-Style Report")
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
    lines.append("## Results")
    lines.append("")
    lines.append(_to_markdown_table(rows))
    lines.append("")

    autojudge_rows = [row for row in rows if row.method == "autojudge"]
    consensus_rows = [row for row in rows if row.method == "consensus_autojudge"]
    topk_rows = [row for row in rows if row.method == "topk"]
    if autojudge_rows:
        lines.append("## AutoJudge threshold sweep")
        lines.append("")
        lines.append(_to_markdown_table(autojudge_rows))
        lines.append("")
    if consensus_rows:
        lines.append("## Consensus AutoJudge sweep")
        lines.append("")
        lines.append(_to_markdown_table(consensus_rows))
        lines.append("")
    if topk_rows:
        lines.append("## Top-K sweep")
        lines.append("")
        lines.append(_to_markdown_table(topk_rows))
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `accuracy, %` = GSM8K exact-match mean * 100 (or `-` for throughput-only tasks).")
    lines.append("- `speculative decoding` = reference speculative tok/s.")
    lines.append("- `speedup (ours)` = row tok/s / speculative tok/s.")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Yandex-style report from benchmark JSONL.")
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
    parser.add_argument(
        "--eval-task",
        default=None,
        help="Filter records to a specific eval_task (e.g. gsm8k, livecodebench).",
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
    rows = build_yandex_report(records, eval_task=args.eval_task)
    write_outputs(
        out_prefix=Path(args.out_prefix),
        rows=rows,
        manifest=manifest_payload,
        inputs=[str(p) for p in input_paths],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
