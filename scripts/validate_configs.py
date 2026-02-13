from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


BENCH_METHODS = {"baseline", "speculative", "autojudge", "both", "all", "specexec"}
DRAFT_REQUIRED_METHODS = {"speculative", "autojudge", "both", "all", "specexec"}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _get_table(payload: Dict[str, Any], key: str, path: Path) -> Dict[str, Dict[str, Any]]:
    table = payload.get(key, {})
    if not isinstance(table, dict):
        raise ValueError(f"Expected '{key}' to be an object in {path}")
    output: Dict[str, Dict[str, Any]] = {}
    for name, preset in table.items():
        if not isinstance(preset, dict):
            raise ValueError(f"Preset '{name}' in {path} must be an object.")
        output[str(name)] = preset
    return output


def validate_config_dir(config_dir: Path) -> Tuple[List[str], List[str], Dict[str, int]]:
    errors: List[str] = []
    warnings: List[str] = []

    models_payload = _load_json(config_dir / "models.json")
    methods_payload = _load_json(config_dir / "methods.json")
    experiments_payload = _load_json(config_dir / "experiments.json")

    models = _get_table(models_payload, "models", config_dir / "models.json")
    methods = _get_table(methods_payload, "methods", config_dir / "methods.json")
    experiments = _get_table(experiments_payload, "experiments", config_dir / "experiments.json")

    for method_name, method_preset in methods.items():
        method = method_preset.get("method")
        if method not in BENCH_METHODS:
            errors.append(
                f"methods.{method_name}: unsupported method '{method}'. "
                f"Allowed: {sorted(BENCH_METHODS)}."
            )
            continue
        if method in DRAFT_REQUIRED_METHODS:
            k = method_preset.get("k", None)
            if not isinstance(k, int) or k <= 0:
                errors.append(
                    f"methods.{method_name}: method '{method}' requires positive integer k."
                )
        if method == "baseline" and "k" in method_preset and method_preset.get("k") not in (0, None):
            warnings.append(
                f"methods.{method_name}: baseline ignores k, current value={method_preset.get('k')}."
            )

    for exp_name, exp in experiments.items():
        target_preset = exp.get("target_preset")
        draft_preset = exp.get("draft_preset")
        method_preset = exp.get("method_preset")

        if not target_preset:
            errors.append(f"experiments.{exp_name}: missing target_preset.")
            continue
        if target_preset not in models:
            errors.append(
                f"experiments.{exp_name}: target_preset '{target_preset}' not found in models."
            )
            continue

        if not method_preset:
            errors.append(f"experiments.{exp_name}: missing method_preset.")
            continue
        if method_preset not in methods:
            errors.append(
                f"experiments.{exp_name}: method_preset '{method_preset}' not found in methods."
            )
            continue

        method = methods[method_preset].get("method")
        if method not in BENCH_METHODS:
            errors.append(f"experiments.{exp_name}: method '{method}' is not supported.")
            continue

        if method in DRAFT_REQUIRED_METHODS and not draft_preset:
            errors.append(
                f"experiments.{exp_name}: method '{method}' requires draft_preset."
            )
            continue

        if draft_preset:
            if draft_preset not in models:
                errors.append(
                    f"experiments.{exp_name}: draft_preset '{draft_preset}' not found in models."
                )
                continue
            if method == "baseline":
                warnings.append(
                    f"experiments.{exp_name}: draft_preset is ignored for baseline method."
                )

        if method in DRAFT_REQUIRED_METHODS and draft_preset:
            target = models[target_preset]
            draft = models[draft_preset]

            target_tokenizer = target.get("tokenizer")
            draft_tokenizer = draft.get("tokenizer")
            if target_tokenizer and draft_tokenizer and target_tokenizer != draft_tokenizer:
                errors.append(
                    f"experiments.{exp_name}: tokenizer mismatch "
                    f"('{target_tokenizer}' vs '{draft_tokenizer}')."
                )
            if not target_tokenizer or not draft_tokenizer:
                warnings.append(
                    f"experiments.{exp_name}: tokenizer missing for target or draft preset."
                )

            if target_preset == draft_preset:
                warnings.append(
                    f"experiments.{exp_name}: target and draft use the same preset "
                    f"('{target_preset}'); speedup may be limited."
                )

    stats = {
        "models": len(models),
        "methods": len(methods),
        "experiments": len(experiments),
    }
    return errors, warnings, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate preset configuration consistency.")
    parser.add_argument("--config-dir", type=str, default="configs")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    try:
        errors, warnings, stats = validate_config_dir(config_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(
        "Validated presets: "
        f"models={stats['models']}, methods={stats['methods']}, experiments={stats['experiments']}"
    )
    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[ERROR] {error}", file=sys.stderr)

    if errors:
        print(f"Validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
