from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_presets(config_dir: Path) -> Dict[str, Dict]:
    models_path = config_dir / "models.json"
    methods_path = config_dir / "methods.json"
    experiments_path = config_dir / "experiments.json"
    presets = {}
    if models_path.exists():
        presets["models"] = _load_json(models_path).get("models", {})
    else:
        presets["models"] = {}
    if methods_path.exists():
        presets["methods"] = _load_json(methods_path).get("methods", {})
    else:
        presets["methods"] = {}
    if experiments_path.exists():
        presets["experiments"] = _load_json(experiments_path).get("experiments", {})
    else:
        presets["experiments"] = {}
    return presets


def _apply_preset(args: argparse.Namespace, preset: Dict) -> None:
    for key, value in preset.items():
        if value is None:
            continue
        if hasattr(args, key):
            setattr(args, key, value)


def _apply_draft_preset(args: argparse.Namespace, preset: Dict) -> None:
    if "hf_model" in preset and preset["hf_model"] is not None:
        args.hf_draft_model = preset["hf_model"]
    if "tokenizer" in preset and preset["tokenizer"] is not None:
        args.draft_tokenizer = preset["tokenizer"]
    if "device" in preset and preset["device"] is not None:
        args.draft_device = preset["device"]
    if "dtype" in preset and preset["dtype"] is not None:
        args.draft_dtype = preset["dtype"]
    if "quant" in preset:
        args.draft_quant = preset["quant"]
    if "bnb_compute_dtype" in preset and preset["bnb_compute_dtype"] is not None:
        args.draft_bnb_compute_dtype = preset["bnb_compute_dtype"]
    if preset.get("trust_remote_code"):
        args.trust_remote_code = True


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--model-preset", type=str, default=None)
    parser.add_argument("--draft-preset", type=str, default=None)
    parser.add_argument("--method-preset", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["baseline", "speculative", "autojudge", "specexec", "both", "all"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--turn-index", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--draft-noise", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--quant", type=str, default=None)
    parser.add_argument("--bnb-compute-dtype", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--draft-tokenizer", type=str, default=None)
    parser.add_argument("--draft-device", type=str, default=None)
    parser.add_argument("--draft-dtype", type=str, default=None)
    parser.add_argument("--draft-quant", type=str, default=None)
    parser.add_argument("--draft-bnb-compute-dtype", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--autojudge-threshold", type=float, default=None)
    parser.add_argument("--autojudge-train-samples", type=int, default=None)
    parser.add_argument("--autojudge-train-steps", type=int, default=None)
    parser.add_argument("--autojudge-train-batch-size", type=int, default=None)
    parser.add_argument("--autojudge-train-lr", type=float, default=None)
    parser.add_argument("--autojudge-audit-ratio", type=float, default=None)
    parser.add_argument("--autojudge-checkpoint", type=str, default=None)
    parser.add_argument("--parallel-branches", type=int, default=None)
    parser.add_argument("--branch-prune-threshold", type=float, default=None)
    parser.add_argument("--require-headless", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speculative sampling CLI runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench_parser = subparsers.add_parser("bench", help="Run benchmark with presets.")
    _add_run_args(bench_parser)

    autojudge_parser = subparsers.add_parser("autojudge", help="Run benchmark in AutoJudge mode.")
    _add_run_args(autojudge_parser)
    specexec_parser = subparsers.add_parser("specexec", help="Run benchmark in SpecExec mode.")
    _add_run_args(specexec_parser)
    list_parser = subparsers.add_parser("list-presets", help="List available presets.")
    list_parser.add_argument("--config-dir", type=str, default="configs")

    return parser


def _handle_list_presets(config_dir: Path) -> int:
    presets = _load_presets(config_dir)
    models = sorted(presets.get("models", {}).keys())
    methods = sorted(presets.get("methods", {}).keys())
    experiments = sorted(presets.get("experiments", {}).keys())
    print("Models:")
    for name in models:
        print(f"  {name}")
    print("Methods:")
    for name in methods:
        print(f"  {name}")
    print("Experiments:")
    for name in experiments:
        print(f"  {name}")
    return 0


def _handle_bench(args: argparse.Namespace) -> int:
    try:
        from benchmarks import bench_speculative
    except (ModuleNotFoundError, ImportError) as exc:
        print(
            "Benchmark dependencies are missing. Install requirements first "
            "(for example, `pip install -r requirements.txt`).",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    config_dir = Path(args.config_dir)
    presets = _load_presets(config_dir)
    model_presets = presets.get("models", {})
    method_presets = presets.get("methods", {})
    experiments = presets.get("experiments", {})

    bench_args = bench_speculative.build_parser().parse_args([])

    if args.experiment:
        if args.experiment not in experiments:
            print(f"Unknown experiment: {args.experiment}", file=sys.stderr)
            return 2
        experiment = experiments[args.experiment]
        target_preset = experiment.get("target_preset")
        draft_preset = experiment.get("draft_preset")
        method_preset = experiment.get("method_preset")
        if target_preset:
            if target_preset not in model_presets:
                print(
                    f"Unknown model preset in experiment: {target_preset}",
                    file=sys.stderr,
                )
                return 2
            _apply_preset(bench_args, model_presets[target_preset])
        if draft_preset:
            if draft_preset not in model_presets:
                print(
                    f"Unknown draft preset in experiment: {draft_preset}",
                    file=sys.stderr,
                )
                return 2
            _apply_draft_preset(bench_args, model_presets[draft_preset])
        if method_preset:
            if method_preset not in method_presets:
                print(
                    f"Unknown method preset in experiment: {method_preset}",
                    file=sys.stderr,
                )
                return 2
            preset = method_presets[method_preset]
            method = preset.get("method")
            if method not in {
                "baseline",
                "speculative",
                "autojudge",
                "specexec",
                "both",
                "all",
            }:
                print(
                    f"Method preset '{method_preset}' is not supported by bench.",
                    file=sys.stderr,
                )
                return 2
            _apply_preset(bench_args, preset)

    if args.model_preset:
        if args.model_preset not in model_presets:
            print(f"Unknown model preset: {args.model_preset}", file=sys.stderr)
            return 2
        _apply_preset(bench_args, model_presets[args.model_preset])

    if args.draft_preset:
        if args.draft_preset not in model_presets:
            print(f"Unknown draft preset: {args.draft_preset}", file=sys.stderr)
            return 2
        _apply_draft_preset(bench_args, model_presets[args.draft_preset])

    if args.method_preset:
        if args.method_preset not in method_presets:
            print(f"Unknown method preset: {args.method_preset}", file=sys.stderr)
            return 2
        preset = method_presets[args.method_preset]
        method = preset.get("method")
        if method not in {
            "baseline",
            "speculative",
            "autojudge",
            "specexec",
            "both",
            "all",
        }:
            print(
                f"Method preset '{args.method_preset}' is not supported by bench.",
                file=sys.stderr,
            )
            return 2
        _apply_preset(bench_args, preset)

    if args.dataset is not None:
        bench_args.dataset = args.dataset
    if args.out is not None:
        bench_args.out = args.out
    if args.method is not None:
        bench_args.method = args.method
    if args.max_samples is not None:
        bench_args.max_samples = args.max_samples
    if args.max_new_tokens is not None:
        bench_args.max_new_tokens = args.max_new_tokens
    if args.turn_index is not None:
        bench_args.turn_index = args.turn_index
    if args.k is not None:
        bench_args.k = args.k
    if args.runs is not None:
        bench_args.runs = args.runs
    if args.vocab_size is not None:
        bench_args.vocab_size = args.vocab_size
    if args.draft_noise is not None:
        bench_args.draft_noise = args.draft_noise
    if args.seed is not None:
        bench_args.seed = args.seed
    if args.device is not None:
        bench_args.device = args.device
    if args.dtype is not None:
        bench_args.dtype = args.dtype
    if args.quant is not None:
        bench_args.quant = args.quant
    if args.bnb_compute_dtype is not None:
        bench_args.bnb_compute_dtype = args.bnb_compute_dtype
    if args.tokenizer is not None:
        bench_args.tokenizer = args.tokenizer
    if args.draft_tokenizer is not None:
        bench_args.draft_tokenizer = args.draft_tokenizer
    if args.draft_device is not None:
        bench_args.draft_device = args.draft_device
    if args.draft_dtype is not None:
        bench_args.draft_dtype = args.draft_dtype
    if args.draft_quant is not None:
        bench_args.draft_quant = args.draft_quant
    if args.draft_bnb_compute_dtype is not None:
        bench_args.draft_bnb_compute_dtype = args.draft_bnb_compute_dtype
    if args.trust_remote_code:
        bench_args.trust_remote_code = True
    if args.use_chat_template:
        bench_args.use_chat_template = True
    if args.system_prompt is not None:
        bench_args.system_prompt = args.system_prompt
    if args.add_special_tokens:
        bench_args.add_special_tokens = True
    if args.autojudge_threshold is not None:
        bench_args.autojudge_threshold = args.autojudge_threshold
    if args.autojudge_train_samples is not None:
        bench_args.autojudge_train_samples = args.autojudge_train_samples
    if args.autojudge_train_steps is not None:
        bench_args.autojudge_train_steps = args.autojudge_train_steps
    if args.autojudge_train_batch_size is not None:
        bench_args.autojudge_train_batch_size = args.autojudge_train_batch_size
    if args.autojudge_train_lr is not None:
        bench_args.autojudge_train_lr = args.autojudge_train_lr
    if args.autojudge_audit_ratio is not None:
        bench_args.autojudge_audit_ratio = args.autojudge_audit_ratio
    if args.autojudge_checkpoint is not None:
        bench_args.autojudge_checkpoint = args.autojudge_checkpoint
    if args.parallel_branches is not None:
        bench_args.parallel_branches = args.parallel_branches
    if args.branch_prune_threshold is not None:
        bench_args.branch_prune_threshold = args.branch_prune_threshold
    if args.require_headless:
        bench_args.require_headless = True

    bench_speculative.run_with_args(bench_args)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "list-presets":
        return _handle_list_presets(Path(args.config_dir))
    if args.command == "bench":
        return _handle_bench(args)
    if args.command == "autojudge":
        args.method = "autojudge"
        return _handle_bench(args)
    if args.command == "specexec":
        args.method = "specexec"
        return _handle_bench(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
