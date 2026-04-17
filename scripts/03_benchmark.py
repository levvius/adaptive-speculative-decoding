from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import pandas as pd
import torch

from jointadaspec.baselines import FixedSDDecoder, FuzzySDDecoder, SpecDecPPDecoder, VanillaARDecoder
from jointadaspec.inference import JointAdaSpecDecoder, JointAdaSpecPolicy
from jointadaspec.utils import ExperimentLogger, load_dataset, load_model_pair


def _experiment_cfg(cfg: DictConfig) -> DictConfig:
    return cfg.experiments if "experiments" in cfg else cfg


def _make_output_dir(cfg: DictConfig) -> Path:
    if cfg.output_dir:
        return Path(get_original_cwd()) / str(cfg.output_dir)
    root = Path(get_original_cwd()) / str(cfg.output_root)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return root / str(cfg.experiment_name) / timestamp


def _build_decoders(cfg: DictConfig, policy: JointAdaSpecPolicy | None, target_model, draft_model):
    decoders = {
        "vanilla_ar": VanillaARDecoder(target_model=target_model),
        "fixed_sd": FixedSDDecoder(
            target_model=target_model,
            draft_model=draft_model,
            gamma=int(cfg.fixed_sd_gamma),
        ),
        "specdecpp": SpecDecPPDecoder(
            target_model=target_model,
            draft_model=draft_model,
            gamma_max=int(cfg.fuzzy_sd_gamma),
            entropy_threshold=float(cfg.specdecpp_entropy_threshold),
        ),
    }
    for threshold in cfg.fuzzy_sd_T_grid:
        key = f"fuzzy_sd_T{float(threshold):g}"
        decoders[key] = FuzzySDDecoder(
            target_model=target_model,
            draft_model=draft_model,
            gamma=int(cfg.fuzzy_sd_gamma),
            threshold=float(threshold),
        )
    if policy is not None:
        decoders["jointadaspec"] = JointAdaSpecDecoder(
            target_model=target_model,
            draft_model=draft_model,
            policy=policy,
        )
    return decoders


@hydra.main(version_base=None, config_path="../configs", config_name="experiments/default")
def main(cfg: DictConfig) -> None:
    exp_cfg = _experiment_cfg(cfg)
    output_dir = _make_output_dir(exp_cfg)
    logger = ExperimentLogger(output_dir=output_dir, config=cfg)
    target_model, draft_model = load_model_pair(exp_cfg.model_pairs)
    prompts = load_dataset(exp_cfg.datasets, split="test")
    generator_seed = int(exp_cfg.seed)
    policy = JointAdaSpecPolicy.load(Path(exp_cfg.policy_path)) if exp_cfg.policy_path else None
    decoders = _build_decoders(exp_cfg, policy, target_model, draft_model)
    active_names = set(str(name) for name in exp_cfg.baselines)
    if policy is not None:
        active_names.add("jointadaspec")

    records: list[dict[str, float | int | str]] = []
    for decoder_name, decoder in decoders.items():
        if decoder_name.startswith("fuzzy_sd_T"):
            if "fuzzy_sd" not in active_names:
                continue
        elif decoder_name not in active_names:
            continue
        for prompt_idx, prompt in enumerate(prompts):
            if hasattr(target_model, "tokenizer"):
                prompt_ids = [int(token) for token in target_model.tokenizer.encode(prompt, add_special_tokens=False)]
            else:
                prompt_ids = [ord(ch) % int(getattr(target_model, "vocab_size", 256)) for ch in prompt[:64]] or [0]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(generator_seed + prompt_idx)
            result = decoder.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=int(exp_cfg.datasets.max_new_tokens),
                generator=generator,
            )
            tokens_per_sec = (
                result.n_tokens_generated / (result.total_time_ms / 1000.0)
                if result.total_time_ms > 0
                else 0.0
            )
            record = {
                "decoder": decoder_name,
                "prompt_idx": prompt_idx,
                "n_tokens_generated": result.n_tokens_generated,
                "acceptance_rate": result.acceptance_rate,
                "total_time_ms": result.total_time_ms,
                "tokens_per_sec": tokens_per_sec,
                "n_target_calls": result.n_target_calls,
                "n_draft_calls": result.n_draft_calls,
            }
            records.append(record)
            logger.log(record)

    frame = pd.DataFrame.from_records(records)
    benchmark_path = output_dir / "benchmark.csv"
    frame.to_csv(benchmark_path, index=False)
    summary = frame.groupby("decoder", as_index=False).agg(
        mean_tokens_per_sec=("tokens_per_sec", "mean"),
        mean_acceptance_rate=("acceptance_rate", "mean"),
        runs=("prompt_idx", "count"),
    )
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    if not summary.empty and "vanilla_ar" in set(summary["decoder"]):
        baseline_tps = float(
            summary.loc[summary["decoder"] == "vanilla_ar", "mean_tokens_per_sec"].iloc[0]
        )
        summary["speedup_vs_vanilla"] = summary["mean_tokens_per_sec"].map(
            lambda value: 0.0 if baseline_tps <= 0.0 else float(value) / baseline_tps
        )
        summary.to_csv(summary_path, index=False)

    logger.finalize(
        {
            "stage": "benchmark",
            "benchmark_path": str(benchmark_path),
            "summary_path": str(summary_path),
            "num_records": len(records),
        }
    )
    print(summary_path)


if __name__ == "__main__":
    main()
