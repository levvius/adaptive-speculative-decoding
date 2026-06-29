from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import torch

from jointadaspec.mdp import MDPConfig, collect_traces
from jointadaspec.utils import ExperimentLogger, load_dataset, load_model_pair


def _experiment_cfg(cfg: DictConfig) -> DictConfig:
    return cfg.experiments if "experiments" in cfg else cfg


def _make_output_dir(cfg: DictConfig) -> Path:
    if cfg.output_dir:
        return Path(get_original_cwd()) / str(cfg.output_dir)
    root = Path(get_original_cwd()) / str(cfg.output_root)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return root / str(cfg.experiment_name) / timestamp


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


@hydra.main(version_base=None, config_path="../configs", config_name="experiments/default")
def main(cfg: DictConfig) -> None:
    exp_cfg = _experiment_cfg(cfg)
    output_dir = _make_output_dir(exp_cfg)
    logger = ExperimentLogger(output_dir=output_dir, config=cfg)
    mdp_config = MDPConfig.from_mapping(exp_cfg)
    target_model, draft_model = load_model_pair(exp_cfg.model_pairs)
    prompts = load_dataset(exp_cfg.datasets, split="train")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(exp_cfg.seed))

    traces_path = collect_traces(
        target_model=target_model,
        draft_model=draft_model,
        prompts=prompts,
        n_traces=int(exp_cfg.n_traces),
        output_path=output_dir / "traces.parquet",
        config=mdp_config,
        generator=generator,
        resume=bool(exp_cfg.get("trace_resume", _env_bool("TRACE_RESUME", True))),
        checkpoint_every=int(
            exp_cfg.get("trace_checkpoint_every", _env_int("TRACE_CHECKPOINT_EVERY", 1))
        ),
        progress_every=int(
            exp_cfg.get("trace_progress_every", _env_int("TRACE_PROGRESS_EVERY", 5))
        ),
    )
    logger.finalize(
        {
            "stage": "collect_traces",
            "traces_path": str(traces_path),
            "num_prompts": len(prompts),
        }
    )
    print(traces_path)


if __name__ == "__main__":
    main()
