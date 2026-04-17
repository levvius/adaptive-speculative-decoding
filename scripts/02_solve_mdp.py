from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from jointadaspec.inference import JointAdaSpecPolicy
from jointadaspec.mdp import MDPConfig, estimate_mdp_parameters, solve_mdp
from jointadaspec.utils import ExperimentLogger


def _experiment_cfg(cfg: DictConfig) -> DictConfig:
    return cfg.experiments if "experiments" in cfg else cfg


def _make_output_dir(cfg: DictConfig) -> Path:
    if cfg.output_dir:
        return Path(get_original_cwd()) / str(cfg.output_dir)
    root = Path(get_original_cwd()) / str(cfg.output_root)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return root / str(cfg.experiment_name) / timestamp


@hydra.main(version_base=None, config_path="../configs", config_name="experiments/default")
def main(cfg: DictConfig) -> None:
    exp_cfg = _experiment_cfg(cfg)
    traces_path = exp_cfg.traces_path
    if not traces_path:
        raise ValueError("Set traces_path=/path/to/traces.parquet before solving the MDP.")

    output_dir = _make_output_dir(exp_cfg)
    logger = ExperimentLogger(output_dir=output_dir, config=cfg)
    base_config = MDPConfig.from_mapping(exp_cfg)
    kappa_values = list(exp_cfg.get("kappa_values", [base_config.kappa]))
    produced_policies: list[str] = []

    for kappa in kappa_values:
        mdp_config = replace(base_config, kappa=float(kappa))
        estimate = estimate_mdp_parameters(traces_path=traces_path, config=mdp_config)
        V_star, pi_star, solve_log = solve_mdp(
            transitions=estimate.transitions,
            rewards=estimate.rewards,
            config=mdp_config,
        )
        policy = JointAdaSpecPolicy(config=mdp_config, pi_star=pi_star, V_star=V_star)
        suffix = f"_kappa_{float(kappa):g}" if len(kappa_values) > 1 else ""
        policy_path = output_dir / f"policy{suffix}.npz"
        solve_log_path = output_dir / f"solve_log{suffix}.json"
        policy.save(policy_path)
        solve_log_path.write_text(json.dumps(solve_log, indent=2), encoding="utf-8")
        produced_policies.append(str(policy_path))
        logger.log(
            {
                "stage": "solve_mdp",
                "kappa": float(kappa),
                "policy_path": str(policy_path),
                "solve_log_path": str(solve_log_path),
            }
        )

    logger.finalize(
        {
            "stage": "solve_mdp",
            "traces_path": str(traces_path),
            "policy_paths": produced_policies,
        }
    )
    for policy_path in produced_policies:
        print(policy_path)


if __name__ == "__main__":
    main()
