"""Shared policy helpers for staged cascade baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from jointadaspec.mdp.spaces import ActionSpace, MDPConfig, StateSpace


def _config_payload(config: MDPConfig) -> dict[str, Any]:
    return {
        "H_max": config.H_max,
        "K_max": config.K_max,
        "gamma_max": config.gamma_max,
        "N_H": config.N_H,
        "N_K": config.N_K,
        "T_levels": list(config.T_levels),
        "lambda_discount": config.lambda_discount,
        "epsilon_convergence": config.epsilon_convergence,
        "max_vi_iterations": config.max_vi_iterations,
        "kappa": config.kappa,
        "alpha_smooth": config.alpha_smooth,
        "nu_min": config.nu_min,
        "c_time": config.c_time,
        "K_init": config.K_init,
    }


class CascadePolicy:
    """Combined staged policy returned by a cascade baseline solve.

    ``length_policy`` and ``verif_policy`` are both stored as action-index
    tables over the shared JointAdaSpec state space. ``pi_star`` is the final
    combined policy after both cascade stages have been solved.
    """

    def __init__(
        self,
        *,
        config: MDPConfig,
        cascade_order: str,
        pi_star: np.ndarray,
        length_policy: np.ndarray,
        verif_policy: np.ndarray,
        V_star: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.cascade_order = str(cascade_order)
        self.state_space = StateSpace(config)
        self.action_space = ActionSpace(config)
        self.pi_star = np.asarray(pi_star, dtype=np.int32)
        self.length_policy = np.asarray(length_policy, dtype=np.int32)
        self.verif_policy = np.asarray(verif_policy, dtype=np.int32)
        expected = (config.num_states,)
        for name, value in {
            "pi_star": self.pi_star,
            "length_policy": self.length_policy,
            "verif_policy": self.verif_policy,
        }.items():
            if value.shape != expected:
                raise ValueError(f"Expected {name} shape {expected}, got {value.shape}.")
        self.V_star = None if V_star is None else np.asarray(V_star, dtype=np.float64)

    def get_action(self, H: float, K: float, k: int) -> tuple[str, float]:
        state_idx = self.state_space.encode(H=H, K=K, k=k)
        action = self.action_space.decode(int(self.pi_star[state_idx]))
        return action.length_action, action.threshold

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "policy_kind": "cascade",
            "cascade_order": self.cascade_order,
            "config": _config_payload(self.config),
        }
        payload: dict[str, Any] = {
            "pi_star": self.pi_star,
            "length_policy": self.length_policy,
            "verif_policy": self.verif_policy,
            "metadata_json": np.array(json.dumps(metadata), dtype=np.str_),
        }
        if self.V_star is not None:
            payload["V_star"] = self.V_star
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "CascadePolicy":
        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata_json"].item()))
        config = MDPConfig.from_mapping(metadata["config"])
        V_star = payload["V_star"] if "V_star" in payload else None
        return cls(
            config=config,
            cascade_order=metadata["cascade_order"],
            pi_star=payload["pi_star"],
            length_policy=payload["length_policy"],
            verif_policy=payload["verif_policy"],
            V_star=V_star,
        )
