"""Tabular policy wrapper for JointAdaSpec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from jointadaspec.mdp.spaces import ActionSpace, MDPConfig, StateSpace


class JointAdaSpecPolicy:
    """O(1) table lookup policy over the discretised MDP state space."""

    def __init__(
        self,
        *,
        config: MDPConfig,
        pi_star: np.ndarray,
        V_star: np.ndarray | None = None,
        Q_star: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.state_space = StateSpace(config)
        self.action_space = ActionSpace(config)
        self.pi_star = np.asarray(pi_star, dtype=np.int32)
        if self.pi_star.shape != (config.num_states,):
            raise ValueError(
                f"Expected pi_star shape {(config.num_states,)}, got {self.pi_star.shape}."
            )
        self.V_star = None if V_star is None else np.asarray(V_star, dtype=np.float64)
        self.Q_star = None if Q_star is None else np.asarray(Q_star, dtype=np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "config": {
                "H_max": self.config.H_max,
                "K_max": self.config.K_max,
                "gamma_max": self.config.gamma_max,
                "N_H": self.config.N_H,
                "N_K": self.config.N_K,
                "T_levels": list(self.config.T_levels),
                "lambda_discount": self.config.lambda_discount,
                "epsilon_convergence": self.config.epsilon_convergence,
                "max_vi_iterations": self.config.max_vi_iterations,
                "kappa": self.config.kappa,
                "alpha_smooth": self.config.alpha_smooth,
                "nu_min": self.config.nu_min,
                "c_time": self.config.c_time,
                "K_init": self.config.K_init,
            }
        }
        payload: dict[str, Any] = {
            "pi_star": self.pi_star,
            "metadata_json": np.array(json.dumps(metadata), dtype=np.str_),
        }
        if self.V_star is not None:
            payload["V_star"] = self.V_star
        if self.Q_star is not None:
            payload["Q_star"] = self.Q_star
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "JointAdaSpecPolicy":
        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata_json"].item()))
        config = MDPConfig.from_mapping(metadata["config"])
        V_star = payload["V_star"] if "V_star" in payload else None
        Q_star = payload["Q_star"] if "Q_star" in payload else None
        return cls(config=config, pi_star=payload["pi_star"], V_star=V_star, Q_star=Q_star)

    def get_action(self, H: float, K: float, k: int) -> tuple[str, float]:
        state_idx = self.state_space.encode(H=H, K=K, k=k)
        action = self.action_space.decode(int(self.pi_star[state_idx]))
        return action.length_action, action.threshold

    def export_threshold_surface(self, k_fixed: int) -> np.ndarray:
        surface = np.full((self.config.N_H, self.config.N_K), np.nan, dtype=np.float64)
        for i_H in range(self.config.N_H):
            for i_K in range(self.config.N_K):
                H = (i_H + 0.5) * self.config.H_max / self.config.N_H
                K = (i_K + 0.5) * self.config.K_max / self.config.N_K
                state_idx = self.state_space.encode(H=H, K=K, k=k_fixed)
                action = self.action_space.decode(int(self.pi_star[state_idx]))
                surface[i_H, i_K] = action.threshold
        return surface

    def export_length_surface(self, k_fixed: int) -> np.ndarray:
        surface = np.full((self.config.N_H, self.config.N_K), np.nan, dtype=np.float64)
        for i_H in range(self.config.N_H):
            for i_K in range(self.config.N_K):
                H = (i_H + 0.5) * self.config.H_max / self.config.N_H
                K = (i_K + 0.5) * self.config.K_max / self.config.N_K
                state_idx = self.state_space.encode(H=H, K=K, k=k_fixed)
                action = self.action_space.decode(int(self.pi_star[state_idx]))
                surface[i_H, i_K] = 1.0 if action.length_action == "continue" else 0.0
        return surface

    def export_surface(self, k_fixed: int) -> np.ndarray:
        return self.export_threshold_surface(k_fixed)
