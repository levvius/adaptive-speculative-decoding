"""Discrete state and action spaces for the JointAdaSpec tabular MDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jointadaspec.core.features import dequantize, quantize


@dataclass(frozen=True)
class JointAction:
    """Single MDP action: draft-length choice and verification threshold."""

    length_action: str
    threshold: float

    @property
    def is_stop(self) -> bool:
        return self.length_action == "stop"


@dataclass(frozen=True)
class MDPConfig:
    H_max: float = 6.0
    K_max: float = 8.0
    gamma_max: int = 8
    N_H: int = 20
    N_K: int = 20
    T_levels: tuple[float, ...] = (1.0, 1.22, 1.49, 1.82, 2.22, 2.71, 3.3, 4.0)
    lambda_discount: float = 0.99
    epsilon_convergence: float = 1.0e-4
    max_vi_iterations: int = 10_000
    kappa: float = 1.0
    alpha_smooth: float = 1.0
    nu_min: int = 5
    c_time: float = 0.01
    K_init: float = 0.0

    def __post_init__(self) -> None:
        if self.H_max <= 0 or self.K_max <= 0:
            raise ValueError("H_max and K_max must be positive.")
        if self.gamma_max < 0:
            raise ValueError("gamma_max must be non-negative.")
        if self.N_H <= 0 or self.N_K <= 0:
            raise ValueError("N_H and N_K must be positive.")
        if not self.T_levels:
            raise ValueError("T_levels must be non-empty.")
        if any(level < 1.0 for level in self.T_levels):
            raise ValueError("All T_levels must be >= 1.0.")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MDPConfig":
        allowed_keys = set(cls.__dataclass_fields__)
        data = {key: value for key, value in dict(mapping).items() if key in allowed_keys}
        if "T_levels" in data:
            data["T_levels"] = tuple(float(value) for value in data["T_levels"])
        return cls(**data)

    @property
    def num_states(self) -> int:
        return self.N_H * self.N_K * (self.gamma_max + 1)

    @property
    def num_actions(self) -> int:
        return 2 * len(self.T_levels)


@dataclass(frozen=True)
class StateSpace:
    config: MDPConfig

    @property
    def num_states(self) -> int:
        return self.config.num_states

    def encode(self, H: float, K: float, k: int) -> int:
        return quantize(H=H, K=K, k=k, config=self.config)

    def decode(self, state_idx: int) -> tuple[int, int, int]:
        return dequantize(state_idx=state_idx, config=self.config)

    def k_of(self, state_idx: int) -> int:
        return self.decode(state_idx)[2]


@dataclass(frozen=True)
class ActionSpace:
    config: MDPConfig

    def __post_init__(self) -> None:
        actions: list[JointAction] = []
        for threshold in self.config.T_levels:
            actions.append(JointAction(length_action="stop", threshold=float(threshold)))
        for threshold in self.config.T_levels:
            actions.append(JointAction(length_action="continue", threshold=float(threshold)))
        object.__setattr__(self, "actions", tuple(actions))
        object.__setattr__(
            self,
            "stop_action_indices",
            tuple(idx for idx, action in enumerate(actions) if action.is_stop),
        )
        object.__setattr__(
            self,
            "continue_action_indices",
            tuple(idx for idx, action in enumerate(actions) if not action.is_stop),
        )

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def decode(self, action_idx: int) -> JointAction:
        return self.actions[action_idx]

    def encode(self, length_action: str, threshold: float) -> int:
        for idx, action in enumerate(self.actions):
            if action.length_action == length_action and action.threshold == float(threshold):
                return idx
        raise KeyError(f"Unknown action ({length_action}, {threshold}).")

    def valid_action_indices(self, k: int) -> tuple[int, ...]:
        if k >= self.config.gamma_max:
            return self.stop_action_indices
        return tuple(range(self.num_actions))
