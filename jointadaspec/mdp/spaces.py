"""Discrete state and action spaces for the JointAdaSpec tabular MDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jointadaspec.core.features import (
    confidence_center,
    dequantize,
    dequantize_full,
    quantize,
)


@dataclass(frozen=True)
class JointAction:
    """Single MDP action: draft-length choice and verification threshold."""

    length_action: str
    threshold: float

    @property
    def is_stop(self) -> bool:
        return self.length_action in {"verify", "stop"}

    @property
    def is_verify(self) -> bool:
        return self.length_action in {"verify", "stop"}


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
    quality_risk_K: float = 0.0
    quality_risk_k: float = 0.0
    quality_risk_form: str = "multiplicative"
    # Draft-confidence state axis (outermost). N_C == 1 reproduces the legacy
    # 3-D (H, K, k) state space exactly. The inference-time early-verify gate
    # (conf_gate_tau) is a decoder knob, not part of the MDP identity, so it does
    # not change the state space, traces, solved policy, or config hash.
    N_C: int = 1
    C_max: float = 1.0
    draft_conf_feature: str = "max_prob"

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
        if self.quality_risk_K < 0.0 or self.quality_risk_k < 0.0:
            raise ValueError("quality_risk_K and quality_risk_k must be non-negative.")
        if self.quality_risk_form not in ("multiplicative", "additive"):
            raise ValueError(
                "quality_risk_form must be 'multiplicative' (legacy) or 'additive' (experimental)."
            )
        if self.N_C < 1:
            raise ValueError("N_C must be >= 1 (1 disables the draft-confidence axis).")
        if self.C_max <= 0.0:
            raise ValueError("C_max must be positive.")
        if self.draft_conf_feature not in ("max_prob", "margin"):
            raise ValueError("draft_conf_feature must be 'max_prob' or 'margin'.")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MDPConfig":
        allowed_keys = set(cls.__dataclass_fields__)
        data = {key: value for key, value in dict(mapping).items() if key in allowed_keys}
        if "T_levels" in data:
            data["T_levels"] = tuple(float(value) for value in data["T_levels"])
        return cls(**data)

    @property
    def num_states(self) -> int:
        return self.N_C * self.N_H * self.N_K * (self.gamma_max + 1)

    @property
    def num_actions(self) -> int:
        return 1 + len(self.T_levels)


@dataclass(frozen=True)
class StateSpace:
    config: MDPConfig

    @property
    def num_states(self) -> int:
        return self.config.num_states

    def encode(self, H: float, K: float, k: int, C: float = 0.0) -> int:
        return quantize(H=H, K=K, k=k, config=self.config, C=C)

    def decode(self, state_idx: int) -> tuple[int, int, int]:
        """Return (i_H, i_K, i_k); the draft-confidence bin is stripped."""
        return dequantize(state_idx=state_idx, config=self.config)

    def decode_full(self, state_idx: int) -> tuple[int, int, int, int]:
        """Return (i_H, i_K, i_C, i_k) including the draft-confidence bin."""
        return dequantize_full(state_idx=state_idx, config=self.config)

    def k_of(self, state_idx: int) -> int:
        return self.decode(state_idx)[2]

    def c_of(self, state_idx: int) -> int:
        return self.decode_full(state_idx)[2]

    def C_center(self, i_C: int) -> float:
        return confidence_center(i_C, self.config)


def quality_risk_weight(config: MDPConfig, *, i_K: int, k: int) -> float:
    """State-dependent multiplier for conservative fuzzy-verification penalties.

    Legacy multiplicative form. Breaks Bellman linearity when multiplied with
    action-dependent terms (e.g. d_step). Prefer ``quality_risk_penalty`` with
    ``quality_risk_form='additive'`` for new policies (see Theorem B).
    """
    K_bin_norm = 0.0 if config.N_K <= 1 else float(i_K) / float(config.N_K - 1)
    k_norm = 0.0 if config.gamma_max <= 0 else float(k) / float(config.gamma_max)
    return float(1.0 + config.quality_risk_K * K_bin_norm + config.quality_risk_k * k_norm)


def quality_risk_penalty(config: MDPConfig, *, i_K: int, k: int) -> float:
    """State-dependent additive penalty kept separate from action terms.

    This experimental branch avoids multiplying action-dependent losses by a
    state-only factor, but a state-only reward shift is not generally
    policy-invariant in the discounted MDP.
    """
    K_bin_norm = 0.0 if config.N_K <= 1 else float(i_K) / float(config.N_K - 1)
    k_norm = 0.0 if config.gamma_max <= 0 else float(k) / float(config.gamma_max)
    return float(config.quality_risk_K * K_bin_norm + config.quality_risk_k * k_norm)


def quality_risk_weight_for_state(config: MDPConfig, state_idx: int) -> float:
    state_space = StateSpace(config)
    _, i_K, k = state_space.decode(state_idx)
    return quality_risk_weight(config, i_K=i_K, k=k)


def quality_risk_penalty_for_state(config: MDPConfig, state_idx: int) -> float:
    state_space = StateSpace(config)
    _, i_K, k = state_space.decode(state_idx)
    return quality_risk_penalty(config, i_K=i_K, k=k)


@dataclass(frozen=True)
class ActionSpace:
    config: MDPConfig

    def __post_init__(self) -> None:
        actions: list[JointAction] = [JointAction(length_action="continue", threshold=1.0)]
        for threshold in self.config.T_levels:
            actions.append(JointAction(length_action="verify", threshold=float(threshold)))
        object.__setattr__(self, "actions", tuple(actions))
        object.__setattr__(
            self,
            "verify_action_indices",
            tuple(idx for idx, action in enumerate(actions) if action.is_verify),
        )
        object.__setattr__(
            self,
            "stop_action_indices",
            tuple(idx for idx, action in enumerate(actions) if action.is_verify),
        )
        object.__setattr__(
            self,
            "continue_action_indices",
            tuple(idx for idx, action in enumerate(actions) if action.length_action == "continue"),
        )

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def decode(self, action_idx: int) -> JointAction:
        return self.actions[action_idx]

    def encode(self, length_action: str, threshold: float) -> int:
        if length_action == "stop":
            length_action = "verify"
        if length_action == "continue":
            return self.continue_action_indices[0]
        for idx, action in enumerate(self.actions):
            if action.length_action == length_action and action.threshold == float(threshold):
                return idx
        raise KeyError(f"Unknown action ({length_action}, {threshold}).")

    def valid_action_indices(self, k: int) -> tuple[int, ...]:
        if k >= self.config.gamma_max:
            return self.verify_action_indices
        return tuple(range(self.num_actions))
