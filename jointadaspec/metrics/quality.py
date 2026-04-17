"""Quality-oriented metrics for JointAdaSpec experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def acceptance_adjusted_accuracy(
    correct_predictions: Sequence[bool],
    acceptance_rates: Sequence[float],
) -> float:
    if len(correct_predictions) != len(acceptance_rates):
        raise ValueError("Inputs must have the same length.")
    if not correct_predictions:
        return 0.0
    weights = np.clip(np.asarray(acceptance_rates, dtype=np.float64), 0.0, 1.0)
    if float(weights.sum()) <= 0.0:
        return float(np.mean(np.asarray(correct_predictions, dtype=np.float64)))
    correct = np.asarray(correct_predictions, dtype=np.float64)
    return float(np.sum(correct * weights) / np.sum(weights))


def tv_distance_to_target(
    target_distribution: Sequence[float],
    approximate_distribution: Sequence[float],
) -> float:
    target = np.asarray(target_distribution, dtype=np.float64)
    approx = np.asarray(approximate_distribution, dtype=np.float64)
    if target.shape != approx.shape:
        raise ValueError("Distributions must have the same shape.")
    if np.any(target < 0) or np.any(approx < 0):
        raise ValueError("Distributions must be non-negative.")
    target = target / target.sum()
    approx = approx / approx.sum()
    return float(0.5 * np.abs(target - approx).sum())
