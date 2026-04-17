"""Benchmark metrics for JointAdaSpec experiments."""

from .pareto import build_pareto_front, dominates, hypervolume
from .quality import acceptance_adjusted_accuracy, tv_distance_to_target
from .speed import compute_speedup, compute_tpot, compute_ttft, measure_latency

__all__ = [
    "acceptance_adjusted_accuracy",
    "build_pareto_front",
    "compute_speedup",
    "compute_tpot",
    "compute_ttft",
    "dominates",
    "hypervolume",
    "measure_latency",
    "tv_distance_to_target",
]
