"""Speed-oriented metrics and timing helpers."""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np
import torch


def _sync(device: str | torch.device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    runs: int = 10,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    for _ in range(warmup):
        fn()

    samples_ms: list[float] = []
    for _ in range(runs):
        _sync(device)
        started = time.perf_counter()
        fn()
        _sync(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    q1, q3 = np.percentile(samples_ms, [25, 75])
    return {
        "median_ms": float(np.median(samples_ms)),
        "iqr_ms": float(q3 - q1),
        "mean_ms": float(np.mean(samples_ms)),
    }


def compute_speedup(reference_latency_ms: float, candidate_latency_ms: float) -> float:
    if reference_latency_ms <= 0 or candidate_latency_ms <= 0:
        raise ValueError("Latencies must be positive.")
    return float(reference_latency_ms / candidate_latency_ms)


def compute_ttft(token_timestamps_ms: Sequence[float]) -> float:
    if not token_timestamps_ms:
        return 0.0
    return float(token_timestamps_ms[0])


def compute_tpot(token_timestamps_ms: Sequence[float]) -> float:
    if len(token_timestamps_ms) <= 1:
        return 0.0
    diffs = np.diff(np.asarray(token_timestamps_ms, dtype=np.float64))
    return float(np.mean(diffs))
