"""Pareto-front helpers for speed/quality trade-off analysis."""

from __future__ import annotations

from typing import Iterable, Sequence


def dominates(lhs: Sequence[float], rhs: Sequence[float]) -> bool:
    if len(lhs) != len(rhs):
        raise ValueError("Points must have the same dimensionality.")
    lhs_values = [float(value) for value in lhs]
    rhs_values = [float(value) for value in rhs]
    return all(l >= r for l, r in zip(lhs_values, rhs_values)) and any(
        l > r for l, r in zip(lhs_values, rhs_values)
    )


def build_pareto_front(points: Iterable[Sequence[float]]) -> list[tuple[float, ...]]:
    materialised = [tuple(float(value) for value in point) for point in points]
    front: list[tuple[float, ...]] = []
    for point in materialised:
        if any(dominates(other, point) for other in materialised if other is not point):
            continue
        front.append(point)
    return sorted(front)


def hypervolume(front: Iterable[Sequence[float]], reference_point: Sequence[float]) -> float:
    ref_x, ref_y = (float(reference_point[0]), float(reference_point[1]))
    sorted_front = sorted((tuple(float(v) for v in point) for point in front), key=lambda p: p[0])
    if not sorted_front:
        return 0.0
    xs = [ref_x] + [point[0] for point in sorted_front]
    area = 0.0
    for idx in range(len(xs) - 1):
        left = xs[idx]
        right = xs[idx + 1]
        width = max(0.0, right - left)
        active = [point[1] for point in sorted_front if point[0] >= right]
        height = max(0.0, max(active, default=ref_y) - ref_y)
        area += width * height
    return area
