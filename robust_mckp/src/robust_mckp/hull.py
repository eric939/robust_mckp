"""Upper-hull filtering for convexified option sets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .utils import EPS, safe_div


@dataclass
class Hull:
    """Upper hull representation for an item.

    Attributes:
        costs: 1D array of hull vertex costs (c values), increasing.
        values: 1D array of hull vertex values (v values), increasing in cost order.
        option_indices: Original option indices for each hull vertex.
        delta_costs: 1D array of segment lengths between vertices.
        slopes: 1D array of segment slopes Δv/Δc.
    """

    costs: np.ndarray
    values: np.ndarray
    option_indices: np.ndarray
    delta_costs: np.ndarray
    slopes: np.ndarray


@dataclass
class Point:
    cost: float
    value: float
    option_index: int


def merge_equal_cost(points: Sequence[Point]) -> List[Point]:
    """Merge points with equal cost by keeping the maximum value.

    Args:
        points: Iterable of Point objects.

    Returns:
        List of points sorted by cost with duplicates merged.
    """

    if not points:
        return []
    pts = sorted(points, key=lambda p: (p.cost, p.value))
    merged: List[Point] = []
    cur = pts[0]
    for p in pts[1:]:
        if abs(p.cost - cur.cost) <= EPS:
            if p.value > cur.value + EPS:
                cur = p
        else:
            merged.append(cur)
            cur = p
    merged.append(cur)
    return merged


def prune_dominated(points: Sequence[Point]) -> List[Point]:
    """Remove points dominated by lower-cost higher-value points.

    Args:
        points: Points sorted by cost.

    Returns:
        List with strictly increasing values in cost order.
    """

    if not points:
        return []
    filtered: List[Point] = []
    max_val = -np.inf
    for p in points:
        if p.value > max_val + EPS:
            filtered.append(p)
            max_val = p.value
    return filtered


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a.cost - o.cost) * (b.value - o.value) - (a.value - o.value) * (b.cost - o.cost)


def upper_hull(points: Sequence[Point]) -> List[Point]:
    """Compute the upper hull (convex envelope) of 2D points.

    Collinear interior points on the hull boundary are removed.

    Args:
        points: Points sorted by cost.

    Returns:
        List of hull vertices ordered by increasing cost.
    """

    if not points:
        return []
    if len(points) <= 2:
        return list(points)

    pts = list(points)
    upper: List[Point] = []
    for p in pts:
        while len(upper) >= 2:
            cross = _cross(upper[-2], upper[-1], p)
            if cross >= -EPS:
                upper.pop()
            else:
                break
        upper.append(p)
    return upper


def build_upper_hull(costs: np.ndarray, values: np.ndarray, option_indices: np.ndarray) -> Hull:
    """Build upper hull with dominance pruning and segment data.

    Args:
        costs: 1D array of costs.
        values: 1D array of values.
        option_indices: 1D array of option indices aligned with costs/values.

    Returns:
        Hull structure with vertices and segment slopes.
    """

    points = [Point(float(c), float(v), int(j)) for c, v, j in zip(costs, values, option_indices)]
    points = merge_equal_cost(points)
    points = prune_dominated(points)
    points = upper_hull(points)

    if not points:
        return Hull(
            costs=np.array([]),
            values=np.array([]),
            option_indices=np.array([], dtype=int),
            delta_costs=np.array([]),
            slopes=np.array([]),
        )
    if len(points) == 1:
        return Hull(
            costs=np.array([points[0].cost], dtype=float),
            values=np.array([points[0].value], dtype=float),
            option_indices=np.array([points[0].option_index], dtype=int),
            delta_costs=np.array([], dtype=float),
            slopes=np.array([], dtype=float),
        )

    costs_arr = np.array([p.cost for p in points], dtype=float)
    values_arr = np.array([p.value for p in points], dtype=float)
    idx_arr = np.array([p.option_index for p in points], dtype=int)
    delta_costs = np.diff(costs_arr)
    delta_values = np.diff(values_arr)
    slopes = np.array([safe_div(dv, dc, default=0.0) for dv, dc in zip(delta_values, delta_costs)], dtype=float)
    return Hull(costs=costs_arr, values=values_arr, option_indices=idx_arr, delta_costs=delta_costs, slopes=slopes)


