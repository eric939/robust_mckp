"""Main solver for the robust MCKP pricing problem."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .certificate import compute_certificate
from .greedy import LPSolution, greedy_lp
from .hull import Hull, build_upper_hull
from .model import PricingInstance, Solution
from .rounding import DiscreteSolution, round_lp_solution
from .utils import EPS


@dataclass
class CandidateResult:
    theta: float
    selections: List[int]
    objective: float
    lp_value: float
    certificate_value: float


def _extract_arrays(instance: PricingInstance) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    v_list: List[np.ndarray] = []
    s_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []
    for group in instance.items:
        v_list.append(np.array([opt.value for opt in group], dtype=float))
        s_list.append(np.array([opt.margin for opt in group], dtype=float))
        t_list.append(np.array([opt.uncertainty for opt in group], dtype=float))
    return v_list, s_list, t_list


def solve(instance: PricingInstance, *, upgrade_completion: bool = True) -> Solution:
    """Solve the robust pricing MCKP via θ-enumeration and hull-greedy.

    The robust constraint is enforced via:
        Σ_i s_i^θ(j(i)) ≥ Γ θ,  where  s_i^θ = s_i - max(0, |t_i| - θ).
    The baseline-slack transform yields a knapsack with capacity
        C^θ = Σ_i max_j s_i^θ(j) - Γ θ.

    Args:
        instance: PricingInstance with per-option (v, s, t).
        upgrade_completion: If True, apply optional upgrade-completion step.

    Returns:
        Solution with best feasible discrete selection found.
    """

    start = time.perf_counter()
    v_list, s_list, t_list = _extract_arrays(instance)

    abs_t = np.concatenate([np.abs(t) for t in t_list])
    candidates = np.unique(np.concatenate([np.array([0.0], dtype=float), abs_t]))
    candidates.sort()

    best: Optional[CandidateResult] = None

    for theta in candidates:
        hulls: List[Hull] = []
        s_star_sum = 0.0
        for i, (v, s, t) in enumerate(zip(v_list, s_list, t_list)):
            penalty = np.maximum(0.0, np.abs(t) - theta)
            s_theta = s - penalty
            j_star = int(np.argmax(s_theta))
            s_star = float(s_theta[j_star])
            s_star_sum += s_star
            costs = s_star - s_theta
            costs = np.maximum(costs, 0.0)
            indices = np.arange(len(v), dtype=int)
            hull = build_upper_hull(costs, v, indices)
            hulls.append(hull)

        capacity = s_star_sum - instance.gamma * theta
        if capacity < -EPS:
            continue

        lp_sol = greedy_lp(hulls, capacity)
        discrete = round_lp_solution(lp_sol, hulls, capacity, upgrade_completion=upgrade_completion)
        if discrete is None:
            continue

        selections = [
            int(hulls[i].option_indices[idx]) for i, idx in enumerate(discrete.vertices)
        ]

        cert = compute_certificate(instance, selections)
        if cert < -EPS:
            continue

        objective = float(sum(v_list[i][sel] for i, sel in enumerate(selections)))
        lp_value = float(lp_sol.lp_value)

        if best is None or objective > best.objective + EPS:
            best = CandidateResult(
                theta=float(theta),
                selections=selections,
                objective=objective,
                lp_value=lp_value,
                certificate_value=float(cert),
            )

    elapsed = time.perf_counter() - start
    if best is None:
        return Solution(
            selections=[-1 for _ in range(instance.n_items)],
            objective=float("-inf"),
            lp_value=float("-inf"),
            gap_to_lp=float("inf"),
            certificate_value=float("-inf"),
            theta=float("nan"),
            elapsed=elapsed,
            is_feasible=False,
            metadata={"message": "No feasible solution found"},
        )

    if abs(best.lp_value) <= EPS:
        gap = 0.0 if abs(best.objective) <= EPS else float("inf")
    else:
        gap = (best.lp_value - best.objective) / best.lp_value

    return Solution(
        selections=best.selections,
        objective=best.objective,
        lp_value=best.lp_value,
        gap_to_lp=float(gap),
        certificate_value=best.certificate_value,
        theta=best.theta,
        elapsed=elapsed,
        is_feasible=True,
        metadata=None,
    )
