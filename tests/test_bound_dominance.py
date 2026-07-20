from __future__ import annotations

import math

import pytest

from research.bound_dominance import exact_minimax_epigraph_bound
from research.compressed_interval_oracle import CompressedThetaIntervalOracle
from research.novelty_go_no_go import build_small_instance
from research.structural_feasibility_study import bounded_theta_clique_lp


@pytest.mark.parametrize("seed", range(6))
def test_exact_minimax_epigraph_matches_group_envelope_oracle(seed: int) -> None:
    instance = build_small_instance(seed=1000 + seed, n=5, m=4, gamma=2)
    oracle = CompressedThetaIntervalOracle(instance)
    intervals = [
        (0, len(oracle.thetas) - 1),
        (0, len(oracle.thetas) // 2),
        (len(oracle.thetas) // 3, len(oracle.thetas) - 1),
    ]
    for lo, hi in intervals:
        exact = exact_minimax_epigraph_bound(instance, lo, hi)
        approximate = oracle.bound(lo, hi)
        if exact.status == "infeasible":
            assert approximate.upper_bound == float("-inf")
        else:
            # The finite multiplier search is a valid restriction of the
            # exact minimization, so it may only be weaker (larger).
            assert approximate.upper_bound >= exact.upper_bound - 2e-7
            assert approximate.upper_bound == pytest.approx(
                exact.upper_bound, abs=3e-5, rel=2e-9
            )


@pytest.mark.parametrize("seed", range(10))
def test_minimax_bound_is_dominated_by_bounded_threshold_clique_lp(seed: int) -> None:
    instance = build_small_instance(seed=2000 + seed, n=5, m=4, gamma=2)
    oracle = CompressedThetaIntervalOracle(instance)
    intervals = [
        (0, len(oracle.thetas) - 1),
        (0, len(oracle.thetas) // 2),
        (len(oracle.thetas) // 2, len(oracle.thetas) - 1),
    ]
    for lo, hi in intervals:
        exact = exact_minimax_epigraph_bound(instance, lo, hi)
        clique, _seconds, _theta = bounded_theta_clique_lp(
            instance,
            float(oracle.thetas[lo]),
            float(oracle.thetas[hi]),
        )
        if math.isfinite(exact.upper_bound):
            assert exact.upper_bound <= clique + 2e-7 * max(1.0, abs(clique))
