import numpy as np

from robust_mckp.greedy import greedy_lp
from robust_mckp.hull import build_upper_hull
from robust_mckp.rounding import round_lp_solution


def test_rounding_feasible():
    costs1 = np.array([0.0, 1.0])
    values1 = np.array([0.0, 2.0])
    idx1 = np.array([0, 1])
    hull1 = build_upper_hull(costs1, values1, idx1)

    costs2 = np.array([0.0, 1.0])
    values2 = np.array([0.0, 1.0])
    idx2 = np.array([0, 1])
    hull2 = build_upper_hull(costs2, values2, idx2)

    lp = greedy_lp([hull1, hull2], capacity=1.5)
    discrete = round_lp_solution(lp, [hull1, hull2], capacity=1.5)
    assert discrete is not None
    assert discrete.cost <= 1.5 + 1e-9


def test_roundup_repair_downgrade():
    costs1 = np.array([0.0, 1.0])
    values1 = np.array([0.0, 10.0])
    idx1 = np.array([0, 1])
    hull1 = build_upper_hull(costs1, values1, idx1)

    costs2 = np.array([0.0, 1.0])
    values2 = np.array([0.0, 9.0])
    idx2 = np.array([0, 1])
    hull2 = build_upper_hull(costs2, values2, idx2)

    lp = greedy_lp([hull1, hull2], capacity=1.2)
    discrete = round_lp_solution(lp, [hull1, hull2], capacity=1.2)
    assert discrete is not None
    assert discrete.cost <= 1.2 + 1e-9

