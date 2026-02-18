import numpy as np

from robust_mckp.greedy import greedy_lp
from robust_mckp.hull import build_upper_hull


def test_greedy_fractional():
    # Item 1: slope 2
    costs1 = np.array([0.0, 1.0])
    values1 = np.array([0.0, 2.0])
    idx1 = np.array([0, 1])
    hull1 = build_upper_hull(costs1, values1, idx1)

    # Item 2: slope 1
    costs2 = np.array([0.0, 1.0])
    values2 = np.array([0.0, 1.0])
    idx2 = np.array([0, 1])
    hull2 = build_upper_hull(costs2, values2, idx2)

    lp = greedy_lp([hull1, hull2], capacity=1.5)
    assert abs(lp.lp_value - 2.5) < 1e-9
    assert lp.fractional_item in (0, 1)


def test_greedy_zero_capacity():
    costs = np.array([0.0, 1.0])
    values = np.array([0.0, 1.0])
    idx = np.array([0, 1])
    hull = build_upper_hull(costs, values, idx)
    lp = greedy_lp([hull], capacity=0.0)
    assert abs(lp.lp_value - 0.0) < 1e-9
    assert lp.positions[0].lower_vertex == 0


def test_greedy_large_capacity():
    costs = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 2.0, 3.0])
    idx = np.array([0, 1, 2])
    hull = build_upper_hull(costs, values, idx)
    lp = greedy_lp([hull], capacity=100.0)
    assert lp.positions[0].lower_vertex == 2
    assert abs(lp.lp_value - 3.0) < 1e-9
