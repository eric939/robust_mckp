import numpy as np

from robust_mckp.hull import build_upper_hull


def test_hull_collinear_keeps_endpoints():
    costs = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 1.0, 2.0])
    idx = np.array([0, 1, 2])
    hull = build_upper_hull(costs, values, idx)
    assert hull.costs.tolist() == [0.0, 2.0]
    assert hull.values.tolist() == [0.0, 2.0]


def test_hull_merge_equal_cost():
    costs = np.array([0.0, 0.0, 1.0])
    values = np.array([1.0, 2.0, 1.5])
    idx = np.array([0, 1, 2])
    hull = build_upper_hull(costs, values, idx)
    # cost 0 keeps max value 2.0
    assert hull.costs[0] == 0.0
    assert hull.values[0] == 2.0


def test_hull_prune_dominated():
    costs = np.array([0.0, 1.0, 2.0])
    values = np.array([2.0, 1.0, 0.5])
    idx = np.array([0, 1, 2])
    hull = build_upper_hull(costs, values, idx)
    assert hull.costs.tolist() == [0.0]
    assert hull.values.tolist() == [2.0]

