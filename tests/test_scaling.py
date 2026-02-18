from robust_mckp.model import Option, PricingInstance
from robust_mckp.solver import solve


def _make_instance(n: int) -> PricingInstance:
    items = [
        [
            Option(1.0, 1.0, 0.1),
            Option(1.5, 0.8, 0.1),
            Option(2.0, 0.4, 0.1),
        ]
        for _ in range(n)
    ]
    return PricingInstance(items=items, gamma=n // 5)


def test_scaling_gap_decreases():
    sol_100 = solve(_make_instance(100))
    sol_1000 = solve(_make_instance(1000))

    assert sol_100.is_feasible
    assert sol_1000.is_feasible
    assert sol_1000.gap_to_lp <= sol_100.gap_to_lp + 1e-12

