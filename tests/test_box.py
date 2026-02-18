from itertools import product

from robust_mckp.certificate import compute_certificate
from robust_mckp.model import Option, PricingInstance
from robust_mckp.solver import solve


def test_box_gamma_full():
    instance = PricingInstance(
        items=[
            [Option(3.0, 1.0, 0.5), Option(4.0, 0.2, 0.1)],
            [Option(2.0, 0.5, 0.3), Option(5.0, 0.1, 0.4)],
            [Option(1.0, 0.2, 0.2), Option(6.0, 0.0, 0.6)],
        ],
        gamma=3,
    )

    best_val = -1e9
    best_sel = None
    for sel in product(*(range(len(g)) for g in instance.items)):
        cert = compute_certificate(instance, sel)
        if cert >= 0:
            val = sum(instance.items[i][sel[i]].value for i in range(len(sel)))
            if val > best_val:
                best_val = val
                best_sel = list(sel)

    sol = solve(instance)
    assert sol.is_feasible
    assert abs(sol.objective - best_val) < 1e-9
    assert sol.selections == best_sel

