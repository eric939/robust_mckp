from itertools import product

from robust_mckp.certificate import compute_certificate
from robust_mckp.model import Option, PricingInstance
from robust_mckp.solver import solve


def test_solver_bruteforce_small():
    instance = PricingInstance(
        items=[
            [
                Option(5.0, 1.0, 0.2),
                Option(6.0, 0.5, 0.4),
                Option(4.0, 1.5, 0.1),
            ],
            [
                Option(4.0, 1.0, 0.3),
                Option(7.0, 0.2, 0.5),
                Option(3.0, 1.2, 0.2),
            ],
        ],
        gamma=1,
    )

    best_val = -1e9
    best_sel = None
    for i, j in product(range(3), range(3)):
        sel = [i, j]
        cert = compute_certificate(instance, sel)
        if cert >= 0:
            val = instance.items[0][i].value + instance.items[1][j].value
            if val > best_val:
                best_val = val
                best_sel = sel

    sol = solve(instance)
    assert sol.is_feasible
    assert abs(sol.objective - best_val) < 1e-9
    assert sol.selections == best_sel

