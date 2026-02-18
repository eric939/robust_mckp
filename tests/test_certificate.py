from robust_mckp.certificate import compute_certificate, is_feasible
from robust_mckp.model import Option, PricingInstance


def test_certificate_gamma_zero():
    instance = PricingInstance(
        items=[
            [Option(1.0, 1.0, 0.5)],
            [Option(2.0, 2.0, 1.5)],
        ],
        gamma=0,
    )
    cert = compute_certificate(instance, [0, 0])
    assert abs(cert - 3.0) < 1e-9
    assert is_feasible(instance, [0, 0])


def test_certificate_gamma_full():
    instance = PricingInstance(
        items=[
            [Option(1.0, 1.0, 0.5)],
            [Option(2.0, 2.0, 1.5)],
        ],
        gamma=2,
    )
    cert = compute_certificate(instance, [0, 0])
    # beta = 0.5 + 1.5 = 2.0
    assert abs(cert - 1.0) < 1e-9

