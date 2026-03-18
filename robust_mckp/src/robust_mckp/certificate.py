"""Exact robust feasibility certificate."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .model import PricingInstance
from .utils import EPS, top_gamma


def compute_certificate(instance: PricingInstance, selections: Sequence[int]) -> float:
    """Compute certificate value Z = Σ s_i(x_i) - β(x, Γ).

    Args:
        instance: PricingInstance with original margins and uncertainties.
        selections: Selected option indices per item.

    Returns:
        Certificate value Z.
    """

    if len(selections) != instance.n_items:
        raise ValueError("selections length must match number of items")

    s_vals = np.zeros(instance.n_items, dtype=float)
    t_vals = np.zeros(instance.n_items, dtype=float)
    for i, idx in enumerate(selections):
        option = instance.items[i][idx]
        s_vals[i] = option.margin
        t_vals[i] = abs(option.uncertainty)

    beta = top_gamma(t_vals, instance.gamma)
    return float(s_vals.sum() - beta)


def is_feasible(instance: PricingInstance, selections: Sequence[int]) -> bool:
    """Return True if selections satisfy robust constraint."""

    cert = compute_certificate(instance, selections)
    return cert >= -EPS


