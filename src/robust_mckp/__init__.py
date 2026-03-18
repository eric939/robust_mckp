"""Robust MCKP pricing optimization package."""

from .model import Option, PricingInstance, Solution
from .preprocessing import filter_admissible_options, from_pricing_data
from .solver import solve

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Option",
    "PricingInstance",
    "Solution",
    "from_pricing_data",
    "filter_admissible_options",
    "solve",
]
