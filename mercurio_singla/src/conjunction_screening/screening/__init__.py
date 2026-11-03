"""Conjunction screening algorithms.

Provides:
- :class:`MercurioSinglaScreening` — K-d tree modified NN search (Algorithm 3)
- :func:`brute_force_screen` — all-pairs ground truth

Both depend only on the :class:`~conjunction_screening.distances.ProbabilisticDistance`
interface, never on a concrete metric.
"""

from .mercurio_singla import MercurioSinglaScreening
from .brute_force import brute_force_screen
from .results import ScreeningResult, PairResult

__all__ = [
    "MercurioSinglaScreening",
    "brute_force_screen",
    "ScreeningResult",
    "PairResult",
]
