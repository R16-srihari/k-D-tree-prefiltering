"""Pluggable probabilistic distance interface and implementations.

The K-d tree screening algorithm depends only on :class:`ProbabilisticDistance`.
Concrete metrics such as :class:`HellingerDistance` are created through
:func:`create_distance` so the screening code never imports them directly.
"""

from .base import ProbabilisticDistance
from .hellinger import (
    HellingerDistance,
    hellinger_gmm,
    hellinger_gmm_component,
    hellinger_gmm_lp,
    hellinger_gmm_approx,
    _hellinger_gaussian_core,
)
from .factory import create_distance

__all__ = [
    "ProbabilisticDistance",
    "HellingerDistance",
    "create_distance",
    "hellinger_gmm",
    "hellinger_gmm_component",
    "hellinger_gmm_lp",
    "hellinger_gmm_approx",
]
