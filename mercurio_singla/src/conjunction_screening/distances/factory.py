"""Factory for creating probabilistic distance metrics.

Kept separate so the screening and tree modules never need to import
concrete metric classes directly.
"""

from __future__ import annotations

from .base import ProbabilisticDistance
from .hellinger import HellingerDistance


def create_distance(name: str = "hellinger", **kwargs) -> ProbabilisticDistance:
    """Create a probabilistic distance metric by name.

    Parameters
    ----------
    name : str
        Metric name. Currently only "hellinger" is supported.
    **kwargs
        Additional keyword arguments passed to the metric constructor.

    Returns
    -------
    ProbabilisticDistance
        The requested metric instance.

    Raises
    ------
    ValueError
        If the metric name is unknown.
    """
    name = name.lower()
    if name == "hellinger":
        return HellingerDistance(**kwargs)
    raise ValueError(
        f"Unknown probabilistic distance: {name!r}. "
        "Available: 'hellinger'."
    )


__all__ = ["create_distance", "ProbabilisticDistance", "HellingerDistance"]
