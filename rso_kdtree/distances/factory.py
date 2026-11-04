"""Distance metric factory for runtime metric selection."""

from __future__ import annotations

from .base import SUPPORTED_METRICS, ProbabilisticDistance
from .gaussian_distances import (
    HellingerDistance,
    MahalanobisDistance,
    SymmetricKLDistance,
    WassersteinDistance,
)

_METRIC_CLASSES = {
    "hellinger": HellingerDistance,
    "wasserstein": WassersteinDistance,
    "mahalanobis": MahalanobisDistance,
    "kl": SymmetricKLDistance,
}


def create_distance_metric(metric_name: str) -> ProbabilisticDistance:
    """Create a distance metric instance by name.

    Parameters
    ----------
    metric_name : str
        One of 'wasserstein', 'mahalanobis', 'hellinger', 'kl'.

    Returns
    -------
    ProbabilisticDistance
        The corresponding distance metric instance.

    Raises
    ------
    ValueError
        If the metric name is not supported.
    """
    key = metric_name.lower().strip()
    if key not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric: {metric_name!r}. "
            f"Supported: {sorted(SUPPORTED_METRICS)}"
        )
    return _METRIC_CLASSES[key]()


def list_supported_metrics() -> list[str]:
    """Return the list of supported metric names."""
    return sorted(SUPPORTED_METRICS)
