"""Brute-force all-pairs screening for validation.

Computes the probabilistic distance for every pair (i, j) with i < j and
returns them sorted. Used as ground truth when measuring K-d tree screening
accuracy (recall/precision).
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import time
import numpy as np

from ..models import RSO
from ..distances.base import ProbabilisticDistance
from ..spatial import dist_point_to_point
from .results import ScreeningResult


def brute_force_screen(rsos: List[RSO],
                       distance_metric: ProbabilisticDistance) -> ScreeningResult:
    """Screen all pairs by brute force.

    Parameters
    ----------
    rsos : list of RSO
        Catalog of objects.
    distance_metric : ProbabilisticDistance
        Pluggable probabilistic distance metric.

    Returns
    -------
    ScreeningResult
        All pairs (i, j) with i < j, with Euclidean and probabilistic
        distances.
    """
    n = len(rsos)
    if n == 0:
        return ScreeningResult([], [], [], {})

    means = np.stack([r.overall_mean for r in rsos], axis=0)
    uncertainties = [r.uncertainty for r in rsos]

    start = time.perf_counter()
    stats: Dict[str, Any] = {
        "objects_processed": n,
        "method": "brute_force",
        "dh_evals": 0,
        "de_evals": 0,
    }

    pairs: List[Tuple[int, int]] = []
    e_dists: List[float] = []
    p_dists: List[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            de = dist_point_to_point(means[i], means[j])
            stats["de_evals"] += 1
            dp = distance_metric.compute(uncertainties[i], uncertainties[j])
            stats["dh_evals"] += 1
            pairs.append((i, j))
            e_dists.append(de)
            p_dists.append(dp)

    elapsed = time.perf_counter() - start
    stats["elapsed_time"] = elapsed
    stats["total_pairs"] = len(pairs)

    return ScreeningResult(
        pairs=pairs,
        euclidean_distances=e_dists,
        probabilistic_distances=p_dists,
        stats=stats,
    )


__all__ = ["brute_force_screen"]
