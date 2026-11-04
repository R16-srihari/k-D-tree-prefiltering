"""Brute-force KNN reference implementation.

Computes the probabilistic distance between a query RSO and every other RSO,
then returns the K nearest by probabilistic distance. Used as the reference
solution for validating KD-tree results.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..distances.base import ProbabilisticDistance
from ..models.rso import RSO
from .results import KNNResult, KNNSearchStats, NeighborResult


def brute_force_knn(
    query_rso: RSO,
    all_rsos: list[RSO],
    k: int,
    distance_metric: ProbabilisticDistance,
) -> tuple[KNNResult, KNNSearchStats]:
    """Brute-force KNN search.

    Parameters
    ----------
    query_rso : RSO
        The query RSO (excluded from its own neighbor list).
    all_rsos : list[RSO]
        All RSOs in the catalogue (includes query_rso).
    k : int
        Number of neighbors to return.
    distance_metric : ProbabilisticDistance
        The probabilistic distance measure.

    Returns
    -------
    result : KNNResult
        K nearest neighbors ranked by probabilistic distance.
    stats : KNNSearchStats
        Search instrumentation statistics.
    """
    start_time = time.perf_counter()

    candidates: list[NeighborResult] = []
    distance_evals = 0

    query_pos = query_rso.nominal_position

    for candidate in all_rsos:
        if candidate.object_id == query_rso.object_id:
            continue

        dist = distance_metric.compute(query_rso.uncertainty, candidate.uncertainty)
        distance_evals += 1

        euclidean_dist = float(np.linalg.norm(query_pos - candidate.nominal_position))

        candidates.append(
            NeighborResult(
                object_id=candidate.object_id,
                probabilistic_distance=dist,
                nominal_euclidean_distance=euclidean_dist,
            )
        )

    # Sort by probabilistic distance, then object_id for tie-breaking
    candidates.sort()

    # Take top K
    k = min(k, len(candidates))
    top_k = candidates[:k]

    query_time = time.perf_counter() - start_time

    stats = KNNSearchStats(
        distance_evaluations=distance_evals,
        nodes_visited=0,
        nodes_pruned=0,
        leaves_visited=0,
        query_time_seconds=query_time,
    )

    result = KNNResult(
        query_id=query_rso.object_id,
        neighbors=top_k,
        metric=distance_metric.name,
        component_count=query_rso.uncertainty.n_components,
    )

    return result, stats
