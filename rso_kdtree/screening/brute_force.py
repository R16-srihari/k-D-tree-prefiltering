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

# Metrics whose GMM distance is symmetric, D(a, b) == D(b, a), so the
# brute-force all-pairs sweep may evaluate each unordered pair once
# (N(N-1)/2 evaluations, per Mercurio & Singla 2016, sec. 3). All currently
# supported metrics qualify: Hellinger, Wasserstein and pooled-covariance
# Mahalanobis are symmetric by construction, and the "kl" implementation is
# the *symmetric* KL divergence. An asymmetric metric must NOT be added here.
SYMMETRIC_METRIC_NAMES = frozenset({"hellinger", "wasserstein", "mahalanobis", "kl"})


def brute_force_knn_all(
    all_rsos: list[RSO],
    k: int,
    distance_metric: ProbabilisticDistance,
) -> tuple[list[KNNResult], KNNSearchStats]:
    """Brute-force KNN for every RSO in the catalogue, exploiting pair symmetry.

    For symmetric metrics each unordered pair (i, j) is evaluated exactly
    once and the value is shared by both directions, giving N(N-1)/2 distance
    evaluations instead of N(N-1) (Mercurio & Singla 2016, sec. 3). Results
    are identical to calling :func:`brute_force_knn` per query.

    Parameters
    ----------
    all_rsos : list[RSO]
        The full catalogue.
    k : int
        Number of neighbors per query.
    distance_metric : ProbabilisticDistance
        The probabilistic distance measure.

    Returns
    -------
    results : list[KNNResult]
        One result per RSO, in catalogue order.
    stats : KNNSearchStats
        Aggregate instrumentation (distance_evaluations reflects the
        symmetry-halved pair count).
    """
    start_time = time.perf_counter()

    n = len(all_rsos)
    k = min(k, n - 1)
    symmetric = distance_metric.name in SYMMETRIC_METRIC_NAMES

    neighbor_lists: list[list[NeighborResult]] = [[] for _ in range(n)]
    distance_evals = 0

    for i in range(n):
        pos_i = all_rsos[i].nominal_position
        for j in range(i + 1, n):
            d_ij = distance_metric.compute(
                all_rsos[i].uncertainty, all_rsos[j].uncertainty
            )
            distance_evals += 1
            e_ij = float(np.linalg.norm(pos_i - all_rsos[j].nominal_position))
            neighbor_lists[i].append(
                NeighborResult(
                    object_id=all_rsos[j].object_id,
                    probabilistic_distance=d_ij,
                    nominal_euclidean_distance=e_ij,
                )
            )
            if symmetric:
                neighbor_lists[j].append(
                    NeighborResult(
                        object_id=all_rsos[i].object_id,
                        probabilistic_distance=d_ij,
                        nominal_euclidean_distance=e_ij,
                    )
                )
            else:
                d_ji = distance_metric.compute(
                    all_rsos[j].uncertainty, all_rsos[i].uncertainty
                )
                distance_evals += 1
                neighbor_lists[j].append(
                    NeighborResult(
                        object_id=all_rsos[i].object_id,
                        probabilistic_distance=d_ji,
                        nominal_euclidean_distance=e_ij,
                    )
                )

    results: list[KNNResult] = []
    for i in range(n):
        top_k = sorted(neighbor_lists[i])[:k]
        results.append(
            KNNResult(
                query_id=all_rsos[i].object_id,
                neighbors=top_k,
                metric=distance_metric.name,
                component_count=all_rsos[i].uncertainty.n_components,
            )
        )

    query_time = time.perf_counter() - start_time
    stats = KNNSearchStats(
        distance_evaluations=distance_evals,
        nodes_visited=0,
        nodes_pruned=0,
        leaves_visited=0,
        query_time_seconds=query_time,
    )
    return results, stats


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
