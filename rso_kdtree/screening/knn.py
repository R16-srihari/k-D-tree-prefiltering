"""Probabilistic KNN search using a scipy.spatial.KDTree with Euclidean pruning.

Implements the Mercurio-Singla-inspired search on top of scipy's KDTree:

1. Query the tree for an initial set of Euclidean nearest neighbors.
2. Evaluate probabilistic distances for those candidates (excluding the
   query itself) and maintain the K best by probabilistic distance.
3. Compute the Euclidean threshold D_E,K: the K-th smallest nominal
   Euclidean distance among the current best candidates.
4. Ball-query the tree at radius D_E,K and evaluate any new candidates;
   repeat until no unevaluated candidate lies within the threshold.
   If fewer than K candidates have been evaluated, expand the Euclidean
   neighbor query instead.
5. Return candidates sorted by probabilistic distance.

Critical caveat: Euclidean pruning is a heuristic and does NOT provide a
guaranteed lower bound on probabilistic distances. It may prune objects
that are among the true probabilistic K nearest neighbors.

Note: scipy's KDTree does not expose node-level traversal, so per-node
instrumentation (nodes visited/pruned/leaves visited) is unavailable and
reported as zero; candidate-set statistics are still collected.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import KDTree

from ..distances.base import ProbabilisticDistance
from ..models.rso import RSO
from .results import KNNResult, KNNSearchStats, NeighborResult


def _update_best(
    best: list[NeighborResult],
    candidate: NeighborResult,
    k: int,
) -> list[NeighborResult]:
    """Insert a candidate into the best-K list, maintaining sort order and size."""
    best.append(candidate)
    best.sort()
    if len(best) > k:
        best = best[:k]
    return best


def _current_euclidean_threshold(
    euclidean_dists: list[float], k: int
) -> float:
    """K-th smallest nominal Euclidean distance among all evaluated candidates.

    Per PLAN.md §10.3, D_E,K is the K-th smallest nominal Euclidean distance
    among the evaluated candidate objects. This value is monotonically
    non-increasing as more candidates are evaluated, so the ball-query radius
    only shrinks or stays the same, guaranteeing termination.

    Returns infinity if fewer than K candidates have been evaluated.
    """
    if len(euclidean_dists) < k:
        return float("inf")
    return sorted(euclidean_dists)[k - 1]


def probabilistic_knn(
    query_rso: RSO,
    kd_tree: KDTree,
    all_rsos: list[RSO],
    k: int,
    distance_metric: ProbabilisticDistance,
) -> tuple[KNNResult, KNNSearchStats]:
    """Probabilistic KNN search using a scipy KDTree with Euclidean pruning.

    Parameters
    ----------
    query_rso : RSO
        The query RSO (excluded from its own neighbor list).
    kd_tree : scipy.spatial.KDTree
        Pre-built KD-tree on nominal positions.
    all_rsos : list[RSO]
        All RSOs in the catalogue (used to look up RSOs by index).
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
    query_position = query_rso.nominal_position
    n = len(all_rsos)
    k_available = min(k, n - 1)

    best: list[NeighborResult] = []
    distance_evals = 0
    query_rso_index = int(query_rso.object_id.split("_")[-1])
    # Pre-mark the query index so it is never re-returned by ball queries.
    evaluated: set[int] = {query_rso_index}
    # Euclidean distances of all evaluated candidates, used for D_E,K threshold.
    evaluated_euclidean_dists: list[float] = []

    def make_result() -> KNNResult:
        return KNNResult(
            query_id=query_rso.object_id,
            neighbors=list(best),
            metric=distance_metric.name,
            component_count=query_rso.uncertainty.n_components,
        )

    if kd_tree is None or n <= 1 or k_available == 0:
        query_time = time.perf_counter() - start_time
        stats = KNNSearchStats(0, 0, 0, 0, query_time)
        return make_result(), stats

    def evaluate(indices) -> None:
        nonlocal distance_evals, best
        for idx in indices:
            idx = int(idx)
            if idx in evaluated or idx == query_rso_index:
                continue
            candidate = all_rsos[idx]
            if candidate.object_id == query_rso.object_id:
                continue
            dist = distance_metric.compute(
                query_rso.uncertainty, candidate.uncertainty
            )
            distance_evals += 1
            euclidean_dist = float(
                np.linalg.norm(query_position - candidate.nominal_position)
            )
            evaluated.add(idx)
            evaluated_euclidean_dists.append(euclidean_dist)
            best = _update_best(
                best,
                NeighborResult(
                    object_id=candidate.object_id,
                    probabilistic_distance=dist,
                    nominal_euclidean_distance=euclidean_dist,
                ),
                k,
            )

    # Phase 1: initial Euclidean neighbor queries until K candidates evaluated
    k_query = min(max(k + 1, 2), n)
    while len(best) < k_available and k_query <= n:
        _, idxs = kd_tree.query(query_position, k=k_query)
        idxs = np.atleast_1d(np.asarray(idxs))
        evaluate(idxs)
        if len(best) >= k_available:
            break
        k_query = min(k_query * 2, n)

    # Phase 2: ball-query expansion at the D_E,K threshold until stable
    while len(best) >= k_available:
        threshold = _current_euclidean_threshold(evaluated_euclidean_dists, k)
        if not np.isfinite(threshold):
            break
        ball_indices = {
            int(i) for i in kd_tree.query_ball_point(query_position, threshold)
        }
        new_indices = [i for i in ball_indices if i not in evaluated]
        if not new_indices:
            break
        evaluate(new_indices)

    query_time = time.perf_counter() - start_time

    stats = KNNSearchStats(
        distance_evaluations=distance_evals,
        nodes_visited=0,  # not exposed by scipy KDTree
        nodes_pruned=0,
        leaves_visited=0,
        query_time_seconds=query_time,
    )

    return make_result(), stats
