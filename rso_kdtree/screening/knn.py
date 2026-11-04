"""Probabilistic KNN search using exact Alg. 3 tree traversal.

Implements Mercurio-Singla Algorithm 3 on a custom KD-tree:

1. Descend from root to the leaf containing the query position.
2. Evaluate the leaf: compute probabilistic (DH) and Euclidean (DE)
   distances for all RSOs in the leaf (excluding the query itself).
   Maintain best-K list sorted by DH. Track DE_min = K-th smallest DE.
3. Backtrack up the tree. For each ancestor's sibling subtree:
   - Compute D_box = Euclidean distance from query to sibling's AABB.
   - If D_box < DE_min: recursively search the sibling subtree.
   - Else: prune (skip the subtree).
4. Continue up to root, repeating step 3 for each ancestor.
5. Return best-K list sorted by probabilistic distance.

Node-level instrumentation tracks nodes_visited, nodes_pruned, and
leaves_visited during traversal.
"""

from __future__ import annotations

import time

import numpy as np

from ..distances.base import ProbabilisticDistance
from ..models.rso import RSO
from ..spatial import KDNode
from ..spatial.geometry import point_to_aabb_distance
from .results import KNNResult, KNNSearchStats, NeighborResult


def _update_best(
    best: list[NeighborResult],
    candidate: NeighborResult,
    k: int,
) -> None:
    """Insert a candidate into the best-K list, maintaining sort order and size.

    Modifies ``best`` in place.
    """
    best.append(candidate)
    best.sort()
    del best[k:]


def probabilistic_knn(
    query_rso: RSO,
    kd_tree: KDNode,
    all_rsos: list[RSO],
    k: int,
    distance_metric: ProbabilisticDistance,
) -> tuple[KNNResult, KNNSearchStats]:
    """Probabilistic KNN search using exact Alg. 3 tree traversal.

    Parameters
    ----------
    query_rso : RSO
        The query RSO (excluded from its own neighbor list).
    kd_tree : KDNode
        Root node of the custom KD-tree.
    all_rsos : list[RSO]
        All RSOs in the catalogue (used to resolve object_id -> index).
    k : int
        Number of neighbors to return.
    distance_metric : ProbabilisticDistance
        The probabilistic distance measure.

    Returns
    -------
    result : KNNResult
        K nearest neighbors ranked by probabilistic distance.
    stats : KNNSearchStats
        Search instrumentation statistics (nodes_visited, nodes_pruned,
        leaves_visited, distance_evaluations, query_time_seconds).
    """
    start_time = time.perf_counter()
    query_position = query_rso.nominal_position
    n = len(all_rsos)
    k_available = min(k, n - 1)

    best: list[NeighborResult] = []
    distance_evals = 0

    # Instrumentation counters
    nodes_visited = 0
    leaves_visited = 0
    nodes_pruned = 0

    # Build object_id -> index map (avoids fragile string parsing of object_id)
    object_id_to_index = {r.object_id: i for i, r in enumerate(all_rsos)}
    _ = object_id_to_index[query_rso.object_id]  # validate query is in catalogue

    def make_result() -> KNNResult:
        return KNNResult(
            query_id=query_rso.object_id,
            neighbors=list(best),
            metric=distance_metric.name,
            component_count=query_rso.uncertainty.n_components,
        )

    def _de_min() -> float:
        """K-th smallest Euclidean distance among evaluated candidates, or inf."""
        if len(best) < k:
            return float("inf")
        de_sorted = sorted(c.nominal_euclidean_distance for c in best)
        return de_sorted[k - 1]

    def _evaluate_leaf(node: KDNode) -> None:
        """Evaluate all RSOs in a leaf node (excluding the query itself)."""
        nonlocal distance_evals, leaves_visited
        leaves_visited += 1
        for rso in node.rsos:
            if rso.object_id == query_rso.object_id:
                continue
            dist = distance_metric.compute(query_rso.uncertainty, rso.uncertainty)
            distance_evals += 1
            euclidean_dist = float(
                np.linalg.norm(query_position - rso.nominal_position)
            )
            _update_best(
                best,
                NeighborResult(
                    object_id=rso.object_id,
                    probabilistic_distance=dist,
                    nominal_euclidean_distance=euclidean_dist,
                ),
                k,
            )

    def _search_subtree(node: KDNode) -> None:
        """Recursively search a subtree, pruning by D_box < DE_min.

        Evaluates all leaves whose AABB is within the current DE_min
        threshold, updating best-K and DE_min as candidates are found.
        """
        nonlocal nodes_visited, nodes_pruned

        if node.is_leaf:
            _evaluate_leaf(node)
            return

        nodes_visited += 1

        for child in (node.left, node.right):
            if child is None:
                continue
            d_box = point_to_aabb_distance(
                query_position, child.aabb_lower, child.aabb_upper
            )
            if d_box < _de_min():
                _search_subtree(child)
            else:
                nodes_pruned += 1

    def _descend_to_leaf(node: KDNode) -> KDNode:
        """Descend from root to the leaf containing the query position."""
        current = node
        while not current.is_leaf:
            axis = current.split_axis
            val = current.split_value
            if query_position[axis] <= val:
                current = current.left
            else:
                current = current.right
        return current

    if kd_tree is None or n <= 1 or k_available == 0:
        query_time = time.perf_counter() - start_time
        stats = KNNSearchStats(0, 0, 0, 0, query_time)
        return make_result(), stats

    # Step 1: Descend to the leaf containing the query position
    leaf = _descend_to_leaf(kd_tree)

    # Step 2: Evaluate the leaf
    _evaluate_leaf(leaf)

    # Step 3: Backtrack up the tree, checking sibling subtrees
    current = leaf
    while current.parent is not None:
        parent = current.parent
        nodes_visited += 1  # internal node entered during backtracking

        # Identify the sibling subtree
        if parent.left is current:
            sibling = parent.right
        else:
            sibling = parent.left

        if sibling is not None:
            d_box = point_to_aabb_distance(
                query_position, sibling.aabb_lower, sibling.aabb_upper
            )
            if d_box < _de_min():
                _search_subtree(sibling)
            else:
                nodes_pruned += 1

        current = parent

    query_time = time.perf_counter() - start_time

    stats = KNNSearchStats(
        distance_evaluations=distance_evals,
        nodes_visited=nodes_visited,
        nodes_pruned=nodes_pruned,
        leaves_visited=leaves_visited,
        query_time_seconds=query_time,
    )

    return make_result(), stats
