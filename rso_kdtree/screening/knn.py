"""Probabilistic KNN search using exact Mercurio Algorithm 3 tree traversal.

Implements Mercurio-Singla Algorithm 3 on a custom KD-tree, split into two
phases exactly as described in the paper:

Phase 1 - Bootstrap (Alg. 3 steps 3-6):
  1. Descend from root to the leaf containing the query position.
2. Evaluate that leaf (Hellinger + Euclidean for each RSO it contains).
  3. Traverse up the tree via parent pointers. At each ancestor, open the
     sibling subtree and evaluate its contents. Continue opening boxes
     going up until the best-K list holds k candidates (or the tree is
     exhausted).  No distance pruning is applied in this phase because
     DE_min is undefined until k candidates exist.
  4. Store DH_min and DE_min (DE_min = K-th smallest Euclidean distance
     among the current best-K).

Phase 2 - Pruning (Alg. 3 steps 7-8):
  5. Traverse the tree from the root. For every box not yet opened:
     - Compute D_box = Euclidean distance from the query to the box's AABB.
     - If D_box < DE_min: open the box, evaluate its RSOs, update DE_min.
     - Else: prune the box.

The implementation follows Alg. 3 with a distinct bootstrap phase
(open boxes until k candidates are found) followed by a pruning phase
(open remaining boxes only when D_box < DE_min).
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

    Two phases per Mercurio-Singla Algorithm 3:
      Phase 1 (bootstrap): descend to query leaf, evaluate it, then walk
      up via parent pointers opening sibling boxes until k candidates are
      collected (no distance pruning yet).
      Phase 2 (pruning): traverse remaining boxes from root, opening only
      those with D_box < DE_min.

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

    # Track opened boxes so they are not re-evaluated in the pruning phase.
    visited: set[int] = set()

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
        """k_available-th smallest Euclidean distance among evaluated candidates, or inf.

        This is the DE_min threshold of Alg. 3 step 6/8. While fewer than
        k_available candidates have been found it returns +inf so that no box
        is pruned during the bootstrap phase. ``best`` is capped at
        k_available entries, so the threshold must be indexed at
        k_available - 1 (not k - 1, which overflows when k > n - 1).
        """
        if len(best) < k_available:
            return float("inf")
        de_sorted = sorted(c.nominal_euclidean_distance for c in best)
        return de_sorted[k_available - 1]

    def _evaluate_leaf(node: KDNode) -> None:
        """Evaluate all RSOs in a leaf node (excluding the query itself)."""
        nonlocal distance_evals, leaves_visited
        leaves_visited += 1
        visited.add(id(node))
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
                k_available,
            )

    def _open_subtree_bounded(node: KDNode) -> None:
        """Open a subtree during the bootstrap phase.

        Evaluates leaves until len(best) >= k_available. No distance-based
        pruning is applied because DE_min is undefined until k candidates
        exist (Alg. 3 step 4).
        """
        nonlocal nodes_visited, leaves_visited, distance_evals

        if id(node) in visited:
            return

        if node.is_leaf:
            _evaluate_leaf(node)
            return

        nodes_visited += 1
        visited.add(id(node))

        for child in (node.left, node.right):
            if child is None:
                continue
            if len(best) >= k_available:
                return
            _open_subtree_bounded(child)

    def _prune_remaining(node: KDNode) -> None:
        """Pruning phase: test all remaining boxes (Alg. 3 steps 7–8).

        For each child box, compute D_box = Euclidean distance from the query
        to the box's AABB. If D_box < DE_min the box is opened recursively;
        otherwise it is pruned. D_box is a lower bound on the distance to
        *every* point in the box, so a subtree whose root was opened during
        the bootstrap is NOT exempt: its unopened remainder must still be
        tested. Only already-evaluated leaves are skipped.
        """
        nonlocal nodes_visited, nodes_pruned

        if node.is_leaf:
            if id(node) not in visited:
                _evaluate_leaf(node)
            return

        # Internal node: count it once if not already visited.
        if id(node) not in visited:
            nodes_visited += 1
            visited.add(id(node))

        for child in (node.left, node.right):
            if child is None:
                continue
            if child.is_leaf and id(child) in visited:
                continue  # already evaluated during the bootstrap phase
            d_box = point_to_aabb_distance(
                query_position, child.aabb_lower, child.aabb_upper
            )
            if d_box < _de_min():
                _prune_remaining(child)
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

    # ------------------------------------------------------------------
    # Phase 1 - Bootstrap (Alg. 3 steps 3-6)
    # ------------------------------------------------------------------
    # Step 3 (continued): Traverse up the tree, opening sibling boxes
    # until the required number of nearest neighbors is met.
    current = leaf
    while current.parent is not None and len(best) < k_available:
        parent = current.parent
        nodes_visited += 1
        visited.add(id(parent))

        # Identify the sibling subtree (the adjacent box at this level).
        if parent.left is current:
            sibling = parent.right
        else:
            sibling = parent.left

        if sibling is not None:
            _open_subtree_bounded(sibling)

        current = parent

    # Step 6: DE_min is now defined (if k candidates were found).
    # DH_min = best[0].probabilistic_distance (kept as a stored value per
    # Alg. 3, though only DE_min drives the pruning test in step 8).

    # ------------------------------------------------------------------
    # Phase 2 - Pruning (Alg. 3 steps 7-8)
    # ------------------------------------------------------------------
    # Step 7-8: Compute D_box for all remaining boxes; open those with
    # D_box < DE_min, prune the rest. Opening a box may lower DE_min,
    # which is naturally honoured because the threshold is re-read at
    # each comparison.
    _prune_remaining(kd_tree)

    query_time = time.perf_counter() - start_time

    stats = KNNSearchStats(
        distance_evaluations=distance_evals,
        nodes_visited=nodes_visited,
        nodes_pruned=nodes_pruned,
        leaves_visited=leaves_visited,
        query_time_seconds=query_time,
    )

    return make_result(), stats
