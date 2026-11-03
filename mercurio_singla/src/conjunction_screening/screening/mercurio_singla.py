"""Mercurio–Singla modified nearest-neighbour screening.

Implements Algorithm 3 from the paper: a K-d tree search on mean positions
that ranks neighbours by a pluggable probabilistic distance while using
Euclidean distance only for tree geometry and box pruning.

The screening algorithm depends exclusively on the ProbabilisticDistance
interface and never imports a concrete metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Callable
from heapq import heappush, heappop
import time
import numpy as np

from ..models import RSO, Gaussian, GaussianMixture
from ..distances.base import ProbabilisticDistance
from ..spatial import KDTree, dist_point_to_box, dist_point_to_point
from .results import ScreeningResult


@dataclass
class MercurioSinglaScreening:
    """Modified nearest-neighbour conjunction screener (Algorithm 3).

    Parameters
    ----------
    tree : KDTree
        K-d tree built on RSO mean positions.
    distance_metric : ProbabilisticDistance
        Pluggable probabilistic distance metric. The tree and search logic
        never reference this type's concrete class.
    k : int
        Number of neighbours to retrieve per query object.
    """

    tree: KDTree
    distance_metric: ProbabilisticDistance
    k: int = 10

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def screen(self, rsos: List[RSO]) -> ScreeningResult:
        """Screen all objects against the tree.

        For each object, finds up to k nearest neighbours by probabilistic
        distance using the modified NN search, then assembles unique pairs.

        Parameters
        ----------
        rsos : list of RSO
            Catalog of objects to screen.

        Returns
        -------
        ScreeningResult
        """
        n = len(rsos)
        if n == 0:
            return ScreeningResult([], [], [], {})

        means = np.stack([r.overall_mean for r in rsos], axis=0)
        uncertainties = [r.uncertainty for r in rsos]

        start = time.perf_counter()
        stats: Dict[str, Any] = {
            "objects_processed": n,
            "k": self.k,
            "nodes_visited": 0,
            "nodes_opened": 0,
            "nodes_pruned": 0,
            "de_evals": 0,
            "dh_evals": 0,
        }

        neighbours: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

        for i in range(n):
            nn_indices, nn_dists, node_stats = self._modified_nn_search(
                means[i], uncertainties[i], means, uncertainties, i
            )
            neighbours[i] = list(zip(nn_indices, nn_dists))

            for key in ("nodes_visited", "nodes_opened", "nodes_pruned",
                        "de_evals", "dh_evals"):
                stats[key] = stats.get(key, 0) + node_stats.get(key, 0)

        elapsed = time.perf_counter() - start
        stats["elapsed_time"] = elapsed

        pairs, e_dists, p_dists = self._assemble_pairs(
            neighbours, means, uncertainties
        )

        return ScreeningResult(
            pairs=pairs,
            euclidean_distances=e_dists,
            probabilistic_distances=p_dists,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Modified nearest-neighbour search (Algorithm 3)
    # ------------------------------------------------------------------
    def _modified_nn_search(
        self,
        mu_query: np.ndarray,
        unc_query,
        mu_database: np.ndarray,
        unc_database: list,
        query_idx: int,
    ) -> Tuple[List[int], List[float], Dict[str, int]]:
        """Run the modified NN search for a single query object.

        Returns neighbour indices, probabilistic distances, and node stats.
        """
        n = mu_database.shape[0]
        k = min(self.k, n - 1)  # exclude self
        if k < 1:
            return [], [], {
                "nodes_visited": 0, "nodes_opened": 0, "nodes_pruned": 0,
                "de_evals": 0, "dh_evals": 0,
            }

        stats = {
            "nodes_visited": 0,
            "nodes_opened": 0,
            "nodes_pruned": 0,
            "de_evals": 0,
            "dh_evals": 0,
        }

        def _prob_dist(idx: int) -> float:
            return self.distance_metric.compute(unc_query, unc_database[idx])

        # Step 1: initial KNN by Euclidean distance
        knn = self.tree.knn_euclidean(mu_query, k)
        stats["de_evals"] += len(knn)

        best_by_dh: List[Tuple[int, float]] = []
        for idx, _ in knn:
            dh = _prob_dist(idx)
            stats["dh_evals"] += 1
            best_by_dh = _insert_sorted(best_by_dh, idx, dh, k)
        best_by_dh = best_by_dh[:k]

        # Current DE_min: min Euclidean distance among current best DH set
        def _update_de_min() -> float:
            if len(best_by_dh) < k:
                return float("inf")
            return min(
                dist_point_to_point(mu_query, mu_database[idx])
                for idx, _ in best_by_dh
            )

        DE_min = _update_de_min()
        stats["de_evals"] += len(best_by_dh)  # recomputed above

        # Step 2: best-bin-first search with box pruning
        pq: List[Tuple[float, object]] = []
        heappush(pq, (0.0, self.tree.root))

        in_best = {idx for idx, _ in best_by_dh}

        while pq:
            Dbox, node = heappop(pq)
            stats["nodes_visited"] += 1

            if Dbox >= DE_min:
                stats["nodes_pruned"] += 1
                continue

            stats["nodes_opened"] += 1

            idx = node.index
            if idx is not None and idx != query_idx and idx not in in_best:
                de = dist_point_to_point(mu_query, mu_database[idx])
                stats["de_evals"] += 1

                dh = _prob_dist(idx)
                stats["dh_evals"] += 1

                best_by_dh = _insert_sorted(best_by_dh, idx, dh, k)
                best_by_dh = best_by_dh[:k]
                in_best = {i for i, _ in best_by_dh}

                if len(best_by_dh) >= k:
                    DE_min = _update_de_min()
                    stats["de_evals"] += len(best_by_dh)

            for child in (node.left, node.right):
                if child is not None:
                    Dbox_child = dist_point_to_box(
                        mu_query, child.bbox_min, child.bbox_max
                    )
                    heappush(pq, (Dbox_child, child))

        indices = [idx for idx, _ in best_by_dh]
        distances = [dist for _, dist in best_by_dh]
        return indices, distances, stats

    # ------------------------------------------------------------------
    # Pair assembly
    # ------------------------------------------------------------------
    def _assemble_pairs(
        self,
        neighbours: List[List[Tuple[int, float]]],
        means: np.ndarray,
        uncertainties: list,
    ) -> Tuple[List[Tuple[int, int]], List[float], List[float]]:
        """Assemble unique (i < j) pairs from neighbour lists.

        Euclidean distance is computed directly from means; probabilistic
        distance is taken from the neighbour list (already computed during
        search).
        """
        pair_dict: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for i in range(len(neighbours)):
            for j_idx, dh in neighbours[i]:
                a, b = (i, j_idx) if i < j_idx else (j_idx, i)
                de = dist_point_to_point(means[a], means[b])
                pair_dict[(a, b)] = (de, dh)

        pairs = sorted(pair_dict.keys())
        e_dists = [pair_dict[p][0] for p in pairs]
        p_dists = [pair_dict[p][1] for p in pairs]
        return pairs, e_dists, p_dists


def _insert_sorted(best: List[Tuple[int, float]],
                   idx: int, dist: float, k: int) -> List[Tuple[int, float]]:
    """Insert (idx, dist) into best list sorted by distance ascending, capped
    at k elements. Avoids duplicates by index."""
    for i in range(len(best)):
        if best[i][0] == idx:
            if dist < best[i][1]:
                best[i] = (idx, dist)
                best.sort(key=lambda x: x[1])
            return best
        if dist < best[i][1]:
            best.insert(i, (idx, dist))
            if len(best) > k:
                best.pop()
            return best
    if len(best) < k:
        best.append((idx, dist))
    return best


__all__ = ["MercurioSinglaScreening"]
