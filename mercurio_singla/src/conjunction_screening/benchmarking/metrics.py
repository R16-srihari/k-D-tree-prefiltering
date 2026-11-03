"""Screening accuracy metrics.

Compares K-d tree screening results against brute-force ground truth to
measure recall and precision of the candidate-pair set.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Set, Optional
import numpy as np

from ..models import RSO
from ..distances.base import ProbabilisticDistance
from ..screening import brute_force_screen, MercurioSinglaScreening
from ..spatial import KDTree


def recall_precision(tree_result, brute_result) -> Dict[str, float]:
    """Compute recall and precision of a tree result vs brute force.

    Recall = |tree pairs ∩ brute pairs| / |brute pairs|
    Precision = |tree pairs ∩ brute pairs| / |tree pairs|

    Parameters
    ----------
    tree_result : ScreeningResult
        Result from MercurioSinglaScreening.
    brute_result : ScreeningResult
        Result from brute_force_screen (all pairs).

    Returns
    -------
    dict with keys 'recall', 'precision', 'f1', 'n_tree', 'n_brute',
    'n_intersection'.
    """
    tree_set: Set[Tuple[int, int]] = set(tree_result.pairs)
    brute_set: Set[Tuple[int, int]] = set(brute_result.pairs)

    intersection = tree_set & brute_set
    n_tree = len(tree_set)
    n_brute = len(brute_set)
    n_intersection = len(intersection)

    recall = n_intersection / n_brute if n_brute else 1.0
    precision = n_intersection / n_tree if n_tree else 0.0
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)

    return {
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "n_tree": n_tree,
        "n_brute": n_brute,
        "n_intersection": n_intersection,
    }


def topk_recall(tree_result, brute_result, k: int) -> Dict[str, float]:
    """Recall of the top-k probabilistic pairs vs brute-force top-k.

    For each object, the brute-force top-k neighbours by probabilistic
    distance form the ground truth; the tree result's top-k is compared
    against it.

    Parameters
    ----------
    tree_result : ScreeningResult
    brute_result : ScreeningResult
    k : int
        Number of neighbours per object.

    Returns
    -------
    dict with 'recall', 'precision', 'n_true_topk', 'n_found'.
    """
    n = max(max(pair) for pair in tree_result.pairs) + 1 if tree_result.pairs else 0

    # Build brute-force top-k per object
    brute_adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for (i, j), dp in zip(brute_result.pairs, brute_result.probabilistic_distances):
        brute_adj[i].append((j, dp))
        brute_adj[j].append((i, dp))

    true_topk: Set[Tuple[int, int]] = set()
    for i in range(n):
        brute_adj[i].sort(key=lambda x: x[1])
        for j, _ in brute_adj[i][:k]:
            true_topk.add((min(i, j), max(i, j)))

    # Tree top-k per object
    tree_adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for (i, j), dp in zip(tree_result.pairs, tree_result.probabilistic_distances):
        tree_adj[i].append((j, dp))
        tree_adj[j].append((i, dp))

    tree_topk: Set[Tuple[int, int]] = set()
    for i in range(n):
        tree_adj[i].sort(key=lambda x: x[1])
        for j, _ in tree_adj[i][:k]:
            tree_topk.add((min(i, j), max(i, j)))

    intersection = tree_topk & true_topk
    recall = len(intersection) / len(true_topk) if true_topk else 1.0
    precision = len(intersection) / len(tree_topk) if tree_topk else 0.0

    return {
        "recall": float(recall),
        "precision": float(precision),
        "n_true_topk": len(true_topk),
        "n_found": len(intersection),
    }


def evaluate_screening(rsos: List[RSO],
                       distance_metric: ProbabilisticDistance,
                       k: int = 10,
                       tree_backend_k: Optional[int] = None) -> Dict[str, Any]:
    """Run tree screening and brute force, then compare.

    Returns a dict with tree result, brute result, recall/precision metrics,
    and timing/evaluation statistics from both.
    """
    tree_k = tree_backend_k if tree_backend_k is not None else k

    tree = KDTree.from_rsos(rsos)
    tree_screener = MercurioSinglaScreening(
        tree=tree,
        distance_metric=distance_metric,
        k=tree_k,
    )

    tree_result = tree_screener.screen(rsos)
    brute_result = brute_force_screen(rsos, distance_metric)

    metrics = recall_precision(tree_result, brute_result)
    metrics_topk = topk_recall(tree_result, brute_result, k)

    return {
        "tree_result": tree_result,
        "brute_result": brute_result,
        "metrics": metrics,
        "metrics_topk": metrics_topk,
        "tree_stats": tree_result.stats,
        "brute_stats": brute_result.stats,
    }
