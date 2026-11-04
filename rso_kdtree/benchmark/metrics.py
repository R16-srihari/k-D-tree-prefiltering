"""Benchmark metrics: Recall@K and distance-difference statistics."""

from __future__ import annotations

import numpy as np

from ..screening.results import KNNResult


def recall_at_k(
    kd_result: KNNResult,
    brute_result: KNNResult,
) -> float:
    """Compute Recall@K between KD-tree and brute-force neighbor sets.

    Recall@K = |N_KD intersect N_BF| / K
    """
    kd_ids = {n.object_id for n in kd_result.neighbors}
    bf_ids = {n.object_id for n in brute_result.neighbors}
    if not bf_ids:
        return 1.0
    return len(kd_ids & bf_ids) / len(bf_ids)


def mean_distance_difference(
    kd_result: KNNResult,
    brute_result: KNNResult,
) -> float:
    """Mean absolute difference between matched neighbor distances.

    Matches neighbors by object_id and averages |d_KD - d_BF|.
    """
    kd_map = {n.object_id: n.probabilistic_distance for n in kd_result.neighbors}
    bf_map = {n.object_id: n.probabilistic_distance for n in brute_result.neighbors}

    common = set(kd_map) & set(bf_map)
    if not common:
        return 0.0

    diffs = [abs(kd_map[oid] - bf_map[oid]) for oid in common]
    return float(np.mean(diffs))