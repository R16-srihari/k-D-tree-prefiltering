"""Benchmark suite for KD-tree vs brute-force KNN screening.

Runs all four distance measures over the same synthetic catalogue and
reports timing, distance evaluations, and Recall@K.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..data.synthetic_catalogue import generate_synthetic_catalogue
from ..distances.factory import create_distance_metric, list_supported_metrics
from ..models.rso import RSO
from ..screening.brute_force import brute_force_knn_all
from ..screening.knn import probabilistic_knn
from ..screening.results import KNNResult
from ..spatial import build_kdtree, build_leaf_map
from .metrics import mean_distance_difference, recall_at_k


@dataclass
class BenchmarkConfig:
    n_rso: int = 100
    k: int = 5
    n_components: int = 6
    r_min: float = 6500.0
    r_max: float = 8000.0
    var_min: float = 1e-4
    var_max: float = 100.0
    seed: int = 42
    leaf_size: int = 1
    metrics: list[str] | None = None


@dataclass
class BenchmarkRow:
    metric: str
    n_components: int
    kdtree_time: float
    brute_time: float
    recall_at_k: float
    distance_evaluations: int
    nodes_visited: int
    nodes_pruned: int
    leaves_visited: int
    mean_distance_difference: float


@dataclass
class BenchmarkResults:
    rows: list[BenchmarkRow] = field(default_factory=list)
    catalogue_build_time: float = 0.0
    kdtree_build_time: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Metric": r.metric,
                    "GMM Components": r.n_components,
                    "K-D Tree Time (s)": r.kdtree_time,
                    "Brute Force Time (s)": r.brute_time,
                    "Recall@K": r.recall_at_k,
                    "Distance Evaluations": r.distance_evaluations,
                    "Nodes Visited": r.nodes_visited,
                    "Nodes Pruned": r.nodes_pruned,
                    "Leaves Visited": r.leaves_visited,
                    "Mean Distance Difference": r.mean_distance_difference,
                }
                for r in self.rows
            ]
        )


def _log_first_n_neighbors(
    metric_name: str,
    catalogue: list[RSO],
    kdtree_results: list[KNNResult],
    brute_results: list[KNNResult],
    n: int = 5,
    write: Callable[[str], None] = print,
) -> None:
    """Write neighbor IDs of the first ``n`` RSOs for KD-tree vs BruteForce.

    Intended for the terminal/log so a user can inspect, per chosen measure,
    which RSOs each method returned as nearest neighbors for the first few
    queries.

    Parameters
    ----------
    metric_name : str
        Name of the distance measure being benchmarked.
    catalogue : list[RSO]
        Full catalogue (query RSOs are taken from the front of this list).
    kdtree_results : list[KNNResult]
        Per-query KD-tree results, aligned with ``catalogue`` order.
    brute_results : list[KNNResult]
        Per-query BruteForce results, aligned with ``catalogue`` order.
    n : int
        Number of leading RSOs to log (default: 5).
    write : Callable[[str], None]
        Sink for the log text (``print`` by default; pass ``tee.write`` to
        mirror to both terminal and the benchmark log file).
    """
    limit = min(n, len(catalogue))
    write(f"\n--- Neighbor IDs of first {limit} RSOs (metric: {metric_name}) ---\n")
    for idx in range(limit):
        query = catalogue[idx]
        kd_res = kdtree_results[idx]
        bf_res = brute_results[idx]
        kd_ids = [neighbor.object_id for neighbor in kd_res.neighbors]
        bf_ids = [neighbor.object_id for neighbor in bf_res.neighbors]
        write(f"Query RSO: {query.object_id}\n")
        write(f"  KD-tree neighbors:   {kd_ids}\n")
        write(f"  BruteForce neighbors: {bf_ids}\n")
    write("\n")


def run_benchmark(
    config: BenchmarkConfig | None = None,
    log_write: Callable[[str], None] = print,
) -> BenchmarkResults:
    """Run the full benchmark suite.

    Parameters
    ----------
    config : BenchmarkConfig | None
        Benchmark configuration. Uses defaults if None.
    log_write : Callable[[str], None]
        Sink for informational log text (``print`` by default; pass
        ``tee.write`` to mirror to both terminal and the benchmark log file).

    Returns
    -------
    BenchmarkResults
        Collected benchmark results.
    """
    if config is None:
        config = BenchmarkConfig()

    metrics = config.metrics or list_supported_metrics()

    # Generate catalogue once (reused across all metrics)
    catalogue_start = time.perf_counter()
    catalogue = generate_synthetic_catalogue(
        n_rso=config.n_rso,
        n_components=config.n_components,
        r_min=config.r_min,
        r_max=config.r_max,
        seed=config.seed,
        var_min=config.var_min,
        var_max=config.var_max,
    )
    catalogue_build_time = time.perf_counter() - catalogue_start

    # Build KD-tree once (same positions for all metrics)
    positions = np.array([r.nominal_position for r in catalogue])
    object_ids = [r.object_id for r in catalogue]

    tree_start = time.perf_counter()
    kd_tree = build_kdtree(positions, catalogue, leaf_size=config.leaf_size)
    leaf_map = build_leaf_map(kd_tree)
    kdtree_build_time = time.perf_counter() - tree_start

    k = min(config.k, config.n_rso - 1)

    results = BenchmarkResults(
        catalogue_build_time=catalogue_build_time,
        kdtree_build_time=kdtree_build_time,
    )

    for metric_name in metrics:
        distance_metric = create_distance_metric(metric_name)

        # Brute-force over all queries in one sweep; symmetric metrics
        # evaluate each unordered pair once (N(N-1)/2 evaluations).
        brute_start = time.perf_counter()
        brute_results, _ = brute_force_knn_all(
            catalogue, k=k, distance_metric=distance_metric
        )
        brute_time = time.perf_counter() - brute_start

        # KD-tree over all queries
        kdtree_start = time.perf_counter()
        kdtree_results = []
        total_distance_evals = 0
        total_nodes_visited = 0
        total_nodes_pruned = 0
        total_leaves_visited = 0

        for query_rso in catalogue:
            result, stats = probabilistic_knn(
                query_rso=query_rso,
                kd_tree=kd_tree,
                all_rsos=catalogue,
                k=k,
                distance_metric=distance_metric,
                leaf_map=leaf_map,
            )
            kdtree_results.append(result)
            total_distance_evals += stats.distance_evaluations
            total_nodes_visited += stats.nodes_visited
            total_nodes_pruned += stats.nodes_pruned
            total_leaves_visited += stats.leaves_visited

        kdtree_time = time.perf_counter() - kdtree_start

        # Log neighbor IDs of the first 5 RSOs for this measure
        _log_first_n_neighbors(
            metric_name=metric_name,
            catalogue=catalogue,
            kdtree_results=kdtree_results,
            brute_results=brute_results,
            n=5,
            write=log_write,
        )

        # Aggregate metrics across all queries
        recalls = []
        dist_diffs = []
        for kd_res, bf_res in zip(kdtree_results, brute_results):
            recalls.append(recall_at_k(kd_res, bf_res))
            dist_diffs.append(mean_distance_difference(kd_res, bf_res))

        results.rows.append(
            BenchmarkRow(
                metric=metric_name,
                n_components=config.n_components,
                kdtree_time=kdtree_time,
                brute_time=brute_time,
                recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
                distance_evaluations=total_distance_evals,
                nodes_visited=total_nodes_visited,
                nodes_pruned=total_nodes_pruned,
                leaves_visited=total_leaves_visited,
                mean_distance_difference=float(np.mean(dist_diffs)) if dist_diffs else 0.0,
            )
        )

    return results
