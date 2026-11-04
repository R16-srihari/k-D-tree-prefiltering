"""Command-line interface for RSO probabilistic KNN screening.

Example:

    python main.py --n-rso 2000 --k 10 --metric hellinger --components 3 \
        --leaf-size 10 --seed 42 --cube-size 1000 --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from .benchmark.metrics import recall_at_k
from .data.synthetic_catalogue import generate_synthetic_catalogue
from .distances.factory import create_distance_metric
from .screening.brute_force import brute_force_knn
from .screening.knn import probabilistic_knn
from .spatial import build_kdtree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probabilistic KNN screening of RSOs using a modular K-D tree."
    )
    parser.add_argument(
        "--n-rso", type=int, default=2000, help="Number of RSOs (default: 2000)"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of neighbors (default: 10)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="hellinger",
        choices=["wasserstein", "mahalanobis", "hellinger", "kl"],
        help="Probabilistic distance metric (default: hellinger)",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=3,
        help="GMM components per RSO (default: 3)",
    )
    parser.add_argument(
        "--leaf-size",
        type=int,
        default=10,
        help="K-D tree leaf size (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--cube-size",
        type=float,
        default=1000.0,
        help="Position cube half-width in km (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path for KNN results",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark suite across all four metrics",
    )
    return parser


def run_single(config: argparse.Namespace) -> None:
    """Run a single KNN query sweep for one metric."""
    distance_metric = create_distance_metric(config.metric)

    catalogue = generate_synthetic_catalogue(
        n_rso=config.n_rso,
        n_components=config.components,
        cube_size=config.cube_size,
        seed=config.seed,
    )

    positions = np.array([r.nominal_position for r in catalogue])
    object_ids = [r.object_id for r in catalogue]

    kd_tree = build_kdtree(positions, leaf_size=config.leaf_size)

    k = min(config.k, config.n_rso - 1)

    all_results = []
    total_kdtree_time = 0.0
    total_brute_time = 0.0
    total_distance_evals = 0
    total_nodes_visited = 0
    total_nodes_pruned = 0
    total_leaves_visited = 0
    recalls = []

    for query_rso in catalogue:
        # KD-tree search
        t0 = time.perf_counter()
        kd_result, kd_stats = probabilistic_knn(
            query_rso=query_rso,
            kd_tree=kd_tree,
            all_rsos=catalogue,
            k=k,
            distance_metric=distance_metric,
        )
        t1 = time.perf_counter()
        total_kdtree_time += t1 - t0
        total_distance_evals += kd_stats.distance_evaluations
        total_nodes_visited += kd_stats.nodes_visited
        total_nodes_pruned += kd_stats.nodes_pruned
        total_leaves_visited += kd_stats.leaves_visited

        # Brute-force reference
        t0 = time.perf_counter()
        bf_result, _ = brute_force_knn(
            query_rso=query_rso,
            all_rsos=catalogue,
            k=k,
            distance_metric=distance_metric,
        )
        t1 = time.perf_counter()
        total_brute_time += t1 - t0

        recalls.append(recall_at_k(kd_result, bf_result))

        for neighbor in kd_result.neighbors:
            all_results.append(
                {
                    "query_id": query_rso.object_id,
                    "neighbor_rank": len([r for r in all_results if r["query_id"] == query_rso.object_id]) + 1,
                    "neighbor_id": neighbor.object_id,
                    "probabilistic_distance": neighbor.probabilistic_distance,
                    "nominal_euclidean_distance": neighbor.nominal_euclidean_distance,
                    "metric": config.metric,
                    "component_count": config.components,
                    "seed": config.seed,
                }
            )

    mean_recall = float(np.mean(recalls)) if recalls else 0.0

    print(f"Metric: {config.metric}")
    print(f"RSOs: {config.n_rso}, K: {k}, Components: {config.components}, Seed: {config.seed}")
    print(f"K-D tree total time: {total_kdtree_time:.4f}s")
    print(f"Brute force total time: {total_brute_time:.4f}s")
    print(f"Mean Recall@K: {mean_recall:.4f}")
    print(f"Total distance evaluations: {total_distance_evals}")
    print(f"Total nodes visited: {total_nodes_visited}")
    print(f"Total nodes pruned: {total_nodes_pruned}")
    print(f"Total leaves visited: {total_leaves_visited}")

    if config.output:
        output_path = Path(config.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "query_id",
            "neighbor_rank",
            "neighbor_id",
            "probabilistic_distance",
            "nominal_euclidean_distance",
            "metric",
            "component_count",
            "seed",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Results written to {output_path}")


def run_benchmark(config: argparse.Namespace) -> None:
    """Run the benchmark suite across all metrics and component counts."""
    from .benchmark.benchmark import run_benchmark as bench_run, BenchmarkConfig

    bench_config = BenchmarkConfig(
        n_rso=config.n_rso,
        k=config.k,
        n_components=config.components,
        cube_size=config.cube_size,
        seed=config.seed,
        leaf_size=config.leaf_size,
    )

    results = bench_run(bench_config)
    df = results.to_dataframe()

    print("\n=== Benchmark Results ===")
    print(df.to_string(index=False))

    print(f"\nCatalogue build time: {results.catalogue_build_time:.4f}s")
    print(f"K-D tree build time: {results.kdtree_build_time:.4f}s")

    if config.output:
        output_path = Path(config.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Benchmark results written to {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.benchmark:
        run_benchmark(args)
    else:
        run_single(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
