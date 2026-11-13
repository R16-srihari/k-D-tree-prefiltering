"""Command-line interface for RSO probabilistic KNN screening.

Example:

    python main.py --n-rso 2000 --k 10 --metric hellinger --components 6 \
        --leaf-size 1 --seed 42 --output results.csv
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
from .screening.brute_force import brute_force_knn_all
from .screening.knn import probabilistic_knn
from .spatial import build_kdtree, build_leaf_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probabilistic KNN screening of RSOs using a modular K-D tree (Mercurio spec)."
    )
    parser.add_argument(
        "--n-rso", type=int, default=100, help="Number of RSOs (default: 100)"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of neighbors (default: 5)"
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
        default=6,
        help="GMM components per RSO (default: 6, per Mercurio)",
    )
    parser.add_argument(
        "--leaf-size",
        type=int,
        default=1,
        help="K-D tree leaf size (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--r-min",
        type=float,
        default=6500.0,
        help="Minimum orbital radius in km (default: 6500, per Mercurio)",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=8000.0,
        help="Maximum orbital radius in km (default: 8000, per Mercurio)",
    )
    parser.add_argument(
        "--var-min",
        type=float,
        default=10000.0,
        help="Minimum covariance variance in km^2 (default: 10000)",
    )
    parser.add_argument(
        "--var-max",
        type=float,
        default=1000000000.0,
        help="Maximum covariance variance in km^2 (default: 1000000000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark.csv",
        help="Output CSV path for KNN results. Default: benchmark.csv since this is the default output for the benchmark suite.",
    )
    parser.add_argument(
        "--log-output",
        type=str,
        default="log.txt",
        help="Output text file path for benchmark terminal log (includes settings and results). Default: log.txt",
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
        r_min=config.r_min,
        r_max=config.r_max,
        seed=config.seed,
        var_min=config.var_min,
        var_max=config.var_max,
    )

    positions = np.array([r.nominal_position for r in catalogue])
    object_ids = [r.object_id for r in catalogue]

    kd_tree = build_kdtree(positions, catalogue, leaf_size=config.leaf_size)
    leaf_map = build_leaf_map(kd_tree)

    k = min(config.k, config.n_rso - 1)

    all_results = []
    total_kdtree_time = 0.0
    total_distance_evals = 0
    total_nodes_visited = 0
    total_nodes_pruned = 0
    total_leaves_visited = 0
    recalls = []

    # Brute-force reference for all queries in one sweep; symmetric metrics
    # evaluate each unordered pair once (N(N-1)/2) per Mercurio & Singla.
    t0 = time.perf_counter()
    brute_results, _ = brute_force_knn_all(catalogue, k=k, distance_metric=distance_metric)
    total_brute_time = time.perf_counter() - t0

    for query_rso, bf_result in zip(catalogue, brute_results):
        # KD-tree search
        t0 = time.perf_counter()
        kd_result, kd_stats = probabilistic_knn(
            query_rso=query_rso,
            kd_tree=kd_tree,
            all_rsos=catalogue,
            k=k,
            distance_metric=distance_metric,
            leaf_map=leaf_map,
        )
        t1 = time.perf_counter()
        total_kdtree_time += t1 - t0
        total_distance_evals += kd_stats.distance_evaluations
        total_nodes_visited += kd_stats.nodes_visited
        total_nodes_pruned += kd_stats.nodes_pruned
        total_leaves_visited += kd_stats.leaves_visited

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
    print(f"Orbital radius: [{config.r_min}, {config.r_max}] km")
    print(f"Covariance variance: [{config.var_min}, {config.var_max}] km^2")
    print(f"K-D tree total time: {total_kdtree_time:.4f}s")
    print(f"Brute force total time: {total_brute_time:.4f}s")
    print(f"Mean Recall@K: {mean_recall:.4f}")
    print(f"Total distance evaluations: {total_distance_evals}")
    print(f"Total nodes visited: {total_nodes_visited}")
    print(f"Total nodes pruned: {total_nodes_pruned}")
    print(f"Total leaves visited: {total_leaves_visited}")

    if config.output:
        output_path = Path("runs") / Path(config.output).name
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


class TeeOutput:
    """Write to both stdout and a file."""
    def __init__(self, file_path: str | None = None):
        self.file = None
        if file_path:
            self.file = open(file_path, "w")
    
    def write(self, text: str) -> None:
        print(text, end="")
        if self.file:
            self.file.write(text)
            self.file.flush()
    
    def flush(self) -> None:
        if self.file:
            self.file.flush()
    
    def close(self) -> None:
        if self.file:
            self.file.close()


def run_benchmark(config: argparse.Namespace) -> None:
    """Run the benchmark suite across all metrics and component counts."""
    from .benchmark.benchmark import BenchmarkConfig
    from .benchmark.benchmark import run_benchmark as bench_run

    bench_config = BenchmarkConfig(
        n_rso=config.n_rso,
        k=config.k,
        n_components=config.components,
        r_min=config.r_min,
        r_max=config.r_max,
        var_min=config.var_min,
        var_max=config.var_max,
        seed=config.seed,
        leaf_size=config.leaf_size,
    )

    # Set up tee output for logging (save to runs folder)
    Path("runs").mkdir(parents=True, exist_ok=True)
    tee = TeeOutput(str(Path("runs") / Path(config.log_output).name))
    
    try:
        # Write log header
        tee.write("=== RSO KNN Benchmark Log ===\n")
        tee.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        tee.write(f"Save log to: runs/{Path(config.log_output).name}\n")
        # Write initial settings (Mercurio spec)
        tee.write("=== Benchmark Configuration (Mercurio Specification) ===\n")
        tee.write(f"n_rso: {config.n_rso}\n")
        tee.write(f"k: {config.k}\n")
        tee.write(f"components: {config.components} (equal weights)\n")
        tee.write(f"Orbital radius range: [{config.r_min}, {config.r_max}] km\n")
        tee.write(f"Covariance variance range: [{config.var_min}, {config.var_max}] km^2 (diagonal)\n")
        tee.write(f"seed: {config.seed}\n")
        tee.write(f"leaf_size: {config.leaf_size}\n")
        tee.write(f"metrics: all (hellinger, wasserstein, mahalanobis, kl)\n")
        tee.write("\n")

        results = bench_run(bench_config, log_write=tee.write)
        df = results.to_dataframe()

        tee.write("\n=== Benchmark Results ===\n")
        tee.write(df.to_string(index=False))
        tee.write("\n")

        tee.write(f"\nCatalogue build time: {results.catalogue_build_time:.4f}s\n")
        tee.write(f"K-D tree build time: {results.kdtree_build_time:.4f}s\n")

        if config.output:
            output_path = Path("runs") / Path(config.output).name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            tee.write(f"Benchmark results written to {output_path}\n")
    finally:
        tee.close()


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
