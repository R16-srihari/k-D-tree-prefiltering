"""Scaling and timing experiments for the K-d tree screener.

Measures execution time, number of distance evaluations, and screening
accuracy across catalog sizes and k values.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import time

import numpy as np

from ..models import RSO, Gaussian
from ..distances import create_distance
from .metrics import evaluate_screening


def generate_gaussian_catalog(n_objects: int = 100,
                              n_clusters: int = 5,
                              cluster_spread: float = 1000.0,
                              base_cov_scale: float = 1.0,
                              anisotropy: float = 0.1,
                              seed: int = 42) -> List[RSO]:
    """Generate a synthetic Gaussian RSO catalog for experiments.

    Parameters
    ----------
    n_objects : int
        Total number of objects.
    n_clusters : int
        Number of spatial clusters.
    cluster_spread : float
        Std dev of cluster centers (km).
    base_cov_scale : float
        Base covariance scale (km^2).
    anisotropy : float
        Ratio of smallest to largest eigenvalue of covariance.
    seed : int
        Random seed.

    Returns
    -------
    list of RSO
    """
    rng = np.random.default_rng(seed)
    earth_radius = 6371.0  # km

    objects: List[RSO] = []
    objects_per_cluster = n_objects // n_clusters
    remainder = n_objects % n_clusters

    for c in range(n_clusters):
        inclination = rng.uniform(0, np.pi)
        raan = rng.uniform(0, 2 * np.pi)
        arg_peri = rng.uniform(0, 2 * np.pi)
        true_anomaly = rng.uniform(0, 2 * np.pi)
        orbit_radius = earth_radius + rng.uniform(400, 800)

        x_orb = orbit_radius * np.cos(true_anomaly)
        y_orb = orbit_radius * np.sin(true_anomaly)
        z_orb = 0.0

        cos_i, sin_i = np.cos(inclination), np.sin(inclination)
        cos_raan, sin_raan = np.cos(raan), np.sin(raan)
        cos_ap, sin_ap = np.cos(arg_peri), np.sin(arg_peri)

        R = np.array([
            [cos_raan * cos_ap - sin_raan * cos_i * sin_ap,
             -cos_raan * sin_ap - sin_raan * cos_i * cos_ap,
             sin_raan * sin_i],
            [sin_raan * cos_ap + cos_raan * cos_i * sin_ap,
             -sin_raan * sin_ap + cos_raan * cos_i * cos_ap,
             -cos_raan * sin_i],
            [sin_i * sin_ap,
             sin_i * cos_ap,
             cos_i],
        ])

        cluster_center = R @ np.array([x_orb, y_orb, z_orb])
        n_this_cluster = objects_per_cluster + (1 if c < remainder else 0)

        for _ in range(n_this_cluster):
            pos = cluster_center + rng.normal(0, cluster_spread / 10, 3)
            evals = base_cov_scale * np.array([1.0, 0.5, anisotropy])
            Q, _ = np.linalg.qr(rng.normal(0, 1, (3, 3)))
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            Sigma = Q @ np.diag(evals) @ Q.T
            Sigma = 0.5 * (Sigma + Sigma.T)

            objects.append(RSO(
                id=f"RSO-{len(objects)}",
                mean_position=pos,
                uncertainty=Gaussian(mean=pos, covariance=Sigma),
            ))

    return objects


def scaling_experiment(n_values: Optional[List[int]] = None,
                       k: int = 10,
                       metric_name: str = "hellinger",
                       seed: int = 42) -> Dict[int, Dict[str, Any]]:
    """Run timing/evaluation experiment across catalog sizes.

    Parameters
    ----------
    n_values : list of int, optional
        Catalog sizes to test. Defaults to [100, 200, 500, 1000, 2000].
    k : int
        Number of neighbours per object.
    metric_name : str
        Probabilistic distance metric name.
    seed : int
        Random seed.

    Returns
    -------
    dict mapping n -> {time, dh_evals, de_evals, recall, precision, ...}
    """
    if n_values is None:
        n_values = [100, 200, 500, 1000, 2000]

    metric = create_distance(metric_name)
    results: Dict[int, Dict[str, Any]] = {}

    for n in n_values:
        rsos = generate_gaussian_catalog(n, seed=seed)

        start = time.perf_counter()
        eval_out = evaluate_screening(rsos, metric, k=k)
        elapsed = time.perf_counter() - start

        results[n] = {
            "n": n,
            "time": elapsed,
            "tree_stats": eval_out["tree_stats"],
            "brute_stats": eval_out["brute_stats"],
            "metrics": eval_out["metrics"],
            "metrics_topk": eval_out["metrics_topk"],
        }

    return results


def knn_sweep(rsos: List[RSO],
              distance_metric,
              k_values: Optional[List[int]] = None) -> Dict[int, Dict[str, Any]]:
    """Sweep over k values measuring recall, precision, and timing.

    Parameters
    ----------
    rsos : list of RSO
    distance_metric : ProbabilisticDistance
    k_values : list of int, optional
        Defaults to [5, 10, 20, 30, 50].

    Returns
    -------
    dict mapping k -> {recall, precision, f1, time, dh_evals, de_evals}
    """
    if k_values is None:
        k_values = [5, 10, 20, 30, 50]

    results: Dict[int, Dict[str, Any]] = {}
    for k in k_values:
        start = time.perf_counter()
        eval_out = evaluate_screening(rsos, distance_metric, k=k)
        elapsed = time.perf_counter() - start

        results[k] = {
            "k": k,
            "time": elapsed,
            "recall": eval_out["metrics"]["recall"],
            "precision": eval_out["metrics"]["precision"],
            "f1": eval_out["metrics"]["f1"],
            "dh_evals": eval_out["tree_stats"].get("dh_evals", 0),
            "de_evals": eval_out["tree_stats"].get("de_evals", 0),
            "nodes_pruned": eval_out["tree_stats"].get("nodes_pruned", 0),
            "nodes_opened": eval_out["tree_stats"].get("nodes_opened", 0),
        }

    return results
