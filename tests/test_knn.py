"""Tests for probabilistic KNN search (Alg. 3 traversal) vs brute-force reference."""

from __future__ import annotations

import numpy as np
import pytest

from rso_kdtree.data.synthetic_catalogue import generate_synthetic_catalogue
from rso_kdtree.distances.factory import create_distance_metric
from rso_kdtree.models.gaussian_component import GaussianComponent
from rso_kdtree.models.gaussian_mixture import GaussianMixture
from rso_kdtree.models.rso import RSO
from rso_kdtree.screening.brute_force import brute_force_knn
from rso_kdtree.screening.knn import probabilistic_knn
from rso_kdtree.spatial import build_kdtree

ALL_METRICS = ["hellinger", "wasserstein", "mahalanobis", "kl"]


def _build_catalogue(n=30, seed=7, var_min=1000.0, var_max=10000.0):
    """6-component GMM catalogue per the Mercurio & Singla (2016) spec."""
    return generate_synthetic_catalogue(
        n_rso=n, n_components=6, seed=seed, var_min=var_min, var_max=var_max
    )


def _build_tree(catalogue, leaf_size=1):
    positions = np.array([r.nominal_position for r in catalogue])
    return build_kdtree(positions, catalogue, leaf_size=leaf_size)


def _build_single_gaussian_catalogue(n=30, seed=7, var=1_000_000.0):
    """RSOs with identical single-Gaussian uncertainty (covariance ``var * I``).

    With identical covariances, every supported metric is a strictly monotone
    function of the Euclidean mean separation, so Alg. 3's deterministic
    distance-to-box pruning is exact. ``var`` must be large enough that
    Hellinger does not saturate at exactly 1.0 for shell-scale separations
    (~2000-10000 km): BC = exp(-d^2 / (8*var)), so var = 1e6 km^2 keeps the
    top-k Hellinger values distinct in double precision.
    """
    rng = np.random.default_rng(seed)
    r3 = rng.uniform(6500.0**3, 8000.0**3, size=n)
    radii = r3 ** (1.0 / 3.0)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions * radii[:, None]
    rsos = []
    for i, pos in enumerate(positions):
        comp = GaussianComponent(mean=pos, covariance=var * np.eye(3))
        gmm = GaussianMixture(weights=np.array([1.0]), components=[comp])
        rsos.append(
            RSO(object_id=f"RSO_{i:04d}", nominal_position=pos, uncertainty=gmm)
        )
    return rsos


def _recall_at_k(kd_result, bf_result) -> float:
    kd_ids = {n.object_id for n in kd_result.neighbors}
    bf_ids = {n.object_id for n in bf_result.neighbors}
    if not bf_ids:
        return 1.0
    return len(kd_ids & bf_ids) / len(bf_ids)


def test_brute_force_excludes_self():
    catalogue = _build_catalogue()
    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    result, stats = brute_force_knn(query, catalogue, k=5, distance_metric=metric)
    ids = {n.object_id for n in result.neighbors}
    assert query.object_id not in ids


def test_brute_force_returns_k():
    catalogue = _build_catalogue()
    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    result, _ = brute_force_knn(query, catalogue, k=10, distance_metric=metric)
    assert len(result.neighbors) == 10


def test_brute_force_sorted_by_probabilistic_distance():
    catalogue = _build_catalogue()
    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    result, _ = brute_force_knn(query, catalogue, k=10, distance_metric=metric)
    dists = [n.probabilistic_distance for n in result.neighbors]
    assert dists == sorted(dists)


def test_kdtree_knn_excludes_self():
    catalogue = _build_catalogue()
    tree = _build_tree(catalogue)
    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    result, _ = probabilistic_knn(query, tree, catalogue, k=5, distance_metric=metric)
    ids_returned = {n.object_id for n in result.neighbors}
    assert query.object_id not in ids_returned


def test_kdtree_knn_returns_k():
    catalogue = _build_catalogue()
    tree = _build_tree(catalogue)
    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    result, _ = probabilistic_knn(query, tree, catalogue, k=10, distance_metric=metric)
    assert len(result.neighbors) == 10


def test_kdtree_knn_sorted():
    catalogue = _build_catalogue()
    tree = _build_tree(catalogue)
    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    result, _ = probabilistic_knn(query, tree, catalogue, k=10, distance_metric=metric)
    dists = [n.probabilistic_distance for n in result.neighbors]
    assert dists == sorted(dists)


def test_kdtree_k_handles_larger_than_available():
    # Regression test for the DE_min threshold fix: k > n - 1 must not
    # crash, because the effective neighbour count is k_available = min(k, n-1).
    catalogue = _build_catalogue(n=5)
    tree = _build_tree(catalogue, leaf_size=2)
    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    # k=10 but only 4 other RSOs available
    result, _ = probabilistic_knn(query, tree, catalogue, k=10, distance_metric=metric)
    assert len(result.neighbors) == 4


def test_kdtree_knn_deterministic():
    catalogue = _build_catalogue()
    metric = create_distance_metric("hellinger")
    query = catalogue[0]

    tree1 = _build_tree(catalogue)
    result1, _ = probabilistic_knn(query, tree1, catalogue, k=5, distance_metric=metric)

    tree2 = _build_tree(catalogue)
    result2, _ = probabilistic_knn(query, tree2, catalogue, k=5, distance_metric=metric)

    ids1 = [n.object_id for n in result1.neighbors]
    ids2 = [n.object_id for n in result2.neighbors]
    assert ids1 == ids2


def test_all_four_metrics_knn():
    catalogue = _build_catalogue()
    tree = _build_tree(catalogue)
    for metric_name in ALL_METRICS:
        metric = create_distance_metric(metric_name)
        query = catalogue[0]
        result, stats = probabilistic_knn(query, tree, catalogue, k=5, distance_metric=metric)
        assert len(result.neighbors) == 5
        assert result.metric == metric_name
        assert stats.distance_evaluations > 0


def test_kdtree_prunes_below_brute_force_evaluations():
    # Alg. 3 must evaluate fewer object pairs than brute force in total.
    catalogue = _build_catalogue(n=40)
    tree = _build_tree(catalogue)
    metric = create_distance_metric("wasserstein")
    total_kd = 0
    for query in catalogue:
        _, stats = probabilistic_knn(query, tree, catalogue, k=5, distance_metric=metric)
        total_kd += stats.distance_evaluations
    brute_total = len(catalogue) * (len(catalogue) - 1)
    assert total_kd < brute_total


# ---------------------------------------------------------------------------
# Correctness of the Alg. 3 traversal against the brute-force reference
# ---------------------------------------------------------------------------


def test_kdtree_knn_exact_recall_identical_covariances():
    """With identical covariances every supported metric is a strictly
    monotone function of the Euclidean mean separation, so the deterministic
    D_box < DE_min pruning of Alg. 3 is exact: the KD-tree must reproduce
    the brute-force top-k for every query (Recall@K == 1.0)."""
    catalogue = _build_single_gaussian_catalogue(n=30, seed=7)
    tree = _build_tree(catalogue)
    k = 5
    for metric_name in ALL_METRICS:
        metric = create_distance_metric(metric_name)
        for query in catalogue:
            kd_result, _ = probabilistic_knn(
                query, tree, catalogue, k=k, distance_metric=metric
            )
            bf_result, _ = brute_force_knn(query, catalogue, k=k, distance_metric=metric)
            recall = _recall_at_k(kd_result, bf_result)
            assert recall == 1.0, f"{metric_name} query {query.object_id}: recall {recall}"


def test_kdtree_knn_distances_match_recomputed():
    # Every neighbour returned by the KD-tree must carry the same probabilistic
    # distance that a fresh metric evaluation produces for that pair.
    catalogue = _build_catalogue(n=20, seed=13)
    tree = _build_tree(catalogue)
    metric = create_distance_metric("hellinger")
    by_id = {r.object_id: r for r in catalogue}
    for query in catalogue[:5]:
        kd_result, _ = probabilistic_knn(query, tree, catalogue, k=5, distance_metric=metric)
        for n in kd_result.neighbors:
            expected = metric.compute(query.uncertainty, by_id[n.object_id].uncertainty)
            assert n.probabilistic_distance == pytest.approx(expected)


def test_kdtree_knn_never_beats_exhaustive_topk():
    # Brute force is exhaustive, so the KD-tree's k-th best probabilistic
    # distance can never be smaller than the true k-th best.
    catalogue = _build_catalogue(n=30, seed=17)
    tree = _build_tree(catalogue)
    metric = create_distance_metric("hellinger")
    k = 5
    for query in catalogue:
        kd_result, _ = probabilistic_knn(query, tree, catalogue, k=k, distance_metric=metric)
        bf_result, _ = brute_force_knn(query, catalogue, k=k, distance_metric=metric)
        assert (
            kd_result.neighbors[-1].probabilistic_distance
            >= bf_result.neighbors[-1].probabilistic_distance - 1e-12
        )


def test_kdtree_knn_recall_default_gmm_catalogue():
    """Recall@K on the 6-component GMM catalogue with default variances.
    Mercurio & Singla report 98.6%+ nearest-neighbour capture; with large
    variances Hellinger saturates near 1, so its floor is lower (the paper's
    own acknowledged deterministic distance-to-box limitation)."""
    catalogue = _build_catalogue(n=40, seed=19)
    tree = _build_tree(catalogue)
    k = 5
    floors = {"hellinger": 0.5, "wasserstein": 0.9, "mahalanobis": 0.9, "kl": 0.9}
    for metric_name in ALL_METRICS:
        metric = create_distance_metric(metric_name)
        recalls = []
        for query in catalogue:
            kd_result, _ = probabilistic_knn(
                query, tree, catalogue, k=k, distance_metric=metric
            )
            bf_result, _ = brute_force_knn(query, catalogue, k=k, distance_metric=metric)
            recalls.append(_recall_at_k(kd_result, bf_result))
        mean_recall = float(np.mean(recalls))
        assert mean_recall >= floors[metric_name], (
            f"{metric_name}: mean Recall@K {mean_recall:.3f} < {floors[metric_name]}"
        )


def test_brute_force_all_metrics():
    catalogue = _build_catalogue()
    for metric_name in ["hellinger", "wasserstein", "mahalanobis", "kl"]:
        metric = create_distance_metric(metric_name)
        query = catalogue[0]
        result, stats = brute_force_knn(query, catalogue, k=5, distance_metric=metric)
        assert len(result.neighbors) == 5
        assert stats.distance_evaluations == len(catalogue) - 1
