"""Tests for probabilistic KNN search and brute-force reference."""

import numpy as np
import pytest

from rso_kdtree.data.synthetic_catalogue import generate_synthetic_catalogue
from rso_kdtree.distances.factory import create_distance_metric
from rso_kdtree.screening.brute_force import brute_force_knn
from rso_kdtree.screening.knn import probabilistic_knn
from rso_kdtree.spatial import build_kdtree


def _build_catalogue(n=30, seed=7, components=2):
    return generate_synthetic_catalogue(
        n_rso=n, n_components=components, cube_size=100.0, seed=seed
    )


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
    positions = np.array([r.nominal_position for r in catalogue])
    ids = [r.object_id for r in catalogue]
    tree = build_kdtree(positions, leaf_size=5)

    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    result, stats = probabilistic_knn(query, tree, catalogue, k=5, distance_metric=metric)
    ids_returned = {n.object_id for n in result.neighbors}
    assert query.object_id not in ids_returned


def test_kdtree_knn_returns_k():
    catalogue = _build_catalogue()
    positions = np.array([r.nominal_position for r in catalogue])
    ids = [r.object_id for r in catalogue]
    tree = build_kdtree(positions, leaf_size=5)

    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    result, _ = probabilistic_knn(query, tree, catalogue, k=10, distance_metric=metric)
    assert len(result.neighbors) == 10


def test_kdtree_knn_sorted():
    catalogue = _build_catalogue()
    positions = np.array([r.nominal_position for r in catalogue])
    ids = [r.object_id for r in catalogue]
    tree = build_kdtree(positions, leaf_size=5)

    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    result, _ = probabilistic_knn(query, tree, catalogue, k=10, distance_metric=metric)
    dists = [n.probabilistic_distance for n in result.neighbors]
    assert dists == sorted(dists)


def test_kdtree_k_handles_larger_than_available():
    catalogue = _build_catalogue(n=5)
    positions = np.array([r.nominal_position for r in catalogue])
    ids = [r.object_id for r in catalogue]
    tree = build_kdtree(positions, leaf_size=2)

    metric = create_distance_metric("hellinger")
    query = catalogue[0]
    # k=10 but only 4 other RSOs available
    result, _ = probabilistic_knn(query, tree, catalogue, k=10, distance_metric=metric)
    assert len(result.neighbors) == 4


def test_kdtree_knn_deterministic():
    catalogue = _build_catalogue()
    positions = np.array([r.nominal_position for r in catalogue])
    ids = [r.object_id for r in catalogue]

    metric = create_distance_metric("hellinger")
    query = catalogue[0]

    tree1 = build_kdtree(positions, leaf_size=5)
    result1, _ = probabilistic_knn(query, tree1, catalogue, k=5, distance_metric=metric)

    tree2 = build_kdtree(positions, leaf_size=5)
    result2, _ = probabilistic_knn(query, tree2, catalogue, k=5, distance_metric=metric)

    ids1 = [n.object_id for n in result1.neighbors]
    ids2 = [n.object_id for n in result2.neighbors]
    assert ids1 == ids2


def test_all_four_metrics_knn():
    catalogue = _build_catalogue()
    positions = np.array([r.nominal_position for r in catalogue])
    ids = [r.object_id for r in catalogue]
    tree = build_kdtree(positions, leaf_size=5)

    for metric_name in ["hellinger", "wasserstein", "mahalanobis", "kl"]:
        metric = create_distance_metric(metric_name)
        query = catalogue[0]
        result, stats = probabilistic_knn(query, tree, catalogue, k=5, distance_metric=metric)
        assert len(result.neighbors) == 5
        assert result.metric == metric_name
        assert stats.distance_evaluations > 0


def test_brute_force_all_metrics():
    catalogue = _build_catalogue()
    for metric_name in ["hellinger", "wasserstein", "mahalanobis", "kl"]:
        metric = create_distance_metric(metric_name)
        query = catalogue[0]
        result, stats = brute_force_knn(query, catalogue, k=5, distance_metric=metric)
        assert len(result.neighbors) == 5
        assert stats.distance_evaluations == len(catalogue) - 1