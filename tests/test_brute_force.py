"""Tests for the symmetric all-pairs brute-force reference (brute_force_knn_all).

Mercurio & Singla (2016, sec. 3) exploit the symmetry of the probabilistic
distance to halve brute-force work to N(N-1)/2 pair evaluations. These tests
pin that (1) every supported metric is indeed symmetric at the GMM level,
(2) the symmetric sweep performs exactly N(N-1)/2 evaluations, and (3) its
per-query results are identical to the per-query brute_force_knn reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from rso_kdtree.data.synthetic_catalogue import generate_synthetic_catalogue
from rso_kdtree.distances.factory import create_distance_metric
from rso_kdtree.screening.brute_force import brute_force_knn, brute_force_knn_all

ALL_METRICS = ["hellinger", "wasserstein", "mahalanobis", "kl"]


def _catalogue(n=12, seed=3):
    return generate_synthetic_catalogue(n_rso=n, n_components=6, seed=seed)


def test_supported_metrics_are_symmetric_at_gmm_level():
    cat = _catalogue(n=8, seed=11)
    for name in ALL_METRICS:
        metric = create_distance_metric(name)
        for i in range(len(cat)):
            for j in range(i + 1, len(cat)):
                d_ij = metric.compute(cat[i].uncertainty, cat[j].uncertainty)
                d_ji = metric.compute(cat[j].uncertainty, cat[i].uncertainty)
                assert d_ij == pytest.approx(d_ji), f"{name} asymmetric for {i},{j}"


def test_all_pairs_halves_evaluations():
    cat = _catalogue()
    n = len(cat)
    for name in ALL_METRICS:
        metric = create_distance_metric(name)
        _, stats = brute_force_knn_all(cat, k=3, distance_metric=metric)
        assert stats.distance_evaluations == n * (n - 1) // 2, name


def test_all_pairs_matches_per_query_reference():
    cat = _catalogue()
    k = 4
    for name in ALL_METRICS:
        metric = create_distance_metric(name)
        all_results, _ = brute_force_knn_all(cat, k=k, distance_metric=metric)
        assert len(all_results) == len(cat)
        for query, agg in zip(cat, all_results):
            ref, _ = brute_force_knn(query, cat, k=k, distance_metric=metric)
            assert agg.query_id == ref.query_id == query.object_id
            agg_ids = [n.object_id for n in agg.neighbors]
            ref_ids = [n.object_id for n in ref.neighbors]
            assert agg_ids == ref_ids, name
            for a, r in zip(agg.neighbors, ref.neighbors):
                # LP solve order differs between d(i,j) and d(j,i) by ~1 ulp
                assert a.probabilistic_distance == pytest.approx(r.probabilistic_distance)
                assert a.nominal_euclidean_distance == r.nominal_euclidean_distance


def test_all_pairs_k_larger_than_available():
    cat = _catalogue(n=5)
    metric = create_distance_metric("hellinger")
    results, _ = brute_force_knn_all(cat, k=10, distance_metric=metric)
    assert all(len(r.neighbors) == 4 for r in results)
