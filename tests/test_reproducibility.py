"""Tests for catalogue reproducibility and metric switching."""

import numpy as np

from rso_kdtree.data.synthetic_catalogue import generate_synthetic_catalogue
from rso_kdtree.distances.factory import create_distance_metric


def test_same_seed_produces_same_catalogue():
    cat1 = generate_synthetic_catalogue(n_rso=50, seed=42)
    cat2 = generate_synthetic_catalogue(n_rso=50, seed=42)

    assert len(cat1) == len(cat2)
    for r1, r2 in zip(cat1, cat2):
        assert r1.object_id == r2.object_id
        assert np.allclose(r1.nominal_position, r2.nominal_position)
        assert np.allclose(r1.uncertainty.weights, r2.uncertainty.weights)
        for c1, c2 in zip(r1.uncertainty.components, r2.uncertainty.components):
            assert np.allclose(c1.mean, c2.mean)
            assert np.allclose(c1.covariance, c2.covariance)


def test_different_seed_produces_different_catalogue():
    cat1 = generate_synthetic_catalogue(n_rso=50, seed=42)
    cat2 = generate_synthetic_catalogue(n_rso=50, seed=99)

    different = any(
        not np.allclose(r1.nominal_position, r2.nominal_position)
        for r1, r2 in zip(cat1, cat2)
    )
    assert different


def test_component_counts_supported():
    for m in [1, 2, 3, 5]:
        cat = generate_synthetic_catalogue(n_rso=10, n_components=m, seed=42)
        for rso in cat:
            assert rso.uncertainty.n_components == m


def test_metric_switching_does_not_regenerate_catalogue():
    # The catalogue is generated once; metric selection is independent.
    cat = generate_synthetic_catalogue(n_rso=20, seed=42)
    for metric_name in ["hellinger", "wasserstein", "mahalanobis", "kl"]:
        metric = create_distance_metric(metric_name)
        assert metric is not None
        # All metrics operate on the same catalogue objects
        rso = cat[0]
        assert rso.uncertainty.n_components == 3