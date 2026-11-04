"""Tests for Gaussian-level distance measures."""

import numpy as np
import pytest

from rso_kdtree.distances.gaussian_distances import (
    hellinger_gaussian,
    wasserstein_gaussian,
    mahalanobis_gaussian,
    symmetric_kl_gaussian,
)
from rso_kdtree.distances.factory import create_distance_metric, list_supported_metrics


def _same_gaussian():
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
    return mean, cov


def test_hellinger_identical_is_zero():
    mean, cov = _same_gaussian()
    d = hellinger_gaussian(mean, cov, mean, cov)
    assert abs(d) < 1e-10


def test_hellinger_nonnegative():
    mean1 = np.zeros(3)
    cov1 = np.eye(3)
    mean2 = np.ones(3)
    cov2 = 2.0 * np.eye(3)
    d = hellinger_gaussian(mean1, cov1, mean2, cov2)
    assert d >= 0.0
    assert d <= 1.0 + 1e-9


def test_hellinger_symmetric():
    mean1 = np.zeros(3)
    cov1 = np.eye(3)
    mean2 = np.array([1.0, 0.5, -0.5])
    cov2 = np.array([[1.5, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 0.8]])
    d1 = hellinger_gaussian(mean1, cov1, mean2, cov2)
    d2 = hellinger_gaussian(mean2, cov2, mean1, cov1)
    assert abs(d1 - d2) < 1e-12


def test_wasserstein_identical_is_zero():
    mean, cov = _same_gaussian()
    d = wasserstein_gaussian(mean, cov, mean, cov)
    # Eigendecomposition roundoff gives ~1e-8 residual for identical inputs
    assert abs(d) < 1e-6


def test_wasserstein_nonnegative():
    mean1 = np.zeros(3)
    cov1 = np.eye(3)
    mean2 = np.ones(3)
    cov2 = 2.0 * np.eye(3)
    d = wasserstein_gaussian(mean1, cov1, mean2, cov2)
    assert d >= 0.0


def test_wasserstein_symmetric():
    mean1 = np.zeros(3)
    cov1 = np.eye(3)
    mean2 = np.array([1.0, 0.5, -0.5])
    cov2 = np.array([[1.5, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 0.8]])
    d1 = wasserstein_gaussian(mean1, cov1, mean2, cov2)
    d2 = wasserstein_gaussian(mean2, cov2, mean1, cov1)
    assert abs(d1 - d2) < 1e-10


def test_mahalanobis_identical_means_is_zero():
    mean = np.array([1.0, 2.0, 3.0])
    cov1 = np.eye(3)
    cov2 = 2.0 * np.eye(3)
    d = mahalanobis_gaussian(mean, cov1, mean, cov2)
    assert abs(d) < 1e-12


def test_mahalanobis_symmetric():
    mean1 = np.zeros(3)
    cov1 = np.eye(3)
    mean2 = np.array([1.0, 0.5, -0.5])
    cov2 = np.array([[1.5, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 0.8]])
    d1 = mahalanobis_gaussian(mean1, cov1, mean2, cov2)
    d2 = mahalanobis_gaussian(mean2, cov2, mean1, cov1)
    assert abs(d1 - d2) < 1e-12


def test_symmetric_kl_identical_is_zero():
    mean, cov = _same_gaussian()
    d = symmetric_kl_gaussian(mean, cov, mean, cov)
    assert abs(d) < 1e-10


def test_symmetric_kl_nonnegative():
    mean1 = np.zeros(3)
    cov1 = np.eye(3)
    mean2 = np.ones(3)
    cov2 = 2.0 * np.eye(3)
    d = symmetric_kl_gaussian(mean1, cov1, mean2, cov2)
    assert d >= -1e-10


def test_symmetric_kl_symmetric():
    mean1 = np.zeros(3)
    cov1 = np.eye(3)
    mean2 = np.array([1.0, 0.5, -0.5])
    cov2 = np.array([[1.5, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 0.8]])
    d1 = symmetric_kl_gaussian(mean1, cov1, mean2, cov2)
    d2 = symmetric_kl_gaussian(mean2, cov2, mean1, cov1)
    assert abs(d1 - d2) < 1e-10


def test_factory_supported_metrics():
    metrics = list_supported_metrics()
    assert set(metrics) == {"hellinger", "mahalanobis", "wasserstein", "kl"}


def test_factory_create_each_metric():
    for name in ["hellinger", "mahalanobis", "wasserstein", "kl"]:
        metric = create_distance_metric(name)
        assert metric.name == name


def test_factory_invalid_metric():
    with pytest.raises(ValueError):
        create_distance_metric("not_a_metric")


def test_distance_compute_interface():
    from rso_kdtree.models.gaussian_component import GaussianComponent
    from rso_kdtree.models.gaussian_mixture import GaussianMixture

    comp1 = GaussianComponent(mean=np.zeros(3), covariance=np.eye(3))
    comp2 = GaussianComponent(mean=np.ones(3), covariance=np.eye(3))
    gmm1 = GaussianMixture(weights=np.array([1.0]), components=[comp1])
    gmm2 = GaussianMixture(weights=np.array([1.0]), components=[comp2])

    metric = create_distance_metric("hellinger")
    d = metric.compute(gmm1, gmm2)
    assert d >= 0.0
    assert d <= 1.0 + 1e-9

    # Identical GMMs -> zero
    d_same = metric.compute(gmm1, gmm1)
    assert abs(d_same) < 1e-10