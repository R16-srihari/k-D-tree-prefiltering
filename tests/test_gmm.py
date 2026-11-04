"""Tests for Gaussian component and GMM data models."""

import numpy as np
import pytest

from rso_kdtree.models.gaussian_component import GaussianComponent
from rso_kdtree.models.gaussian_mixture import GaussianMixture
from rso_kdtree.models.rso import RSO


def _make_component(mean=None, cov=None):
    if mean is None:
        mean = np.array([0.0, 0.0, 0.0])
    if cov is None:
        cov = np.eye(3)
    return GaussianComponent(mean=mean, covariance=cov)


def test_gaussian_component_valid():
    comp = _make_component()
    assert comp.mean.shape == (3,)
    assert comp.covariance.shape == (3, 3)


def test_gaussian_component_invalid_mean_shape():
    with pytest.raises(ValueError):
        GaussianComponent(mean=np.zeros(2), covariance=np.eye(3))


def test_gaussian_component_invalid_cov_shape():
    with pytest.raises(ValueError):
        GaussianComponent(mean=np.zeros(3), covariance=np.eye(2))


def test_gaussian_component_nonsymmetric_cov():
    cov = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        GaussianComponent(mean=np.zeros(3), covariance=cov)


def test_gaussian_component_not_positive_definite():
    cov = np.diag([1.0, 1.0, 0.0])  # singular
    with pytest.raises(ValueError):
        GaussianComponent(mean=np.zeros(3), covariance=cov)


def test_gaussian_component_sqrtm():
    comp = _make_component(cov=np.diag([4.0, 9.0, 16.0]))
    sqrtm = comp.sqrtm()
    reconstructed = sqrtm @ sqrtm
    assert np.allclose(reconstructed, comp.covariance)


def test_gmm_weights_must_sum_to_one():
    comp = _make_component()
    with pytest.raises(ValueError):
        GaussianMixture(weights=np.array([0.6, 0.5]), components=[comp, comp])


def test_gmm_weights_nonnegative():
    comp = _make_component()
    with pytest.raises(ValueError):
        GaussianMixture(weights=np.array([1.2, -0.2]), components=[comp, comp])


def test_gmm_weighted_mean():
    comp1 = _make_component(mean=np.array([0.0, 0.0, 0.0]))
    comp2 = _make_component(mean=np.array([10.0, 0.0, 0.0]))
    gmm = GaussianMixture(weights=np.array([0.5, 0.5]), components=[comp1, comp2])
    assert np.allclose(gmm.weighted_mean, np.array([5.0, 0.0, 0.0]))


def test_gmm_is_single_gaussian():
    comp = _make_component()
    gmm = GaussianMixture(weights=np.array([1.0]), components=[comp])
    assert gmm.is_single_gaussian()
    mu, cov = gmm.to_single_gaussian()
    assert np.allclose(mu, comp.mean)
    assert np.allclose(cov, comp.covariance)


def test_gmm_to_single_gaussian_raises():
    comp = _make_component()
    gmm = GaussianMixture(weights=np.array([0.5, 0.5]), components=[comp, comp])
    with pytest.raises(ValueError):
        gmm.to_single_gaussian()


def test_rso_creation():
    comp = _make_component()
    gmm = GaussianMixture(weights=np.array([1.0]), components=[comp])
    rso = RSO(
        object_id="RSO_0001",
        nominal_position=np.array([1.0, 2.0, 3.0]),
        uncertainty=gmm,
    )
    assert rso.object_id == "RSO_0001"
    assert rso.nominal_position.shape == (3,)
    assert rso.uncertainty is gmm


def test_rso_invalid_position_shape():
    comp = _make_component()
    gmm = GaussianMixture(weights=np.array([1.0]), components=[comp])
    with pytest.raises(ValueError):
        RSO(
            object_id="RSO_0001",
            nominal_position=np.array([1.0, 2.0]),
            uncertainty=gmm,
        )