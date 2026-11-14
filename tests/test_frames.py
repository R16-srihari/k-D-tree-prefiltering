"""Unit tests for RTN-to-ECI frame conversion and covariance transformation."""

import numpy as np
import pytest

from rso_kdtree.data.frames import (
    compute_rtn_frame_vectors,
    compute_rtn_to_eci_rotation,
    transform_covariance_rtn_to_eci,
    transform_covariance_eci_to_rtn,
)


def _random_position_velocity(rng, r_mag=7000.0, v_mag=7.5):
    """Generate a random ECI position/velocity pair with non-zero angular momentum."""
    # Position: random direction
    r_dir = rng.normal(size=3)
    r_dir /= np.linalg.norm(r_dir)
    r = r_dir * r_mag

    # Velocity: mostly perpendicular to position (circular-ish orbit)
    v_dir = rng.normal(size=3)
    v_dir -= np.dot(v_dir, r_dir) * r_dir  # remove radial component
    v_dir /= np.linalg.norm(v_dir)
    # Add small radial component
    v_dir += 0.1 * r_dir
    v_dir /= np.linalg.norm(v_dir)
    v = v_dir * v_mag
    return r, v


def test_rtn_frame_vectors_are_orthonormal():
    """The RTN basis vectors should form an orthonormal right-handed triad."""
    rng = np.random.default_rng(42)
    r, v = _random_position_velocity(rng)
    r_hat, t_hat, n_hat = compute_rtn_frame_vectors(r, v)

    # Each vector is unit length
    assert abs(np.linalg.norm(r_hat) - 1.0) < 1e-12
    assert abs(np.linalg.norm(t_hat) - 1.0) < 1e-12
    assert abs(np.linalg.norm(n_hat) - 1.0) < 1e-12

    # Vectors are mutually orthogonal
    assert abs(np.dot(r_hat, t_hat)) < 1e-12
    assert abs(np.dot(r_hat, n_hat)) < 1e-12
    assert abs(np.dot(t_hat, n_hat)) < 1e-12

    # Right-handed: r x t = n
    assert np.allclose(np.cross(r_hat, t_hat), n_hat)


def test_rtn_frame_radial_aligned_with_position():
    """The radial basis vector should point along the position vector."""
    rng = np.random.default_rng(7)
    r, v = _random_position_velocity(rng)
    r_hat, _, _ = compute_rtn_frame_vectors(r, v)
    r_expected = r / np.linalg.norm(r)
    assert np.allclose(r_hat, r_expected)


def test_rotation_matrix_is_orthogonal():
    """The RTN-to-ECI rotation matrix should be orthogonal (R.T @ R = I)."""
    rng = np.random.default_rng(42)
    r, v = _random_position_velocity(rng)
    R = compute_rtn_to_eci_rotation(r, v)

    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12  # proper rotation


def test_rtn_to_eci_and_back_are_inverse():
    """Transforming RTN -> ECI -> RTN should recover the original covariance."""
    rng = np.random.default_rng(123)
    r, v = _random_position_velocity(rng)
    R = compute_rtn_to_eci_rotation(r, v)

    # Random symmetric positive definite covariance in RTN
    A = rng.normal(size=(3, 3))
    Sigma_rtn = A @ A.T + 0.1 * np.eye(3)

    Sigma_eci = transform_covariance_rtn_to_eci(Sigma_rtn, R)
    Sigma_rtn_back = transform_covariance_eci_to_rtn(Sigma_eci, R)

    assert np.allclose(Sigma_rtn, Sigma_rtn_back, atol=1e-12)


def test_covariance_transformation_preserves_eigenvalues():
    """Rotating a covariance should preserve its eigenvalues (trace and determinant)."""
    rng = np.random.default_rng(99)
    r, v = _random_position_velocity(rng)
    R = compute_rtn_to_eci_rotation(r, v)

    A = rng.normal(size=(3, 3))
    Sigma_rtn = A @ A.T + 0.1 * np.eye(3)
    Sigma_eci = transform_covariance_rtn_to_eci(Sigma_rtn, R)

    assert np.allclose(
        np.linalg.eigvalsh(Sigma_rtn), np.linalg.eigvalsh(Sigma_eci), atol=1e-10
    )
    assert abs(np.linalg.det(Sigma_rtn) - np.linalg.det(Sigma_eci)) < 1e-10
    assert abs(np.trace(Sigma_rtn) - np.trace(Sigma_eci)) < 1e-10


def test_identity_covariance_stays_identity():
    """An identity covariance should remain identity under any rotation."""
    rng = np.random.default_rng(5)
    r, v = _random_position_velocity(rng)
    R = compute_rtn_to_eci_rotation(r, v)

    Sigma_eci = transform_covariance_rtn_to_eci(np.eye(3), R)
    assert np.allclose(Sigma_eci, np.eye(3), atol=1e-12)


def test_zero_position_raises():
    """A zero position vector should raise ValueError."""
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Position vector is zero"):
        compute_rtn_to_eci_rotation(np.zeros(3), v)


def test_parallel_position_velocity_raises():
    """Parallel position and velocity (zero angular momentum) should raise ValueError."""
    r = np.array([1.0, 0.0, 0.0])
    v = np.array([2.0, 0.0, 0.0])  # parallel to r
    with pytest.raises(ValueError, match="parallel|angular momentum"):
        compute_rtn_to_eci_rotation(r, v)


def test_invalid_shapes_raise():
    """Invalid input shapes should raise ValueError."""
    with pytest.raises(ValueError):
        compute_rtn_to_eci_rotation(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        compute_rtn_to_eci_rotation(np.zeros(3), np.zeros(2))

    R = np.eye(3)
    with pytest.raises(ValueError):
        transform_covariance_rtn_to_eci(np.eye(2), R)
    with pytest.raises(ValueError):
        transform_covariance_rtn_to_eci(np.eye(3), np.eye(2))


def test_diagonal_rtn_covariance_rotation():
    """A diagonal RTN covariance rotated to ECI should have the expected structure."""
    rng = np.random.default_rng(314)
    r, v = _random_position_velocity(rng)
    R = compute_rtn_to_eci_rotation(r, v)

    # Diagonal RTN covariance with different variances per axis
    Sigma_rtn = np.diag([1.0, 4.0, 9.0])  # sigma_R=1, sigma_T=2, sigma_N=3 km
    Sigma_eci = transform_covariance_rtn_to_eci(Sigma_rtn, R)

    # Result should be symmetric positive definite
    assert np.allclose(Sigma_eci, Sigma_eci.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(Sigma_eci) > 0)

    # Trace should be preserved (1 + 4 + 9 = 14)
    assert abs(np.trace(Sigma_eci) - 14.0) < 1e-10

    # Transforming back should recover the diagonal
    Sigma_rtn_back = transform_covariance_eci_to_rtn(Sigma_eci, R)
    assert np.allclose(Sigma_rtn_back, Sigma_rtn, atol=1e-12)
