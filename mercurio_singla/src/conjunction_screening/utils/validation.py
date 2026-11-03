"""Shared validation and numerical utilities.

Used by the models and distances modules to validate Gaussian and GMM
parameters before computation.
"""

from __future__ import annotations

import numpy as np


def validate_gaussian(mean, covariance) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize a Gaussian's mean and covariance.

    Parameters
    ----------
    mean : array-like, shape (d,)
    covariance : array-like, shape (d, d)

    Returns
    -------
    tuple of (mean, covariance)
        Validated, symmetrized arrays.

    Raises
    ------
    ValueError
        If mean is not 1D, covariance is not (d, d), either contains
        non-finite values, or covariance is not positive definite.
    """
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    if mean.ndim != 1:
        raise ValueError("Mean must be a 1D array.")
    d = mean.size
    if covariance.shape != (d, d):
        raise ValueError("Covariance shape must be (d, d).")
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(mean)):
        raise ValueError("Mean contains non-finite values.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance contains non-finite values.")
    if np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise ValueError("Covariance must be positive definite.")
    return mean, covariance


def logdet_spd(matrix: np.ndarray) -> float:
    """Log-determinant of a symmetric positive definite matrix."""
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("Matrix must have positive determinant.")
    return float(value)


def validate_mixture_weights(weights: np.ndarray) -> np.ndarray:
    """Validate mixture weights: 1D, finite, non-negative, sum to 1."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("Weights must be a 1D array.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights contain non-finite values.")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1, got {weights.sum()}.")
    return weights


def validate_position(position) -> np.ndarray:
    """Validate a 3D Cartesian position vector."""
    position = np.asarray(position, dtype=float)
    if position.ndim != 1 or position.size != 3:
        raise ValueError(
            f"Position must be a 3D vector, got shape {position.shape}."
        )
    if not np.all(np.isfinite(position)):
        raise ValueError("Position contains non-finite values.")
    return position
