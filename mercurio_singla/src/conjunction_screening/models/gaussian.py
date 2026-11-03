"""Gaussian (normal) distribution representation for RSO uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Gaussian:
    """A multivariate Gaussian distribution.

    Attributes
    ----------
    mean : np.ndarray, shape (d,)
        Mean vector of the distribution.
    covariance : np.ndarray, shape (d, d)
        Covariance matrix. Must be symmetric positive definite.
    """

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        self.mean, self.covariance = validate_gaussian(self.mean, self.covariance)

    @property
    def dim(self) -> int:
        """Dimensionality of the distribution."""
        return int(self.mean.size)

    def is_valid(self) -> bool:
        """Return True if the distribution is well-formed."""
        try:
            validate_gaussian(self.mean, self.covariance)
            return True
        except ValueError:
            return False


def validate_gaussian(mean, covariance) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize a Gaussian's mean and covariance.

    Parameters
    ----------
    mean : array-like, shape (d,)
        Mean vector.
    covariance : array-like, shape (d, d)
        Covariance matrix.

    Returns
    -------
    tuple of (mean, covariance)
        Validated, symmetrized arrays.

    Raises
    ------
    ValueError
        If the mean is not 1D, the covariance is not (d, d), or either
        contains non-finite values, or the covariance is not positive
        definite.
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
    """Log-determinant of a symmetric positive definite matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (d, d)
        SPD matrix.

    Returns
    -------
    float
        Natural logarithm of the determinant.

    Raises
    ------
    ValueError
        If the matrix does not have a positive determinant.
    """
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("Matrix must have positive determinant.")
    return float(value)
