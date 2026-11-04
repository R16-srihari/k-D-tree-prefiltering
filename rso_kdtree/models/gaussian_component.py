"""Gaussian component data model with validation and numerical safeguards."""

from __future__ import annotations

import numpy as np


class GaussianComponent:
    """A single 3D Gaussian component.

    Attributes
    ----------
    mean : np.ndarray, shape (3,)
        Mean vector of the Gaussian.
    covariance : np.ndarray, shape (3, 3)
        Covariance matrix (symmetric positive definite).
    """

    def __init__(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=float).ravel()
        self.covariance = np.asarray(covariance, dtype=float)
        self._validate()

    def _validate(self) -> None:
        """Validate dimensions, symmetry, and positive definiteness."""
        if self.mean.shape != (3,):
            raise ValueError(
                f"GaussianComponent mean must have shape (3,), got {self.mean.shape}"
            )
        if self.covariance.shape != (3, 3):
            raise ValueError(
                f"GaussianComponent covariance must have shape (3, 3), got {self.covariance.shape}"
            )

        # Symmetry check
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-8):
            raise ValueError("GaussianComponent covariance must be symmetric")

        # Positive definiteness check via eigenvalues
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        if np.any(eigenvalues <= 1e-12):
            raise ValueError(
                f"GaussianComponent covariance must be positive definite, "
                f"min eigenvalue = {eigenvalues[0]:.3e}"
            )

    @property
    def determinant(self) -> float:
        """Determinant of the covariance matrix."""
        return float(np.linalg.det(self.covariance))

    @property
    def log_det(self) -> tuple[float, float]:
        """Sign and log-determinant via slogdet for numerical stability."""
        return np.linalg.slogdet(self.covariance)

    def inv(self) -> np.ndarray:
        """Inverse of the covariance matrix."""
        return np.linalg.inv(self.covariance)

    def sqrtm(self) -> np.ndarray:
        """Symmetric matrix square root via eigendecomposition.

        Handles small negative eigenvalues from numerical roundoff by
        clamping them to zero before taking the square root.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return (eigenvectors * sqrt_eigenvalues) @ eigenvectors.T

    def __repr__(self) -> str:
        return (
            f"GaussianComponent(mean={self.mean}, covariance={self.covariance})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GaussianComponent):
            return NotImplemented
        return (
            np.allclose(self.mean, other.mean)
            and np.allclose(self.covariance, other.covariance)
        )