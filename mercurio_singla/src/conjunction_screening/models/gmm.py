"""Gaussian Mixture Model representation for RSO uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .gaussian import Gaussian, validate_gaussian


@dataclass
class GaussianMixture:
    """A Gaussian mixture model.

    Attributes
    ----------
    weights : np.ndarray, shape (K,)
        Mixture weights. Must be non-negative and sum to 1.
    components : list[Gaussian]
        Component Gaussian distributions.
    """

    weights: np.ndarray
    components: list[Gaussian]

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        K = self.weights.shape[0]
        if self.weights.ndim != 1:
            raise ValueError("Weights must be a 1D array.")
        if len(self.components) != K:
            raise ValueError(
                f"Number of components ({len(self.components)}) must match "
                f"number of weights ({K})."
            )
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("Weights contain non-finite values.")
        if np.any(self.weights < 0):
            raise ValueError("Weights must be non-negative.")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1, got {self.weights.sum()}.")
        for i, comp in enumerate(self.components):
            if not isinstance(comp, Gaussian):
                raise ValueError(f"Component {i} is not a Gaussian instance.")

    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        return int(self.weights.size)

    @property
    def means(self) -> np.ndarray:
        """Component means, shape (K, d)."""
        return np.stack([comp.mean for comp in self.components], axis=0)

    @property
    def covariances(self) -> np.ndarray:
        """Component covariances, shape (K, d, d)."""
        return np.stack([comp.covariance for comp in self.components], axis=0)

    @property
    def dim(self) -> int:
        """Dimensionality of the distribution."""
        return int(self.components[0].mean.size)

    @property
    def mean(self) -> np.ndarray:
        """Overall mean of the mixture: sum_i w_i * mu_i."""
        return np.sum(self.weights[:, None] * self.means, axis=0)

    def is_gaussian(self) -> bool:
        """Return True if the mixture is a single Gaussian (K == 1)."""
        return self.n_components == 1

    def to_gaussian(self) -> Gaussian:
        """Convert a single-component GMM to a Gaussian."""
        if not self.is_gaussian():
            raise ValueError("Only a single-component GMM can be converted to Gaussian.")
        return self.components[0]
