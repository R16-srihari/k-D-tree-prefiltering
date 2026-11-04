"""Gaussian Mixture Model data model with validation."""

from __future__ import annotations

import numpy as np

from .gaussian_component import GaussianComponent


class GaussianMixture:
    """A 3D Gaussian Mixture Model.

    Represents p(x) = sum_{i=1}^{M} alpha_i * N(x | mu_i, Sigma_i).

    Attributes
    ----------
    weights : np.ndarray, shape (M,)
        Mixture weights (nonnegative, sum to 1).
    components : list[GaussianComponent]
        List of Gaussian components.
    """

    def __init__(
        self,
        weights: np.ndarray,
        components: list[GaussianComponent],
    ) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.components = list(components)
        self._validate()

    def _validate(self) -> None:
        """Validate weights, component count, and dimensionality."""
        n_components = len(self.components)

        if self.weights.shape != (n_components,):
            raise ValueError(
                f"GaussianMixture weights must have shape ({n_components},), "
                f"got {self.weights.shape}"
            )

        if np.any(self.weights < 0):
            raise ValueError("GaussianMixture weights must be nonnegative")

        if not np.isclose(self.weights.sum(), 1.0, atol=1e-8):
            raise ValueError(
                f"GaussianMixture weights must sum to 1, got {self.weights.sum()}"
            )

        for i, comp in enumerate(self.components):
            if not isinstance(comp, GaussianComponent):
                raise TypeError(
                    f"Component {i} is not a GaussianComponent: {type(comp)}"
                )

    @property
    def n_components(self) -> int:
        """Number of Gaussian components."""
        return len(self.components)

    @property
    def means(self) -> np.ndarray:
        """Component means, shape (M, 3)."""
        return np.array([c.mean for c in self.components])

    @property
    def covariances(self) -> np.ndarray:
        """Component covariances, shape (M, 3, 3)."""
        return np.array([c.covariance for c in self.components])

    @property
    def weighted_mean(self) -> np.ndarray:
        """Mean position of the mixture (weighted average of component means)."""
        return np.sum(self.weights[:, None] * self.means, axis=0)

    def is_single_gaussian(self) -> bool:
        """Check if this is a single Gaussian (M=1)."""
        return self.n_components == 1

    def to_single_gaussian(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert M=1 GMM to (mean, covariance). Raises if M != 1."""
        if not self.is_single_gaussian():
            raise ValueError(
                f"GMM has {self.n_components} components, expected 1"
            )
        return self.components[0].mean, self.components[0].covariance

    def __repr__(self) -> str:
        return (
            f"GaussianMixture(n_components={self.n_components}, "
            f"weights={self.weights})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GaussianMixture):
            return NotImplemented
        if self.n_components != other.n_components:
            return False
        if not np.allclose(self.weights, other.weights):
            return False
        return all(c1 == c2 for c1, c2 in zip(self.components, other.components))