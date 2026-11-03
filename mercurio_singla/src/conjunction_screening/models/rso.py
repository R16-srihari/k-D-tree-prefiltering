"""Resident Space Object (RSO) data model."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .gaussian import Gaussian, validate_gaussian
from .gmm import GaussianMixture


@dataclass
class RSO:
    """A Resident Space Object with mean position and uncertainty distribution.

    Attributes
    ----------
    id : str
        Original catalog identifier, preserved unchanged.
    mean_position : np.ndarray, shape (3,)
        Three-dimensional Cartesian mean position.
    uncertainty : Gaussian | GaussianMixture
        Uncertainty distribution, stored independently of the spatial index.
    """

    id: str
    mean_position: np.ndarray
    uncertainty: Gaussian | GaussianMixture

    def __post_init__(self) -> None:
        self.mean_position = np.asarray(self.mean_position, dtype=float)
        if self.mean_position.ndim != 1 or self.mean_position.size != 3:
            raise ValueError(
                f"mean_position must be a 3D vector, got shape {self.mean_position.shape}."
            )
        if not np.all(np.isfinite(self.mean_position)):
            raise ValueError("mean_position contains non-finite values.")
        if not isinstance(self.uncertainty, (Gaussian, GaussianMixture)):
            raise ValueError(
                "uncertainty must be a Gaussian or GaussianMixture instance."
            )

    @property
    def dim(self) -> int:
        """Dimensionality of the position space (always 3)."""
        return 3

    @property
    def overall_mean(self) -> np.ndarray:
        """Overall mean position used by the K-d tree.

        For a Gaussian this is the mean; for a GMM it is the mixture's
        weighted mean (sum_i w_i * mu_i).
        """
        return self.mean_position
