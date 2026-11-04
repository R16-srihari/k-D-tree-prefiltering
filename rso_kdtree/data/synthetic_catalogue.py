"""Synthetic catalogue generation for RSO KNN screening.

Generates a reproducible catalogue of RSOs with nominal 3D positions and
3D Gaussian Mixture Model (GMM) positional uncertainty.

The same catalogue is reused across all distance metrics to ensure fair
metric comparisons.
"""

from __future__ import annotations

import numpy as np

from ..models.gaussian_component import GaussianComponent
from ..models.gaussian_mixture import GaussianMixture
from ..models.rso import RSO


def _generate_valid_covariance(rng: np.random.Generator, scale: float) -> np.ndarray:
    """Generate a valid 3x3 symmetric positive definite covariance matrix.

    Uses Sigma = A @ A.T + epsilon * I to guarantee positive definiteness.
    """
    a = rng.normal(0.0, 1.0, size=(3, 3)) * np.sqrt(scale)
    cov = a @ a.T
    cov += 1e-6 * np.eye(3)  # numerical safeguard
    # Symmetrize
    cov = 0.5 * (cov + cov.T)
    return cov


def generate_synthetic_catalogue(
    n_rso: int = 2000,
    n_components: int = 3,
    cube_size: float = 1000.0,
    seed: int = 42,
    component_spread: float = 50.0,
    cov_scale: float = 100.0,
) -> list[RSO]:
    """Generate a synthetic catalogue of RSOs with GMM uncertainty.

    Parameters
    ----------
    n_rso : int
        Number of RSOs to generate.
    n_components : int
        Number of Gaussian components per RSO (M).
    cube_size : float
        Half-width L of the position cube [-L, L]^3 in km.
    seed : int
        Random seed for reproducibility.
    component_spread : float
        Standard deviation of component mean offsets from nominal position (km).
    cov_scale : float
        Base scale for component covariance matrices (km^2).

    Returns
    -------
    list[RSO]
        Catalogue of RSOs with nominal positions and GMM uncertainty.
    """
    rng = np.random.default_rng(seed)

    # Nominal positions uniformly within [-L, L]^3
    positions = rng.uniform(-cube_size, cube_size, size=(n_rso, 3))

    rsos: list[RSO] = []

    for i in range(n_rso):
        object_id = f"RSO_{i:04d}"
        nominal_position = positions[i]

        # Generate component weights and normalize
        weights = rng.random(n_components)
        weights = weights / weights.sum()

        # Generate components
        components: list[GaussianComponent] = []
        for j in range(n_components):
            # Component mean near nominal position
            mean = nominal_position + rng.normal(0.0, component_spread, size=3)

            # Component covariance
            cov = _generate_valid_covariance(rng, cov_scale)

            components.append(GaussianComponent(mean=mean, covariance=cov))

        uncertainty = GaussianMixture(weights=weights, components=components)

        rsos.append(
            RSO(
                object_id=object_id,
                nominal_position=nominal_position,
                uncertainty=uncertainty,
            )
        )

    return rsos