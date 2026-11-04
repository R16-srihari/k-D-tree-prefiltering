"""Synthetic catalogue generation for RSO KNN screening.

Generates a reproducible catalogue of RSOs with nominal 3D positions and
3D Gaussian Mixture Model (GMM) positional uncertainty, following the
Mercurio report specifications:

- RSO positions: spherical shell between 6,500 and 8,000 km from origin
- Diagonal covariance matrices with variances between 1,000 and 10,000 km^2
- GMM with 6 components with equal weights

The same catalogue is reused across all distance metrics to ensure fair
metric comparisons.
"""

from __future__ import annotations

import numpy as np

from ..models.gaussian_component import GaussianComponent
from ..models.gaussian_mixture import GaussianMixture
from ..models.rso import RSO


def _generate_diagonal_covariance(rng: np.random.Generator, var_min: float, var_max: float) -> np.ndarray:
    """Generate a 3x3 diagonal covariance matrix with random variances.

    Variances are uniformly sampled from [var_min, var_max] for each axis.
    """
    variances = rng.uniform(var_min, var_max, size=3)
    return np.diag(variances)


def _sample_spherical_shell(rng: np.random.Generator, r_min: float, r_max: float, n: int) -> np.ndarray:
    """Sample n points uniformly from a spherical shell [r_min, r_max].

    Uses uniform sampling in radius^3 and uniform direction on sphere.
    """
    # Uniform in volume: r^3 uniform in [r_min^3, r_max^3]
    r3 = rng.uniform(r_min**3, r_max**3, size=n)
    radii = r3 ** (1.0 / 3.0)

    # Uniform direction on sphere (normal distribution + normalize)
    directions = rng.normal(0.0, 1.0, size=(n, 3))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    return directions * radii[:, np.newaxis]


def generate_synthetic_catalogue(
    n_rso: int = 2000,
    n_components: int = 6,
    r_min: float = 6500.0,
    r_max: float = 8000.0,
    seed: int = 42,
    var_min: float = 1000.0,
    var_max: float = 10000.0,
) -> list[RSO]:
    """Generate a synthetic catalogue of RSOs with GMM uncertainty per Mercurio report.

    Parameters
    ----------
    n_rso : int
        Number of RSOs to generate.
    n_components : int
        Number of Gaussian components per RSO (M=6 per Mercurio).
    r_min : float
        Minimum distance from origin (km). Default 6500 km.
    r_max : float
        Maximum distance from origin (km). Default 8000 km.
    seed : int
        Random seed for reproducibility.
    var_min : float
        Minimum variance for diagonal covariance (km^2). Default 1000.
    var_max : float
        Maximum variance for diagonal covariance (km^2). Default 10000.

    Returns
    -------
    list[RSO]
        Catalogue of RSOs with nominal positions and GMM uncertainty.
    """
    rng = np.random.default_rng(seed)

    # Nominal positions uniformly in spherical shell [r_min, r_max]
    positions = _sample_spherical_shell(rng, r_min, r_max, n_rso)

    rsos: list[RSO] = []

    # Equal weights for all components (per Mercurio)
    weights = np.ones(n_components) / n_components

    for i in range(n_rso):
        object_id = f"RSO_{i:04d}"
        nominal_position = positions[i]

        # Generate components
        components: list[GaussianComponent] = []
        for j in range(n_components):
            # Component mean near nominal position (small perturbation)
            # Using a small spread relative to orbital radius
            mean = nominal_position + rng.normal(0.0, 10.0, size=3)

            # Diagonal covariance matrix with variances in [var_min, var_max]
            cov = _generate_diagonal_covariance(rng, var_min, var_max)

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
