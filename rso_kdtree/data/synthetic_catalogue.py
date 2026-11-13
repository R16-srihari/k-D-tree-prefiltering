"""Synthetic catalogue generation for RSO KNN screening.

Generates a reproducible catalogue of RSOs with nominal 3D positions and
3D Gaussian Mixture Model (GMM) positional uncertainty, following the
Mercurio thesis (§4) and Mercurio & Singla (2016) specifications:

- RSO positions: spherical shell between 6,500 and 8,000 km from origin
- Diagonal covariance matrix Σ with variances between var_min and var_max km^2
  (default [100, 10000] per Mercurio & Singla 2016, Sec. 3: "variances between
  100 and 10,000 km"; the 2014 thesis states 1,000-10,000, but the more recent
  and specific CMES 2016 source takes precedence)
- GMM with exactly 6 components (2·N for N=3 dimensions)
- Component means: µ ± (√(3Σ))_i for i=1,2,3, where (√(3Σ))_i is the i-th
  column of the matrix square root of 3·Σ
- Component covariance: Σ/3 (per Mercurio thesis §4)
- Equal weights 1/6 each

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


def _matrix_sqrtm(matrix: np.ndarray) -> np.ndarray:
    """Symmetric matrix square root via eigendecomposition.

    Clamps small negative eigenvalues from numerical roundoff to zero.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return (eigenvectors * sqrt_eigenvalues) @ eigenvectors.T


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
    var_min: float = 100.0,
    var_max: float = 10000.0,
) -> list[RSO]:
    """Generate a synthetic catalogue of RSOs with GMM uncertainty per Mercurio.

    Follows Mercurio thesis §4 / 2016 paper §3:

    - Component means: χ_i = µ + (√(3Σ))_i and χ_{i+3} = µ − (√(3Σ))_i
      for i = 1, 2, 3, yielding 6 components in 3D.
    - Component covariance: Σ/3 (per thesis §4).
    - Equal weights 1/6 each.

    Parameters
    ----------
    n_rso : int
        Number of RSOs to generate.
    n_components : int
        Number of Gaussian components per RSO. Must equal 6 per Mercurio;
        any other value raises ValueError.
    r_min : float
        Minimum distance from origin (km). Default 6500 km.
    r_max : float
        Maximum distance from origin (km). Default 8000 km.
    seed : int
        Random seed for reproducibility.
    var_min : float
        Minimum variance for diagonal covariance (km^2). Default 100.
    var_max : float
        Maximum variance for diagonal covariance (km^2). Default 10000.

    Returns
    -------
    list[RSO]
        Catalogue of RSOs with nominal positions and GMM uncertainty.

    Raises
    ------
    ValueError
        If n_components != 6 (Mercurio requires exactly 6 components).
    """
    if n_components != 6:
        raise ValueError(
            f"n_components must be 6 per Mercurio specification, got {n_components}"
        )

    rng = np.random.default_rng(seed)

    # Nominal positions uniformly in spherical shell [r_min, r_max]
    positions = _sample_spherical_shell(rng, r_min, r_max, n_rso)

    rsos: list[RSO] = []

    # Equal weights: 1/6 each (per Mercurio)
    weights = np.ones(6) / 6.0

    for i in range(n_rso):
        object_id = f"RSO_{i:04d}"
        nominal_position = positions[i]

        # Generate diagonal covariance Σ with variances in [var_min, var_max]
        Sigma = _generate_diagonal_covariance(rng, var_min, var_max)

        # Compute matrix square root of 3·Σ via eigendecomposition
        # (√(3Σ))_i = i-th column of sqrtm(3·Σ)
        sqrt_3Sigma = _matrix_sqrtm(3.0 * Sigma)

        # Create 6 components: means µ ± δ_i for i = 1, 2, 3
        components: list[GaussianComponent] = []
        for j in range(3):
            delta = sqrt_3Sigma[:, j]
            # Plus direction: χ_j = µ + (√(3Σ))_j
            components.append(
                GaussianComponent(
                    mean=nominal_position + delta,
                    covariance=Sigma ,
                )
            )
            # Minus direction: χ_{j+3} = µ − (√(3Σ))_j
            components.append(
                GaussianComponent(
                    mean=nominal_position - delta,
                    covariance=Sigma ,
                )
            )

        uncertainty = GaussianMixture(weights=weights, components=components)

        rsos.append(
            RSO(
                object_id=object_id,
                nominal_position=nominal_position,
                uncertainty=uncertainty,
            )
        )

    return rsos
