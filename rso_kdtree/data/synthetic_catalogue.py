"""Synthetic catalogue generation for RSO KNN screening.

Generates a reproducible catalogue of RSOs with nominal 3D positions taken from
``data/satellite_propagated.csv`` (SGP4-propagated Space-Track GP data produced by
``data/space-track_catalogue/space-track_data-gen.py``) and 3D Gaussian Mixture Model
(GMM) positional uncertainty, following the Mercurio thesis (§4) and Mercurio & Singla
(2016) specifications:

- RSO positions: the first ``n_rso`` usable rows of ``satellite_propagated.csv`` whose
  X/Y/Z fields are finite (km). Rows are taken in file order.
- Covariance matrix Σ: diagonal with variances between var_min and var_max km^2,
  generated in the RTN frame of each orbit and rotated to ECI using the propagated
  position and velocity (per the SP-672 reference, where covariances are stored in
  RTN coordinates).
- GMM with exactly 6 components (2·N for N=3 dimensions)
- Component means: µ ± (√(3Σ))_i for i=1,2,3, where (√(3Σ))_i is the i-th
  column of the matrix square root of 3·Σ
- Component covariance: Σ/3 (per Mercurio thesis §4)
- Equal weights 1/6 each

The same catalogue is reused across all distance metrics to ensure fair
metric comparisons. Nominal positions are deterministic given the CSV file; the
``seed`` parameter controls only the random diagonal covariances.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..models.gaussian_component import GaussianComponent
from ..models.gaussian_mixture import GaussianMixture
from ..models.rso import RSO
from .frames import compute_rtn_to_eci_rotation, transform_covariance_rtn_to_eci

_DEFAULT_CSV_PATH = Path(__file__).parent / "satellite_propagated.csv"


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


def _load_csv_positions_and_velocities(
    csv_path: Path, n_rso: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load the first ``n_rso`` usable nominal positions and velocities from a satellite CSV.

    The CSV (produced by ``space-track_data-gen.py``) has columns
    ``NORAD_CAT_ID, OBJECT_NAME, EPOCH, X, Y, Z, VX, VY, VZ`` (km / km/s). Rows whose
    X/Y/Z are not finite are skipped (defensive: a regenerated file may contain
    ``nan`` entries from failed SGP4 propagations). Returns ``(object_id, position,
    velocity)`` triples in file order, where ``object_id`` is the sequential index
    within the returned list (``RSO_{i:04d}`` is assigned by the caller).

    Parameters
    ----------
    csv_path : Path
        Path to ``satellite_propagated.csv``.
    n_rso : int
        Number of position/velocity pairs to return.

    Returns
    -------
    list[tuple[str, np.ndarray, np.ndarray]]
        ``(object_id, position_km, velocity_km_per_s)`` triples, length ``n_rso``.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If fewer than ``n_rso`` rows have finite X/Y/Z.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"satellite catalogue CSV not found: {csv_path}")

    entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["X"])
                y = float(row["Y"])
                z = float(row["Z"])
                vx = float(row["VX"])
                vy = float(row["VY"])
                vz = float(row["VZ"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            object_id = str(row.get("NORAD_CAT_ID", "")).strip() or f"RSO_{len(entries):04d}"
            entries.append(
                (
                    object_id,
                    np.array([x, y, z], dtype=float),
                    np.array([vx, vy, vz], dtype=float),
                )
            )
            if len(entries) >= n_rso:
                break

    if len(entries) < n_rso:
        raise ValueError(
            f"satellite catalogue CSV {csv_path} has only {len(entries)} usable rows "
            f"(finite X/Y/Z), but n_rso={n_rso} requested"
        )
    return entries


def generate_synthetic_catalogue(
    n_rso: int = 200,
    n_components: int = 6,
    r_min: float = 6500.0,
    r_max: float = 8000.0,
    seed: int = 42,
    var_min: float = 1e-4,
    var_max: float = 100.0,
    cov_frame: str = "RTN",
    csv_path: Path | str | None = None,
) -> list[RSO]:
    """Generate a synthetic catalogue of RSOs with GMM uncertainty per Mercurio.

    Follows Mercurio thesis §4 / 2016 paper §3:

    - Nominal positions: first ``n_rso`` usable rows of ``satellite_propagated.csv``
      (finite X/Y/Z), taken in file order. Positions are deterministic given the CSV.
    - Covariance matrix Σ: diagonal with variances in [var_min, var_max] km^2.
      The diagonal covariance is generated in the RTN frame of each orbit
      (per the SP-672 reference, where covariances are stored in RTN coordinates)
      and rotated to ECI using the propagated position and velocity.
    - Component means: χ_i = µ + (√(3Σ))_i and χ_{i+3} = µ − (√(3Σ))_i
      for i = 1, 2, 3, yielding 6 components in 3D.
    - Component covariance: Σ/3 (per thesis §4).
    - Equal weights 1/6 each.

    Parameters
    ----------
    n_rso : int
        Number of RSOs to generate (default 200). Must not exceed the number of usable
        rows in the CSV.
    n_components : int
        Number of Gaussian components per RSO. Must equal 6 per Mercurio;
        any other value raises ValueError.
    r_min : float
        Deprecated; retained for backward compatibility. Nominal positions now come from
        ``satellite_propagated.csv`` rather than a spherical shell.
    r_max : float
        Deprecated; retained for backward compatibility. See ``r_min``.
    seed : int
        Random seed controlling only the random diagonal covariances (positions are
        deterministic from the CSV). Default 42.
    var_min : float
        Minimum variance for diagonal covariance (km^2). Default 1e-4 (sigma = 0.01 km),
        per the realistic 0.01–10 km standard deviation scales from the SP-672 report.
    var_max : float
        Maximum variance for diagonal covariance (km^2). Default 100.0 (sigma = 10 km),
        per the realistic 0.01–10 km standard deviation scales from the SP-672 report.
    cov_frame : str
        Frame in which the diagonal covariance is specified. Only "RTN" is supported;
        the covariance is generated in the RTN frame of each orbit and rotated to ECI
        using the propagated position and velocity from the CSV.
    csv_path : Path | str | None
        Path to ``satellite_propagated.csv``. Defaults to the file next to this module.

    Returns
    -------
    list[RSO]
        Catalogue of RSOs with nominal positions and GMM uncertainty.

    Raises
    ------
    ValueError
        If ``n_components != 6`` (Mercurio requires exactly 6 components), or if the CSV
        has fewer than ``n_rso`` usable rows, or if ``cov_frame`` is not "RTN".
    FileNotFoundError
        If ``csv_path`` does not exist.
    """
    if n_components != 6:
        raise ValueError(
            f"n_components must be 6 per Mercurio specification, got {n_components}"
        )

    cov_frame = cov_frame.upper()
    if cov_frame != "RTN":
        raise ValueError(
            f"cov_frame must be 'RTN', got {cov_frame!r}"
        )

    if csv_path is None:
        csv_path = _DEFAULT_CSV_PATH
    else:
        csv_path = Path(csv_path)

    rng = np.random.default_rng(seed)

    # Equal weights: 1/6 each (per Mercurio)
    weights = np.ones(6) / 6.0

    rsos: list[RSO] = []

    # Load positions AND velocities from CSV (needed to build the RTN frame)
    csv_entries = _load_csv_positions_and_velocities(csv_path, n_rso)

    for i in range(n_rso):
        object_id, nominal_position, nominal_velocity = csv_entries[i]

        # Generate diagonal covariance Σ with variances in [var_min, var_max] in RTN frame
        Sigma_rtn = _generate_diagonal_covariance(rng, var_min, var_max)

        # Rotate covariance from RTN to ECI using the propagated ephemeris
        rotation = compute_rtn_to_eci_rotation(
            nominal_position, nominal_velocity
        )
        Sigma = transform_covariance_rtn_to_eci(Sigma_rtn, rotation)
        rtn_covariance = Sigma_rtn

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
                    covariance=Sigma,
                )
            )
            # Minus direction: χ_{j+3} = µ − (√(3Σ))_j
            components.append(
                GaussianComponent(
                    mean=nominal_position - delta,
                    covariance=Sigma,
                )
            )

        uncertainty = GaussianMixture(weights=weights, components=components)

        rsos.append(
            RSO(
                object_id=object_id,
                nominal_position=nominal_position,
                nominal_velocity=nominal_velocity,
                uncertainty=uncertainty,
                rtn_covariance=rtn_covariance,
            )
        )

    return rsos
