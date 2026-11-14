"""RTN (Radial, Transverse, Normal) to ECI frame conversion for satellite ephemeris covariance.

Implements the frame conversion described in the SP-672 reference (Fifth European
Conference on Space Debris): covariances are stored in the RTN frame of each orbit
and must be rotated into ECI for use with the K-D tree conjunction screening.

RTN frame definition (right-handed, centred on the satellite):
  - R (radial): along the ECI position vector from Earth centre to satellite
  - T (transverse / in-track): perpendicular to R in the orbital plane, along the
    direction of motion
  - N (normal / cross-track): orbit normal, completes the right-handed triad

Given ECI position r and velocity v:

    r_hat = r / |r|
    h     = r x v
    n_hat = h / |h|
    t_hat = n_hat x r_hat

The rotation matrix from RTN to ECI has columns [r_hat, t_hat, n_hat] expressed in
ECI, so a vector x_RTN maps to ECI as x_ECI = R @ x_RTN. A covariance
transforms as Sigma_ECI = R @ Sigma_RTN @ R.T.
"""

from __future__ import annotations

import numpy as np


def compute_rtn_frame_vectors(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the RTN frame basis vectors expressed in ECI.

    Parameters
    ----------
    position_eci : np.ndarray, shape (3,)
        Satellite position in ECI (km).
    velocity_eci : np.ndarray, shape (3,)
        Satellite velocity in ECI (km/s).

    Returns
    -------
    r_hat : np.ndarray, shape (3,)
        Radial unit vector (along position), in ECI.
    t_hat : np.ndarray, shape (3,)
        Transverse unit vector (along-track), in ECI.
    n_hat : np.ndarray, shape (3,)
        Normal unit vector (orbit normal), in ECI.

    Raises
    ------
    ValueError
        If the position vector is zero, or if position and velocity are parallel
        (zero angular momentum), in which case the RTN frame is undefined.
    """
    r = np.asarray(position_eci, dtype=float).ravel()
    v = np.asarray(velocity_eci, dtype=float).ravel()

    if r.shape != (3,) or v.shape != (3,):
        raise ValueError(
            f"position_eci and velocity_eci must have shape (3,), "
            f"got {r.shape} and {v.shape}"
        )

    r_norm = float(np.linalg.norm(r))
    if r_norm == 0.0:
        raise ValueError("Position vector is zero; RTN frame is undefined.")

    r_hat = r / r_norm

    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if h_norm == 0.0:
        raise ValueError(
            "Position and velocity are parallel (zero angular momentum); "
            "RTN frame is undefined."
        )

    n_hat = h / h_norm
    t_hat = np.cross(n_hat, r_hat)

    return r_hat, t_hat, n_hat


def compute_rtn_to_eci_rotation(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
) -> np.ndarray:
    """Compute the rotation matrix from RTN to ECI frame.

    The returned 3x3 matrix R has columns [r_hat, t_hat, n_hat] expressed in
    ECI, so that x_ECI = R @ x_RTN and Sigma_ECI = R @ Sigma_RTN @ R.T.

    Parameters
    ----------
    position_eci : np.ndarray, shape (3,)
        Satellite position in ECI (km).
    velocity_eci : np.ndarray, shape (3,)
        Satellite velocity in ECI (km/s).

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix mapping RTN vectors to ECI.

    Raises
    ------
    ValueError
        If the RTN frame is undefined (see compute_rtn_frame_vectors).
    """
    r_hat, t_hat, n_hat = compute_rtn_frame_vectors(position_eci, velocity_eci)
    return np.column_stack([r_hat, t_hat, n_hat])


def transform_covariance_rtn_to_eci(
    cov_rtn: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Transform a covariance matrix from RTN to ECI frame.

    Sigma_ECI = R @ Sigma_RTN @ R.T

    Parameters
    ----------
    cov_rtn : np.ndarray, shape (3, 3)
        Covariance matrix in RTN frame (km^2).
    rotation : np.ndarray, shape (3, 3)
        RTN-to-ECI rotation matrix from compute_rtn_to_eci_rotation.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Covariance matrix in ECI frame (km^2).
    """
    cov_rtn = np.asarray(cov_rtn, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if cov_rtn.shape != (3, 3):
        raise ValueError(f"cov_rtn must have shape (3, 3), got {cov_rtn.shape}")
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    return rotation @ cov_rtn @ rotation.T


def transform_covariance_eci_to_rtn(
    cov_eci: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Transform a covariance matrix from ECI to RTN frame.

    Sigma_RTN = R.T @ Sigma_ECI @ R

    Parameters
    ----------
    cov_eci : np.ndarray, shape (3, 3)
        Covariance matrix in ECI frame (km^2).
    rotation : np.ndarray, shape (3, 3)
        RTN-to-ECI rotation matrix from compute_rtn_to_eci_rotation.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Covariance matrix in RTN frame (km^2).
    """
    cov_eci = np.asarray(cov_eci, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if cov_eci.shape != (3, 3):
        raise ValueError(f"cov_eci must have shape (3, 3), got {cov_eci.shape}")
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    return rotation.T @ cov_eci @ rotation
