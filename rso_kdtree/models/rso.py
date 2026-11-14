"""Resident Space Object (RSO) data model."""

from __future__ import annotations

import numpy as np

from .gaussian_mixture import GaussianMixture


class RSO:
    """A Resident Space Object with nominal position, velocity, and GMM uncertainty.

    Attributes
    ----------
    object_id : str
        Unique object identifier.
    nominal_position : np.ndarray, shape (3,)
        Nominal 3D position used for spatial indexing (ECI, km).
    nominal_velocity : np.ndarray, shape (3,)
        Nominal 3D velocity used to build the RTN frame (ECI, km/s).
        Defaults to zeros if not provided.
    uncertainty : GaussianMixture
        Positional uncertainty represented as a 3D GMM (ECI frame).
    rtn_covariance : np.ndarray, shape (3, 3) or None
        Original covariance matrix in the RTN frame (km^2), before rotation
        to ECI. ``None`` when the covariance was generated directly in ECI.
    """

    def __init__(
        self,
        object_id: str,
        nominal_position: np.ndarray,
        uncertainty: GaussianMixture,
        nominal_velocity: np.ndarray | None = None,
        rtn_covariance: np.ndarray | None = None,
    ) -> None:
        self.object_id = str(object_id)
        self.nominal_position = np.asarray(nominal_position, dtype=float).ravel()
        self.nominal_velocity = (
            np.asarray(nominal_velocity, dtype=float).ravel()
            if nominal_velocity is not None
            else np.zeros(3)
        )
        self.uncertainty = uncertainty
        self.rtn_covariance = (
            np.asarray(rtn_covariance, dtype=float)
            if rtn_covariance is not None
            else None
        )
        self._validate()

    def _validate(self) -> None:
        """Validate dimensions and types."""
        if self.nominal_position.shape != (3,):
            raise ValueError(
                f"RSO nominal_position must have shape (3,), got {self.nominal_position.shape}"
            )
        if self.nominal_velocity.shape != (3,):
            raise ValueError(
                f"RSO nominal_velocity must have shape (3,), got {self.nominal_velocity.shape}"
            )
        if not isinstance(self.uncertainty, GaussianMixture):
            raise TypeError(
                f"RSO uncertainty must be a GaussianMixture, got {type(self.uncertainty)}"
            )
        if self.rtn_covariance is not None and self.rtn_covariance.shape != (3, 3):
            raise ValueError(
                f"RSO rtn_covariance must have shape (3, 3), got {self.rtn_covariance.shape}"
            )

    def __repr__(self) -> str:
        return (
            f"RSO(object_id={self.object_id!r}, "
            f"nominal_position={self.nominal_position}, "
            f"nominal_velocity={self.nominal_velocity}, "
            f"uncertainty={self.uncertainty!r}, "
            f"rtn_covariance={self.rtn_covariance!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RSO):
            return NotImplemented
        return (
            self.object_id == other.object_id
            and np.allclose(self.nominal_position, other.nominal_position)
            and np.allclose(self.nominal_velocity, other.nominal_velocity)
            and self.uncertainty == other.uncertainty
            and np.allclose(self.rtn_covariance, other.rtn_covariance)
            if self.rtn_covariance is not None and other.rtn_covariance is not None
            else self.rtn_covariance is other.rtn_covariance
        )