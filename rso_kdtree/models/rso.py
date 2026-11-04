"""Resident Space Object (RSO) data model."""

from __future__ import annotations

import numpy as np

from .gaussian_mixture import GaussianMixture


class RSO:
    """A Resident Space Object with nominal position and GMM uncertainty.

    Attributes
    ----------
    object_id : str
        Unique object identifier.
    nominal_position : np.ndarray, shape (3,)
        Nominal 3D position used for spatial indexing.
    uncertainty : GaussianMixture
        Positional uncertainty represented as a 3D GMM.
    """

    def __init__(
        self,
        object_id: str,
        nominal_position: np.ndarray,
        uncertainty: GaussianMixture,
    ) -> None:
        self.object_id = str(object_id)
        self.nominal_position = np.asarray(nominal_position, dtype=float).ravel()
        self.uncertainty = uncertainty
        self._validate()

    def _validate(self) -> None:
        """Validate dimensions and types."""
        if self.nominal_position.shape != (3,):
            raise ValueError(
                f"RSO nominal_position must have shape (3,), got {self.nominal_position.shape}"
            )
        if not isinstance(self.uncertainty, GaussianMixture):
            raise TypeError(
                f"RSO uncertainty must be a GaussianMixture, got {type(self.uncertainty)}"
            )

    def __repr__(self) -> str:
        return (
            f"RSO(object_id={self.object_id!r}, "
            f"nominal_position={self.nominal_position}, "
            f"uncertainty={self.uncertainty!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RSO):
            return NotImplemented
        return (
            self.object_id == other.object_id
            and np.allclose(self.nominal_position, other.nominal_position)
            and self.uncertainty == other.uncertainty
        )