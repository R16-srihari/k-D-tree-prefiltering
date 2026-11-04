"""AABB geometry helpers for KD-tree traversal.

Provides point-to-axis-aligned-bounding-box Euclidean distance used for
pruning decisions during Alg. 3 tree traversal.
"""

from __future__ import annotations

import numpy as np


def point_to_aabb_distance(
    point: np.ndarray,
    aabb_lower: np.ndarray,
    aabb_upper: np.ndarray,
) -> float:
    """Compute Euclidean distance from a point to an axis-aligned bounding box.

    Returns 0.0 if the point lies inside the box.

    Parameters
    ----------
    point : np.ndarray, shape (3,)
        Query point.
    aabb_lower : np.ndarray, shape (3,)
        Lower bound of the AABB.
    aabb_upper : np.ndarray, shape (3,)
        Upper bound of the AABB.

    Returns
    -------
    float
        Euclidean distance from point to the AABB (0 if inside).
    """
    point = np.asarray(point, dtype=float)
    lower = np.asarray(aabb_lower, dtype=float)
    upper = np.asarray(aabb_upper, dtype=float)

    # Clamp point to box bounds; distance is 0 if already inside
    clamped = np.clip(point, lower, upper)
    return float(np.linalg.norm(point - clamped))