"""Geometry utilities for K-d tree operations.

Provides Euclidean distance computations used purely for tree geometry and
pruning. These are distinct from the probabilistic distance metric used to
rank neighbours.
"""

from __future__ import annotations
import numpy as np


def dist_point_to_box(point: np.ndarray, bbox_min: np.ndarray,
                      bbox_max: np.ndarray) -> float:
    """Euclidean distance from a point to an axis-aligned bounding box.

    Returns 0 if the point lies inside the box, otherwise the distance to
    the nearest point on the box surface.

    Parameters
    ----------
    point : np.ndarray, shape (d,)
    bbox_min : np.ndarray, shape (d,)
    bbox_max : np.ndarray, shape (d,)

    Returns
    -------
    float
        Euclidean distance from point to box.
    """
    point = np.asarray(point, dtype=float)
    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)

    clamped = np.maximum(bbox_min, np.minimum(point, bbox_max))
    diff = point - clamped
    return float(np.sqrt(float(np.dot(diff, diff))))


def dist_point_to_point(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))
