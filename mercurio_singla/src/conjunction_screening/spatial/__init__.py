"""Spatial indexing: K-d tree on mean positions.

The tree is metric-agnostic: it stores mean positions and bounding boxes and
supports Euclidean KNN search. Probabilistic ranking is handled by the
screening layer.
"""

from .kd_node import KDTreeNode
from .kd_tree import KDTree
from .geometry import dist_point_to_box, dist_point_to_point

__all__ = [
    "KDTreeNode",
    "KDTree",
    "dist_point_to_box",
    "dist_point_to_point",
]
