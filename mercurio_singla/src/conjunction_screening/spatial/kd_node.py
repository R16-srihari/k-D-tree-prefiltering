"""K-d tree node data structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class KDTreeNode:
    """A node in the K-d tree.

    Attributes
    ----------
    point : np.ndarray, shape (d,)
        Mean position stored at this node.
    index : int
        Original index of the point in the input array.
    split_dim : int
        Dimension used for splitting at this node.
    split_val : float
        Split value along split_dim.
    left : KDTreeNode | None
        Left subtree (points with coordinate < split_val on split_dim).
    right : KDTreeNode | None
        Right subtree (points with coordinate >= split_val on split_dim).
    bbox_min : np.ndarray | None
        Bounding box minimum coordinates of the subtree rooted here.
    bbox_max : np.ndarray | None
        Bounding box maximum coordinates of the subtree rooted here.
    """

    point: np.ndarray
    index: int
    split_dim: int
    split_val: float
    left: Optional["KDTreeNode"] = None
    right: Optional["KDTreeNode"] = None
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None
