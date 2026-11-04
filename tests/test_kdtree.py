"""Tests for the scipy KDTree wrapper."""

import numpy as np
import pytest

from rso_kdtree.spatial import build_kdtree, KDTree


def test_build_kdtree_returns_scipy_tree():
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tree = build_kdtree(positions)
    assert isinstance(tree, KDTree)


def test_kdtree_query_nearest():
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    tree = build_kdtree(positions)
    dist, idx = tree.query([0.1, 0.0, 0.0], k=1)
    assert int(idx) == 0
    assert abs(dist - 0.1) < 1e-9


def test_kdtree_query_k_neighbors():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    tree = build_kdtree(positions)
    dists, idxs = tree.query([0.0, 0.0, 0.0], k=3)
    idxs = np.atleast_1d(idxs)
    assert set(int(i) for i in idxs) == {0, 1, 2}


def test_kdtree_query_ball_point():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    tree = build_kdtree(positions)
    idxs = tree.query_ball_point([0.0, 0.0, 0.0], r=2.0)
    assert set(int(i) for i in idxs) == {0, 1}
