"""Tests for the custom KD-tree construction (spatial/kd_tree.py, kd_node.py).

Exercises the top-down median-split builder per Mercurio & Singla (2016)
Alg. 1: root AABB from dataset bounds, split along the longest-spread axis
at the median, leaf RSO storage, and parent/child wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

from rso_kdtree.models.gaussian_component import GaussianComponent
from rso_kdtree.models.gaussian_mixture import GaussianMixture
from rso_kdtree.models.rso import RSO
from rso_kdtree.spatial import KDNode, build_kdtree


def _make_rsos(positions: np.ndarray) -> list[RSO]:
    """Build single-Gaussian RSOs from an (n, 3) position array."""
    rsos: list[RSO] = []
    for i, pos in enumerate(np.asarray(positions, dtype=float)):
        comp = GaussianComponent(mean=pos, covariance=np.eye(3))
        gmm = GaussianMixture(weights=np.array([1.0]), components=[comp])
        rsos.append(
            RSO(object_id=f"RSO_{i:04d}", nominal_position=pos, uncertainty=gmm)
        )
    return rsos


def _leaves(node: KDNode) -> list[KDNode]:
    if node.is_leaf:
        return [node]
    out: list[KDNode] = []
    for child in (node.left, node.right):
        if child is not None:
            out.extend(_leaves(child))
    return out


def _internal_nodes(node: KDNode) -> list[KDNode]:
    if node.is_leaf:
        return []
    out = [node]
    for child in (node.left, node.right):
        if child is not None:
            out.extend(_internal_nodes(child))
    return out


def test_build_returns_kdnode_root():
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rsos = _make_rsos(positions)
    root = build_kdtree(positions, rsos, leaf_size=1)
    assert isinstance(root, KDNode)
    assert root.parent is None
    assert not root.is_leaf


def test_root_aabb_matches_dataset_bounds():
    rng = np.random.default_rng(0)
    positions = rng.uniform(-100.0, 100.0, size=(50, 3))
    rsos = _make_rsos(positions)
    root = build_kdtree(positions, rsos, leaf_size=1)
    assert np.allclose(root.aabb_lower, positions.min(axis=0))
    assert np.allclose(root.aabb_upper, positions.max(axis=0))


def test_every_rso_in_exactly_one_leaf():
    rng = np.random.default_rng(1)
    positions = rng.uniform(0.0, 1000.0, size=(40, 3))
    rsos = _make_rsos(positions)
    root = build_kdtree(positions, rsos, leaf_size=1)
    leaves = _leaves(root)
    assert len(leaves) == 40
    leaf_ids = [r.object_id for leaf in leaves for r in leaf.rsos]
    assert sorted(leaf_ids) == sorted(r.object_id for r in rsos)


def test_leaf_size_respected():
    rng = np.random.default_rng(2)
    positions = rng.uniform(0.0, 1000.0, size=(25, 3))
    rsos = _make_rsos(positions)
    root = build_kdtree(positions, rsos, leaf_size=4)
    leaves = _leaves(root)
    assert all(len(leaf.rsos) <= 4 for leaf in leaves)
    assert sum(len(leaf.rsos) for leaf in leaves) == 25
    assert all(leaf.is_leaf for leaf in leaves)


def test_split_axis_is_longest_spread():
    # x spread = 100, y spread = 10, z spread = 1 -> root splits on x
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 10.0, 1.0],
            [50.0, 5.0, 0.5],
            [25.0, 2.0, 0.2],
        ]
    )
    rsos = _make_rsos(positions)
    root = build_kdtree(positions, rsos, leaf_size=1)
    assert root.split_axis == 0


def test_children_aabbs_partition_parent_box():
    rng = np.random.default_rng(3)
    positions = rng.uniform(0.0, 100.0, size=(30, 3))
    rsos = _make_rsos(positions)
    root = build_kdtree(positions, rsos, leaf_size=1)
    for node in _internal_nodes(root):
        axis = node.split_axis
        val = node.split_value
        assert node.left is not None and node.right is not None
        assert node.left.aabb_upper[axis] == pytest.approx(val)
        assert node.right.aabb_lower[axis] == pytest.approx(val)
        for child in (node.left, node.right):
            assert np.all(child.aabb_lower >= node.aabb_lower - 1e-9)
            assert np.all(child.aabb_upper <= node.aabb_upper + 1e-9)


def test_parent_pointers_wired():
    rng = np.random.default_rng(4)
    positions = rng.uniform(0.0, 100.0, size=(16, 3))
    rsos = _make_rsos(positions)
    root = build_kdtree(positions, rsos, leaf_size=1)
    assert root.parent is None
    for node in _internal_nodes(root):
        assert node.left.parent is node
        assert node.right.parent is node


def test_median_split_keeps_tree_balanced():
    # Balanced median split => depth ~ log2(n)
    rng = np.random.default_rng(5)
    positions = rng.uniform(0.0, 1.0, size=(128, 3))
    rsos = _make_rsos(positions)

    def depth(node: KDNode) -> int:
        if node.is_leaf:
            return 0
        return 1 + max(depth(node.left), depth(node.right))

    root = build_kdtree(positions, rsos, leaf_size=1)
    assert depth(root) <= 10  # log2(128) = 7, allow slack for the median point


def test_degenerate_identical_positions():
    positions = np.tile(np.array([5.0, 5.0, 5.0]), (8, 1))
    rsos = _make_rsos(positions)
    root = build_kdtree(positions, rsos, leaf_size=1)
    leaves = _leaves(root)
    assert sum(len(leaf.rsos) for leaf in leaves) == 8


def test_single_point_root_is_leaf():
    positions = np.array([[1.0, 2.0, 3.0]])
    rsos = _make_rsos(positions)
    root = build_kdtree(positions, rsos, leaf_size=1)
    assert root.is_leaf
    assert len(root.rsos) == 1


def test_empty_positions_raises():
    with pytest.raises(ValueError):
        build_kdtree(np.empty((0, 3)), [], leaf_size=1)
