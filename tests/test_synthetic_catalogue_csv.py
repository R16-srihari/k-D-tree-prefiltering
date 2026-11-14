"""Tests for the CSV-backed synthetic catalogue generator.

Verifies that nominal positions come from the first n usable rows of
``data/satellite_propagated.csv`` (finite X/Y/Z), that the Mercurio 6-component GMM
uncertainty construction is unchanged, and that ``seed`` controls only the random
covariances now that positions are deterministic.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from rso_kdtree.data.synthetic_catalogue import generate_synthetic_catalogue

CSV_PATH = Path(__file__).resolve().parent.parent / "rso_kdtree" / "data" / "satellite_propagated.csv"


def _usable_csv_rows(path: Path, n: int) -> list[tuple[str, np.ndarray]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                x, y, z = float(row["X"]), float(row["Y"]), float(row["Z"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            oid = str(row.get("NORAD_CAT_ID", "")).strip() or f"RSO_{len(rows):04d}"
            rows.append((oid, np.array([x, y, z], dtype=float)))
            if len(rows) >= n:
                break
    return rows


def test_positions_match_first_n_csv_rows():
    n = 25
    cat = generate_synthetic_catalogue(n_rso=n, seed=42, csv_path=CSV_PATH)
    expected = _usable_csv_rows(CSV_PATH, n)

    assert len(cat) == n
    for rso, (oid, pos) in zip(cat, expected):
        assert rso.object_id == oid
        assert np.allclose(rso.nominal_position, pos)


def test_positions_are_deterministic_across_seeds():
    cat1 = generate_synthetic_catalogue(n_rso=30, seed=42, csv_path=CSV_PATH)
    cat2 = generate_synthetic_catalogue(n_rso=30, seed=99, csv_path=CSV_PATH)
    for r1, r2 in zip(cat1, cat2):
        assert r1.object_id == r2.object_id
        assert np.allclose(r1.nominal_position, r2.nominal_position)


def test_different_seeds_yield_different_covariances():
    cat1 = generate_synthetic_catalogue(n_rso=30, seed=42, csv_path=CSV_PATH)
    cat2 = generate_synthetic_catalogue(n_rso=30, seed=99, csv_path=CSV_PATH)
    different_cov = any(
        not np.allclose(c1.covariance, c2.covariance)
        for r1, r2 in zip(cat1, cat2)
        for c1, c2 in zip(r1.uncertainty.components, r2.uncertainty.components)
    )
    assert different_cov


def test_same_seed_fully_reproducible():
    cat1 = generate_synthetic_catalogue(n_rso=20, seed=7, csv_path=CSV_PATH)
    cat2 = generate_synthetic_catalogue(n_rso=20, seed=7, csv_path=CSV_PATH)
    for r1, r2 in zip(cat1, cat2):
        assert r1.object_id == r2.object_id
        assert np.allclose(r1.nominal_position, r2.nominal_position)
        assert np.allclose(r1.uncertainty.weights, r2.uncertainty.weights)
        for c1, c2 in zip(r1.uncertainty.components, r2.uncertainty.components):
            assert np.allclose(c1.mean, c2.mean)
            assert np.allclose(c1.covariance, c2.covariance)


def test_component_count_is_six_per_mercurio():
    cat = generate_synthetic_catalogue(n_rso=10, n_components=6, seed=42, csv_path=CSV_PATH)
    for rso in cat:
        assert rso.uncertainty.n_components == 6


def test_component_count_other_than_six_raises():
    for m in [1, 2, 3, 5]:
        with pytest.raises(ValueError):
            generate_synthetic_catalogue(n_rso=10, n_components=m, seed=42, csv_path=CSV_PATH)


def test_n_rso_exceeding_usable_rows_raises():
    with pytest.raises(ValueError):
        generate_synthetic_catalogue(n_rso=100_000, seed=42, csv_path=CSV_PATH)


def test_missing_csv_raises():
    with pytest.raises(FileNotFoundError):
        generate_synthetic_catalogue(n_rso=5, seed=42, csv_path=Path("does_not_exist.csv"))


def test_gmm_weights_and_covariance_per_mercurio():
    cat = generate_synthetic_catalogue(n_rso=5, seed=42, csv_path=CSV_PATH)
    for rso in cat:
        gmm = rso.uncertainty
        assert np.allclose(gmm.weights.sum(), 1.0)
        assert np.allclose(gmm.weights, np.ones(6) / 6.0)
        for comp in gmm.components:
            # Component covariance = Sigma / 3 (thesis §4)
            eig = np.linalg.eigvalsh(comp.covariance)
            assert np.all(eig > 0)  # positive definite


def test_default_csv_path_works():
    # Default csv_path resolves to the file next to this module's package.
    cat = generate_synthetic_catalogue(n_rso=10, seed=42)
    assert len(cat) == 10
    # First usable CSV row is a real propagated position.
    assert np.all(np.isfinite(cat[0].nominal_position))
