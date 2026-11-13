"""Shared floating-point precision configuration for the RSO k-D tree project.

All floating-point arrays and computations use ``np.longdouble`` (extended
precision, typically 80-bit on x86-64 Linux) instead of ``np.float64``.
This reduces roundoff accumulation in the Hellinger distance (Eq. 15 of
Mercurio & Singla 2016), matrix square roots, slogdet, and the
component-coupling LP solves.

Per the project's source documentation (Sources/Mercurio_Singla_KD_Tree_based
approach for efficient and accurate conjunction.md), the algorithms
(kd-tree construction, modified probabilistic nearest-neighbor search,
Hellinger distance, GMM LP coupling) are unchanged; only the working
precision is widened.
"""

from __future__ import annotations

import numpy as np

# Working floating-point dtype: extended precision long double.
LONGDOUBLE = np.longdouble

# Tolerances appropriate for long double (~18-19 decimal digits).
# These are tighter than the float64 defaults (1e-8 / 1e-12) used previously.
ATOL = np.longdouble("1e-14")
RTOL = np.longdouble("1e-14")

# Positive-definiteness eigenvalue floor for long double.
POSDEF_FLOOR = np.longdouble("1e-14")

# Log-BC clipping bounds for long double (log of long double min is far below -700).
LOG_BC_MIN = np.longdouble("-700.0")
LOG_BC_MAX = np.longdouble("0.0")


def as_longdouble(x, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert ``x`` to a NumPy array with ``LONGDOUBLE`` dtype."""
    return np.asarray(x, dtype=LONGDOUBLE if dtype is None else dtype)


def longdouble_inf() -> np.longdouble:
    """Return positive infinity as ``np.longdouble``."""
    return np.longdouble(np.inf)


def longdouble_nan() -> np.longdouble:
    """Return NaN as ``np.longdouble``."""
    return np.longdouble(np.nan)
