# Mercurio–Singla K-d Tree Conjunction Screening

A modular Python framework implementing the K-d tree-based conjunction screening
approach described in:

1. *A Tree-Based Approach for Efficient and Accurate Conjunction Analysis* —
   Michael Mercurio and Puneet Singla.
2. *A Non-Combinatorial Approach for Efficient Conjunction Analysis* —
   accompanying thesis.

The framework reduces the computational cost of identifying potentially
significant Resident Space Object (RSO) pairs from an orbital catalog while
accounting for positional uncertainty.

## Key design principle

The probabilistic distance metric is an independent, pluggable module. The
K-d tree screening algorithm knows nothing about the specific metric — it
interacts exclusively through the `ProbabilisticDistance` interface. This lets
you substitute alternative probabilistic distance measures (Bhattacharyya,
Mahalanobis, Wasserstein, …) without touching the tree implementation.

## Project structure

```
mercurio_singla/
├── pyproject.toml
├── README.md
├── src/
│   └── conjunction_screening/
│       ├── __init__.py
│       ├── models/          # RSO, Gaussian, GMM uncertainty representations
│       ├── distances/       # ProbabilisticDistance interface + Hellinger
│       ├── spatial/         # KD-tree node, tree, geometry
│       ├── screening/       # Mercurio-Singla search, brute force, results
│       ├── benchmarking/    # Metrics and scaling experiments
│       └── utils/           # Validation utilities
├── tests/                   # Unit tests
└── examples/                # Usage examples
```

## Installation

```bash
cd mercurio_singla
pip install -e .
```

## Quick example

```python
import numpy as np
from conjunction_screening.models import RSO, Gaussian
from conjunction_screening.distances import HellingerDistance
from conjunction_screening.spatial import KDTree
from conjunction_screening.screening import MercurioSinglaScreening

# Build catalog
rsos = [
    RSO(id="RSO-1", mean_position=np.array([1000.0, 0.0, 0.0]),
        uncertainty=Gaussian(mean=np.array([1000.0, 0.0, 0.0]),
                             covariance=np.diag([10.0, 10.0, 10.0]))),
    RSO(id="RSO-2", mean_position=np.array([1005.0, 2.0, 1.0]),
        uncertainty=Gaussian(mean=np.array([1005.0, 2.0, 1.0]),
                             covariance=np.diag([10.0, 10.0, 10.0]))),
]

# Build KD-tree on mean positions
tree = KDTree.build([r.mean_position for r in rsos])

# Screen with Hellinger distance
metric = HellingerDistance()
screener = MercurioSinglaScreening(tree=tree, distance_metric=metric, k=5)
result = screener.screen(rsos)

print(result.pairs)
print(result.distances)
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
