# k-D-tree-prefiltering

Probabilistic K-Nearest Neighbor Screening of Resident Space Objects (RSOs)
using a modular K-D tree.

## Overview

This project implements a K-d tree-based algorithm for finding the K nearest
RSOs according to their positional uncertainty distributions, represented as
3D Gaussian Mixture Models (GMMs).

Four probabilistic distance measures are supported:

1. **Wasserstein-2** — optimal transport distance between Gaussians
2. **Mahalanobis** — pooled-covariance mean separation
3. **Hellinger** — affinity-based distance in [0, 1]
4. **Symmetric KL** — divergence between Gaussians

The K-d tree indexes nominal 3D positions for spatial acceleration, while
probabilistic distances rank candidates. A brute-force reference implementation
is provided for validation via Recall@K.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run a single metric:

```bash
python -m rso_kdtree.main \
    --n-rso 2000 \
    --k 10 \
    --metric hellinger \
    --components 3 \
    --leaf-size 10 \
    --seed 42 \
    --output results.csv
```

Run the full benchmark suite across all four metrics:

```bash
python -m rso_kdtree.main --benchmark --output benchmark.csv
```

## Package Structure

```
rso_kdtree/
├── models/          # GaussianComponent, GaussianMixture, RSO
├── distances/       # Probabilistic distance measures + factory
├── spatial/         # KD-tree, KD-tree node, AABB geometry
├── screening/       # Probabilistic KNN, brute force, results
├── data/            # Synthetic catalogue generation
├── benchmark/       # Benchmark suite + metrics
├── main.py          # CLI
└── requirements.txt
```

## Mathematical Assumptions & Limitations

### Euclidean pruning is a heuristic

The K-d tree prunes nodes using Euclidean AABB distance against the
K-th smallest nominal Euclidean distance among evaluated candidates
(D_E,K). This does **not** provide a guaranteed lower bound on
Wasserstein, Mahalanobis, Hellinger, or KL distances between GMMs.
Objects with distant nominal positions may have similar uncertainty
distributions, particularly when uncertainties are large or anisotropic.
Consequently, KD-tree results are approximate; brute-force results are
the reference.

### GMM distance approximations

GMM distances use a component-coupling formulation solved via linear
programming (scipy.optimize.linprog, HiGHS). For Wasserstein this is a
transport approximation. For Hellinger, Mahalanobis, and symmetric KL,
the component-coupling approach is a component-level comparison, not the
exact distribution-level distance between the full GMMs.

### Mahalanobis limitation

The pooled-covariance Mahalanobis distance compares mean separation
normalized by pooled covariance and does not independently account for
covariance differences. It is not a complete distribution distance.

## Testing

```bash
pytest tests/
```

## Reproducibility

The synthetic catalogue uses `np.random.default_rng(seed)` with a fixed
seed (default 42). The same catalogue is reused across all distance
metrics, ensuring fair metric comparisons.