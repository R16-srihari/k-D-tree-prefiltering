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
    --leaf-size 1 \
    --seed 42 \
    --output results.csv
```

Run the full benchmark suite across all four metrics:

```bash
python -m rso_kdtree.main --benchmark --output benchmark.csv
```
Default paramters are set in `main.py` and can be overridden via command-line arguments. See `python -m rso_kdtree.main --help` for details.

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

This limitation is fundamental to using Euclidean space partitioning for
probabilistic nearest-neighbor search. Mercurio & Singla (2016) report
~98.6% nearest-neighbor capture rather than 100% for this reason. Recall
depends on the catalogue's uncertainty characteristics:

- **Small, isotropic uncertainties**: Euclidean distance is a good proxy
  for probabilistic distance; Recall@K approaches 100%.
- **Large or highly anisotropic uncertainties**: Euclidean distance can
  be a poor proxy; boxes far in Euclidean space may contain RSOs with
  small probabilistic distance. Recall@K drops accordingly.

#### Tuning guidance

To match the paper's reported Recall@K (~98.6%+):

1. **Increase `k` slightly** above the target neighbor count to give the
   pruning threshold more headroom (e.g., use k=8 when you need k=5).
2. **Use smaller `leaf_size`** (default 1) for finer-grained pruning,
   at the cost of a deeper tree.
3. **Validate against brute force** on a representative subset of your
   catalogue to measure actual Recall@K before deploying at scale.
4. **Prefer Hellinger or Wasserstein** over Mahalanobis or KL when
   uncertainty anisotropy is significant; these metrics better capture
   distributional overlap.

#### Hellinger distance normalization

Per Eq. 15 of Mercurio & Singla (2016), the Gaussian-level Hellinger
distance is:

$$
\mathcal{D}_H(p,q) = \sqrt{1 - \frac{|\Sigma_p|^{1/4}|\Sigma_q|^{1/4}}{|\Sigma_{pq}|^{1/2}} e^{-\frac{1}{2}(\mu_p-\mu_q)^T\Sigma_{pq}^{-1}(\mu_p-\mu_q)}}
$$

where $\Sigma_p$, $\Sigma_q$, and $\Sigma_{pq}$ are the covariances of the two Gaussians and their average, respectively.

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

## Benchmark Output (`benchmark.csv`)

Running `python -m rso_kdtree.main --benchmark --output benchmark.csv` writes one row per distance metric with the following instrumentation columns:

| Column | Definition |
|--------|------------|
| **Metric** | Distance measure used (`hellinger`, `kl`, `mahalanobis`, `wasserstein`). |
| **GMM Components** | Number of Gaussian mixture components per RSO (default 6). |
| **K-D Tree Time (s)** | Wall-clock time to run KD-tree KNN search across all query RSOs (tree build time excluded). |
| **Brute Force Time (s)** | Wall-clock time to run brute-force KNN search across all query RSOs. |
| **Recall@K** | Fraction of brute-force true K nearest neighbors recovered by the KD-tree, averaged across all queries: `|N_KD ∩ N_BF| / K`. |
| **Distance Evaluations** | Total number of probabilistic distance computations performed across all queries (one per candidate RSO evaluated in a visited leaf). |
| **Nodes Visited** | Total number of internal KD-tree nodes entered during traversal and backtracking across all queries. |
| **Nodes Pruned** | Total number of sibling subtrees skipped because their AABB distance to the query (`D_box`) was not less than the current K-th Euclidean threshold (`DE_min`). |
| **Leaves Visited** | Total number of leaf nodes whose RSOs were distance-evaluated across all queries. With `leaf_size=1` there are as many leaves as RSOs in the tree, but a single query can visit multiple leaves during backtracking, so this count is cumulative over all 100 queries and typically exceeds the number of RSOs. |
| **Mean Distance Difference** | Mean absolute difference between KD-tree and brute-force probabilistic distances for neighbors found by **both** methods: `mean(|d_KD − d_BF|)` over the intersection of neighbor ID sets. Returns `0.0` when no neighbors overlap; since both methods use the same deterministic distance function, shared neighbors have identical distances and this value is `0.0`. |

## Testing

```bash
pytest tests/
```

## Reproducibility

The synthetic catalogue uses `np.random.default_rng(seed)` with a fixed
seed (default 42). The same catalogue is reused across all distance
metrics, ensuring fair metric comparisons.

## References
1. M. Mercurio and P. Singla, “A Tree-Based Approach for Efficient and Accurate Conjunction Analysis,” Comput. Model. Eng. Sci., vol. 111, no. 3, pp. 229–256, 2016. https://doi.org/10.3970/cmes.2016.111.229
2. M. Mercurio, "A Noncombinatorial Approach for Efficient Conjunction Analysis." Order No. 1566932, State University of New York at Buffalo, United States -- New York, 2014.
