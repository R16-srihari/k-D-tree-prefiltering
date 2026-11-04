# Implementation Plan: Probabilistic K-Nearest Neighbor Screening of RSOs Using a Modular K-D Tree

## 1. Project Objective

Develop a Python implementation of a K-d tree-based algorithm for finding the **K nearest resident space objects (RSOs)** according to their positional uncertainty distributions.

Each RSO is represented by:

- A nominal 3D position.
- A 3D Gaussian Mixture Model (GMM) representing positional uncertainty.
- A unique object identifier.

The algorithm must support four selectable probabilistic distance measures:

1. Wasserstein distance
2. Mahalanobis distance
3. Hellinger distance
4. Symmetric KL divergence

The implementation should use a K-d tree for spatial acceleration, while also providing a brute-force reference implementation to evaluate accuracy and performance.

### Main objectives

- Generate a catalogue of 2,000 synthetic RSOs.
- Represent each RSO's positional uncertainty using a configurable 3D GMM.
- Find the K nearest RSOs using the selected probabilistic distance.
- Compare probabilistic metrics using an identical catalogue.
- Benchmark the K-d tree against brute force.
- Quantify the accuracy limitations introduced by Euclidean spatial pruning.
- Keep the implementation modular and extensible.

---

## 2. Core Design Decisions

### 2.1 Definition of K-nearest neighbors

Define K-nearest neighbors as the K RSOs ranked by the **selected probabilistic distance between their uncertainty distributions**.

The nominal Euclidean distance must not determine the final ranking.

For a query RSO \(q\), define:

\[
\mathcal{N}_K(q)
=
\operatorname{arg\,Kmin}_{j\ne q}
D(P_q,P_j)
\]

where:

- \(P_q\) is the query RSO's positional uncertainty distribution.
- \(P_j\) is the candidate RSO's positional uncertainty distribution.
- \(D\) is the selected probabilistic distance measure.

Requirements:

- Exclude the query RSO from its own neighbor list.
- Rank results by probabilistic distance.
- Break ties deterministically using `object_id`.
- Keep Euclidean distance separate from probabilistic distance.

### 2.2 Use the same random catalogue for every metric

Generate one synthetic catalogue of 2,000 RSOs and reuse it for all metric comparisons.

The catalogue must include the same:

- Nominal positions
- GMM component means
- Covariance matrices
- Component weights
- Object identifiers

Use a fixed random seed, defaulting to `42`.

Switching between Wasserstein, Mahalanobis, Hellinger, and KL must not regenerate the catalogue.

This ensures that metric comparisons are fair and reproducible.

### 2.3 Brute-force validation

Implement a brute-force KNN algorithm that computes the selected probabilistic distance between each query RSO and every other RSO.

Treat brute-force results as the reference solution for each selected metric.

Compare K-d tree results against brute force using:

\[
\operatorname{Recall@K}(q)
=
\frac{
|\mathcal{N}^{KD}_K(q)\cap
\mathcal{N}^{BF}_K(q)|
}{K}
\]

where:

- \(\mathcal{N}^{KD}_K(q)\) is the K-d tree neighbor set.
- \(\mathcal{N}^{BF}_K(q)\) is the brute-force neighbor set.

Report the mean Recall@K across all query RSOs.

**Important limitation:** Euclidean AABB pruning is not guaranteed to preserve probabilistic KNN rankings. Objects with distant nominal positions may have similar uncertainty distributions, particularly when uncertainties are large or anisotropic.

Therefore, clearly distinguish:

- Exact brute-force probabilistic KNN
- K-d tree probabilistic ranking with Euclidean spatial pruning
- Approximation error caused by pruning

Do not claim that the K-d tree produces exact probabilistic KNN unless the pruning rule is mathematically proven safe for the selected distance.

### 2.4 Configurable GMM component count

Make the number of Gaussian components per RSO configurable.

Default:

\[
M=3
\]

Support experiments with:

\[
M\in\{1,2,3,5\}
\]

For \(M=1\), the GMM reduces to a single Gaussian distribution.

Evaluate how component count affects:

- Distance computation time
- K-d tree query runtime
- Brute-force runtime
- Recall@K
- Number of probabilistic distance evaluations

Keep the catalogue generation reproducible across component-count experiments.

---

## 3. Project Architecture

Create the following package structure:

```text
rso_kdtree/
│
├── models/
│   ├── gaussian_component.py
│   ├── gaussian_mixture.py
│   └── rso.py
│
├── distances/
│   ├── base.py
│   ├── gaussian_distances.py
│   ├── gmm_distances.py
│   └── factory.py
│
├── spatial/
│   ├── kd_tree.py
│   ├── kd_node.py
│   └── geometry.py
│
├── screening/
│   ├── knn.py
│   ├── brute_force.py
│   └── results.py
│
├── data/
│   └── synthetic_catalogue.py
│
├── benchmark/
│   ├── benchmark.py
│   └── metrics.py
│
├── tests/
│   ├── test_distances.py
│   ├── test_gmm.py
│   ├── test_kdtree.py
│   ├── test_knn.py
│   └── test_reproducibility.py
│
├── main.py
├── requirements.txt
└── README.md
```

Keep the probabilistic distance implementation independent from the K-d tree.

The tree should not need to know the internal structure of a specific distance measure.

---

## 4. Data Models

### 4.1 Gaussian component

Implement a Gaussian component:

```python
class GaussianComponent:
    mean: np.ndarray
    covariance: np.ndarray
```

Requirements:

- Mean shape: `(3,)`
- Covariance shape: `(3, 3)`
- Covariance must be symmetric.
- Covariance must be positive definite for Gaussian KL and standard Gaussian distance calculations.
- Validate input dimensions.
- Include numerical safeguards for covariance matrices.

### 4.2 Gaussian mixture model

Implement:

```python
class GaussianMixture:
    weights: np.ndarray
    components: list[GaussianComponent]
```

Requirements:

- Component weights must be nonnegative.
- Weights must sum to one.
- All components must be 3D.
- Number of components must be configurable.

Represent the GMM as:

\[
p(\mathbf{x})
=
\sum_{i=1}^{M}
\alpha_i
\mathcal{N}(\mathbf{x}\mid
\boldsymbol{\mu}_i,\Sigma_i)
\]

where:

\[
\sum_{i=1}^{M}\alpha_i=1
\]

### 4.3 RSO model

Implement:

```python
class RSO:
    object_id: str
    nominal_position: np.ndarray
    uncertainty: GaussianMixture
```

Requirements:

- `nominal_position` has shape `(3,)`.
- `uncertainty` is a 3D GMM.
- The nominal position is used for spatial indexing.
- The GMM is used for probabilistic distance calculations.

The nominal position may differ from the GMM mean, but the synthetic catalogue should normally keep them consistent or define the relationship explicitly.

---

## 5. Probabilistic Distance Module

### 5.1 Distance interface

Implement a common interface:

```python
from abc import ABC, abstractmethod

class ProbabilisticDistance(ABC):

    @abstractmethod
    def compute(self, gmm_a, gmm_b) -> float:
        pass
```

The KNN algorithm should accept a distance object:

```python
knn_search(
    query_rso,
    kd_tree,
    k,
    distance_metric
)
```

### 5.2 Runtime metric selection

Implement a factory:

```python
def create_distance_metric(metric_name):
    ...
```

Supported metric strings:

```text
"wasserstein"
"mahalanobis"
"hellinger"
"kl"
```

The CLI should allow:

```bash
python main.py --metric hellinger
```

Changing the metric should require no changes to the K-d tree or screening algorithm.

---

## 6. Gaussian Distance Measures

Implement Gaussian-level distance functions first.

### 6.1 Hellinger distance

For Gaussian distributions:

\[
P=\mathcal{N}(\mu_1,\Sigma_1)
\]

\[
Q=\mathcal{N}(\mu_2,\Sigma_2)
\]

Define:

\[
\bar{\Sigma}
=
\frac{\Sigma_1+\Sigma_2}{2}
\]

\[
\Delta\mu=\mu_1-\mu_2
\]

The Bhattacharyya coefficient is:

\[
BC(P,Q)
=
\frac{
|\Sigma_1|^{1/4}
|\Sigma_2|^{1/4}
}{
|\bar{\Sigma}|^{1/2}
}
\exp\left(
-\frac18
\Delta\mu^T
\bar{\Sigma}^{-1}
\Delta\mu
\right)
\]

Then:

\[
D_H(P,Q)=\sqrt{1-BC(P,Q)}
\]

Requirements:

- Symmetric
- Nonnegative
- Zero for identical distributions
- Numerically stable
- Return values in \([0,1]\), up to numerical tolerance

Use `slogdet` and linear solves rather than direct matrix inversion where possible.

### 6.2 Wasserstein distance

Implement the Gaussian 2-Wasserstein distance:

\[
W_2^2(P,Q)
=
\|\mu_1-\mu_2\|^2
+
\operatorname{tr}
\left[
\Sigma_1+\Sigma_2
-
2(\Sigma_1^{1/2}
\Sigma_2
\Sigma_1^{1/2})^{1/2}
\right]
\]

Return:

\[
W_2(P,Q)=\sqrt{W_2^2(P,Q)}
\]

Requirements:

- Use symmetric matrix square roots.
- Use eigendecomposition or another numerically stable PSD square-root method.
- Handle small negative eigenvalues caused by numerical roundoff.
- Reject materially invalid covariance matrices.

### 6.3 Mahalanobis distance

Implement the pooled-covariance Mahalanobis distance:

\[
\Sigma_{\text{pool}}
=
\frac{\Sigma_1+\Sigma_2}{2}
\]

\[
D_M(P,Q)
=
\sqrt{
(\mu_1-\mu_2)^T
\Sigma_{\text{pool}}^{-1}
(\mu_1-\mu_2)
}
\]

Important:

This measure compares mean separation normalized by pooled covariance. It is **not a complete distribution distance** because it does not independently account for covariance differences.

Document this limitation in the README and benchmark results.

### 6.4 Symmetric KL divergence

Implement:

\[
D_{\text{SKL}}(P,Q)
=
\frac12
\left[
KL(P\|Q)+KL(Q\|P)
\right]
\]

For Gaussian distributions:

\[
D_{\text{SKL}}
=
\frac14
\left[
\operatorname{tr}
\left(
\Sigma_2^{-1}\Sigma_1
+
\Sigma_1^{-1}\Sigma_2
\right)
-2d
+
\Delta\mu^T
\left(
\Sigma_1^{-1}
+
\Sigma_2^{-1}
\right)
\Delta\mu
\right]
\]

where \(d=3\).

Requirements:

- Symmetric
- Nonnegative up to numerical tolerance
- Zero for identical distributions
- Clearly identified as a divergence rather than a metric

It does not generally satisfy the triangle inequality.

---

## 7. GMM Distance Calculation

For GMMs:

\[
P=\sum_{i=1}^{M}\alpha_iP_i
\]

\[
Q=\sum_{j=1}^{N}\beta_jQ_j
\]

Implement a component-coupling formulation using linear programming.

### 7.1 Component cost matrix

Construct:

\[
C_{ij}=d(P_i,Q_j)
\]

where \(d\) is the selected Gaussian-level distance.

### 7.2 Optimal transport coupling

Solve:

\[
\min_{w_{ij}}
\sum_{i=1}^{M}
\sum_{j=1}^{N}
w_{ij}C_{ij}
\]

subject to:

\[
\sum_jw_{ij}=\alpha_i
\]

\[
\sum_iw_{ij}=\beta_j
\]

\[
w_{ij}\geq0
\]

Use:

```python
scipy.optimize.linprog
```

Use the HiGHS solver.

### 7.3 Interpretation

For Wasserstein, this component-coupling construction provides a GMM transport approximation based on Gaussian component costs.

For Hellinger and symmetric KL, the component-coupling approach is a component-level comparison or approximation. It is **not generally equal to the exact Hellinger distance or exact symmetric KL divergence between the full GMM distributions**.

Document these distinctions clearly.

For Mahalanobis, the component coupling similarly defines a component-level comparison based on the chosen Gaussian Mahalanobis cost.

---

## 8. Synthetic Catalogue Generation

Generate 2,000 RSOs.

### 8.1 Nominal positions

Generate nominal positions uniformly within a configurable 3D cube:

\[
\mathbf{r}_i\sim
\mathcal{U}([-L,L]^3)
\]

Default:

\[
L=1000\ \text{km}
\]

Make the cube size configurable.

### 8.2 GMM generation

For each RSO:

- Generate \(M\) Gaussian components.
- Use \(M=3\) by default.
- Generate component weights and normalize them.
- Generate component means near the RSO nominal position.
- Generate valid 3D covariance matrices.

A suitable covariance generation approach is:

\[
\Sigma=AA^T+\epsilon I
\]

where:

- \(A\) is a randomly generated matrix.
- \(\epsilon>0\) ensures positive definiteness.

Component means may be generated using:

\[
\mu_j=\mathbf{r}_{nominal}+\delta_j
\]

where \(\delta_j\) is a configurable random offset.

### 8.3 Reproducibility

Use a single controlled random generator:

```python
rng = np.random.default_rng(seed)
```

Default:

```text
seed = 42
```

The generated catalogue must be reusable across all distance metrics.

---

## 9. K-D Tree Construction

### 9.1 Spatial indexing

Construct the K-d tree using **nominal 3D positions**, not GMM component means.

Each node should contain:

- AABB lower bound
- AABB upper bound
- Split axis
- Split value
- Parent pointer
- Left child
- Right child
- RSOs stored in the node if it is a leaf

### 9.2 Split strategy

At each internal node:

1. Compute the spatial spread along each dimension.
2. Select the dimension with the largest spread.
3. Split at the median nominal position along that dimension.
4. Recursively construct child nodes.

Make leaf size configurable.

Default:

```text
leaf_size = 1
```

### 9.3 AABB geometry

Implement Euclidean distance from a query point \(\mathbf{x}\) to an axis-aligned bounding box.

For each dimension \(i\):

\[
\delta_i=
\begin{cases}
l_i-x_i & x_i<l_i\\
0 & l_i\leq x_i\leq u_i\\
x_i-u_i & x_i>u_i
\end{cases}
\]

Then:

\[
D_{\text{AABB}}(\mathbf{x})
=
\sqrt{\sum_i\delta_i^2}
\]

This function belongs in the spatial geometry module.

---

## 10. K-D Tree KNN Search

### 10.1 Search procedure

Implement the KNN algorithm:

1. Start at the root node.
2. Descend to the leaf containing the query nominal position.
3. Evaluate candidate RSOs in the leaf.
4. Exclude the query RSO itself.
5. Calculate probabilistic distances using the selected metric.
6. Maintain the current K best candidates.
7. Backtrack to parent nodes.
8. Evaluate neighboring or sibling nodes when required.
9. Use AABB distances and a spatial pruning threshold to decide whether to open nodes.
10. Return the K candidates sorted by probabilistic distance.

### 10.2 Candidate list

Maintain a structure containing:

```text
object_id
probabilistic_distance
nominal_euclidean_distance
```

The candidate list must be sorted by:

1. Probabilistic distance
2. Object identifier for deterministic tie-breaking

### 10.3 Spatial pruning strategy

Implement the Mercurio–Singla-inspired Euclidean pruning heuristic.

Generalize the Euclidean threshold to KNN:

\[
D_{E,K}
\]

where \(D_{E,K}\) is the K-th smallest nominal Euclidean distance among the evaluated candidate objects.

Use a rule such as:

\[
D_{\text{AABB}}<D_{E,K}
\]

to determine whether a node should be opened.

Keep this pruning rule isolated in a dedicated method so alternative strategies can be implemented later.

**Critical caveat:** This is a spatial heuristic. It does not provide a guaranteed lower bound on Wasserstein, Mahalanobis, Hellinger, or KL distances between GMMs.

Consequently, it may prune objects that are among the true probabilistic K nearest neighbors.

Do not conflate:

- Euclidean candidate pruning
- Probabilistic distance ranking

### 10.4 Instrumentation

Record:

- Number of probabilistic distance evaluations
- Number of tree nodes visited
- Number of leaves visited
- Number of nodes pruned
- Query runtime

These statistics are needed to assess the effectiveness of the K-d tree.

---

## 11. Brute-Force KNN

Implement a separate reference algorithm.

For each query RSO:

1. Iterate through every other RSO.
2. Calculate the selected probabilistic distance.
3. Sort all candidates by probabilistic distance.
4. Return the first K.

Complexity:

\[
O(N^2)
\]

for all catalogue queries, ignoring the cost of individual GMM distance evaluations.

Use brute force to validate K-d tree results.

Do not reuse the K-d tree's pruning logic in brute force.

---

## 12. Command-Line Interface

Implement:

```bash
python main.py \
    --n-rso 2000 \
    --k 10 \
    --metric hellinger \
    --components 6 \
    --leaf-size 1 \
    --seed 42
```

Supported arguments:

| Argument | Description | Default |
|---|---|---|
| `--n-rso` | Number of RSOs | 2000 |
| `--k` | Number of neighbors | 10 |
| `--metric` | Probabilistic distance | `hellinger` |
| `--components` | GMM components per RSO | 6 |
| `--leaf-size` | K-d tree leaf size | 1 |
| `--seed` | Random seed | 42 |
| `--cube-size` | Position cube half-width, km | 1000 |
| `--benchmark` | Run benchmark suite | False |

Valid metric choices:

```text
wasserstein
mahalanobis
hellinger
kl
```

---

## 13. Output Format

Save KNN results to CSV.

Suggested columns:

```text
query_id
neighbor_rank
neighbor_id
probabilistic_distance
nominal_euclidean_distance
metric
component_count
seed
```

Example:

```text
query_id,neighbor_rank,neighbor_id,probabilistic_distance,nominal_euclidean_distance,metric
RSO_0001,1,RSO_0345,0.021,120.4,hellinger
RSO_0001,2,RSO_1290,0.034,85.2,hellinger
```

The CSV should contain the probabilistic ranking, not the Euclidean ranking.

---

## 14. Benchmarking

### 14.1 Benchmark scenarios

Evaluate all four distance measures:

- Wasserstein
- Mahalanobis
- Hellinger
- Symmetric KL

Use the same 2,000-RSO catalogue.

Start with:

```text
N = 2000
K = 10
M = 3
leaf_size = 10
seed = 42
```

Then evaluate component counts:

```text
M = 1, 2, 3, 5
```

### 14.2 Performance metrics

Record:

- K-d tree construction time
- K-d tree total query time
- Average query time
- Brute-force total query time
- Number of distance evaluations
- Number of nodes visited
- Number of leaves visited
- Number of nodes pruned
- Mean Recall@K
- Minimum Recall@K
- Mean difference between K-d tree and brute-force neighbor distances

### 14.3 Benchmark table

Generate a summary table:

| Metric | GMM Components | K-D Tree Time | Brute Force Time | Recall@K | Distance Evaluations |
|---|---:|---:|---:|---:|---:|
| Hellinger | 3 | | | | |
| Wasserstein | 3 | | | | |
| Mahalanobis | 3 | | | | |
| Symmetric KL | 3 | | | | |

Use measured values only.

Do not fabricate benchmark results.

---

## 15. Testing

### 15.1 Gaussian distance tests

For every Gaussian-level distance:

- Identical distributions should return zero.
- Distances should be nonnegative within numerical tolerance.
- Symmetric distances should satisfy \(D(P,Q)=D(Q,P)\).
- Invalid covariance matrices should be rejected or handled explicitly.
- Test different means and covariance matrices.

### 15.2 GMM tests

Verify:

- GMM weights sum to one.
- Component count is correct.
- LP coupling constraints are satisfied.
- Identical GMMs return zero for supported distance formulations.
- GMM distance computation is deterministic.

### 15.3 K-d tree tests

Verify:

- All RSOs are represented in the tree.
- Every RSO appears in exactly one leaf.
- AABBs correctly contain their descendants.
- Tree construction is deterministic for a fixed catalogue and seed.
- Duplicate nominal positions do not break tree construction.

### 15.4 KNN tests

Verify:

- Query RSO is not returned as its own neighbor.
- Exactly K neighbors are returned when \(K\leq N-1\).
- Results are sorted by probabilistic distance.
- K=1 works.
- K=N−1 works.
- K larger than the number of available neighbors is handled gracefully.
- Brute-force results match a manually computed small test case.

### 15.5 Reproducibility tests

Verify that:

- The same seed produces the same catalogue.
- Metric switching does not regenerate the catalogue.
- Repeated runs produce identical results for deterministic algorithms.

---

## 16. Dependencies

Use:

```text
Python >= 3.11
NumPy
SciPy
pandas
Matplotlib
pytest
```

Use:

- NumPy for array operations
- SciPy for linear algebra and LP optimization
- pandas for CSV and benchmark tables
- Matplotlib for benchmark plots
- pytest for testing

Do not use scikit-learn for the K-d tree implementation. Implement the tree directly to retain control over the spatial pruning algorithm and instrumentation.

---

## 17. Required Deliverables

Produce:

1. Complete Python package.
2. Gaussian and GMM data models.
3. Four Gaussian-level distance implementations.
4. GMM component-coupling distance implementations.
5. Modular distance factory.
6. K-d tree implementation.
7. Euclidean AABB geometry module.
8. Probabilistic KNN search.
9. Brute-force KNN reference implementation.
10. Synthetic catalogue generator.
11. CLI.
12. CSV neighbor results.
13. Benchmark suite.
14. Recall@K evaluation.
15. Unit tests.
16. README with installation, usage, mathematical assumptions, and limitations.

---

## 18. Final Acceptance Criteria

The implementation is complete when:

- It generates 2,000 reproducible RSOs with 3D GMM uncertainty.
- GMM component count can be changed to 1, 2, 3, or 5.
- All four distance measures can be selected through a CLI argument.
- The same catalogue is used for every metric.
- KNN results are ranked by probabilistic distance.
- Brute-force results are available for validation.
- K-d tree performance and Recall@K are reported.
- Euclidean pruning is explicitly identified as a heuristic unless proven exact.
- GMM distance approximations are documented.
- Results are saved to CSV.
- Tests pass.

The implementation should prioritize **correct modular design, reproducibility, and transparent accuracy evaluation** over premature optimization.
