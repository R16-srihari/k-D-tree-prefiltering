# Task: Implement Mercurio–Singla K-d Tree-Based Conjunction Screening in Python

## 1. Objective

Implement a modular, well-tested Python framework replicating the K-d tree-based conjunction screening approach described in:

1. *A Tree-Based Approach for Efficient and Accurate Conjunction Analysis* — Michael Mercurio and Puneet Singla.
2. *A Non-Combinatorial Approach for Efficient Conjunction Analysis* — accompanying thesis.

The objective is to reduce the computational cost of identifying potentially significant Resident Space Object (RSO) pairs from an orbital catalog while accounting for positional uncertainty.

The implementation must reproduce the original method as closely as possible before introducing improvements.

**Critical architectural requirement:** The probabilistic distance metric must be implemented as an independent module. The K-d tree screening algorithm must not contain metric-specific logic. This will allow alternative probabilistic distance measures to be substituted without modifying the tree implementation.

---

## 2. Scope

Implement the following capabilities:

- Represent RSOs with mean Cartesian position and uncertainty distributions.
- Construct a balanced K-d tree using RSO mean positions.
- Implement the modified Mercurio–Singla nearest-neighbor search.
- Compute Euclidean distance between RSO mean positions.
- Compute Euclidean distance from a query point to a K-d tree bounding box.
- Compute probabilistic distances through a pluggable distance interface.
- Support Gaussian distributions initially.
- Support Gaussian Mixture Models (GMMs).
- Implement Mercurio–Singla's Hellinger distance formulation.
- Compare K-d tree results against brute-force screening.
- Measure execution time, number of distance evaluations, and screening accuracy.

Do not implement TCA determination or probability-of-collision calculation in this project. These belong to subsequent conjunction-analysis modules.

---

## 3. Required software architecture

Create the following project structure:

```text
mercurio_singla/
│

├── pyproject.toml
├── README.md
│
├── src/
│   └── conjunction_screening/
│       ├── __init__.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── rso.py
│       │   ├── gaussian.py
│       │   └── gmm.py
│       │
│       ├── distances/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── hellinger.py
│       │   └── factory.py
│       │
│       ├── spatial/
│       │   ├── __init__.py
│       │   ├── kd_node.py
│       │   ├── kd_tree.py
│       │   └── geometry.py
│       │
│       ├── screening/
│       │   ├── __init__.py
│       │   ├── mercurio_singla.py
│       │   ├── brute_force.py
│       │   └── results.py
│       │
│       ├── benchmarking/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── experiments.py
│       │
│       └── utils/
│           └── validation.py
│
├── tests/
│   ├── test_gaussian_distance.py
│   ├── test_gmm_distance.py
│   ├── test_kd_tree.py
│   ├── test_geometry.py
│   ├── test_screening.py
│   └── test_brute_force.py
│
└── examples/
    ├── gaussian_example.py
    └── gmm_example.py
```

The structure may be refined if necessary, but preserve the separation between spatial indexing, screening logic, probabilistic distance computation, and uncertainty representation.


## 4. RSO data model

Create an RSO class with:

`python
class RSO:
    id: str
    mean_position: np.ndarray
    uncertainty: object
`

Requirements:
- mean_position must be a three-dimensional Cartesian position vector.
- Position units must be consistent across all RSOs.
- The uncertainty distribution must be stored independently of the spatial index.
- Validate dimensions and covariance matrices.
- Support both Gaussian and GMM uncertainty representations.
- Preserve the original catalog identifier.

The K-d tree must use the RSO's overall mean position.

For a GMM:

\[
oldsymbol{\mu}=
\sum_{i=1}^{M}w_ioldsymbol{\mu}_i
\]

The overall mean must be calculated from the component means and weights.

---

## 5. Probabilistic distance interface

Implement a generic abstract interface in:

\\python
from abc import ABC, abstractmethod

class ProbabilisticDistance(ABC):

    @abstractmethod
    def compute(self, distribution_a, distribution_b) -> float:
        pass
\
The K-d tree screening algorithm must interact with this interface exclusively.

Example:

`python
distance_metric = HellingerDistance()

screening = MercurioSinglaScreening(
    tree=tree,
    distance_metric=distance_metric,
)
`

The screening algorithm must not import \HellingerDistance\ directly.

The following metrics should eventually be interchangeable:
- Hellinger distance
- Bhattacharyya distance or coefficient-based dissimilarity
- Mahalanobis distance
- Wasserstein distance
- Other user-defined probabilistic distances

Do not implement all these metrics initially. Implement the interface and Hellinger first.

**Important:** The metric interface must clearly distinguish the probabilistic distance from Euclidean mean-position distance, which is used for tree geometry.

---


## 6. Probabilistic distance specification

Below is an implementable specification for four probabilistic distance measures designed to plug into the Mercurio Singla K-d tree implementation.

The main design rule is to keep the probabilistic distance calculation independent of the K-d tree. The K-d tree can call any distance class through a common interface.

One important distinction: in Mercurio Singla, Hellinger is a distance between probability distributions. Mahalanobis distance is fundamentally a distance between vectors, so we need to define how it compares two uncertainty distributions. I use a pooled-covariance formulation below and identify it as a chosen extension, rather than claiming it is the paper's definition.

### 1. Common interface

from abc import ABC, abstractmethod

class ProbabilisticDistance(ABC):

    @abstractmethod
    def compute(self, distribution_a, distribution_b) -> float:
        Return a non-negative distance or dissimilarity.
        raise NotImplementedError

Suggested input representations:

Gaussian(
    mean: np.ndarray, shape (d,)
    covariance: np.ndarray shape (d, d)
)
GaussianMixture(
    weights: np.ndarray, shape (K,)
    components: list[Gaussian])

Assume both distributions describe the same random variable in the same coordinate system. For orbital applications, that means comparing distributions in a common representation for example, both Cartesian position distributions at the same epoch.

For GMMs, use the component-level distance d_ij to construct a cost matrix, then apply the Mercurio Singla mixture LP:

D(P,Q)=min_w_ij sum_i sum_j w_ij d_ij

subject to

sum_j w_ij = alpha_i, sum_i w_ij = beta_j, w_ij >= 0.

This is an optimal-transport-style coupling of mixture weights. It is the mixture procedure described in the uploaded Mercurio Singla paper.

### 2. Shared Gaussian and GMM utilities

All four algorithms should use common validation and numerical utilities. Covariances must be symmetric positive definite for the Gaussian closed-form formulas below.

import numpy as np
from scipy.linalg import solve, eigh
from scipy.optimize import linprog

def validate_gaussian(mean, covariance):
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mean.ndim != 1:
        raise ValueError Mean must be a 1D array.)
    d = mean.size
    if covariance.shape != (d, d):
        raise ValueError Covariance shape must be (d, d).)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(mean)):
        raise ValueError Mean contains non-finite values.)
    if not np.all(np.isfinite(covariance)):
        raise ValueError Covariance contains non-finite values.)
    if np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise ValueError Covariance must be positive definite.)
    return mean, covariance

def logdet_spd(matrix):
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError Matrix must have positive determinant.

```
import numpy as npfrom scipy.linalg import solve, eighfrom scipy.optimize import linprogdef validate_gaussian(mean, covariance):    mean = np.asarray(mean, dtype=float)    covariance = np.asarray(covariance, dtype=float)    if mean.ndim != 1:        raise ValueError("Mean must be a 1D array.")    d = mean.size    if covariance.shape != (d, d):        raise ValueError("Covariance shape must be (d, d).")    covariance = 0.5 * (covariance + covariance.T)    if not np.all(np.isfinite(mean)):        raise ValueError("Mean contains non-finite values.")    if not np.all(np.isfinite(covariance)):        raise ValueError("Covariance contains non-finite values.")    if np.min(np.linalg.eigvalsh(covariance)) <= 0:        raise ValueError("Covariance must be positive definite.")    return mean, covariancedef logdet_spd(matrix):    sign, value = np.linalg.slogdet(matrix)    if sign <= 0:        raise ValueError("Matrix must have positive determinant.")
```

The code below assumes each Gaussian is represented as a tuple:

```
(mean, covariance)
```

# 3. Hellinger distance

## 3.1 Gaussian formulation

For two Gaussian distributions,

\\[ P=\mathcal N(\mu_1,\Sigma_1),\qquad Q=\mathcal N(\mu_2,\Sigma_2), \\]

define

\\[ \bar{\Sigma}=\frac{\Sigma_1+\Sigma_2}{2}, \qquad \Delta\mu=\mu_1-\mu_2. \\]

The Bhattacharyya coefficient is

\\[ BC(P,Q)= \frac{|\Sigma_1|^{1/4}|\Sigma_2|^{1/4}} {|\bar{\Sigma}|^{1/2}} \exp\left( -\frac18\Delta\mu^T\bar{\Sigma}^{-1}\Delta\mu \right). \\]

Then the Hellinger distance used by Mercurio–Singla is

\\[ \boxed{D_H(P,Q)=\sqrt{1-BC(P,Q)}}. \\]

It ranges from 0 to 1 for continuous probability distributions. It is symmetric and satisfies the triangle inequality.&#x20;

Mercurio_Singla_KD_Tree_based approack for efficient and accurate conjunction.pdf



## 3.2 Implementable Gaussian algorithm

```
class HellingerDistance(ProbabilisticDistance):    def gaussian(self, gaussian_a, gaussian_b):        mu1, S1 = validate_gaussian(*gaussian_a)        mu2, S2 = validate_gaussian(*gaussian_b)        delta = mu1 - mu2        Sbar = 0.5 * (S1 + S2)        # log of determinant factor        log_bc = (            0.25 * logdet_spd(S1)            + 0.25 * logdet_spd(S2)            - 0.50 * logdet_spd(Sbar)        )        # delta.T @ inv(Sbar) @ delta, without explicitly inverting        quadratic = delta @ solve(Sbar, delta, assume_a="pos")        log_bc -= 0.125 * quadratic        # Numerical safety        bc = np.exp(min(log_bc, 0.0))        bc = np.clip(bc, 0.0, 1.0)        return np.sqrt(max(0.0, 1.0 - bc))    def compute(self, distribution_a, distribution_b):        return self.gaussian(distribution_a, distribution_b)
```

## 3.3 GMM algorithm

For

\\[ P=\sum\_{i=1}^{N}\alpha_iP_i,\qquad Q=\sum\_{j=1}^{M}\beta_jQ_j, \\]

calculate each Gaussian component distance:

\\[ C\_{ij}=D_H(P_i,Q_j). \\]

Then solve the LP:

\\[ \min\_{w\_{ij}}\sum\_{i,j}w\_{ij}C\_{ij} \\]

subject to the mixture-weight constraints.

This follows the component-coupling method described in the paper.&#x20;

Mercurio_Singla_KD_Tree_based approack for efficient and accurate conjunction.pdf



### Generic GMM LP

```
def gmm_lp_distance(weights_a, components_a,                    weights_b, components_b,                    component_distance):    alpha = validate_gmm(weights_a, components_a)    beta = validate_gmm(weights_b, components_b)    N = len(components_a)    M = len(components_b)    # Component cost matrix    C = np.empty((N, M))    for i in range(N):        for j in range(M):            C[i, j] = component_distance(                components_a[i],                components_b[j]            )    # LP variables are w_ij, flattened    c = C.ravel()    A_eq = []    b_eq = []    # Each row of the coupling sums to alpha_i    for i in range(N):        row = np.zeros((N, M))        row[i, :] = 1.0        A_eq.append(row.ravel())        b_eq.append(alpha[i])    # Each column sums to beta_j    for j in range(M):        row = np.zeros((N, M))
```

For the Hellinger GMM calculation:

```
hellinger = HellingerDistance()D = gmm_lp_distance(    weights_a, components_a,    weights_b, components_b,    component_distance=hellinger.gaussian)
```

# 4. Wasserstein distance

## 4.1 Gaussian \\(2\\)-Wasserstein distance

Use the \\(2\\)-Wasserstein distance, denoted \\(W_2\\). For Gaussian distributions, it has a closed-form solution:

\\[ P=\mathcal N(\mu_1,\Sigma_1),\quad Q=\mathcal N(\mu_2,\Sigma_2) \\]

\\[ \boxed{ W_2^2(P,Q)= \\|\mu_1-\mu_2\\|\_2^2+ \operatorname{tr}\left( \Sigma_1+\Sigma_2 -2(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2} \right) } \\]

The distance is \\(W_2=\sqrt{W_2^2}\\).

Unlike Hellinger, Wasserstein incorporates both mean displacement and covariance shape/scale. Its units are the same as the units of the state variable. For example, position-only Wasserstein distance is in metres.

## 4.2 Algorithm

1. Validate both means and covariances.
2. Calculate the squared mean separation.
3. Compute the symmetric positive-semidefinite square root of \\(\Sigma_1\\).
4. Construct \\(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}\\).
5. Compute its square root.
6. Evaluate the trace expression.
7. Clip tiny negative values caused by floating-point error to zero.
8. Return the square root.

```
def matrix_sqrt_psd(A, tolerance=1e-10):    A = 0.5 * (A + A.T)    eigenvalues, eigenvectors = eigh(A)    if eigenvalues.min() < -tolerance:        raise ValueError("Matrix is not positive semidefinite.")    eigenvalues = np.clip(eigenvalues, 0.0, None)    return (        eigenvectors        @ np.diag(np.sqrt(eigenvalues))        @ eigenvectors.T    )class WassersteinDistance(ProbabilisticDistance):    def gaussian_squared(self, gaussian_a, gaussian_b):        mu1, S1 = validate_gaussian(*gaussian_a)        mu2, S2 = validate_gaussian(*gaussian_b)        delta = mu1 - mu2        mean_term = delta @ delta        S1_sqrt = matrix_sqrt_psd(S1)        middle = S1_sqrt @ S2 @ S1_sqrt        middle_sqrt = matrix_sqrt_psd(middle)        covariance_term = np.trace(            S1 + S2 - 2.0 * middle_sqrt        )        return max(0.0, mean_term + covariance_term)    def gaussian(self, gaussian_a, gaussian_b):        return np.sqrt(            self.gaussian_squared(gaussian_a, gaussian_b)        )    def compute(self, distribution_a, distribution_b):        return self.gaussian(distribution_a, distribution_b)
```

## 4.3 GMM Wasserstein approximation

The exact Wasserstein distance between two arbitrary GMMs is generally not obtained simply by comparing Gaussian components. A practical approximation for this project is to transport the mixture weights between Gaussian components.

First compute:

\\[ C\_{ij}=W_2^2(P_i,Q_j). \\]

Then solve:

\\[ \min\_{w\_{ij}}\sum\_{ij}w\_{ij}W_2^2(P_i,Q_j) \\]

subject to the mixture coupling constraints.

This is an optimal transport between mixture components, using Gaussian \\(W_2^2\\) as the ground cost. It is not generally the exact Wasserstein distance between the full continuous GMM PDFs.

```
def gmm_wasserstein(weights_a, components_a,                    weights_b, components_b):    metric = WassersteinDistance()    return gmm_lp_distance(        weights_a, components_a,        weights_b, components_b,        component_distance=metric.gaussian_squared    )
```

Important: this function returns the optimal transport cost based on squared component distances. It does not take the square root of the final LP objective. If you want the conventional component-transport distance, use:

```
D_transport = np.sqrt(    gmm_wasserstein(        weights_a, components_a,        weights_b, components_b    ))
```

Keep both conventions explicit in the API or documentation.

# 5. Mahalanobis distance

## 5.1 Definition for uncertainty distributions

Mahalanobis distance normally measures separation between vectors:

\\[ D_M(x,y)= \sqrt{(x-y)^T\Sigma^{-1}(x-y)}. \\]

For two uncertain RSOs, we need to decide what covariance \\(\Sigma\\) should represent. A reasonable, implementable choice is the pooled covariance:

\\[ \Sigma\_{\mathrm{pool}}=\frac{\Sigma_1+\Sigma_2}{2}. \\]

Then define the distribution-level dissimilarity:

\\[ \boxed{ D_M(P,Q)= \sqrt{ (\mu_1-\mu_2)^T \Sigma\_{\mathrm{pool}}^{-1} (\mu_1-\mu_2) } } \\]

Interpretation: mean separation measured in units of the combined uncertainty.

This is useful for orbital screening because it answers a question like:

> How many pooled standard deviations apart are the two mean positions?

However, it does not fully compare the distributions. Two distributions with identical means but different covariances receive distance zero.

## 5.2 Algorithm

1. Calculate \\(\Delta\mu=\mu_1-\mu_2\\).
2. Form the pooled covariance.
3. Solve \\(\Sigma\_{\mathrm{pool}}z=\Delta\mu\\).
4. Compute \\(\sqrt{\Delta\mu^Tz}\\).

```
class MahalanobisDistance(ProbabilisticDistance):    def gaussian(self, gaussian_a, gaussian_b):        mu1, S1 = validate_gaussian(*gaussian_a)        mu2, S2 = validate_gaussian(*gaussian_b)        delta = mu1 - mu2        S_pool = 0.5 * (S1 + S2)        z = solve(S_pool, delta, assume_a="pos")        squared_distance = delta @ z        return np.sqrt(max(0.0, squared_distance))    def compute(self, distribution_a, distribution_b):        return self.gaussian(distribution_a, distribution_b)
```

## 5.3 GMM version

For the LP-based mixture comparison, calculate:

\\[ C\_{ij}=D_M(P_i,Q_j) \\]

between every pair of Gaussian component means, using their pooled covariance.

```
def gmm_mahalanobis(weights_a, components_a,                    weights_b, components_b):    metric = MahalanobisDistance()    return gmm_lp_distance(        weights_a, components_a,        weights_b, components_b,        component_distance=metric.gaussian    )
```

### Limitation to document

The pooled covariance makes the Mahalanobis distance uncertainty-aware, but it remains a mean-separation metric. It is not a full distribution distance and does not satisfy all properties of a metric over Gaussian distributions.

For this reason, I would treat it as an experimental ranking measure rather than a direct replacement for Hellinger.

# 6. Symmetric KL divergence

## 6.1 Definition

The Kullback–Leibler divergence is asymmetric:

\\[ D\_{\mathrm{KL}}(P\\|Q)\neq D\_{\mathrm{KL}}(Q\\|P). \\]

A common symmetrization is the average KL divergence, also called the half-Jeffreys divergence:

\\[ \boxed{ D\_{\mathrm{SKL}}(P,Q) =\frac12\left[ D\_{\mathrm{KL}}(P\\|Q)+ D\_{\mathrm{KL}}(Q\\|P) \right] } \\]

This is symmetric and non-negative, but it is not a metric because it does not generally satisfy the triangle inequality.

Mercurio–Singla specifically note the symmetry and triangle-inequality limitations of KL divergence.&#x20;

Mercurio_Singla_KD_Tree_based approack for efficient and accurate conjunction.pdf



## 6.2 Gaussian closed-form expression

For two Gaussians in \\(d\\) dimensions, define

\\[ \Delta\mu=\mu_1-\mu_2. \\]

The KL divergence is

\\[ D\_{\mathrm{KL}}(P\\|Q) =\frac12\left[ \log\frac{|\Sigma_2|}{|\Sigma_1|} +\operatorname{tr}(\Sigma_2^{-1}\Sigma_1) -d +\Delta\mu^T\Sigma_2^{-1}\Delta\mu \right]. \\]

After symmetrization, the log-determinant terms cancel:

\\[ \boxed{ D\_{\mathrm{SKL}}(P,Q)= \frac14\left[ \operatorname{tr}(\Sigma_2^{-1}\Sigma_1+                   \Sigma_1^{-1}\Sigma_2) -2d +\Delta\mu^T(\Sigma_1^{-1}+\Sigma_2^{-1})\Delta\mu \right] } \\]

This is a divergence, not a distance in the strict metric sense. Its value is unbounded above.

## 6.3 Algorithm

1. Validate means and covariance matrices.
2. Calculate \\(\Delta\mu\\).
3. Solve for \\(\Sigma_1^{-1}\Sigma_2\\) and \\(\Sigma_2^{-1}\Sigma_1\\) using linear solves.
4. Compute trace terms.
5. Compute the mean-separation term.
6. Combine the terms.

```
class SymmetricKLDistance(ProbabilisticDistance):    def gaussian(self, gaussian_a, gaussian_b):        mu1, S1 = validate_gaussian(*gaussian_a)        mu2, S2 = validate_gaussian(*gaussian_b)        d = len(mu1)        delta = mu1 - mu2        # Tr(S2^-1 S1)        trace_21 = np.trace(            solve(S2, S1, assume_a="pos")        )        # Tr(S1^-1 S2)        trace_12 = np.trace(            solve(S1, S2, assume_a="pos")        )        # delta.T @ S1^-1 @ delta        mean_term_1 = delta @ solve(            S1, delta, assume_a="pos"        )        # delta.T @ S2^-1 @ delta        mean_term_2 = delta @ solve(            S2, delta, assume_a="pos"        )        value = 0.25 * (            trace_21 + trace_12 - 2.0 * d            + mean_term_1 + mean_term_2        )        return max(0.0, value)
```

## 6.4 GMM version

For each component pair, calculate

\\[ C\_{ij}=D\_{\mathrm{SKL}}(P_i,Q_j). \\]

Then use the same mixture LP:

```
def gmm_symmetric_kl(weights_a, components_a,                     weights_b, components_b):    metric = SymmetricKLDistance()    return gmm_lp_distance(        weights_a, components_a,        weights_b, components_b,        component_distance=metric.gaussian    )
```

Terminology note: this LP-based GMM result is a component-coupled symmetric-KL dissimilarity. It is not generally equal to the symmetric KL divergence between the full mixture PDFs.

# 7. Unified comparison table

| Measure               | Mean separation | Covariance difference       | Range            | Metric?                | Gaussian closed form |
| --------------------- | --------------- | --------------------------- | ---------------- | ---------------------- | -------------------- |
| Hellinger             | Yes             | Yes                         | \\([0,1]\\)      | Yes                    | Yes                  |
| \\(W_2\\) Wasserstein | Yes             | Yes                         | \\([0,\infty)\\) | Yes                    | Yes                  |
| Pooled Mahalanobis    | Yes             | Through pooled scaling only | \\([0,\infty)\\) | Not over distributions | Yes                  |
| Symmetric KL          | Yes             | Yes                         | \\([0,\infty)\\) | No triangle inequality | Yes                  |

For the K-d tree experiment, the measures provide different notions of similarity:

- Hellinger: PDF overlap / similarity.
- Wasserstein: distributional displacement, with covariance shape contributing.
- Mahalanobis: normalized mean separation.
- Symmetric KL: information divergence, penalizing mismatch in means and covariance.

## 8. Recommended project structure

```
src/distances/
├── base.py
├── hellinger.py
├── wasserstein.py
├── mahalanobis.py
├── symmetric_kl.py
├── gmm_transport.py
└── factory.py
```

Keep Gaussian component calculations separate from GMM coupling. This allows each distance class to expose its Gaussian implementation, while `gmm_transport.py` handles the mixture-level LP.

A factory could be:

```
def get_distance(name):    distances = {        "hellinger": HellingerDistance,        "wasserstein": WassersteinDistance,        "mahalanobis": MahalanobisDistance,        "symmetric_kl": SymmetricKLDistance,    }    try:        return distances[name.lower()]()    except KeyError:        raise ValueError(f"Unknown distance measure: {name}")
```
