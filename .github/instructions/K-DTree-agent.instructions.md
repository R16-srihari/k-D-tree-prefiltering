---
description: Workspace instructions for the Github Copilot agent and Cline agents, governing code generation and adherence to authoritative documentation.
---

# Copilot and Cline Agent Instructions

## Purpose

These instructions govern how the copilot agent behaves and, more importantly, how all downstream code generation must be performed. They instruct the agent that **every piece of code generated must be derived from and consistent with the authoritative documentation located in the Sources/ folder** of this repository. Code that does not trace to a documented algorithm, formula, or approach in Sources/ must not be produced without an explicit justification.

## Core Directive: Follow the Sources

> **All code generation must follow the documentation in the Sources folder.**

When generating any code for this project, you must:

1. **Identify the relevant documentation file** in Sources/ before writing any code.
2. **Use the algorithms, mathematical formulations, and design decisions** documented in those files as the sole basis for implementation.
3. **Do not introduce new algorithms, heuristics, or formulas** that are not grounded in the Sources/ documentation, unless the user explicitly authorizes such additions.
4. **Cross-reference code back to the documentation** and include a comment or note in the generated code indicating which source file and section the code implements.
5. **If the Sources/ documentation is ambiguous or incomplete**, state the ambiguity, propose the most faithful interpretation, and ask for confirmation before proceeding.

## Authoritative Documentation in Sources/

| File | What it documents | Relevance to code generation |
|---|---|---|
| Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md | The full tree-based conjunction analysis methodology: kd-tree construction, conventional and **modified probabilistic nearest-neighbor search**, Hellinger distance, Gaussian Mixture Models (GMMs), probability of collision, and numerical validation. | Primary reference for the Kd-Tree prefiltering algorithm, tree construction, nearest-neighbor search, GMM handling, and the distance metric computations. |
| Sources/A_noncombinatorial_approach_for_efficient_conjunction analysis.md | Michael Mercurio's 2014 PhD thesis: uncertainty propagation, Gaussian Mixture Models, probabilistic distance metrics, kd-trees, nearest-neighbor search, and brute-force vs. tree-based comparison. | Reference for the theoretical foundations: uncertainty characterization, probability-of-collision computation, probabilistic distance metrics, and tree-based search. |
| Sources/Distance_formulae_algo.md | Mathematical formulation and algorithmic implementation of symmetric KL divergence, mixture Wasserstein-2 distance, and Mahalanobis distance between GMMs. | Reference for distance/metrics between GMMs; useful for variants, verification, or extensions of the Hellinger-based distance. |
| Sources/Hellinger_distance_lecture.md | Lecture notes on the Hellinger distance: definition, relation to the Bhattacharyya coefficient/Fidelity, cut-and-paste property, and the triangle inequality. | Reference for the Hellinger distance and its metric properties; confirms the Hellinger distance is a proper metric (self-contained definition). |


## Core Directive: Follow the Sources

> **All code generation must follow the documentation in the Sources folder.**

When generating any code for this project, you must:

1. **Identify the relevant documentation file** in Sources/ before writing any code.
2. **Use the algorithms, mathematical formulations, and design decisions** documented in those files as the sole basis for implementation.
3. **Do not introduce new algorithms, heuristics, or formulas** that are not grounded in the Sources/ documentation, unless the user explicitly authorizes such additions.
4. **Cross-reference code back to the documentation** and include a comment or note in the generated code indicating which source file and section the code implements.
5. **If the Sources/ documentation is ambiguous or incomplete**, state the ambiguity, propose the most faithful interpretation, and ask for confirmation before proceeding.

## Authoritative Documentation in Sources/

| File | What it documents | Relevance to code generation |
|---|---|---|
| Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md | The full tree-based conjunction analysis methodology: kd-tree construction, conventional and **modified probabilistic nearest-neighbor search**, Hellinger distance, Gaussian Mixture Models (GMMs), probability of collision, and numerical validation. | Primary reference for the Kd-Tree prefiltering algorithm, tree construction, nearest-neighbor search, GMM handling, and the distance metric computations. |
| Sources/A_noncombinatorial_approach_for_efficient_conjunction analysis.md | Michael Mercurio's 2014 PhD thesis: uncertainty propagation, Gaussian Mixture Models, probabilistic distance metrics, kd-trees, nearest-neighbor search, and brute-force vs. tree-based comparison. | Reference for the theoretical foundations: uncertainty characterization, probability-of-collision computation, probabilistic distance metrics, and tree-based search. |
| Sources/Distance_formulae_algo.md | Mathematical formulation and algorithmic implementation of symmetric KL divergence, mixture Wasserstein-2 distance, and Mahalanobis distance between GMMs. | Reference for distance/metrics between GMMs; useful for variants, verification, or extensions of the Hellinger-based distance. |
| Sources/Hellinger_distance_lecture.md | Lecture notes on the Hellinger distance: definition, relation to the Bhattacharyya coefficient/Fidelity, cut-and-paste property, and the triangle inequality. | Reference for the Hellinger distance and its metric properties; confirms the Hellinger distance is a proper metric (self-contained definition). |


## Code Generation Constraints

### 1. kd-tree construction

- Follow the **top-down, median-based** construction described in Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md (Section 2.1 and Algorithm 1) and Sources/A_noncombinatorial_approach_for_efficient_conjunction analysis.md (Section 3.4).
- The tree partitions space along the **longest available dimension** at each node to maintain a balanced tree (O(N log N) construction).
- Each internal node represents a bounding box; each leaf holds exactly one data point.
- Use the **mean location of each RSO (Gaussian component)** as the spatial key for tree indexing, as documented in the modified nearest-neighbor section.

### 2. Nearest-neighbor search

- Use the **modified probabilistic nearest-neighbor search** (Algorithm 3 in Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md), not the standard deterministic one, because RSO positions carry uncertainty.
- Retain the structure of the standard NN search:
  1. Locate the item of interest in the kd-tree.
  2. Traverse up the tree, opening boxes until the required number of nearest neighbors is met.
  3. Compute both the **Hellinger distance** and the **Euclidean distance** to the nearest neighbors found.
  4. Store minimum distances DHmin and DEmin.
  5. Compute the Euclidean distance to all remaining nodes/boxes Dbox.
  6. If Dbox < DEmin, open the box and compute distances to its enclosed items; update DEmin and the neighbor list.
- **Continue using Euclidean distance for the distance-to-box comparison** (the Dbox < DEmin test), while using **Hellinger distance for the inter-RSO comparison** inside opened boxes.
- The reason: Hellinger distance is normalized (range [0, 1]) and heavily favored over raw Euclidean distance in a non-normalized space; using it for the box comparison produces incorrect pruning. This is a documented pitfall in the source paper (Section 2.5).

### 3. Probabilistic distance metrics

- **Hellinger distance** is the metric of choice for inter-RSO distance because it is a proper metric (satisfies non-negativity, symmetry, and the triangle inequality). Reference: Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md (Section 2.3) and Sources/Hellinger_distance_lecture.md.
- The closed-form Hellinger distance between two Gaussian components is given in Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md (Eq. 15):
  DH(p,q) = (1/sqrt(2)) * sqrt(1 - (|Sigmap|^(1/4) * |Sigmaq|^(1/4)) / (|Sigmapq|^(1/2)) * exp(-(1/8) * (mu_p - mu_q)^T Sigmapq^(-1) (mu_p - mu_q)))
  where Sigmapq = (1/2)(Sigmap + Sigmaq). Note the exponent on the exponential is -(1/8), not -(1/2).
- For **Gaussian Mixture Models (GMMs)**, do **not** fall back to numerical integration where a closed or linear-programming approach is documented. Use the linear programming framework of Liu & Huang (2000) described in Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md (Section 2.4):
  D(P,Q) = min_W sum_i sum_j w_ij DH(p_i, q_j)
  subject to w_ij >= 0, sum_j w_ij = alpha_i (row sums), and sum_i w_ij = beta_j (column sums), where alpha_i, beta_j are the mixture weights of the component GMMs and DH(p_i, q_j) is the closed-form Hellinger distance between Gaussian components.
- The LP problem is convex; a feasible (though typically non-optimal) solution is w_ij = alpha_i * beta_j.

### 4. Probability of collision

- When implementing probability-of-collision computation (for verification or optional extension), follow the fusion/overlap approach in Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md (Section 2.3) and Sources/A_noncombinatorial_approach_for_efficient_conjunction analysis.md (Section 3.2).
- For two i.i.d. Gaussians: P_coll = c, where c = (1 / ((2*pi)^(3/2) * |Sigma1 + Sigma2|^(1/2))) * exp(-(1/2) * (mu1-mu2)^T (Sigma1+Sigma2)^(-1) (mu1-mu2)).
- For two GMMs, the collision probability is obtained by summing over all component pairs: P_coll = sum_i sum_j alpha_i * beta_j * c_ij.

### 5. Uncertainty representation

- Each RSO's position uncertainty is represented by a **Gaussian Mixture Model (GMM)**: P(x|w) = sum_i w_i * N(x | mu_i, Sigma_i).
- GMM components are typically generated with means chi_i = mu +/- sqrt(N*Sigma) * e_i (as described in Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md Section 3, Eqs. 29-30), each component having covariance (1/3)Sigma or the full Sigma.
- The weights of the GMM sum to 1: sum_i w_i = 1.

### 6. Assumptions to preserve

- Linear (high-speed) relative motion is assumed: the relative velocity vector is constant during the encounter; position uncertainties are constant and represented by GMMs; RSOs are assumed spherical.
- The encounter-plane transformation (Eqs. 2.16-2.27) is the correct frame for 2D probability-of-collision integration: i-hat along the relative velocity, k-hat = i-hat x j-hat, with j-hat = (v2 x v1) / |v2 x v1|.


## Handling Discrepancies and Ambiguities

- **When a task does not obviously map to Sources/ documentation:** Search Sources/ first. If no matching documentation exists, ask the user before generating code.
- **When Sources/ documentation conflicts with itself:** Prefer the more recent and more specific source (Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md, CMES 2016). Cite both sources in the justification.
- **When Sources/ documentation is incomplete for the requested feature:** State the gap, document your interpretation, and confirm with the user before generating the full implementation.
- **Never silently combine algorithms from unrelated sources** (e.g., mixing a Wasserstein-based distance metric into a pipeline that the Sources specify should use Hellinger distance), unless the user explicitly requests the combination.

## Code Output Standards

- Every generated code file should include a header comment identifying the **source documentation file** and **section** that the implementation follows.
- Where algorithms differ between sources, include a comment citing the differing sources and the chosen interpretation.
- If you generate numerical comparisons or validation tables, ground them in the **numerical results described in Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md** (e.g., 200, 2,000, 5,000, and 10,000 RSO sets; CPU time reductions of ~3 orders of magnitude; accuracy of ~98.6-99.5%).
- Do not generate code that contradicts the documented computational complexity (O(N log N) tree-based vs. O(N^2) brute-force) unless the user explicitly requests a change in complexity.

## Source Reference Summary

- Primary implementation reference: Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md
- Theoretical foundations: Sources/A_noncombinatorial_approach_for_efficient_conjunction analysis.md
- Distance-metric variants (optional): Sources/Distance_formulae_algo.md and Sources/Hellinger_distance_lecture.md
