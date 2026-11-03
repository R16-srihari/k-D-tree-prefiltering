# Algorithm: Mercurio–Singla K-d Tree-Based Conjunction Screening

The following is a replication-oriented algorithm based on Michael Mercurio and Puneet Singla’s A Tree-Based Approach for Efficient and Accurate Conjunction Analysis (2016), cross-referenced with the accompanying thesis, A Non-Combinatorial Approach for Efficient Conjunction Analysis.

The central idea is to replace exhaustive all-on-all conjunction screening with a K-d tree nearest-neighbor search, modified to account for orbital uncertainty using the Hellinger distance. Gaussian Mixture Models (GMMs) extend the approach to non-Gaussian uncertainty.

---
---

## 4. Stage II — Distance metrics

## 1. Algorithm overview

The method has four main stages:

1. Construct a K-d tree using the mean positions of the resident space objects (RSOs).
2. Locate each RSO in the tree and find an initial set of neighbors.
3. Evaluate probabilistic distances between candidate RSOs using the Hellinger distance.
4. Search other tree nodes using Euclidean distance-to-box pruning.

The important distinction is:

- **Tree geometry**: Euclidean positions of RSO means.
- **Candidate-pair comparison**: Hellinger distance between uncertainty distributions.
- **Tree pruning**: Euclidean distance from an RSO mean to a node's bounding box.

The authors retain the conventional K-d tree structure while modifying its nearest-neighbor metric and search procedure.

---

## 2. Input data and notation

For each RSO $i$, define:

$$ \mathcal R_i = \left\{ \boldsymbol{\mu}_i,\; P_i,\; \text{ID}_i \right\} $$

where:

- $\boldsymbol{\mu}_i\in\mathbb R^3$: mean Cartesian position at the screening epoch.
- $P_i$: position uncertainty PDF.
- $\text{ID}_i$: catalog identifier.

For a Gaussian uncertainty:

$$ P_i=\mathcal N(\boldsymbol{\mu}_i,\Sigma_i) $$

For a Gaussian mixture:

$$ P_i(\mathbf x)= \sum_{a=1}^{M_i} \alpha_{ia} \mathcal N(\mathbf x\mid\boldsymbol{\mu}_{ia},\Sigma_{ia}) $$

with

$$ \sum_{a=1}^{M_i}\alpha_{ia}=1 $$

The position mean used for tree construction is the overall distribution mean:

$$ \boldsymbol{\mu}_i= \sum_{a=1}^{M_i}\alpha_{ia}\boldsymbol{\mu}_{ia} $$

For a Gaussian, this is simply the Gaussian mean.

**Important**: The K-d tree is constructed from one mean position per RSO—not from every individual Gaussian mixture component.

---

## 3. Stage I — Construct the K-d tree

### Algorithm 1: Balanced K-d tree construction

The paper describes recursive partitioning around median values, with divisions along a suitable dimension, typically the longest available dimension. The tree terminates when leaves contain one or two objects in the paper's implementation.

```
FUNCTION BUILD_KDTREE(RSOs):

    root_box.lower = componentwise_minimum(RSO.mean_position)
    root_box.upper = componentwise_maximum(RSO.mean_position)

    RETURN BUILD_NODE(RSOs, root_box)


FUNCTION BUILD_NODE(points, box):

    node.box = box

    IF number_of_points(points) <= LEAF_CAPACITY:
        node.points = points
        node.is_leaf = TRUE
        RETURN node

    # Choose spatial splitting dimension
    dimension = argmax(box.upper - box.lower)

    # Split around median mean-position coordinate
    median = median(points[:, dimension])

    left_points  = points where coordinate < median
    right_points = points where coordinate >= median

    # Split bounding box
    left_box  = box restricted to lower half
    right_box = box restricted to upper half

    node.is_leaf = FALSE

    node.left  = BUILD_NODE(left_points, left_box)
    node.right = BUILD_NODE(right_points, right_box)

    RETURN node
$$

### Bounding box representation

Each node stores:

$$ \mathcal B= [\ell_1,u_1]\times[\ell_2,u_2]\times[\ell_3,u_3] $$

where $\ell_d$ and $u_d$ are lower and upper limits in dimension $d$.

---

## 5. Stage III — Modified nearest-neighbor search

This is the main algorithm from the paper.

The original nearest-neighbor K-d tree searches using Euclidean distance. Mercurio–Singla modifies the search by:

- Computing Hellinger distances between RSOs.
- Maintaining Euclidean distances for tree pruning.
- Using Euclidean distance-to-box to decide whether another node should be opened.

### Algorithm 2: Modified probabilistic nearest-neighbor search

```
INPUT:
    RSOs
    K-d tree
    K = required number of nearest neighbors

FOR each RSO p:

    1. Locate p in the K-d tree.

    2. Traverse upward through the tree.

       Open containing boxes until at least K candidate neighbors
       have been found.

    3. Compute for each candidate q:

           DH(p,q) = Hellinger distance
           DE(p,q) = Euclidean distance between mean positions

    4. Store:
           probabilistic nearest neighbors
           DH_min = current minimum Hellinger distance
           DE_min = current minimum Euclidean distance

    5. Examine remaining K-d tree boxes.

       For each box B:

           Dbox = Euclidean distance from mean position of p
                  to bounding box B

           IF Dbox < DE_min:

               Open box B.

               Compute distances to all enclosed RSOs.

               Update:
                   nearest neighbors
                   DH_min
                   DE_min

           ELSE:

               Skip box B.

    6. Return probabilistic nearest neighbors of p.

END FOR
```

This follows the paper's Algorithm 3.

### 5.1 Distance-to-box function

For a query mean position:

$$ \boldsymbol{\mu}_p= (\mu_{p,1},\mu_{p,2},\mu_{p,3}) $$

and box:

$$ \mathcal B=[\ell_1,u_1]\times[\ell_2,u_2]\times[\ell_3,u_3] $$

define the distance in each coordinate:

$$ \delta_d= \begin{cases} \ell_d-\mu_{p,d}, & \mu_{p,d}<\ell_d\\\ 0, & \ell_d\le\mu_{p,d}\le u_d\\\ \mu_{p,d}-u_d, & \mu_{p,d}>u_d \end{cases} $$

Then:

$$ \boxed{ D\_{\mathrm{box}}= \sqrt{\delta_1^2+\delta_2^2+\delta_3^2} } $$

This gives the minimum Euclidean distance from the query mean to any point inside the bounding box.

### 5.2 Neighbor update logic

For a practical implementation, maintain a list of the $K$ smallest Hellinger distances:

$$ \mathcal N_K(p)= \operatorname{arg\,Kmin}_{q\ne p} D_H(P_p,P_q) $$

Also maintain the corresponding Euclidean distances.

For the paper's single-nearest-neighbor formulation:

$$ D_{H,\min}=\min_q D_H(P_p,P_q) $$

$$ D_{E,\min}=\min_q D_E(p,q) $$

For the $K$-neighbor extension, use the $K$-th smallest distance as the current threshold:

$$ D_{E,K}=\operatorname{KthSmallest}_q D_E(p,q) $$

This is a natural implementation generalization of the paper's nearest-neighbor algorithm. It should be distinguished from its literal minimum-distance description.

---

## 6. Complete end-to-end screening algorithm

This combines the tree construction and modified neighbor search into one workflow.

```
    candidate_pairs = set()    for p in RSOs:        # Initial candidate set from tree traversal        candidates = initial_candidates(tree, p, K)        # Store distances for candidate RSOs        for q in candidates:            if p.id == q.id:                continue            DH = hellinger(p.uncertainty, q.uncertainty)            DE = euclidean(p.mean_position, q.mean_position)            update_neighbor_list(p, q, DH, DE, K)        # Search remaining nodes        for node in remaining_nodes(tree, p):            DE_box = box_distance(p.mean_position, node.box)            # Paper's Euclidean distance-to-box pruning rule            if DE_box < current_euclidean_threshold(p, K):                for q in node.points_or_descendants():                    if p.id == q.id:                        continue                    DH = hellinger(                        p.uncertainty,                        q.uncertainty                    )                    DE = euclidean(                        p.mean_position,                        q.mean_position                    )                    update_neighbor_list(p, q, DH, DE, K)        # Save directed neighbor relations as undirected pairs        for q in p.neighbors:            pair = tuple(sorted((p.id, q.id)))            candidate_pairs.add(pair)    return candidate_pairs
```

The K-d tree stage is therefore not equivalent to a conventional geometric conjunction filter. It ranks uncertain objects according to PDF similarity and uses spatial indexing to reduce comparisons.

---

## 8. Important limitations for a faithful replication
---

## 9. Complexity

The motivation is the reduction from the all-pairs calculation:

$$ N_{\mathrm{pairs}} = \frac{N(N-1)}{2} = O(N^2) $$

to a K-d tree construction and nearest-neighbor search whose average behavior is approximately:

$$ O(N\log N) $$

The paper reports the same $O(N\log N)$ scaling for the modified probabilistic approach, with the additional Euclidean calculations contributing only a constant-factor increase.

There is, however, an important limitation that is particularly relevant to your research: **the distance-to-box test is deterministic**. It uses the mean positions rather than the full uncertainty distributions. Consequently, a probabilistically close object can be missed if its mean lies sufficiently far away for its K-d tree node to be pruned. The paper explicitly identifies this as the source of its imperfect nearest-neighbour capture.

That gives you a useful research formulation:

$$ \boxed{ \text{KD-tree} \overset{\text{mean-position pruning}}{\longrightarrow} \text{probabilistic NN candidates} \overset{D_H}{\longrightarrow} \text{candidate pairs} \overset{\text{TCA},\,P_c}{\longrightarrow} \text{conjunctions} } $$

and also highlights the natural place where you could improve the original method: **replace the deterministic $D_{\mathrm{box}}$ pruning criterion with an uncertainty-aware conservative bound**, while retaining the K-d tree's spatial efficiency.

The authors explicitly report shortcomings that matter when implementing this algorithm.

### A. Euclidean pruning is not a guaranteed probabilistic pruning rule

The algorithm opens a box only if:

$$ D_{\mathrm{box}}<D_{E,\min} $$

But the actual neighbor ranking is based on:

$$ D_H(P_i,P_j) $$

These metrics are not equivalent.

Two RSOs may have mean positions that are far apart but possess broad or overlapping uncertainty distributions. A box can therefore be skipped even when it contains a probabilistically close neighbor.

The thesis reports a case involving 200 RSOs where brute-force analysis identified a probable collision that the modified K-d tree missed. The authors attribute this possibility to the Euclidean box-pruning strategy.

### B. The original algorithm is single-epoch

The authors explicitly state that the method is implemented for a single time instant. Propagating uncertainties and rebuilding or updating the tree over time were identified as future research directions.

### C. Nearest neighbors are not necessarily collision pairs

The K-d tree identifies probabilistically similar distributions. It does not directly enforce:

$$ \|\mathbf r_i-\mathbf r_j\|<R_{\mathrm{HBR}} $$

Nor does it guarantee that all pairs satisfying a collision-distance threshold are retained.

This means the method is best understood as a candidate selection mechanism, not a collision-complete screening guarantee.

---

## 7. Candidate-pair screening vs. collision probability

The algorithm returns candidate pairs:

$$ \mathcal C= \{(i,j):j\in\mathcal N_K(i)\} $$

These pairs are then passed to a detailed conjunction-analysis module.

A subsequent pipeline would typically be:

1. Propagate candidate states
2. Determine TCA (Time of Closest Approach)
3. Calculate relative state/covariance
4. Calculate probability of collision $P_c$
5. Apply conjunction threshold

The K-d tree stage is therefore not equivalent to a conventional geometric conjunction filter. It ranks uncertain objects according to PDF similarity and uses spatial indexing to reduce comparisons.