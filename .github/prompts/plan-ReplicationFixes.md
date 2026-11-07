# Plan: Fixes to Replicate Mercurio's kd-tree Results

## Problem
The current kd-tree implementation does not fully replicate Mercurio & Singla (2016) results (reported 98.6%+ nearest-neighbor capture, ~3 orders of magnitude speedup over brute force). Several discrepancies between the implementation and the source documentation affect accuracy and performance.

## Source Documentation Reference
- Primary: `Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md` (Alg. 1, Alg. 3, Eq. 15, GMM LP framework)
- Theoretical: `Sources/A_noncombinatorial_approach_for_efficient_conjunction analysis.md`

## Identified Issues

### 1. Hellinger distance normalization mismatch (affects absolute values, not ranking)
- **Paper Eq. 15**: DH = (1/√2) × √(1 - BC)

- **Implementation** (`gaussian_distances.py:hellinger_gaussian`): H = √(1 - BC), missing the 1/√2 factor
- **Impact**: Constant √2 scaling on all Hellinger distances; ranking unchanged but absolute values don't match the paper

### 2. KD-tree median split creates unbalanced trees with duplicate values (affects performance)
- **Implementation** (`kd_tree.py`): Left mask = `positions <= median`, right = `~left_mask`. All values equal to median go left, creating highly unbalanced trees when duplicates exist.
- **Impact**: Tree depth degrades from O(log N) toward O(N), reducing pruning effectiveness and slowing search

### 3. Phase 2 pruning re-tests already-visited boxes (affects performance)
- **Implementation** (`knn.py:_prune_remaining`): Starts from root and recomputes D_box for all boxes, including those fully evaluated during bootstrap
- **Impact**: Redundant distance-to-box computations; slower than necessary

### 4. DE_min recomputed by sorting on every call (affects performance)
- **Implementation** (`knn.py:_de_min`): Sorts all best candidates' Euclidean distances on every invocation
- **Impact**: O(k log k) per box test; unnecessary overhead in the pruning loop

### 5. Fundamental limitation: Euclidean AABB distance is not a lower bound on probabilistic distance (affects recall)
- **Issue**: D_box (Euclidean) < DE_min (Euclidean threshold) is used to prune boxes, but ranking is by Hellinger/probabilistic distance. Boxes far in Euclidean space can contain RSOs with small probabilistic distance (large overlapping uncertainties).
- **Impact**: Recall@K < 100%; this is the core reason Mercurio reports ~98.6% rather than 100%
- **Mitigation**: Acknowledge as approximate; tune parameters to match paper's reported recall

## Proposed Fixes

| # | Fix | File | Expected Impact |
|---|-----|------|-----------------|
| 1 | Add 1/√2 normalization to Hellinger distance to match Eq. 15 | `distances/gaussian_distances.py` | Absolute Hellinger values match paper; ranking unchanged |
| 2 | Fix median split to balance left/right even with duplicate median values | `spatial/kd_tree.py` | Better tree balance, O(log N) depth, faster search |
| 3 | Skip fully-evaluated subtrees in Phase 2 pruning | `screening/knn.py` | Fewer redundant D_box computations |
| 4 | Maintain DE_min incrementally (max-heap) instead of re-sorting | `screening/knn.py` | Faster DE_min lookups in pruning loop |
| 5 | Document the Euclidean pruning limitation and parameter tuning guidance | `README.md`, code comments | Sets correct expectations for recall |

## Files to Change
1. `rso_kdtree/distances/gaussian_distances.py` — Hellinger normalization
2. `rso_kdtree/spatial/kd_tree.py` — Median split balancing
3. `rso_kdtree/screening/knn.py` — Phase 2 optimization, DE_min caching
4. `README.md` — Document limitations and tuning

## Verification
- Run `pytest tests/` to ensure no regressions
- Run benchmark to compare Recall@K and distance evaluations before/after
- Verify Hellinger distance values match Eq. 15 for known test cases
- Verify tree balance (depth ~ log2(N)) for datasets with duplicate values

## Notes
- Fix #5 (fundamental limitation) cannot be "fixed" without abandoning the kd-tree approach; it's inherent to using Euclidean pruning for probabilistic distances. The paper itself acknowledges this (~98.6% recall).
- All fixes are surgical and preserve existing behavior where correct.
