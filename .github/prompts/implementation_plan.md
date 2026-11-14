# Implementation Plan: Align `rso_kdtree` with Mercurio–Singla Sources

## Context

This plan addresses 7 required changes to bring the `rso_kdtree` implementation into fidelity with the Mercurio–Singla references (Mercurio 2014 thesis + Mercurio & Singla 2016 CMES paper) and PLAN.md §§9–10.

The benchmark experiment compares **Hellinger distance** (source-prescribed metric) against **Wasserstein-2**, **Mahalanobis**, and **symmetric KL divergence**. All modular distance formulae must remain intact.

---

## File Map

| File | Role | Changes Required |
|---|---|---|
| `rso_kdtree/spatial/__init__.py` | KD-tree build entry point | Replace scipy KDTree with custom tree + `KDNode` class |
| `rso_kdtree/spatial/kd_node.py` | NEW: KD-tree node data model | Node with AABB, split axis/value, parent, children, leaf RSOs |
| `rso_kdtree/spatial/kd_tree.py` | NEW: Custom KD-tree construction | Median split on longest spread, leaf-size configurable |
| `rso_kdtree/spatial/geometry.py` | NEW: AABB geometry helpers | Point-to-AABB Euclidean distance |
| `rso_kdtree/screening/knn.py` | Probabilistic KNN search | Exact Alg. 3 traversal (descend → backtrack → prune by `D_AABB < DE_min`) |
| `rso_kdtree/data/synthetic_catalogue.py` | Catalogue generation | Mercurio GMM: component means along `±√(NΣ)`, covariance = Σ/3 (thesis) |
| `rso_kdtree/main.py` | CLI | Defaults: `n_rso=100`, `components=6`, `leaf_size=1` |
| `rso_kdtree/screening/results.py` | Result containers | No change (already supports instrumentation fields) |
| `rso_kdtree/benchmark/benchmark.py` | Benchmark orchestrator | No structural change; will receive real instrumentation stats |
| `rso_kdtree/distances/` | Distance module | No change — keep all four metrics modular |


---

## Task 1: Implement Custom KD-tree (PLAN §9–10)

### Current state
`spatial/__init__.py` wraps `scipy.spatial.KDTree`. This prevents:
- Control over split axis (scipy uses variance-based, not longest-spread)
- Access to node-level AABB bounds for pruning decisions
- Node-level instrumentation (visited/pruned/leaves)

### Target state
Create `spatial/kd_node.py` and `spatial/kd_tree.py` implementing a custom top-down KD-tree:

**`KDNode`** (`kd_node.py`):
- `aabb_lower`: np.ndarray (3,) — lower bound of axis-aligned bounding box
- `aabb_upper`: np.ndarray (3,) — upper bound of AABB
- `split_axis`: int — dimension used for splitting (internal nodes only)
- `split_value`: float — split coordinate (internal nodes only)
- `parent`: KDNode | None
- `left`: KDNode | None
- `right`: KDNode | None
- `rsos`: list[RSO] — RSOs stored at this node (leaf nodes only, up to `leaf_size`)
- `is_leaf`: bool

**`build_kdtree(positions, rsos, leaf_size=1)`** (`kd_tree.py`):
1. Compute root AABB from min/max of all positions
2. Recursively split:
   - If `len(rsos) <= leaf_size`: make leaf, store RSOs
   - Else: compute spread per dimension, split along dimension with largest spread at median
   - Partition RSOs by median value along split axis
   - Recurse on left and right subsets
3. Return root `KDNode`

Update `spatial/__init__.py` to re-export `KDNode`, `build_kdtree` from new modules.

### Acceptance criteria
- Tree constructed without scipy dependency in `spatial/`
- Split axis = dimension of largest spread
- Split value = median nominal position along that axis
- Leaf size configurable (default 1 per item 4)
- Every RSO appears in exactly one leaf

---

## Task 2: Implement Exact Alg. 3 Tree Traversal (PLAN §10.1)

### Current state
`screening/knn.py` uses a two-phase ball-query approximation:
- Phase 1: `kd_tree.query()` for k+1 Euclidean neighbors, doubling
- Phase 2: `kd_tree.query_ball_point()` at radius `D_E,K`

This does not replicate Mercurio–Singla Alg. 3 tree traversal.

### Target state
Rewrite `probabilistic_knn()` to implement exact Alg. 3:

```
1. Determine required number of nearest neighbors (k)
2. for each query RSO p:
   a. Locate p in the kd-tree (descend to leaf containing nominal position)
   b. Traverse up the tree, opening boxes until K nearest neighbors met
   c. Compute Hellinger distance (DH) and Euclidean distance (DE) to nearest neighbors found
   d. Store minimum values: DH_min and DE_min
   e. Compute Euclidean distance from p to all remaining nodes (boxes): D_box
   f. if D_box < DE_min: open the box, compute distance to enclosed items, update DE_min and neighbors
   g. else: move to next possible box
3. Return K candidates sorted by probabilistic distance
```

### Detailed algorithm
1. **Descend**: From root, follow child whose AABB contains query position until reaching a leaf
2. **Evaluate leaf**: Compute probabilistic distance (DH) and Euclidean distance (DE) for all RSOs in leaf (excluding query itself). Maintain best-K list sorted by DH. Track `DE_min` = K-th smallest DE among evaluated candidates.
3. **Backtrack**: Move to parent. For each sibling subtree:
   - Compute `D_box` = Euclidean distance from query position to sibling's AABB
   - If `D_box < DE_min`: recursively search that sibling subtree
   - Else: prune (do not visit)
4. Continue up to root, repeating step 3 for each ancestor's other child
5. Return best-K list sorted by probabilistic distance

### Key distinction from ball-query approach
- Ball query: spherical radius search in one shot
- Alg. 3: hierarchical tree traversal with per-node AABB distance checks, enabling node-level instrumentation

### Acceptance criteria
- Search uses `KDNode` tree structure (not scipy KDTree)
- `D_box < DE_min` pruning rule applied per node
- Node-level traversal (descend to leaf, backtrack up, open siblings)
- Returns same KNNResult interface (sorted by probabilistic distance)

---

## Task 3: Fix GMM Generation to Match Mercurio

### Current state
`data/synthetic_catalogue.py` generates:
- Component means: `nominal_position + rng.normal(0, 10, size=3)` (isotropic small perturbation)
- Component covariances: independent random diagonal matrices in [var_min, var_max]

### Target state
Match Mercurio thesis §4 / 2016 paper §3:

**Component means** (thesis Eq. 4.1–4.2 / 2016 paper Eq. 29–30):
```
χ_i   = µ + (√NΣ)_i     for i = 1, 2, 3   (plus direction)
χ_{i+N} = µ − (√NΣ)_i   for i = 1, 2, 3   (minus direction)
```
where `N` = dimension (3), `(√NΣ)_i` = i-th column of matrix square root of `N·Σ` (i.e., `sqrt(3·Σ)`).
This yields `2·N = 6` components for 3D, matching Mercurio's 6 components.

**Component covariance**: Use `Σ/3` (per thesis §4: "each Gaussian component is assumed to possess a covariance matrix equal to 1/3 Σ"). Document this choice (2016 paper uses Σ; thesis uses Σ/3; we follow the thesis).

**Weights**: Equal weights `1/6` each (already correct).

### Implementation
For each RSO:
1. Generate diagonal covariance `Σ` with variances in [var_min, var_max]
2. Compute matrix square root of `3·Σ` via eigendecomposition: `sqrtm(3·Σ)`
3. Column `i` of `sqrtm(3·Σ)` gives offset vector `δ_i`
4. Create 6 components: means `µ ± δ_i` for i=1,2,3, each with covariance `Σ/3`
5. Equal weights `1/6`

### Acceptance criteria
- `n_components` must equal 6; any other value raises `ValueError` (Option A: strict Mercurio-faithful)
- Component means offset along covariance principal axes: `µ ± (√(3Σ))_i` for i=1,2,3
- Component covariance = `Σ/3` (per thesis §4)
- Equal weights `1/6` each


---

## Task 4: Align CLI Defaults with Mercurio Prescription

### Current state
`main.py` defaults: `n_rso=100`, `k=5`, `components=6`, `leaf_size=1`

### Target state
Defaults already match Mercurio prescription — no changes needed:

| Argument | Current Default | Mercurio Spec |
|---|---|---|
| `--n-rso` | 100 | 100 (keep) |
| `--k` | 5 | 5 (keep) |
| `--components` | 6 | 6 (strict, per Mercurio) |
| `--leaf-size` | 1 | 1 (Mercurio uses 1–2 points per leaf) |
| `--seed` | 42 | 42 (keep) |
| `--r-min` | 6500.0 | 6500.0 (keep) |
| `--r-max` | 8000.0 | 8000.0 (keep) |
| `--var-min` | 1e-4 | 1e-4 (sigma = 0.01 km) |
| `--var-max` | 100.0 | 100.0 (sigma = 10 km) |

Update `benchmark/BenchmarkConfig` defaults to match (currently `leaf_size=10`, `n_components=6`).

### Acceptance criteria
- CLI defaults: `n_rso=100`, `components=6`, `leaf_size=1`
- Benchmark config defaults aligned

---

## Task 5: Fix Fragile `object_id` Parsing

### Current state
`screening/knn.py` line 103:
```python
query_rso_index = int(query_rso.object_id.split("_")[-1])
```
This breaks if `object_id` format changes (e.g., not `"RSO_XXXX"`).

### Target state
Build a position-to-index map in `run_single()` / `run_benchmark()` before the query loop:
```python
position_to_index = {id(r.nominal_position.tobytes()): i for i, r in enumerate(catalogue)}
```
Or simpler: pass a `dict[str, int]` mapping `object_id -> index` to `probabilistic_knn()`, and use it to look up the query index instead of parsing the string.

### Acceptance criteria
- No string parsing of `object_id` to derive array index
- Query index resolved via explicit mapping
- Works with any `object_id` format

---

## Task 6: Add Node-level Instrumentation

### Current state
`KNNSearchStats.nodes_visited`, `nodes_pruned`, `leaves_visited` are all hardcoded to 0 because scipy KDTree doesn't expose node traversal.

### Target state
With the custom KD-tree (Task 1) and exact traversal (Task 2), track during search:
- `nodes_visited`: number of internal nodes entered during backtracking
- `leaves_visited`: number of leaf nodes whose RSOs were evaluated
- `nodes_pruned`: number of sibling subtrees skipped because `D_box >= DE_min`

Update `probabilistic_knn()` to populate these in `KNNSearchStats`.

### Acceptance criteria
- `nodes_visited > 0` for non-trivial searches
- `nodes_pruned >= 0` reflects pruned subtrees
- `leaves_visited` counts leaves examined
- Stats aggregated correctly in benchmark

---

## Task 7: Keep Modular Distance Formulae Intact

### Current state
`rso_kdtree/distances/` already implements:
- `base.py`: `ProbabilisticDistance` ABC
- `gaussian_distances.py`: Hellinger, Wasserstein, Mahalanobis, symmetric KL at Gaussian level
- `gmm_distances.py`: Component-coupling LP via `scipy.optimize.linprog` (HiGHS)
- `factory.py`: Runtime metric selection

### Target state
No changes to distance module. All four metrics remain available:
- `hellinger` (source-prescribed, proper metric in [0,1])
- `wasserstein` (Gaussian 2-Wasserstein)
- `mahalanobis` (pooled covariance, documented limitation)
- `kl` (symmetric KL divergence, documented as divergence not metric)

The benchmark experiment will compare all four on the same catalogue.

### Acceptance criteria
- `create_distance_metric("hellinger")` through `"kl"` all work
- GMM LP coupling unchanged
- Gaussian-level closed forms unchanged

---

## Implementation Order

1. **Create `spatial/kd_node.py`** — KDNode data model
2. **Create `spatial/kd_tree.py`** — Custom tree construction (median split, longest spread)
3. **Create `spatial/geometry.py`** — AABB point-to-box distance
4. **Update `spatial/__init__.py`** — Re-export from new modules, remove scipy dependency
5. **Update `data/synthetic_catalogue.py`** — Mercurio GMM generation (means along `±√(NΣ)`, cov = Σ/3)
6. **Update `screening/knn.py`** — Exact Alg. 3 traversal + instrumentation + fix object_id parsing
7. **Update `main.py`** — CLI defaults (already mostly correct; verify `leaf_size=1`)
8. **Update `benchmark/benchmark.py`** — Align `BenchmarkConfig` defaults
9. **Update `screening/results.py`** — No change needed (already has fields)
10. **Test** — Run single query + benchmark, verify instrumentation > 0

---

## Testing Checklist

- [ ] Custom KD-tree builds without scipy
- [ ] Every RSO appears in exactly one leaf
- [ ] Split axis = dimension of largest spread
- [ ] KNN results sorted by probabilistic distance
- [ ] Query RSO excluded from its own neighbor list
- [ ] `nodes_visited`, `nodes_pruned`, `leaves_visited` all > 0 for typical search
- [ ] GMM has 6 components with means at `µ ± (√(3Σ))_i` and cov = Σ/3
- [ ] All 4 distance metrics produce results
- [ ] Recall@K computed vs brute force
- [ ] CLI defaults: `n_rso=100`, `components=6`, `leaf_size=1`
