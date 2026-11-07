# Plan: Replicate Mercurio & Singla (2016) kd-tree Results (k=10, N=200)

## Goal
Close the gap vs `runs/benchmark_k10_N200.csv`: Hellinger Recall@K = 0.725 with
28,960 distance evals (> brute force's N(N-1)/2 = 19,900) and KD-tree slower than
brute force (122.7s vs 78.8s). Paper target: ~98.6%+ capture, ~3 orders of magnitude
fewer evaluations than brute force at N=10,000.

## Source grounding (workspace rules: code must trace to Sources/)
- `Sources/Mercurio_Singla_KD_Tree_based approach for efficient and accurate conjunction.md` (CMES 2016, primary):
  - Alg. 3 step 6: "Store the **minimum** distance values: DHmin and DEmin."
  - Alg. 3 step 8: "if Dbox < DEmin then open the box ... Update DEmin and nearest neighbors."
  - Fig. 2 / Alg. 1: root box from dataset bounds, median split on longest axis, **leaves store the objects**.
  - Sec. 3 (numerical validation): "diagonal covariance matrix, with variances between **100 and 10,000 km**"; RSO sets of 200/2,000/5,000/10,000.
- `Sources/A_noncombinatorial_approach_for_efficient_conjunction analysis.md` (thesis 2014) says variances 1,000–10,000; per workspace rules the more recent, more specific CMES 2016 source wins (var_min=100), cited alongside the thesis.
- Hellinger formula: **unchanged** (user confirmed current implementation matches CMES 2016 Eq. 15 as intended).

## Agreed fixes (in order)

### Fix 1 — DE_min = minimum Euclidean among current best-k (do first)
- File: `rso_kdtree/screening/knn.py`, `_de_min()`.
- Current: `de_sorted[k_available - 1]` (k-th smallest = max Euclidean of best-k). Paper Alg. 3 step 6 stores the **minimum**.
- Change: `de_sorted[0]`; update docstrings (module header + `_de_min` + Phase-1 comment) to cite Alg. 3 step 6/8.
- Keep the existing cache invalidation (`de_min_valid`) — "Update DEmin" happens whenever `best` changes.
- Risk noted: min-Euclidean threshold prunes more aggressively; `test_kdtree_knn_exact_recall_identical_covariances` asserts recall == 1.0 and may need its expectation revised to the paper's approximate behavior if it fails. Will check after running.

### Fix 2 — exact query-leaf location via leaf map (Fig. 2: leaves store objects)
- `rso_kdtree/spatial/kd_tree.py`: add `build_leaf_map(root) -> dict[str, KDNode]` (one O(N) traversal mapping object_id -> leaf). Export from `spatial/__init__.py`.
- `rso_kdtree/screening/knn.py`: `probabilistic_knn(..., leaf_map=None)`; if given, `leaf = leaf_map[query_rso.object_id]` (Alg. 3 step 2 "Locate the item of interest in the kd-tree"), else fall back to `_descend_to_leaf` (needed because builder splits by sorted index, value-descent can land on the wrong leaf when duplicate split values exist).
- `rso_kdtree/benchmark/benchmark.py` and `rso_kdtree/main.py`: build the map once after `build_kdtree`, pass to every `probabilistic_knn` call.
- New test in `tests/test_kdtree.py` / `test_knn.py`: with duplicate positions, `build_leaf_map` maps each RSO to a leaf containing it; `probabilistic_knn` with the map works.

### Fix 3 — var_min default 1000 -> 100 (CMES 2016 Sec. 3), var_max stays 10,000
- `rso_kdtree/data/synthetic_catalogue.py`: `var_min: float = 100.0` + docstring (cite CMES 2016 §3, note thesis says 1,000 per conflict rule).
- `rso_kdtree/main.py`: `--var-min` default 100.0, help text "default: 100, per Mercurio & Singla 2016".
- `rso_kdtree/benchmark/benchmark.py`: `BenchmarkConfig.var_min = 100.0`.
- `tests/test_knn.py`: `_build_catalogue` default `var_min=100.0` to match.

## Verification
1. `python -m pytest tests/ -v` — all tests pass (or the one exact-recall test is consciously revised if the Alg. 3 min-DE_min semantics change its expectation).
2. Re-run benchmark k=10, N=200, all 4 metrics, seed 42; compare Recall@K / distance evals / times vs `runs/benchmark_k10_N200.csv`.
3. Target: Hellinger distance evals << 19,900 and KD-tree time < brute-force time; Recall@K toward ~0.986.

## Out of scope
- Hellinger formula (user: do not change).
- Issue-2 bootstrap ordering (closest-first sibling opening) — deferred until Fix 1's effect is observed ("then we will see").
