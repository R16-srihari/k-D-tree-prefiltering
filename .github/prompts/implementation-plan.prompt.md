# Implementation Plan: Fix Code Flaws Preventing Replication of Mercurio Paper Results

## Overview
This plan addresses the critical flaws identified in the codebase that prevent replication of the Mercurio & Singla (2016) paper results.

## Critical Flaw (BLOCKING)

### Flaw 1: Hellinger Distance Exponent is Wrong
**File:** `rso_kdtree/distances/gaussian_distances.py`, function `hellinger_gaussian()`

**Current code (line ~83):**
```python
log_bc -= 0.5 * mahal   # computes exp(-1/2 * mahal)
```

**Required per Mercurio & Singla (2016) Eq. 15:**
```python
log_bc -= 0.125 * mahal  # computes exp(-1/8 * mahal)
```

**Fix:** Change `0.5 * mahal` to `0.125 * mahal` in `hellinger_gaussian()`.

---

## Additional Issues (Non-Blocking)

### Issue 2: Missing Source Documentation References in Code Headers
Add header comments to all code files identifying the source documentation file and section per K-DTree-agent instructions.

Files needing headers:
- `spatial/kd_tree.py` → Mercurio_Singla §2.1, Alg. 1, Fig. 2
- `spatial/kd_node.py` → Mercurio_Singla §2.1
- `spatial/geometry.py` → Mercurio_Singla §2.5 (distance-to-box)
- `distances/gaussian_distances.py` → Mercurio_Singla Eq. 15 (Hellinger), Distance_formulae_algo.md (other metrics)
- `distances/gmm_distances.py` → Mercurio_Singla §2.4 (LP framework), Distance_formulae_algo.md
- `distances/base.py`, `distances/factory.py` → Mercurio_Singla §2.3-2.4
- `models/gaussian_component.py`, `gaussian_mixture.py`, `rso.py` → Mercurio_Singla §2.4, thesis §3.1.2
- `screening/results.py` → Mercurio_Singla §3 (results format)
- `main.py`, `benchmark/benchmark.py`, `benchmark/metrics.py` → Mercurio_Singla §3 (benchmarking)

### Issue 3: No Probability of Collision Implementation (Optional)
Implement per Mercurio & Singla 2016 §2.3 and thesis §3.2:
- Two i.i.d. Gaussians: P_coll formula
- Two GMMs: sum over component pairs

### Issue 4: Minor — Wasted Median Computation in KD-tree Build
`kd_tree.py` computes `np.median(split_values)` then overwrites it. Remove the wasted computation.

---

## Verification Steps
1. Fix the Hellinger exponent in `gaussian_distances.py`
2. Add source documentation header comments to all code files
3. Optionally implement probability of collision computation
4. Run existing test suite: `python -m pytest tests/ -v`
5. Validate Hellinger distance: identical Gaussians = 0, well-separated → 1

---

## Acceptance Criteria
- [ ] Hellinger distance uses exponent -1/8 (not -1/2)
- [ ] All code files have source documentation header comments
- [ ] Test suite passes
- [ ] Benchmark results approach paper's reported 98.6-99.5% nearest-neighbor capture
