# Hellinger recall and KD-tree conformance: findings

Date: 2026-11-13

Both checks are done: the code matches the paper as printed, but the Hellinger recall of 0.753 is an artefact of saturated distances and tie-breaking, not a KD-tree or distance bug.

## 1. Hellinger distance

- **Code vs paper:** `hellinger_gaussian` (`k-D-tree-prefiltering/rso_kdtree/distances/gaussian_distances.py:51`) implements the paper's Eq. 15 exactly, including the `-1/2` factor in the exponent.
- **Eq. 15 itself is wrong:** the correct factor is `-1/8`, which is what the paper's own Eq. 12 and 14 imply. Checked against numerical integration on three random Gaussian pairs:

  | Numerical H | Code (`-1/2`) | With `-1/8` |
  |---|---|---|
  | 0.2845 | 0.3770 | 0.2845 |
  | 0.6618 | 0.9362 | 0.6618 |
  | 0.6003 | 0.9102 | 0.6003 |

  The fix is `log_bc -= 0.125 * mahal` at line 77. What the authors' C++ code actually used is unknown.
- **This is not the cause of the bad recall:** with `-1/8` the k=10 recall is still 0.754.

## 2. Why the recall is bad

The catalogue is so sparse relative to the covariances that Hellinger is numerically 1 for almost every pair. The N=200, k=10 run was reproduced exactly (recall 0.753, 30,482 evaluations) and measured:

- **Saturation:** 99.97% of the 19,900 pairs have D ≥ 1 − 1e-12; only one pair is below 0.999, and the minimum is 0.943.
- **Ties decide the ranking:** `NeighborResult.__lt__` breaks ties by `object_id`, and the LP adds noise of about 1e-16. This is why the log shows brute-force neighbours `RSO_0001 … RSO_0010` for every query.
- **Recall measures the wrong thing:** brute force returns the lowest IDs in the whole catalogue, while the KD-tree returns the lowest IDs among the boxes it opened. The 0.753 is the overlap of those two arbitrary sets.
- **Tie-aware recall is 1.000:** counting a KD-tree neighbour as correct when its distance is ≤ the brute-force k-th distance gives 1.000 for both k=1 and k=10. On the 12 queries whose nearest neighbour is actually resolved, ID-based k=1 recall is also 1.000.
- **Cost side effect:** the best-K set holds arbitrary far-away objects, so the pruning radius is huge. That is why Hellinger needs about 30k evaluations against about 7k for Wasserstein.

KL, Mahalanobis and Wasserstein are unbounded, so they never saturate and do not show this.

Measured numbers for N=200, seed 42:

| Exponent factor | Pairs with D ≥ 1 − 1e-12 | Pairs with D < 0.999 | Min D | Queries with saturated nearest neighbour | k=1 ID recall | k=10 ID recall | Tie-aware recall (k=1, k=10) |
|---|---|---|---|---|---|---|---|
| `1/2` (as coded) | 99.97% | 1 | 0.9433 | 188 / 200 | 0.195 | 0.753 | 1.000, 1.000 |
| `1/8` (correct) | 99.87% | 6 | 0.9071 | 154 / 200 | 0.400 | 0.754 | 1.000, 1.000 |

The paper's catalogue must be much denser than this one: its Table 2 lists ten pairs with D_H between 0.19 and 0.92 for 200 RSOs. A uniform shell from 6500 to 8000 km with σ ≤ 100 km gives a typical nearest-neighbour spacing of roughly 1,700 km, far too wide to overlap.

## 3. KD-tree vs paper

The implementation is in line with Algorithms 1 and 3; no deviation that affects correctness was found.

- **Construction:** root box from min/max, split on the longest dimension at the median, leaves hold the points. `leaf_size=1` is within the paper's "one or two".
- **Search:** locate the leaf, walk up opening sibling boxes until k candidates exist, then open remaining boxes only if `D_box < DE_min`, ranking by D_H.
- **One interpretation:** the paper does not define `DE_min` for k > 1. The code uses the largest Euclidean distance among the current best-K by D_H, which is a reasonable reading.
- **One data deviation:** GMM components use covariance `Σ/3` (`rso_kdtree/data/synthetic_catalogue.py:147`, following the thesis), whereas the paper's Sec. 3 says `Σ`. The smaller covariance makes saturation slightly worse.

## Recommendations

1. Change the exponent factor to `1/8`.
2. Make `recall_at_k` tie-aware, or report recall only for queries whose k-th distance is below 1.
3. To reproduce the paper's regime, increase the overlap: a thinner shell, larger N, or larger variances, plus component covariance `Σ` if following the 2016 paper.

No project files were changed during this analysis.
