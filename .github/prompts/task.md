
# Past task 1

We have the code in this directory which is using KD Tree algorithm from Mercurio and Singla for prefiltering.

We have compared the performance of various divergence measures. We are facing an issue with hellinger norm which is giving drastically bad recall rates as compared to the other norms and as reported by the authors.

We have implemented the KD-tree algo. as close as possible to the implementation details given in the paper.

Tasks:

1. Confirm that the hellinger distance calculation is implemented currectly.
2. Confirm that the KDtree algo is in line with the paper.

Take a look at the results in k-D-tree-prefiltering/runs, particularly, benchmark_k10_N200.csv and the corresponding log file.

the paper is also in this directory.

** Response **


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


# Past task 2

you can read your response in a separate file in the folder. We have following questions.

1. We wish to know more about the saturation. So it can happen if the covariances are small or the distances are large. for this purpose we need to know realistic values (ranges) for the covariance matrices. Also we need to download a snapshot from the celestrak and plot the histogram of the distances, and lowest values to compare them with the realistic values of covariances matrices. This will tell us if hellinger is going to saturate.

Can you down the celestrak data and create the histogram. Whatever can be done in half an hour. If more time is required, then we can do the run the scripts generated by you.

2. In any case, I expect that in any realistic dataset of LEO satellites, a large proportion of pairs will be far away from each other. So hellinger distance will be saturated for most of them. Will this create trouble and should I move to some other norm. I need feedback on this.

3. the kdtree just finds the nearest neighbours but does not use the velocity information. It doesn't have to. It is calculating the collision probabilities at that instant. But we have to propagate the satellites to next snapshot time and again create the kdtree. Can you think of a way of using the velocity vectors of the objects to cheaply create the tree at the next snapshot? Or any other way to use the velocity information for filtering.

Please do an indepth analysis and store your comments and recommendations in a markdown file.

**Response**


# Hellinger saturation on real LEO data, choice of norm, and use of velocity

Date: 2026-11-13. Follows `findings.md`. Scripts, data and figures are in `celestrak_analysis/`.

## Summary

1. **Hellinger will saturate on any realistic LEO snapshot.** In a CelesTrak snapshot of 18,885 LEO objects the median nearest-neighbour distance is 132 km, while realistic position sigmas are 0.01–1 km (up to about 10 km in-track for poorly tracked debris days ahead). With TLE-level covariances, 7 of 178 million pairs have a Hellinger distance distinguishable from 1.
2. **The paper's covariances are 100–1000 times larger than real ones.** Its sigmas of 10–100 km are the only regime in which a Hellinger k-nearest-neighbour search is meaningful. Put on the real LEO positions, they give 99.8% of objects ten or more resolved neighbours.
3. **Saturation is harmless physically but breaks the k-NN formulation.** A saturated pair has zero collision probability at that instant. The trouble is asking for k neighbours when fewer than k exist. I recommend ranking by the Bhattacharyya distance (the logarithm inside Hellinger, same ordering, no saturation) and replacing "k nearest" by "all neighbours below a threshold".
4. **Rebuilding the tree is not worth optimising; missing conjunctions between snapshots is the real problem.** Propagating all objects and rebuilding the tree takes about 12 ms. But testing distance only at snapshots 10 s apart finds 11% of the pairs that come within 5 km. Using velocity for a closed-form closest approach inside each interval finds 100%, with snapshots 30–60 s apart.

## 1. Saturation: realistic covariances against real distances

### 1.1 When Hellinger saturates

For two Gaussians (with the corrected 1/8 factor from `findings.md`):

- Bhattacharyya distance: `D_B = (1/8) dᵀ Σ_pq⁻¹ d + ½ ln( |Σ_pq| / sqrt(|Σ_p||Σ_q|) )`, with `Σ_pq = (Σ_p + Σ_q)/2`.
- Hellinger distance: `D_H = sqrt(1 − exp(−D_B))`.

For equal isotropic covariances this is `D_B = d² / (8σ²)`, so saturation depends only on the separation measured in sigmas:

| Level | D_B | Separation |
|---|---|---|
| D_H > 0.999 (no useful contrast) | 6.2 | 7.1 σ |
| D_H > 1 − 1e-12 (ties, as in `findings.md`) | 26.9 | 14.7 σ |
| `1 − BC == 1` exactly in float64 | 36.7 | 17.1 σ |

So you are right that it is the ratio that matters: Hellinger carries information only for pairs closer than roughly 15 sigma.

### 1.2 Realistic covariance values

| Source of the state | Radial σ | In-track σ | Cross-track σ | Reference |
|---|---|---|---|---|
| Precise orbit predictions (ENVISAT, ERS-2) | 1–2 m | 9–33 m | 5–10 m | Flohrer et al. 2009, Table 2 |
| TLE, average over 8,454 LEO objects | 102 m | 471 m | 126 m | Flohrer et al. 2009, Table 1 |
| TLE, individual LEO objects | 75–440 m | 0.2–2.9 km | 90–470 m | Flohrer et al. 2009, Table 3 |
| CDM secondary objects, 7 days to TCA down to TCA | up to ~200 m | up to ~25 km | up to ~160 m | Hasan 2018 (NASA EOS, 150,000 CDMs) |
| Mercurio and Singla synthetic | 10–100 km | 10–100 km | 10–100 km | paper Sec. 3 |

- The CDM row is read from the axis ranges of the plots in the NASA slides, not from tabulated values. The slides show the in-track value falling steeply towards TCA and with the number of tracks per day.
- Real covariances are strongly anisotropic: the in-track sigma is 5–100 times the other two, and the long axis follows the velocity vector.
- I did not find a public table of operator (GPS-based) ephemeris covariances. Tens of metres is the usual order of magnitude, consistent with the first row.

Sources: [Flohrer, Krag, Klinkrad et al., 5th European Conference on Space Debris, 2009](https://conference.sdo.esoc.esa.int/proceedings/sdc5/paper/111/SDC5-paper111.pdf); [Hasan, "Analyzing the Relationship between Tracking and Covariance for Satellite Collision Avoidance", NASA, 2018](https://ntrs.nasa.gov/api/citations/20180007758/downloads/20180007758.pdf).

### 1.3 The CelesTrak snapshot

- **Download:** CelesTrak GP data in CSV for the groups `active`, `analyst` and the four debris clouds (Fengyun-1C, Cosmos-2251, Iridium-33, Cosmos-1408): 20,151 unique objects.
- **LEO only:** objects with apogee altitude of 2000 km or more were removed (881, which covers all MEO, GEO and HEO objects). A further 367 with element sets older than 5 days and 18 docked duplicates (ISS and CSS modules and visiting vehicles) were removed.
- **Result:** 18,885 LEO objects, propagated with SGP4 to 2026-11-13 12:00 UTC. Altitudes: 5th, 50th and 95th percentiles are 364, 481 and 1175 km.
- **Coverage caveat:** this is not the full tracked population. Most rocket bodies and debris are only on Space-Track (login needed). Nearest-neighbour distance scales roughly as N^(−1/3), so doubling the catalogue would shorten the distances below by about 20%. That does not change any conclusion.

Distances between nominal positions:

| Quantity | Min | 1% | 5% | Median | 95% | Max |
|---|---|---|---|---|---|---|
| Nearest neighbour (km) | 0.66 | 21.9 | 43.7 | 131.7 | 337 | 893 |
| 10th nearest neighbour (km) | 139 | 206 | 243 | 366 | 662 | 1285 |

| Pairs closer than | 1 km | 5 km | 10 km | 50 km | 100 km | 500 km | 1000 km |
|---|---|---|---|---|---|---|---|
| Count (of 178,312,170) | 1 | 6 | 16 | 726 | 4,035 | 191,206 | 893,472 |

The six closest pairs are at 0.66, 1.23, 1.49, 2.49, 3.55 and 4.63 km.

![All-pairs distance histogram](celestrak_analysis/figures/pair_distance_histogram.png)

![Nearest-neighbour histograms](celestrak_analysis/figures/nearest_neighbour_histogram.png)

The dashed lines mark 14.7σ for four values of σ. Pairs to the right of a line are tied at D_H = 1 for that σ.

### 1.4 Will Hellinger saturate? Yes

Fraction of the snapshot that Hellinger can rank (D_H < 1 − 1e-12), if every object had the same isotropic σ:

| σ (km) | Tie radius (km) | Pairs resolved | Objects with a resolved nearest neighbour | Objects with 10 resolved neighbours |
|---|---|---|---|---|
| 0.01 | 0.15 | 0 | 0 | 0 |
| 0.1 | 1.5 | 1.1e-8 | 0.02% | 0 |
| 1 | 14.7 | 2.1e-7 | 0.4% | 0 |
| 5 | 73 | 1.0e-5 | 17% | 0 |
| 10 | 147 | 5.9e-5 | 58% | 0.01% |
| 30 | 440 | 8.0e-4 | 98% | 77% |
| 100 | 1468 | 1.1e-2 | 100% | 100% |

![Resolved fraction against sigma](celestrak_analysis/figures/hellinger_resolved_vs_sigma.png)

With anisotropic covariances aligned to each object's radial, in-track and cross-track axes (built from the velocity vector), single Gaussian per object:

| Covariance scenario | Pairs with D_H < 1 − 1e-12 | Pairs with D_H < 0.999 | Objects with ≥ 1 resolved neighbour | Objects with ≥ 10 | Smallest D_H |
|---|---|---|---|---|---|
| Operator or well-tracked CDM: 20 m, 200 m, 20 m | 0 | 0 | 0 | 0 | 1.000 |
| TLE at epoch: 0.10, 0.47, 0.13 km | 7 | 3 | 0.07% | 0 | 0.962 |
| Poorly tracked debris, days ahead: 0.2, 10, 0.15 km | 367 | 38 | 3.7% | 0 | 0.336 |
| Paper synthetic: diagonal variances 100–10,000 km² | 891,000 | 169,000 | 100% | 99.8% | 0.086 |

- **Realistic covariances:** essentially every pair is saturated. A k=10 Hellinger search returns ten arbitrary tie-broken objects for every query.
- **Paper covariances:** the search is meaningful, which is why the paper reports sensible numbers. The paper's result does not transfer to real data.
- **Your N=200 benchmark:** it combines the paper's large covariances with a catalogue about 100 times sparser than real LEO, so it lands in the saturated regime for a different reason (the one in `findings.md`).

## 2. Should you move to another norm?

### 2.1 What saturation does and does not break

- **It does not break the physics.** A pair with D_H = 1 to twelve digits has no overlap, so no collision probability at that instant. Declaring those pairs "not neighbours" is the correct answer.
- **It breaks k-NN.** k-NN must return k objects and must order them. With realistic data, almost no object has even one resolved neighbour, so the output is decided by tie-breaking and the recall figure is meaningless.
- **It breaks pruning efficiency.** As noted in `findings.md`, the best-k set fills with arbitrary far objects, the pruning radius stays large, and Hellinger costs about four times the evaluations of the other measures.

### 2.2 Recommendation: keep the Hellinger family, but in log form

Use the Bhattacharyya distance `D_B` as the ranking quantity and report `D_H = sqrt(1 − exp(−D_B))` only for display.

- **Same ordering.** D_H is a monotone function of D_B, so every pair that Hellinger can rank is ranked identically.
- **Never saturates.** D_B grows as d²/8σ², so far pairs stay ordered.
- **Directly tied to collision probability.** The probability density of the two objects being at the same place is `N(d; 0, Σ_p + Σ_q)`, whose exponent is `−¼ dᵀ Σ_pq⁻¹ d`, exactly twice the exponent of the Bhattacharyya coefficient. Of the measures in the code, D_B is the one closest to what a conjunction screen wants.
- **Small code change.** `hellinger_gaussian` already computes `log_bc`; return `−log_bc` (with 0.125) and drop the clip at −700.
- **Caveat for mixtures.** The component-coupling LP with D_B costs is not a monotone transform of the LP with D_H costs, so mixture-level values change. Neither is the exact mixture Hellinger distance, so nothing rigorous is lost.

**Checked on your benchmark.** I re-ran your N=200, k=10, seed 42 Hellinger benchmark with only the component cost swapped to D_B, in memory, without editing the repository (`celestrak_analysis/scripts/bhattacharyya_benchmark.py`):

| Component cost | Recall@10 | Distance evaluations | Nodes pruned |
|---|---|---|---|
| Hellinger (your run) | 0.753 | 30,482 | 2,572 |
| Bhattacharyya | 0.999 | 8,470 | 4,296 |
| Mahalanobis (your run) | 0.999 | 8,470 | 4,296 |

The recall problem disappears and the cost drops to that of Mahalanobis. The counts are identical to the Mahalanobis run, which shows the two measures order this catalogue almost the same way.

### 2.3 Recommendation: threshold query instead of k nearest

Ask for "all objects with D_B below τ" (for example τ = 27, which is D_H < 1 − 1e-12, or a τ derived from a collision-probability floor). The answer may be empty, which is correct.

This also makes the KD-tree pruning exact instead of heuristic, because D_B has a Euclidean lower bound:

- `D_B ≥ d² / (8 λ_max(Σ_pq))`, since the log-determinant term is non-negative.
- So every neighbour with `D_B < τ` lies within `R = sqrt(8 τ λ_max)`, where `λ_max` can be bounded by the mean of the two objects' largest eigenvalues, or by the catalogue maximum.
- A ball query of radius R on the KD-tree therefore has 100% recall by construction. For mixtures, add the largest component-mean offset of each object to R.

I used this bound to select candidate pairs in `snapshot_histograms.py`. Example: τ = 27 with TLE covariances gives R = 6.9 km, which is 10 candidate pairs in the whole LEO snapshot instead of 178 million.

### 2.4 The other measures

| Measure | Saturates | Behaviour on realistic data | Verdict |
|---|---|---|---|
| Hellinger | Yes | Tied at 1 for essentially all pairs | Reporting only |
| Bhattacharyya | No | Same ordering as Hellinger, equals log collision density up to a factor | Recommended |
| Mahalanobis (pooled) | No | Equals 8·D_B without the covariance-mismatch term | Fine; nearly equivalent |
| Symmetric KL | No | Similar, but weights each object's own covariance, so a tight object far from a loose one scores very large | Usable, less physical |
| Wasserstein-2 | No | When σ ≪ d it reduces to the Euclidean distance between means, so it ignores covariance | Not informative here |

The Wasserstein row explains why it shows recall 1.000 and the lowest cost in your benchmark: it is close to a plain Euclidean nearest-neighbour search, which is exactly what the KD-tree prunes on.

### 2.5 Smaller points

- Make `recall_at_k` tie-aware regardless of the measure chosen (`findings.md`, recommendation 2).
- A fair comparison with the paper needs the paper's regime: use the real LEO positions with the paper's covariances, or increase the synthetic density.

## 3. Using velocity

### 3.1 Cheaper tree updates are not needed

Measured on the 18,885-object snapshot (one core):

| Step | Time |
|---|---|
| Propagate all objects with SGP4 (vectorised `SatrecArray`) | 8 ms |
| Build `scipy.spatial.cKDTree` | 4 ms |
| 10 nearest neighbours for every object | 45 ms |

- **Rebuild cost is negligible** next to the distance evaluations (your benchmark spends minutes in the LP for 200 objects). An incremental or kinetic tree would save milliseconds.
- **A stale tree goes wrong quickly.** Average overlap between an object's 10 nearest neighbours now and later:

  | Δt | 1 s | 5 s | 10 s | 30 s | 60 s | 300 s |
  |---|---|---|---|---|---|---|
  | Shared neighbours out of 10 | 9.8 | 9.2 | 8.5 | 6.0 | 3.8 | 1.7 |

- **If tree build time hurts in your code,** it is because the tree is pure Python. Use `cKDTree` for the spatial part and keep your code for the probabilistic ranking.

### 3.2 The real problem: conjunctions happen between snapshots

Ground truth for a 10-minute window, from a 0.5 s sweep with closest-approach refinement: **367 pairs come within 5 km**. Their median relative speed is 10.4 km/s, and 357 of them exceed 1 km/s.

Recall of "distance below 5 km at a snapshot" against that truth:

| Snapshot spacing | 0.5 s | 1 s | 2 s | 5 s | 10 s | 30 s | 60 s |
|---|---|---|---|---|---|---|---|
| Recall | 83% | 55% | 33% | 18% | 11% | 5% | 3% |

A pair closing at 10 km/s spends about one second inside a 5 km sphere. Any snapshot-only scheme, with any norm, misses most conjunctions unless snapshots are sub-second.

### 3.3 Recommended scheme: one tree per interval plus closed-form closest approach

For each interval of length Δt, with the snapshot at its midpoint:

1. Propagate all objects to the midpoint and build the tree.
2. Query pairs within `R = D + v_rel,max · Δt/2`, with `v_rel,max = 2 · max speed` (15.7 km/s here). No pair outside R can come within D during the interval.
3. Drop pairs that cannot close the gap with their own relative speed: `|Δr| − |Δv| · Δt/2 > D`.
4. For the rest, compute the closest approach assuming straight relative motion: `τ* = −(Δr·Δv)/|Δv|²` clipped to the interval, miss distance `|Δr + Δv τ*|`. Optionally add the two-body relative acceleration term (quadratic).
5. Keep pairs with miss distance below D, and evaluate D_B or the collision probability at `τ*`.

Results on the same 10-minute window (D = 5 km):

| Δt | Query radius | Candidate pairs (all intervals) | After step 3 | Recall | Max miss-distance error, linear | Quadratic | Python time for the window |
|---|---|---|---|---|---|---|---|
| 2 s | 21 km | 23,372 | 8,509 | 100% | 1 m | 1 m | 6.0 s |
| 10 s | 83 km | 156,342 | 40,459 | 100% | 1 m | 1 m | 1.3 s |
| 30 s | 240 km | 693,935 | 185,429 | 100% | 4 m | 3 m | 0.6 s |
| 60 s | 475 km | 1,687,468 | 491,068 | 100% | 14 m | 5 m | 0.6 s |
| 120 s | 944 km | 3,927,523 | 1,278,286 | 100% | 66 m | 6 m | 1.0 s |

- **All 367 conjunctions are found at every Δt,** with no false positives except one borderline pair at 120 s (linear).
- **Δt of 30–60 s is the sweet spot:** the whole 10-minute all-on-all screen takes 0.6 s in NumPy, against 178 million pair checks per snapshot for brute force.
- **The straight-line model is accurate** because its error is mostly along the relative velocity, which shifts the time of closest approach but hardly changes the miss distance.
- **About 1 m of the error is the resolution of the ground truth itself.** Truth and test use the same SGP4 propagator, so this validates the screening logic, not the orbit accuracy.

This is the idea behind the classical "smart sieve" conjunction filters (Alarcón-Rodríguez, Martínez-Fadrique and Klinkrad) and the Hoots, Crawford and Roehrich filter chain; I am citing those from memory, so check the references before quoting them.

### 3.4 Other uses of velocity

- **Covariance orientation.** The in-track axis is the velocity direction, and realistic covariances are needles along it. Building `Σ = Q diag(σ_R², σ_I², σ_C²) Qᵀ` from position and velocity is what makes a synthetic catalogue realistic. In the debris scenario of Section 1.4 the isotropic bound gave 10,500 candidates of which 367 were resolved, so an orientation-aware bound would cut candidates about 30 times.
- **Encounter geometry for collision probability.** The standard short-encounter collision probability is a 2D integral in the plane perpendicular to the relative velocity. The `τ*` and `Δv` from step 4 are exactly the inputs it needs.
- **Apogee and perigee filter.** Pairs whose altitude bands do not overlap can never meet. On this catalogue it keeps 16% of pairs with a 5 km pad and 36% with a 35 km pad, and rejected none of the 367 true conjunctions. It is a one-off sort, but in today's crowded shells it only removes a factor of 3–6, far less than the tree.
- **A 6D tree on position and velocity is not useful.** Colliding objects are close in position and far apart in velocity, so velocity proximity is not a screening criterion.

## 4. Recommendations

1. Rank by Bhattacharyya distance; keep Hellinger for reporting, with the 1/8 factor.
2. Replace k-NN by a threshold query with the rigorous radius `sqrt(8 τ λ_max)`, or at least make recall tie-aware.
3. Generate test catalogues from real LEO positions with velocity-aligned covariances at realistic sizes (Section 1.2), and separately with the paper's sizes for comparison with the paper.
4. Do not spend effort on incremental tree updates. Rebuild at each interval midpoint and use the velocity-based closest approach of Section 3.3 with Δt of 30–60 s.
5. Evaluate the probabilistic measure at the time of closest approach of each surviving pair, not at the snapshot.

## 5. Limitations

- One epoch, one propagator (SGP4 on public GP elements), and the public CelesTrak subset of the catalogue.
- Saturation statistics use one Gaussian per object with the same covariance for all objects in a scenario; real covariances vary by object. The 6-component mixtures and the LP were not used on the real snapshot.
- Covariances were held fixed over the 10-minute window; over days they grow, mainly in-track.
- The velocity experiment used a 5 km Euclidean threshold. With large in-track sigmas the threshold from Section 2.3 becomes tens of kilometres and the candidate counts rise accordingly.

## 6. Reproducing

```bash
python -m venv venv && venv/bin/pip install numpy scipy matplotlib sgp4
cd celestrak_analysis
# optional refresh of the data (one request per group):
#   curl -o data/active.csv "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv"
../venv/bin/python scripts/snapshot_histograms.py data figures     # ~25 s
../venv/bin/python scripts/velocity_screening.py data figures      # ~35 s
../venv/bin/python scripts/bhattacharyya_benchmark.py ../k-D-tree-prefiltering   # several minutes
```

- All numbers in this report are in `celestrak_analysis/figures/summary.json` and `velocity_summary.json`.
- If the data are refreshed, pass `--epoch` close to the download time; element sets older than 5 days are dropped.
- No file inside `k-D-tree-prefiltering/` was changed.

# Current task











