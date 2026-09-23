# GRID complete available-results report

## Corrected common-reference update (2026-09-23)

This section supersedes the older structural and RandomGIN tables below. All methods now use the exact same serialized 20-graph reference, equal graph counts, the same structural implementation, and 10 RandomGIN evaluator seeds. DeFoG uses independent seeds 0, 4, and 5.

| Method | Degree MMD | Clustering MMD | Orbit MMD | Spectral MMD | Diameter MMD | Triangle MMD | Sparsity MMD | Mean-edge error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Motif=False | 0.135944 | 0.120726 | 0.761679 | 0.021480 | 0.207147 | 7.944e-6 | **2.315e-10** | 226.483 |
| Motif=True full | **0.003347** | **0.002963** | **0.007801** | **0.010271** | **0.127829** | **1.680e-8** | 6.893e-10 | 65.433 |
| DeFoG | 0.006341 +/- 0.008510 | 0.009601 +/- 0.012100 | 0.018098 +/- 0.018100 | 0.016965 +/- 0.003040 | 0.583752 +/- 0.304000 | 6.999e-8 +/- 8.790e-8 | 3.638e-10 +/- 7.580e-11 | **63.817 +/- 11.000** |

| Method | F1-PR | Precision | Recall | MMD-RBF |
|---|---:|---:|---:|---:|
| Motif=False | 0.392779 | 0.341667 | 0.775000 | 0.806865 |
| Motif=True full | 0.913467 | 0.846667 | 0.996667 | **0.160813** |
| DeFoG | **0.956353 +/- 0.042200** | **0.928333 +/- 0.063700** | **1.000000** | 0.218746 +/- 0.083200 |

Compared with the earlier two-seed DeFoG table, DeFoG F1-PR rises from `0.900152` to `0.956353`. Motif=True still wins degree, clustering, orbit, spectral, diameter, triangle, and RandomGIN MMD-RBF; DeFoG now wins F1-PR, precision, recall, and mean-edge error.

Updated 2026-09-17 after evaluating replacement DeFoG seed 4.

## Primary reporting policy and current completeness

- GraphVAE motif=False, motif=True total, and motif=True full each use seeds
  0, 1, and 2.
- DeFoG paper-facing results use healthy seeds 0 and replacement seed 4
  (`n=2`). Anomalous seeds 1, 2, and 3 are excluded and their values are not
  included in this report's aggregates.
- Seed 4 completed successfully and is independently evaluated on the same
  20-graph held-out protocol as seed 0.
- GraphVAE and DeFoG use the same GRID definition and 70/10/20 split policy,
  but their archived held-out graph identities differ. This is a dataset-level
  comparison, not an identical-reference paired evaluation.
- GRID has no native node or edge features. “Structural-feature Random-GIN”
  uses degree, clustering, and square-clustering derived from adjacency;
  constant-input Random-GIN is topology-only.
- The LinkCorrelation rule is
  `edges(nodes0,nodes1) AND edges(nodes1,nodes2)`. Its retained training
  support is the four binary states `FF`, `TT`, `FT`, and `TF`; the TV metric
  sums counts across the entire collection, normalizes once on this same state
  list for every method, and computes `0.5 * sum |p_gen-p_ref|`.

The paper-facing sections below use only declared healthy DeFoG seeds 0 and 4.

## Phase 1 — Structural metrics

| Metric | Better | DeFoG filtered | GraphVAE false | False advantage | GraphVAE total | Total advantage | GraphVAE full | Full advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Degree MMD | ↓ | 0.009410 ± 0.012100 (n=2) | 0.135944 ± 0.066671 | -1344.66% | 0.008652 ± 0.008443 | +8.06% | 0.003347 ± 0.001894 | +64.43% |
| Clustering MMD | ↓ | 0.013811 ± 0.013675 (n=2) | 0.120726 ± 0.027632 | -774.12% | 0.004751 ± 0.005211 | +65.60% | 0.002963 ± 0.002331 | +78.55% |
| Orbit MMD | ↓ | 0.019388 ± 0.020219 (n=2) | 0.761679 ± 0.165471 | -3828.51% | 0.008304 ± 0.006896 | +57.17% | 0.007801 ± 0.011440 | +59.76% |
| Spectral MMD | ↓ | 0.022768 ± 0.002686 (n=2) | 0.021480 ± 0.004223 | +5.66% | 0.013937 ± 0.001741 | +38.79% | 0.010271 ± 0.001345 | +54.89% |
| Diameter MMD | ↓ | 0.456019 ± 0.361624 (n=2) | 0.207147 ± 0.070912 | +54.57% | 0.089761 ± 0.063279 | +80.32% | 0.127829 ± 0.026073 | +71.97% |
| Triangle MMD | ↓ | 1.029e-7 ± 9.461e-8 (n=2) | 7.944124e-06 ± 2.232988e-06 | -7620.41% | 0.001667 ± 0.002887 | -1619955.28% | 1.679718e-08 ± 1.631362e-08 | +83.68% |
| Sparsity MMD | ↓ | 8.840e-11 ± 3.681e-12 (n=2) | 2.315187e-10 ± 3.281282e-10 | -161.91% | 4.679553e-08 ± 8.069439e-08 | -52837.68% | 6.892942e-10 ± 2.192168e-10 | -679.77% |
| Edge-count absolute error | ↓ | 12.900 ± 9.617 (n=2) | 226.483333 ± 33.954614 | -1655.68% | 1121.500000 ± 1872.564010 | -8593.80% | 65.433333 ± 10.272820 | -407.24% |

### Retained DeFoG structural seeds

| Seed | Role | Degree MMD ↓ | Clustering MMD ↓ | Orbit MMD ↓ | Edge-count error ↓ |
|---:|---|---:|---:|---:|---:|
| 0 | Original healthy seed | 0.017966 | 0.023481 | 0.033686 | 19.700 |
| 4 | Healthy replacement | 0.000854 | 0.004142 | 0.005091 | 6.100 |

## Phase 2 — Random-GIN without native node features

GRID has no native node attributes. The common evaluator below uses only
degree, clustering, and square-clustering calculated from adjacency.

### GRID comparison after DeFoG outlier exclusion

DeFoG seeds retained: `0, 4`; anomalous seeds removed: `1, 2, 3`.
GraphVAE false/total/full remain unchanged at three seeds.


### Primary topology Random-GIN

| Metric | Better | DeFoG filtered | GraphVAE false | False advantage | GraphVAE total | Total advantage | GraphVAE full | Full advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1-PR | ↑ | 0.900152 ± 0.105925 (n=2) | 0.392779 ± 0.142617 | -56.37% | 0.842757 ± 0.053776 | -6.38% | 0.913467 ± 0.030861 | +1.48% |
| Precision | ↑ | 0.867500 ± 0.123744 (n=2) | 0.341667 ± 0.089768 | -60.61% | 0.736667 ± 0.075719 | -15.08% | 0.846667 ± 0.049329 | -2.40% |
| Recall | ↑ | 0.965000 ± 0.049497 (n=2) | 0.775000 ± 0.368273 | -19.69% | 1.000000 ± 0.000000 | +3.63% | 0.996667 ± 0.002887 | +3.28% |
| MMD-RBF | ↓ | 0.250932 ± 0.081010 (n=2) | 0.806865 ± 0.153998 | -221.55% | 0.094722 ± 0.082571 | +62.25% | 0.160813 ± 0.017471 | +35.91% |
| MMD-linear mean | ↓ | 7.243556 ± 8.944336 (n=2) | 7074.556824 ± 4922.887057 | -97566.90% | 2.185083e+12 ± 3.784675e+12 | -30165886727167.26% | 41.943895 ± 10.558208 | -479.05% |
| MMD-linear median | ↓ | 1.379962 ± 1.627697 (n=2) | 1323.220754 ± 693.530730 | -95788.17% | 1.499455e+11 ± 2.597132e+11 | -10865911855130.22% | 32.903332 ± 8.703001 | -2284.36% |
| MMD-linear 10%-trimmed | ↓ | 4.620334 ± 6.133772 (n=2) | 2016.097254 ± 1311.313696 | -43535.31% | 4.706730e+11 ± 8.152295e+11 | -10186990935057.90% | 33.644433 ± 8.178085 | -628.18% |

### Retained DeFoG topology Random-GIN seeds

| Seed | Role | F1-PR ↑ | Precision ↑ | Recall ↑ | MMD-RBF ↓ |
|---:|---|---:|---:|---:|---:|
| 0 | Original healthy seed | 0.825252 | 0.780000 | 0.930000 | 0.308215 |
| 4 | Healthy replacement | 0.975052 | 0.955000 | 1.000000 | 0.193650 |


## Phase 3 — Motif-correlation metrics on the identical pruned rule set

| DeFoG filtered | GraphVAE false (historical 70-graph) | False comparison | GraphVAE total | Total advantage | GraphVAE full | Full advantage |
| --- | --- | --- | --- | --- | --- | --- |
| 0.004296 ± 0.000289 (n=2) | 0.012910 ± 0.003124 | N/C: different protocol | 0.050785 ± 0.0788217 (n=3) | -1082.02% | 0.00676398 ± 0.00200851 (n=3) | -57.43% |

### Retained DeFoG motif-TV seeds

| Seed | Role | Pruned LinkCorrelation TV ↓ |
|---:|---|---:|
| 0 | Original healthy seed | 0.004501 |
| 4 | Healthy replacement | 0.004092 |

## Result addresses

- DeFoG seed 0: `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/metrics/grid_seed0.json`
- DeFoG replacement seed 4: `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/metrics/grid_seed4.json`
- Replacement seed-4 model: `/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/frozen_eval/jobs/grid/seed_4/training/checkpoints/frozen_grid_seed_4`
