# MUTAG complete verified available-results report

Updated 2026-09-17 with the new four-mode attributed RandomGIN evaluation.
The established structural comparison below uses the matched `alpha_motif=0.1`
campaign. The new RandomGIN appendix at the end uses the later full-matrix
`alpha_motif=0.2` campaign and is kept separate so the two hyperparameter
campaigns are not silently mixed.

Verified 2026-09-14. This report's internally matched three-way campaign uses
the edge-aware GraphVAE configuration described below (motif=True full matrix,
`alpha_motif=0.1`) and the common 39-graph topology evaluation. Additional
motif-weight sweep directories exist, including `0.01`, `0.05`, `0.15`, `0.2`,
`0.25`, and `0.3`; they are separate tuning campaigns and are not silently
pooled into this three-seed aggregate.

## Comparability and motif-correlation audit

- Motif=True and motif=False GraphVAE are matched on split, architecture,
  20,000 epochs, node/edge categorical targets, and three seeds; motif loss is
  the intended difference.
- DeFoG is evaluated against the same 39 held-out reference adjacencies for the
  common topology table, but its archived seed-0 and seed-1 adjacency files are
  byte-identical. Its displayed SD is therefore not a valid three-independent-
  seed uncertainty estimate.
- The archived common DeFoG files contain adjacency only. Consequently, an
  exact edge-label-aware motif correlation on the same **pruned training
  states** is not available for all three DeFoG seeds. The existing 5,620-state
  complete/unpruned diagnostic is not substituted as the primary motif
  correlation. That paper-facing cell remains N/A until attributed samples are
  regenerated for three independent DeFoG checkpoints and restricted to the
  retained cache state list.
- Structural and topology Random-GIN metrics remain valid under their stated
  common adjacency protocol.

## Scope and fairness

This report covers the new 20,000-epoch MUTAG campaign using training seeds 0, 1, and 2:

- GraphVAE-REQ motif=True, full-matrix calibrated-Gaussian motif loss (`motif_weight=0.1`)
- GraphVAE motif=False, otherwise matched configuration
- DeFoG with its native node and edge categorical cross-entropy objectives

The GraphVAE variants use seven categorical atom labels and four categorical bond labels, with `node_loss_weight=1` and `edge_loss_weight=1`. Motif=True reads `_CP_smoothed` and trains on the pruned full-matrix rules. All GraphVAE metrics below use 39 held-out reference graphs and 39 generated graphs from the saved best-validation-MMD model.

For DeFoG, a common topology-only Random-GIN evaluation was recomputed against the same 39 held-out reference graphs, using the first 39 of its 40 saved samples. The saved `generated_adjs.npz` contains adjacency only, so node/bond-aware evaluation and edge-labelled motif-rule evaluation are not recoverable from that artifact without regenerating attributed samples from the checkpoint.

Important DeFoG caveat: seed 0 and seed 1 `generated_adjs.npz` files are byte-identical even though their Hydra configurations say `train.seed=0` and `train.seed=1`. Consequently, the displayed DeFoG mean/SD is descriptive of the three files, but is not a valid estimate from three independent generated samples.

## Aggregated results (mean ± sample SD across three training seeds)

Lower is better for MMD/error metrics; higher is better for precision, recall, and F1-PR. “Improvement” is motif=True relative to motif=False, with positive values meaning better after respecting metric direction.

## Phase 1 — Structural metrics

| Metric | Motif=True full | Motif=False | Motif=True improvement |
|---|---:|---:|---:|
| Degree MMD ↓ | 0.005115 ± 0.001710 | **0.004816 ± 0.001039** | -6.2% |
| Clustering MMD ↓ | **0.041306 ± 0.025511** | 0.127815 ± 0.038014 | **+67.7%** |
| Orbit MMD ↓ | **0.000564 ± 0.000616** | 0.002792 ± 0.001088 | **+79.8%** |
| Spectral MMD ↓ | 0.025664 ± 0.006150 | **0.019104 ± 0.004216** | -34.3% |
| Diameter MMD ↓ | 0.172634 ± 0.099278 | **0.090911 ± 0.031976** | -89.9% |
| Sparsity MMD ↓ | 8.877e-8 ± 4.426e-8 | **3.902e-8 ± 1.488e-8** | -127.5% |
| Triangle MMD ↓ | **6.349e-7 ± 2.899e-7** | 2.551e-6 ± 9.532e-7 | **+75.1%** |
| Mean generated edges | 15.675 ± 1.417 | **19.179 ± 0.308** | — |
| Absolute error from reference mean (20.590) ↓ | 4.915 | **1.410** | -248.5% |

## Phase 2 — Random-GIN without native node features

| Metric | Motif=True full | Motif=False | DeFoG | Motif=True vs false | Motif=True vs DeFoG |
|---|---:|---:|---:|---:|---:|
| F1-PR ↑ | 0.873220 ± 0.039450 | 0.754209 ± 0.059084 | **0.914758 ± 0.015008*** | **+15.8%** | -4.5% |
| Precision ↑ | 0.780342 ± 0.059878 | 0.612821 ± 0.074182 | **0.884615 ± 0.035818*** | **+27.3%** | -11.8% |
| Recall ↑ | **0.998291 ± 0.001480** | 0.994872 ± 0.006784 | 0.958120 ± 0.007402* | **+0.3%** | **+4.2%** |
| MMD-RBF ↓ | 0.231320 ± 0.101045 | 0.106591 ± 0.022432 | **0.070435 ± 0.003410*** | -117.0% | -228.4% |
| MMD-linear mean ↓ | 383.376 ± 514.050 | 1390.391 ± 1307.717 | **21.771 ± 16.206*** | **+72.4%** | -1660.9% |

\* DeFoG seed 0 and seed 1 generated adjacency artifacts are identical, so these DeFoG aggregate uncertainty values are not independent-seed estimates.

### Secondary GraphVAE-local topology evaluator

This is reported separately because it is not the same evaluator implementation as the common third-party table.

| Metric | Motif=True full | Motif=False |
|---|---:|---:|
| F1-PR ↑ | **0.925939 ± 0.008439** | 0.810462 ± 0.048936 |
| Precision ↑ | **0.867521 ± 0.010363** | 0.694872 ± 0.069798 |
| Recall ↑ | **0.996581 ± 0.005922** | 0.994872 ± 0.006784 |
| MMD-RBF ↓ | 0.253142 ± 0.122655 | **0.107224 ± 0.025680** |

## Phase 3 — Random-GIN with native node features

A common attributed evaluation is **N/A**: GraphVAE attributed samples exist,
but the archived DeFoG comparison samples contain adjacency matrices only.
Regenerating node-labelled samples from every DeFoG checkpoint is required for
a matched three-way result.

## Phase 4 — Motif-correlation metrics on the identical pruned rule set

**N/A for the matched three-way comparison.** The retained MUTAG rules include
categorical labels, while the archived DeFoG samples are adjacency-only. The
available 5,620-state complete/unpruned diagnostic is deliberately not
substituted for the pruned-state metric.

## Supporting per-seed structural results

### Motif=True full matrix

| Seed | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ | Sparsity ↓ | Triangle ↓ | Mean edges |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.006479 | 0.041995 | 0.000402 | 0.032326 | 0.283485 | 1.323e-7 | 8.093e-7 | 14.769 |
| 1 | 0.005669 | 0.066465 | 0.001245 | 0.024463 | 0.142511 | 9.022e-8 | 7.952e-7 | 14.949 |
| 2 | 0.003197 | 0.015457 | 0.000046 | 0.020203 | 0.091907 | 4.379e-8 | 3.003e-7 | 17.308 |

### Motif=False

| Seed | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ | Sparsity ↓ | Triangle ↓ | Mean edges |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.005617 | 0.086563 | 0.002865 | 0.023361 | 0.106873 | 4.553e-8 | 1.978e-6 | 18.872 |
| 1 | 0.005189 | 0.161432 | 0.003842 | 0.019020 | 0.054096 | 4.953e-8 | 3.651e-6 | 19.179 |
| 2 | 0.003641 | 0.135448 | 0.001670 | 0.014930 | 0.111763 | 2.199e-8 | 2.023e-6 | 19.487 |

## Per-seed common topology Random-GIN

| Method | Seed | F1-PR ↑ | Precision ↑ | Recall ↑ | MMD-RBF ↓ | MMD-linear mean ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Motif=True full | 0 | 0.830260 | 0.715385 | 1.000000 | 0.312479 | 970.460 |
| Motif=True full | 1 | 0.881579 | 0.792308 | 0.997436 | 0.263335 | 165.641 |
| Motif=True full | 2 | 0.907820 | 0.833333 | 0.997436 | 0.118146 | 14.026 |
| Motif=False | 0 | 0.777303 | 0.643590 | 0.987179 | 0.103978 | 2617.132 |
| Motif=False | 1 | 0.687066 | 0.528205 | 1.000000 | 0.130215 | 1539.551 |
| Motif=False | 2 | 0.798259 | 0.666667 | 0.997436 | 0.085580 | 14.489 |
| DeFoG | 0 | 0.923422 | 0.905128 | 0.953846 | 0.068466 | 12.415 |
| DeFoG | 1 | 0.923422 | 0.905128 | 0.953846 | 0.068466 | 12.415 |
| DeFoG | 2 | 0.897429 | 0.843590 | 0.966667 | 0.074372 | 40.483 |

## Interpretation

The new motif=True full-matrix model is not uniformly superior. Relative to matched motif=False GraphVAE, it is clearly better on clustering, orbit, triangle, topology F1-PR, precision, and recall. It is worse on degree, spectral, diameter, sparsity, mean edge-count error, and topology MMD-RBF. The edge-count result is the clearest failure mode: motif=True generates about 15.7 edges per graph versus 20.6 in the held-out set, while motif=False generates about 19.2.

On the common topology-only GIN evaluation, DeFoG has the best F1-PR, precision, and both MMD values; motif=True has the best recall. This conclusion must remain topology-only until attributed DeFoG samples are regenerated. The duplicate DeFoG seed-0/seed-1 output must also be fixed before treating its three-seed SD as publication-ready.

## Result locations

Original campaign roots:

- Seed 0 (`cs-cl-18`): `/local-scratch2/mirzaei/mutag_nodefeat_3way_20260910`
- Seed 1 (`cs-cl-19`): `/local-scratch2/mirzaei/mutag_nodefeat_3way_20260910`
- Seed 2 (`cs-cl-09`): `/local-scratch2/mirzaei/mutag_nodefeat_3way_20260910`

Collected metrics and common DeFoG topology evaluation:

`/local-scratch2/mirzaei/mutag_new_results_20260911`

## Metrics not reported as comparable

- Node-feature Random-GIN and node+edge-feature Random-GIN: GraphVAE attributed graphs are available, but DeFoG saved only adjacency matrices.
- Pruned-rule count distance, Gaussian-aligned count distance, and motif-state correlation: the new database rules include labels; therefore DeFoG adjacency-only files cannot support the same labelled rules.
- Native DeFoG structural metrics are omitted from the main comparison because its log evaluated 40 generated graphs against 18 internal reference graphs, not the common 39-graph held-out reference used above.

## New attributed RandomGIN evaluation: alpha_motif=0.2 campaign

All modes use the same 39 held-out reference graphs and ten evaluator
initializations per generator seed. GraphVAE Motif=True and Motif=False use
training seeds 0, 1, and 2. Only one independently generated attributed DeFoG
seed was available to this evaluator, so DeFoG is shown as `n=1` with no
training-seed SD and must not be described as a three-seed aggregate.

Mode definitions:

- `topology_control`: adjacency only; constant node input and zero edge input.
- `decoded_node`: adjacency plus native categorical atom attributes.
- `decoded_edge`: adjacency plus native categorical bond attributes.
- `decoded_node_edge`: adjacency plus both atom and bond attributes.

### Topology-only RandomGIN

| Metric | Motif=True full, alpha 0.2 (n=3) | Motif=False (n=3) | DeFoG (n=1) |
|---|---:|---:|---:|
| F1-PR ↑ | 0.923762 ± 0.013209 | 0.807315 ± 0.064058 | **0.975791** |
| Precision ↑ | 0.866667 ± 0.020513 | 0.687179 ± 0.088712 | **0.956410** |
| Recall ↑ | **1.000000 ± 0** | 0.995726 ± 0.007402 | 0.997436 |
| MMD-RBF ↓ | 0.233522 ± 0.096721 | 0.123245 ± 0.029924 | **0.103895** |
| MMD-linear ↓ | 34.366672 ± 16.367897 | 14.711385 ± 4.827392 | **5.910006** |

### Node-feature RandomGIN

| Metric | Motif=True full, alpha 0.2 (n=3) | Motif=False (n=3) | DeFoG (n=1) |
|---|---:|---:|---:|
| F1-PR ↑ | 0.786591 ± 0.013197 | 0.754854 ± 0.050931 | **0.926597** |
| Precision ↑ | 0.700000 ± 0.026023 | 0.641880 ± 0.087041 | **0.928205** |
| Recall ↑ | 0.906838 ± 0.007402 | **0.939316 ± 0.029496** | 0.925641 |
| MMD-RBF ↓ | 0.236782 ± 0.055547 | 0.134519 ± 0.007019 | **0.100318** |
| MMD-linear ↓ | 64.687407 ± 15.194239 | 42.755797 ± 22.907350 | **9.987804** |

### Edge-feature RandomGIN

| Metric | Motif=True full, alpha 0.2 (n=3) | Motif=False (n=3) | DeFoG (n=1) |
|---|---:|---:|---:|
| F1-PR ↑ | 0.869954 ± 0.039038 | 0.887305 ± 0.007218 | **0.952916** |
| Precision ↑ | 0.776923 ± 0.058245 | 0.812821 ± 0.011177 | **0.961538** |
| Recall ↑ | **0.995726 ± 0.005338** | 0.982051 ± 0.005128 | 0.946154 |
| MMD-RBF ↓ | 0.184179 ± 0.036661 | 0.079099 ± 0.006708 | **0.066322** |
| MMD-linear ↓ | 36.595493 ± 6.422140 | 16.224447 ± 9.918066 | **5.892152** |

### Node-plus-edge-feature RandomGIN

| Metric | Motif=True full, alpha 0.2 (n=3) | Motif=False (n=3) | DeFoG (n=1) |
|---|---:|---:|---:|
| F1-PR ↑ | 0.709201 ± 0.049093 | 0.707778 ± 0.036857 | **0.927136** |
| Precision ↑ | 0.570940 ± 0.061823 | 0.569231 ± 0.049521 | **0.920513** |
| Recall ↑ | **0.969231 ± 0.002564** | 0.959829 ± 0.016485 | 0.935897 |
| MMD-RBF ↓ | 0.206759 ± 0.027136 | 0.128540 ± 0.009242 | **0.075612** |
| MMD-linear ↓ | 175.978782 ± 112.174321 | 112.244700 ± 44.017696 | **9.180813** |

New evaluation root:
`/local-scratch2/mirzaei/edge_feature_random_gin_20260917/results/mutag`.
