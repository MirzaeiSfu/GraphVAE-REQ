# TRIANGULAR_GRID complete available-results report

## Corrected common-reference update (2026-09-23)

This section supersedes the older structural and RandomGIN tables below. All methods use the exact same serialized 20-graph reference, equal graph counts, one structural implementation, and 10 RandomGIN evaluator seeds. DeFoG uses healthy seeds 0, 1, and 3.

| Method | Degree MMD | Clustering MMD | Orbit MMD | Spectral MMD | Diameter MMD | Triangle MMD | Sparsity MMD | Mean-edge error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Motif=False | 0.008932 | 0.137583 | 0.077736 | 0.018747 | 0.081433 | 6.223e-5 | 1.494e-9 | 72.717 |
| Motif=True full | **0.001669** | **0.081536** | **0.013882** | **0.016453** | **0.031367** | **4.134e-7** | **1.291e-9** | 46.750 |
| DeFoG | 0.013730 +/- 0.009230 | 0.996492 +/- 0.179000 | 0.083866 +/- 0.121000 | 0.030942 +/- 0.003100 | 0.214346 +/- 0.024000 | 4.029e-5 +/- 5.950e-5 | 2.832e-9 +/- 3.310e-9 | **40.767 +/- 15.300** |

| Method | F1-PR | Precision | Recall | MMD-RBF |
|---|---:|---:|---:|---:|
| Motif=False | 0.903451 | 0.828333 | **1.000000** | 0.206531 |
| Motif=True full | **0.927428** | **0.870000** | **1.000000** | **0.181505** |
| DeFoG | 0.794559 +/- 0.070100 | 0.703333 +/- 0.099300 | **1.000000** | 0.240898 +/- 0.039900 |

DeFoG F1-PR increases from the older `0.637496` result to `0.794559`, but Motif=True remains best on F1-PR, precision, MMD-RBF, and every listed structural MMD. Recall is tied at 1.0; DeFoG wins mean-edge error.

## Archive status (2026-09-21)

LGD evaluation has been launched on system 09 in tmux `lgd_eval_triangular_s2`.
It uses the completed seed-2 diffusion checkpoint at epoch 1999 (2,000 epochs).
Outputs will be copied automatically to [LGD results](evaluations/lgd_seed_2/LGD_RESULTS.md), with raw samples, `metrics.json`, and `pipeline.log` alongside it.
This is one generator training seed, evaluated using ten Random-GIN initializations.
The evaluation reports structural metrics and four-state pruned-rule TV against test and training references. The training-reference calculation uses the same generated collection as the test evaluation and is labeled accordingly.

- Archive root: `/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/TRIANGULAR_GRID` on `cs-cl-19`.
- The transfer runs in tmux session `archive_triangular_grid_20260921`; its live log is `manifests/transfer.log`.
- GraphVAE motif=False, motif=True total-count, and motif=True full-matrix each archive seeds 0, 1, and 2, including their saved models and run artifacts.
- DeFoG archives healthy seeds 0, 1, and 3. Seed 2 is deliberately excluded because the documented multi-metric audit classified it as a broad outlier.
- LGD encoder and diffusion training for seed 2 are complete and their checkpoints are being archived under `experiments/lgd/`. No LGD samples or downstream structural/Random-GIN/motif-correlation results were found, so LGD is not included in the numerical comparison below.
- A completed transfer creates `manifests/transfer_finished.txt`, `manifests/file_manifest.tsv`, and `manifests/archive_size.txt`.

### Archive layout

| Content | Relative directory |
| --- | --- |
| GraphVAE motif=False | `experiments/graphvae/motif_false/seed_{0,1,2}` |
| GraphVAE motif=True total count | `experiments/graphvae/motif_true_total_count/seed_{0,1,2}` |
| GraphVAE motif=True full matrix | `experiments/graphvae/motif_true_full_matrix/seed_{0,1,2}` |
| DeFoG healthy seeds | `experiments/defog/seed_{0,1,3}` |
| LGD trained seed 2 | `experiments/lgd/{encoder_seed_2,diffusion_seed_2}` |
| Evaluation evidence | `metrics/` |
| Reports | `reports/` |

Updated 2026-09-17 after evaluating replacement DeFoG seed 3.

## Primary reporting policy and current completeness

- GraphVAE motif=False, motif=True total, and motif=True full each use seeds
  0, 1, and 2.
- DeFoG seed 2 was a broad outlier and is removed from the paper-facing report.
  Completed healthy replacement seed 3 is used instead.
- The final DeFoG aggregate uses seeds 0, 1, and 3 (`n=3`).
- GraphVAE and DeFoG use the same dataset definition and split policy, but the
  archived held-out graph identities differ. Results are dataset-level rather
  than identical-reference paired comparisons.
- This dataset has no native node or edge features. Structural-feature
  Random-GIN derives its inputs from adjacency.
- Motif correlation is TV on the identical retained four-state support of
  `edges(nodes0,nodes1) AND edges(nodes1,nodes2)`: counts are summed over the
  collection, normalized once, then compared by `0.5 * sum |p_gen-p_ref|`.

The paper-facing sections below use healthy DeFoG seeds 0, 1, and 3 only.

## Phase 1 — Structural metrics

| Metric | Better | DeFoG filtered | GraphVAE false | False advantage | GraphVAE total | Total advantage | GraphVAE full | Full advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Degree MMD | ↓ | 0.015198 ± 0.009987 (n=3) | 0.008932 ± 0.008190 | +41.23% | 0.001775 ± 0.001276 | +88.32% | 0.001669 ± 0.001221 | +89.02% |
| Clustering MMD | ↓ | 0.989717 ± 0.177959 (n=3) | 0.137583 ± 0.074300 | +86.10% | 0.133882 ± 0.049311 | +86.47% | 0.081536 ± 0.029108 | +91.76% |
| Orbit MMD | ↓ | 0.065953 ± 0.103605 (n=3) | 0.077736 ± 0.057113 | -17.87% | 0.024459 ± 0.008265 | +62.91% | 0.013882 ± 0.007069 | +78.95% |
| Spectral MMD | ↓ | 0.020778 ± 0.002146 (n=3) | 0.018747 ± 0.003187 | +9.77% | 0.023009 ± 0.001093 | -10.74% | 0.016453 ± 0.000848 | +20.82% |
| Diameter MMD | ↓ | 0.184917 ± 0.035703 (n=3) | 0.081433 ± 0.027672 | +55.96% | 0.163010 ± 0.055931 | +11.85% | 0.031367 ± 0.023510 | +83.04% |
| Triangle MMD | ↓ | 3.229e-05 ± 5.009e-05 (n=3) | 6.223435e-05 ± 6.006294e-05 | -92.76% | 5.111818e-06 ± 2.600772e-06 | +84.17% | 4.133542e-07 ± 3.546051e-07 | +98.72% |
| Sparsity MMD | ↓ | 3.752e-09 ± 5.800e-09 (n=3) | 1.493624e-09 ± 7.156814e-10 | +60.19% | 4.504044e-09 ± 7.004047e-10 | -20.06% | 1.291103e-09 ± 6.267800e-10 | +65.58% |
| Edge-count absolute error | ↓ | 26.067 ± 35.737 (n=3) | 72.716667 ± 8.589868 | -178.96% | 74.750000 ± 13.159027 | -186.76% | 46.750000 ± 6.957550 | -79.35% |

### Retained DeFoG structural seeds

| Seed | Role | Degree MMD ↓ | Clustering MMD ↓ | Orbit MMD ↓ | Edge-count error ↓ |
|---:|---|---:|---:|---:|---:|
| 0 | Original healthy seed | 0.020211 | 0.937091 | 0.185585 | 6.850 |
| 1 | Original healthy seed | 0.003697 | 0.844005 | 0.006144 | 67.300 |
| 3 | Healthy replacement | 0.021685 | 1.188054 | 0.006129 | 4.050 |

## Phase 2 — Random-GIN without native node features

TRIANGULAR_GRID has no native node attributes. The common evaluator below uses
only degree, clustering, and square-clustering calculated from adjacency.

### TRIANGULAR_GRID comparison after DeFoG outlier exclusion

DeFoG seeds retained: `0, 1, 3`; outlier seed 2 removed. GraphVAE
false/total/full remain unchanged at three seeds.


### Primary topology Random-GIN

| Metric | Better | DeFoG filtered | GraphVAE false | False advantage | GraphVAE total | Total advantage | GraphVAE full | Full advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1-PR | ↑ | 0.637496 ± 0.085773 (n=3) | 0.903451 ± 0.023010 | +41.72% | 0.928475 ± 0.044078 | +45.64% | 0.927428 ± 0.039461 | +45.48% |
| Precision | ↑ | 0.551667 ± 0.098277 (n=3) | 0.828333 ± 0.036171 | +50.15% | 0.875000 ± 0.070887 | +58.61% | 0.870000 ± 0.067639 | +57.70% |
| Recall | ↑ | 0.993333 ± 0.007638 (n=3) | 1.000000 ± 0.000000 | +0.67% | 1.000000 ± 0.000000 | +0.67% | 1.000000 ± 0.000000 | +0.67% |
| MMD-RBF | ↓ | 0.273197 ± 0.023457 (n=3) | 0.206531 ± 0.066011 | +24.40% | 0.261688 ± 0.022518 | +4.21% | 0.181505 ± 0.018954 | +33.56% |
| MMD-linear mean | ↓ | 34.537096 ± 11.425772 (n=3) | 137.946756 ± 40.902745 | -299.42% | 112.274997 ± 63.962418 | -225.09% | 31.034648 ± 11.575211 | +10.14% |
| MMD-linear median | ↓ | 8.970074 ± 5.598702 (n=3) | 81.556480 ± 26.040791 | -809.21% | 61.876726 ± 17.836195 | -589.81% | 25.732992 ± 7.588006 | -186.88% |
| MMD-linear 10%-trimmed | ↓ | 16.693869 ± 7.831946 (n=3) | 96.052583 ± 29.855321 | -475.38% | 60.767090 ± 13.812732 | -264.01% | 28.232151 ± 10.971760 | -69.12% |

### Retained DeFoG topology Random-GIN seeds

| Seed | Role | F1-PR ↑ | Precision ↑ | Recall ↑ | MMD-RBF ↓ |
|---:|---|---:|---:|---:|---:|
| 0 | Original healthy seed | 0.600166 | 0.500000 | 0.985000 | 0.300282 |
| 1 | Original healthy seed | 0.576713 | 0.490000 | 1.000000 | 0.259558 |
| 3 | Healthy replacement | 0.735608 | 0.665000 | 0.995000 | 0.259751 |


## Phase 3 — Motif-correlation metrics on the identical pruned rule set

| DeFoG filtered | GraphVAE false (historical 70-graph) | False comparison | GraphVAE total | Total advantage | GraphVAE full | Full advantage |
| --- | --- | --- | --- | --- | --- | --- |
| 0.012997 ± 0.006056 (n=3) | 0.018245 ± 0.003490 | N/C: different protocol | 0.0148587 ± 0.00109577 (n=3) | -14.32% | 0.011586 ± 0.00101271 (n=3) | +10.86% |

### Retained DeFoG motif-TV seeds

| Seed | Role | Pruned LinkCorrelation TV ↓ |
|---:|---|---:|
| 0 | Original healthy seed | 0.012480 |
| 1 | Original healthy seed | 0.019295 |
| 3 | Healthy replacement | 0.007217 |

## Result addresses

- Original DeFoG metrics: `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/metrics/triangular_grid_seed{0,1}.json`
- Replacement DeFoG seed 3: `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/metrics/triangular_grid_seed3.json`
- Replacement generated graphs: `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/artifacts/triangular_grid/generated/seed_3/generated_graphs.pt`
