# Triangular-grid results

Generated from the completed intrinsic-feature triangular-grid v3 experiments in `grid_tri_v3_20260715`.

- Dataset: `TRIANGULAR_GRID`
- Database: `tri_grid_v3_intrinsic_undir`
- BFS strategy: `legacy_first_component`
- Epochs: `20000`
- Training batch size: `200`
- Best-checkpoint metric: `table3_priority`
- Final evaluation uses the saved `best_validation_mmd_model` on the test split.

## Settings

| Setting | Meaning | Motif loss | Retained motif combinations | alpha motif | Motif mode |
|---|---|---:|---:|---:|---|
| 01 | GraphVAE baseline | False | none | 0.00 | `calibrated_gaussian` |
| 03 | GraphVAE + motif, no temperature annealing | True | globally top 70 | 0.01 | `calibrated_gaussian` |

Both settings use the same intrinsic features, data split, model losses, and evaluation procedure. Setting 03 adds the formula-pruned motif loss.

## Aggregate final test metrics

Values are mean +/- sample standard deviation across the completed result files available on the cluster. Higher F1/precision/recall is better; lower MMD is better.

| Setting | Completed seeds | Test F1-PR | Test MMD RBF | Precision | Recall | Degree MMD | Clustering MMD | Orbit MMD | Spectral MMD | Diameter MMD | 3rd-party F1-PR | 3rd-party MMD RBF | 3rd-party linear MMD trimmed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 3 | 0.6470 +/- 0.0159 | 0.2732 +/- 0.0101 | 0.5033 +/- 0.0153 | 0.9850 +/- 0.0150 | 0.0634 +/- 0.0205 | 0.4263 +/- 0.0621 | 0.3365 +/- 0.1268 | 0.0220 +/- 0.0013 | 0.1374 +/- 0.0601 | 0.5914 +/- 0.0657 | 0.2794 +/- 0.0411 | 311.3372 +/- 170.4033 |
| 03 | 3 | 0.5840 +/- 0.0452 | 0.3393 +/- 0.0526 | 0.4767 +/- 0.0431 | 0.9300 +/- 0.0614 | 0.0480 +/- 0.0222 | 0.6048 +/- 0.1245 | 0.2912 +/- 0.2203 | 0.0234 +/- 0.0032 | 0.0728 +/- 0.0365 | 0.5488 +/- 0.0959 | 0.3063 +/- 0.0965 | 227.3798 +/- 130.7254 |

## Per-seed final test metrics

| Setting | Seed | Host | Test F1-PR | Test MMD RBF | Precision | Recall | Degree | Clustering | Orbit | Spectral | Diameter | 3rd-party F1-PR | 3rd-party MMD RBF | 3rd-party linear MMD trimmed |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0 | `cs-cl-16` | 0.6584 | 0.2846 | 0.5200 | 0.9850 | 0.0838 | 0.4889 | 0.4607 | 0.0214 | 0.1021 | 0.5156 | 0.3264 | 186.8794 |
| 01 | 1 | `cs-cl-26` | 0.6289 | 0.2654 | 0.4900 | 0.9700 | 0.0428 | 0.4251 | 0.2072 | 0.0234 | 0.2067 | 0.6295 | 0.2503 | 241.5813 |
| 01 | 2 | `cs-cl-36` | 0.6538 | 0.2697 | 0.5000 | 1.0000 | 0.0635 | 0.3648 | 0.3415 | 0.0211 | 0.1033 | 0.6291 | 0.2614 | 505.5509 |
| 03 | 0 | `cs-cl-17` | 0.6361 | 0.2827 | 0.5150 | 0.8850 | 0.0337 | 0.5568 | 0.1470 | 0.0219 | 0.0477 | 0.6496 | 0.2431 | 145.7453 |
| 03 | 1 | `cs-cl-17` | 0.5603 | 0.3866 | 0.4850 | 0.9050 | 0.0735 | 0.7462 | 0.5448 | 0.0271 | 0.1146 | 0.4587 | 0.4174 | 158.2381 |
| 03 | 2 | `cs-cl-26` | 0.5556 | 0.3487 | 0.4300 | 1.0000 | 0.0367 | 0.5115 | 0.1818 | 0.0211 | 0.0559 | 0.5380 | 0.2585 | 378.1561 |

## Interpretation and completeness

- All three seeds of both settings completed 20,000 epochs, returned exit code `0`, and produced final Table 2, Table 3, and combined metric summaries.
- Setting 03 does not improve the primary aggregate metrics in this experiment: its mean local and third-party F1-PR values are lower, while its mean local and third-party RBF MMD values are higher than Setting 01.
- Setting 03 improves mean degree and diameter MMD and slightly improves orbit MMD, but clustering and spectral MMD are worse.

## Source files

- Setting 01 seed 0: `cs-cl-16:/local-scratch/mirzaei/grid_tri_v3_20260715/GraphVAE-REQ/experiments/grid_tri_v3_20260715/results/triangular_grid/setting_01/seed_0/final_metrics_summary.json`
- Setting 01 seed 1: `cs-cl-26:/local-scratch/mirzaei/grid_tri_v3_20260715/GraphVAE-REQ/experiments/grid_tri_v3_20260715/results/triangular_grid/setting_01/seed_1/final_metrics_summary.json`
- Setting 01 seed 2: `cs-cl-36:/var/tmp/mirzaei/grid_tri_v3_20260715/GraphVAE-REQ/experiments/grid_tri_v3_20260715/results/triangular_grid/setting_01/seed_2/final_metrics_summary.json`
- Setting 03 seeds 0 and 1: `cs-cl-17:/local-scratch/mirzaei/grid_tri_v3_20260715/GraphVAE-REQ/experiments/grid_tri_v3_20260715/results/triangular_grid/setting_03/seed_*/final_metrics_summary.json`
- Setting 03 seed 2: `cs-cl-26:/local-scratch/mirzaei/grid_tri_v3_20260715/GraphVAE-REQ/experiments/grid_tri_v3_20260715/results/triangular_grid/setting_03/seed_2/final_metrics_summary.json`
