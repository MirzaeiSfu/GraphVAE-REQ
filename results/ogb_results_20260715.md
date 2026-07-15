# OGBG-MOLBBBP results

Collected from completed OGB runs on `cs-cl-17` and the run snapshot transferred to `cs-cl-18`.

- Dataset: `ogbg-molbbbp`
- BFS strategy: `legacy_first_component`
- Epochs: `20000`
- Best-checkpoint metric: `table3_priority`
- Final evaluation uses the saved `best_validation_mmd_model` on the test split.

## Settings

| Setting | Meaning | Motif loss | Temperature | alpha motif | Motif mode |
|---|---|---:|---|---:|---|
| 01 | GraphVAE baseline | False | none | 0.00 | `abs_log_ratio` |
| 03 | GraphVAE + motif, no temperature annealing | True | 1.0 to 1.0 | 0.10 | `calibrated_gaussian` |
| 06 | GraphVAE + motif, temperature annealing | True | 1.0 to 0.5 | 0.10 | `calibrated_gaussian` |

The motif runs use the pruned OGB motif cache with fewer than 100 retained combinations.

## Aggregate final test metrics

Values are mean +/- sample standard deviation across the completed result files available on `cs-cl-18` and `cs-cl-17`. Higher F1/precision/recall is better; lower MMD is better.

| Setting | Completed seeds | Test F1-PR | Test MMD RBF | Precision | Recall | Degree MMD | Clustering MMD | Orbit MMD | Spectral MMD | Diameter MMD | 3rd-party F1-PR | 3rd-party MMD RBF | 3rd-party linear MMD trimmed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 2 | 0.7259 +/- 0.0600 | 0.0756 +/- 0.1018 | 0.6142 +/- 0.0641 | 0.8930 +/- 0.0466 | 0.0132 +/- 0.0004 | 0.2310 +/- 0.0646 | 0.0023 +/- 0.0024 | 0.0189 +/- 0.0040 | 0.1356 +/- 0.0421 | 0.7447 +/- 0.0471 | 0.0914 +/- 0.1040 | 39771.7493 +/- 56194.9798 |
| 03 | 3 | 0.7757 +/- 0.0258 | 0.0542 +/- 0.0902 | 0.7105 +/- 0.0099 | 0.8588 +/- 0.0523 | 0.0125 +/- 0.0020 | 0.1818 +/- 0.0422 | 0.0012 +/- 0.0007 | 0.0210 +/- 0.0016 | 0.1829 +/- 0.0126 | 0.8007 +/- 0.0103 | 0.0654 +/- 0.1010 | 638711.9063 +/- 1046267.8800 |
| 06 | 2 | 0.7726 +/- 0.0590 | 0.0583 +/- 0.0801 | 0.7048 +/- 0.0750 | 0.8588 +/- 0.0325 | 0.0112 +/- 0.0060 | 0.2007 +/- 0.0178 | 0.0011 +/- 0.0006 | 0.0172 +/- 0.0043 | 0.1604 +/- 0.0703 | 0.7880 +/- 0.0572 | 0.1018 +/- 0.1318 | 129060.4592 +/- 181213.3827 |

## Per-seed final test metrics

| Setting | Seed | Test F1-PR | Test MMD RBF | Precision | Recall | Degree | Clustering | Orbit | Spectral | Diameter | 3rd-party F1-PR | 3rd-party MMD RBF | 3rd-party linear MMD trimmed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0 | 0.7684 | 0.1476 | 0.6595 | 0.9259 | 0.0129 | 0.1852 | 0.0005 | 0.0217 | 0.1654 | 0.7780 | 0.1649 | 35.8980 |
| 01 | 1 | 0.6835 | 0.0036 | 0.5688 | 0.8600 | 0.0134 | 0.2767 | 0.0040 | 0.0161 | 0.1058 | 0.7114 | 0.0179 | 79507.6006 |
| 03 | 0 | 0.7691 | 0.0018 | 0.7015 | 0.8593 | 0.0103 | 0.1529 | 0.0009 | 0.0211 | 0.1935 | 0.7897 | 0.0038 | 1846163.5620 |
| 03 | 1 | 0.8042 | 0.1584 | 0.7211 | 0.9109 | 0.0128 | 0.1624 | 0.0006 | 0.0227 | 0.1689 | 0.8101 | 0.1819 | 31.1006 |
| 03 | 2 | 0.7538 | 0.0025 | 0.7090 | 0.8062 | 0.0143 | 0.2302 | 0.0020 | 0.0194 | 0.1861 | 0.8023 | 0.0104 | 69941.0557 |
| 06 | 1 | 0.7309 | 0.0017 | 0.6518 | 0.8358 | 0.0069 | 0.2133 | 0.0015 | 0.0142 | 0.1107 | 0.7475 | 0.0086 | 257197.6709 |
| 06 | 2 | 0.8143 | 0.1150 | 0.7579 | 0.8817 | 0.0154 | 0.1881 | 0.0007 | 0.0203 | 0.2102 | 0.8284 | 0.1950 | 923.2474 |

## Interpretation and completeness

- Setting 03 is a complete three-seed result: seeds 0 and 2 are on `cs-cl-17`, and seed 1 is the transferred snapshot on `cs-cl-18`.
- Setting 03 improves mean local F1-PR and third-party F1-PR over the two locally available Setting 01 seeds. Its mean MMD RBF is also lower, but variability is high.
- Only two completed result files were available for Setting 01 and Setting 06 while creating this report, so those rows are explicitly `n=2` and should not be presented as three-seed aggregates.
- Random-GIN linear MMD contains extreme outliers. The values are retained exactly, but F1-PR and RBF MMD are safer primary comparison metrics.

## Source files

- Setting 01 seeds 0 and 1: `/local-scratch2/new/ogb_table2_20260712/GraphVAE-REQ-main-check/runs/cluster_tests/ogbg_molbbbp_table2/01_graphvae/seed_*/final_metrics_summary.json`
- Setting 03 seeds 0 and 2: `cs-cl-17:/localhome/mirzaei/ogb_table2_20260712/GraphVAE-REQ-main-check/runs/cluster_tests/ogbg_molbbbp_table2/03_graphvae_motif_original_no_temp/seed_*/final_metrics_summary.json`
- Setting 03 seed 1: `/local-scratch2/new/jie_ogb_full_snapshot_20260713/runs/cluster_tests/ogbg_molbbbp_table2/03_graphvae_motif_original_no_temp/seed_1/final_metrics_summary.json`
- Setting 06 seeds 1 and 2: `cs-cl-17:/localhome/mirzaei/ogb_table2_20260712/GraphVAE-REQ-main-check/runs/cluster_tests/ogbg_molbbbp_table2/06_graphvae_motif_original_temp/seed_*/final_metrics_summary.json`
