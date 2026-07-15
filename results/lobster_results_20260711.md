# LOBSTER results

Generated from completed cluster runs in `graphvae_grid_lobster_20260711_073601/GraphVAE-REQ-main-check`.

All runs used `database_name: lobster_undir_feat_snap_85093d_cfg410000000`, `bfs_strategy: legacy_first_component`, `rule_prune: true`, `best_validation_mmd_metric: table3_priority`, `checkpoint_interval_epochs: 1000`, and `epoch_number: 20000`. The final test evaluation uses the saved `best_validation_mmd_model`, selected on the validation split.

## Settings

| Setting | Host | Meaning | Motif loss | alpha_motif_loss | motif_loss_mode |
|---|---|---|---:|---:|---|
| 01 | `cs-cl-26` | GraphVAE baseline | False | 0.0000 | `abs_log_ratio` |
| 03 | `cs-cl-09` | GraphVAE + motif, no temp | True | 0.1000 | `calibrated_gaussian` |
| 06 | `cs-cl-36` | GraphVAE + motif, temp | True | 0.1000 | `calibrated_gaussian` |

## Aggregate final test metrics across seeds

For F1/precision/recall, higher is better. For MMD metrics, lower is better.

| Setting | Test F1-PR ↑ | Test MMD RBF ↓ | Precision ↑ | Recall ↑ | Degree MMD ↓ | Clustering MMD ↓ | Orbit MMD ↓ | Spectral MMD ↓ | Diameter MMD ↓ | 3rd-party F1-PR ↑ | 3rd-party MMD RBF ↓ | 3rd-party linear MMD trimmed ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 0.8954 ± 0.0288 | 0.2449 ± 0.1040 | 0.8433 ± 0.0293 | 0.9567 ± 0.0355 | 0.0766 ± 0.0099 | 0.5928 ± 0.0504 | 0.1215 ± 0.0264 | 0.0548 ± 0.0038 | 0.2858 ± 0.2234 | 0.8182 ± 0.0065 | 0.2619 ± 0.0823 | 28.1929 ± 3.5274 |
| 03 | 0.8395 ± 0.0102 | 0.0387 ± 0.0335 | 0.7367 ± 0.0247 | 0.9783 ± 0.0202 | 0.0642 ± 0.0127 | 0.4772 ± 0.0361 | 0.1269 ± 0.0123 | 0.0785 ± 0.0107 | 0.3719 ± 0.0761 | 0.7991 ± 0.0141 | 0.0387 ± 0.0336 | 1563748318.0000 ± 1363631461.6945 |
| 06 | 0.8279 ± 0.0809 | 0.0351 ± 0.0173 | 0.7167 ± 0.1154 | 0.9917 ± 0.0104 | 0.0651 ± 0.0349 | 0.4461 ± 0.1516 | 0.1296 ± 0.0906 | 0.0776 ± 0.0291 | 0.2881 ± 0.1644 | 0.7943 ± 0.0912 | 0.0355 ± 0.0177 | 1197878282.6667 ± 806944219.5474 |

## Per-seed final test metrics

| Setting | Seed | Best val epoch | Best val score ↓ | Val F1-PR ↑ | Val MMD RBF ↓ | Test F1-PR ↑ | Test MMD RBF ↓ | Precision ↑ | Recall ↑ | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ | 3rd-party F1-PR ↑ | 3rd-party MMD RBF ↓ | 3rd-party linear MMD trimmed ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 0 | 6999 | 0.1747 | 0.9842 | 0.2026 | 0.9051 | 0.2181 | 0.8650 | 0.9500 | 0.0665 | 0.6412 | 0.1034 | 0.0542 | 0.2309 | 0.8122 | 0.2496 | 24.7749 |
| 01 | 1 | 12999 | 0.1794 | 0.9947 | 0.2073 | 0.9180 | 0.1569 | 0.8550 | 0.9950 | 0.0768 | 0.5407 | 0.1519 | 0.0513 | 0.0950 | 0.8252 | 0.1864 | 27.9834 |
| 01 | 2 | 3999 | 0.1880 | 1.0000 | 0.2433 | 0.8630 | 0.3596 | 0.8100 | 0.9250 | 0.0864 | 0.5966 | 0.1092 | 0.0589 | 0.5315 | 0.8173 | 0.3496 | 31.8204 |
| 03 | 0 | 16999 | 0.1104 | 0.9474 | 0.0200 | 0.8309 | 0.0774 | 0.7250 | 0.9750 | 0.0695 | 0.4681 | 0.1387 | 0.0909 | 0.4458 | 0.8034 | 0.0775 | 3136086224.0000 |
| 03 | 1 | 3999 | 0.1387 | 0.9474 | 0.0200 | 0.8370 | 0.0187 | 0.7200 | 1.0000 | 0.0497 | 0.4466 | 0.1277 | 0.0722 | 0.2938 | 0.7833 | 0.0187 | 704737788.0000 |
| 03 | 2 | 4999 | 0.1747 | 0.9276 | 0.0200 | 0.8507 | 0.0200 | 0.7650 | 0.9600 | 0.0733 | 0.5170 | 0.1142 | 0.0724 | 0.3762 | 0.8106 | 0.0200 | 850420942.0000 |
| 06 | 0 | 2999 | 0.1686 | 0.9357 | 0.0200 | 0.7362 | 0.0451 | 0.5850 | 0.9950 | 0.1040 | 0.6157 | 0.2341 | 0.1100 | 0.3357 | 0.6896 | 0.0464 | 1349987928.0000 |
| 06 | 1 | 18999 | 0.1126 | 0.9474 | 0.0200 | 0.8889 | 0.0151 | 0.8000 | 1.0000 | 0.0367 | 0.3236 | 0.0714 | 0.0536 | 0.1051 | 0.8556 | 0.0151 | 325704144.0000 |
| 06 | 2 | 7999 | 0.1516 | 0.9415 | 0.0200 | 0.8587 | 0.0450 | 0.7650 | 0.9800 | 0.0545 | 0.3988 | 0.0834 | 0.0692 | 0.4234 | 0.8379 | 0.0450 | 1917942776.0000 |

## Per-seed Table 2 structural metrics

| Setting | Seed | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 0 | 0.0665 | 0.6412 | 0.1034 | 0.0542 | 0.2309 |
| 01 | 1 | 0.0768 | 0.5407 | 0.1519 | 0.0513 | 0.0950 |
| 01 | 2 | 0.0864 | 0.5966 | 0.1092 | 0.0589 | 0.5315 |
| 03 | 0 | 0.0695 | 0.4681 | 0.1387 | 0.0909 | 0.4458 |
| 03 | 1 | 0.0497 | 0.4466 | 0.1277 | 0.0722 | 0.2938 |
| 03 | 2 | 0.0733 | 0.5170 | 0.1142 | 0.0724 | 0.3762 |
| 06 | 0 | 0.1040 | 0.6157 | 0.2341 | 0.1100 | 0.3357 |
| 06 | 1 | 0.0367 | 0.3236 | 0.0714 | 0.0536 | 0.1051 |
| 06 | 2 | 0.0545 | 0.3988 | 0.0834 | 0.0692 | 0.4234 |

## Per-seed third-party Random-GIN metrics

| Setting | Seed | F1-PR ↑ | MMD RBF ↓ | Linear MMD mean ↓ | Linear MMD median ↓ | Linear MMD trimmed mean ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 0 | 0.8122 | 0.2496 | 52.8954 | 22.3145 | 24.7749 |
| 01 | 1 | 0.8252 | 0.1864 | 31.2734 | 21.3109 | 27.9834 |
| 01 | 2 | 0.8173 | 0.3496 | 44.7971 | 25.7777 | 31.8204 |
| 03 | 0 | 0.8034 | 0.0775 | 21424957760.0000 | 1551926400.0000 | 3136086224.0000 |
| 03 | 1 | 0.7833 | 0.0187 | 4766131209.6000 | 346049296.0000 | 704737788.0000 |
| 03 | 2 | 0.8106 | 0.0200 | 5863492316.8000 | 424271728.0000 | 850420942.0000 |
| 06 | 0 | 0.6896 | 0.0464 | 8982016336.0000 | 663177280.0000 | 1349987928.0000 |
| 06 | 1 | 0.8556 | 0.0151 | 2094898054.4000 | 160126848.0000 | 325704144.0000 |
| 06 | 2 | 0.8379 | 0.0450 | 13229507116.8000 | 957003008.0000 | 1917942776.0000 |

## Source files

- Setting 01: `cs-cl-26:~/graphvae_grid_lobster_20260711_073601/GraphVAE-REQ-main-check/runs/cluster_tests/lobster_table2/01_graphvae/seed_*/`
- Setting 03: `cs-cl-09:~/graphvae_grid_lobster_20260711_073601/GraphVAE-REQ-main-check/runs/cluster_tests/lobster_table2/03_graphvae_motif_original_no_temp/seed_*/`
- Setting 06: `cs-cl-36:~/graphvae_grid_lobster_20260711_073601/GraphVAE-REQ-main-check/runs/cluster_tests/lobster_table2/06_graphvae_motif_original_temp/seed_*/`
