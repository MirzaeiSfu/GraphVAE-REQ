# PROTEINS results

Generated from `reports/protein_results_bundle_20260712_014819.tar.gz`.

- `database`: `proteins_dir_feat_snap_0094da`
- `bfs_strategy`: `legacy_first_component`
- `rule_prune`: `True`
- `best_validation_mmd_metric`: `table3_priority`
- final evaluation uses the saved `best_validation_mmd_model` selected on the validation split.

## Settings

| Setting | Meaning | Motif loss | alpha_motif_loss | motif_loss_mode |
|---|---|---:|---:|---|
| 00 | GraphVAE baseline | False | 0.0000 | `calibrated_gaussian` |
| 03 | GraphVAE + motif, no temp | True | 0.1000 | `calibrated_gaussian` |
| 06 | GraphVAE + motif, temp | True | 0.1000 | `calibrated_gaussian` |

## Aggregate final test metrics across seeds

For F1/precision/recall, higher is better. For MMD metrics, lower is better.

| Setting | Test F1-PR ↑ | Test MMD RBF ↓ | Precision ↑ | Recall ↑ | Degree MMD ↓ | Clustering MMD ↓ | Orbit MMD ↓ | Spectral MMD ↓ | Diameter MMD ↓ | 3rd-party F1-PR ↑ | 3rd-party MMD RBF ↓ | 3rd-party linear MMD trimmed ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 | 0.8915 ± 0.0433 | 0.0739 ± 0.0167 | 0.8949 ± 0.0485 | 0.8902 ± 0.0508 | 0.0465 ± 0.0020 | 0.0296 ± 0.0020 | 0.0266 ± 0.0039 | 0.0238 ± 0.0034 | 0.0420 ± 0.0165 | 0.9005 ± 0.0269 | 0.0928 ± 0.0165 | 2.9466 ± 0.7713 |
| 03 | 0.9291 ± 0.0237 | 0.0260 ± 0.0439 | 0.8995 ± 0.0466 | 0.9633 ± 0.0129 | 0.0428 ± 0.0066 | 0.0290 ± 0.0033 | 0.0126 ± 0.0055 | 0.0231 ± 0.0043 | 0.0273 ± 0.0162 | 0.9199 ± 0.0393 | 0.0190 ± 0.0316 | 4724108.0570 ± 4470962.6688 |
| 06 | 0.9271 ± 0.0308 | 0.0260 ± 0.0439 | 0.8971 ± 0.0520 | 0.9616 ± 0.0154 | 0.0420 ± 0.0044 | 0.0285 ± 0.0032 | 0.0154 ± 0.0108 | 0.0213 ± 0.0028 | 0.0217 ± 0.0101 | 0.9197 ± 0.0381 | 0.0190 ± 0.0316 | 4962797.3695 ± 4601984.5640 |

## Per-seed final test metrics

| Setting | Seed | Best val epoch | Best val score ↓ | Val F1-PR ↑ | Val MMD RBF ↓ | Test F1-PR ↑ | Test MMD RBF ↓ | Precision ↑ | Recall ↑ | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ | 3rd-party F1-PR ↑ | 3rd-party MMD RBF ↓ | 3rd-party linear MMD trimmed ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 | 0 | 3333 | 0.2070 | 0.8822 | 0.0561 | 0.9047 | 0.0928 | 0.9348 | 0.8771 | 0.0485 | 0.0296 | 0.0238 | 0.0235 | 0.0574 | 0.9080 | 0.1108 | 3.4070 |
| 00 | 1 | 8000 | 0.1923 | 0.9331 | 0.0698 | 0.9267 | 0.0683 | 0.9090 | 0.9462 | 0.0465 | 0.0277 | 0.0310 | 0.0206 | 0.0247 | 0.9229 | 0.0783 | 2.0561 |
| 00 | 2 | 3667 | 0.1712 | 0.9338 | 0.0387 | 0.8431 | 0.0607 | 0.8410 | 0.8471 | 0.0446 | 0.0316 | 0.0250 | 0.0273 | 0.0439 | 0.8706 | 0.0893 | 3.3767 |
| 03 | 0 | 20000 | 0.1427 | 0.9462 | 0.0003 | 0.9480 | 0.0004 | 0.9257 | 0.9724 | 0.0501 | 0.0314 | 0.0173 | 0.0254 | 0.0434 | 0.9536 | 0.0004 | 8889386.7812 |
| 03 | 1 | 9333 | 0.1111 | 0.9405 | 0.0002 | 0.9367 | 0.0767 | 0.9271 | 0.9486 | 0.0373 | 0.0252 | 0.0067 | 0.0181 | 0.0110 | 0.8767 | 0.0555 | 1.8585 |
| 03 | 2 | 10333 | 0.1230 | 0.9481 | 0.0007 | 0.9025 | 0.0009 | 0.8457 | 0.9690 | 0.0411 | 0.0304 | 0.0139 | 0.0258 | 0.0273 | 0.9295 | 0.0009 | 5282935.5312 |
| 06 | 0 | 17333 | 0.1551 | 0.9116 | 0.0007 | 0.9519 | 0.0004 | 0.9271 | 0.9786 | 0.0460 | 0.0289 | 0.0275 | 0.0232 | 0.0311 | 0.9495 | 0.0004 | 9089265.7812 |
| 06 | 1 | 9333 | 0.1111 | 0.9405 | 0.0002 | 0.9367 | 0.0767 | 0.9271 | 0.9486 | 0.0373 | 0.0252 | 0.0067 | 0.0181 | 0.0110 | 0.8767 | 0.0555 | 1.8585 |
| 06 | 2 | 11667 | 0.1268 | 0.9490 | 0.0007 | 0.8927 | 0.0010 | 0.8371 | 0.9576 | 0.0429 | 0.0315 | 0.0120 | 0.0227 | 0.0230 | 0.9328 | 0.0010 | 5799124.4688 |

## Per-seed Table 2 structural metrics

| Setting | Seed | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 0 | 0.0485 | 0.0296 | 0.0238 | 0.0235 | 0.0574 |
| 00 | 1 | 0.0465 | 0.0277 | 0.0310 | 0.0206 | 0.0247 |
| 00 | 2 | 0.0446 | 0.0316 | 0.0250 | 0.0273 | 0.0439 |
| 03 | 0 | 0.0501 | 0.0314 | 0.0173 | 0.0254 | 0.0434 |
| 03 | 1 | 0.0373 | 0.0252 | 0.0067 | 0.0181 | 0.0110 |
| 03 | 2 | 0.0411 | 0.0304 | 0.0139 | 0.0258 | 0.0273 |
| 06 | 0 | 0.0460 | 0.0289 | 0.0275 | 0.0232 | 0.0311 |
| 06 | 1 | 0.0373 | 0.0252 | 0.0067 | 0.0181 | 0.0110 |
| 06 | 2 | 0.0429 | 0.0315 | 0.0120 | 0.0227 | 0.0230 |

## Per-seed third-party Random-GIN metrics

| Setting | Seed | F1-PR ↑ | MMD RBF ↓ | Linear MMD mean ↓ | Linear MMD median ↓ | Linear MMD trimmed mean ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 0 | 0.9080 | 0.1108 | 3.3668 | 3.3598 | 3.4070 |
| 00 | 1 | 0.9229 | 0.0783 | 2.3007 | 1.8934 | 2.0561 |
| 00 | 2 | 0.8706 | 0.0893 | 3.3379 | 3.4403 | 3.3767 |
| 03 | 0 | 0.9536 | 0.0004 | 1663055367.8000 | 1813480.3125 | 8889386.7812 |
| 03 | 1 | 0.8767 | 0.0555 | 2.2640 | 1.3021 | 1.8585 |
| 03 | 2 | 0.9295 | 0.0009 | 17287327.1000 | 4551753.2500 | 5282935.5312 |
| 06 | 0 | 0.9495 | 0.0004 | 1711445278.9813 | 1871320.8125 | 9089265.7812 |
| 06 | 1 | 0.8767 | 0.0555 | 2.2640 | 1.3021 | 1.8585 |
| 06 | 2 | 0.9328 | 0.0010 | 18993631.2250 | 4999043.0000 | 5799124.4688 |

## Notes

- Protein motif settings `03` and `06` improve aggregate test F1-PR and MMD RBF versus baseline `00`.
- Third-party Random-GIN linear MMD has very large outliers for motif seeds `0` and `2`; those values are preserved exactly as reported by the run outputs.

## Source bundle paths

- Setting 00: `runs/cluster_tests/proteins_table2/00_graphvae/seed_*/`
- Setting 03: `runs/cluster_tests/proteins_table2/03_graphvae_motif_original_no_temp/seed_*/`
- Setting 06: `runs/cluster_tests/proteins_table2/06_graphvae_motif_original_temp/seed_*/`
