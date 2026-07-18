# PROTEINS best-model stability analysis

## Scope

Analyzed the three saved best models under:

`/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp`

Each model was resampled with the same random-seed schedule:

- 10 validation rollouts, each targeting the 104 validation graphs.
- 50 held-out test rollouts, each targeting the 210 test graphs.
- Adjacency threshold: 0.5.
- Structural score: mean of degree, clustering, orbit, spectral, and diameter MMD, normalized by the PROTEINS GraphVAE paper values.
- Dense graph: raw generated edge count greater than the real split mean plus three standard deviations.
- Validation dense threshold: 164.53 edges; held-out test threshold: 179.68 edges.
- Real mean edges: 52.60 on validation and 54.61 on held-out test.

Table 2 structural MMD uses the largest connected component for compatibility, while density diagnostics use raw generated graphs before component filtering.

## Saved-model audit

All three `best_validation_mmd_model` files exist, are nonempty (73,447,942 bytes), have distinct SHA-256 hashes, and loaded successfully for generation.

| Seed | Selected epoch | Original selection score | Model present and loadable |
| ---: | ---: | ---: | :---: |
| 0 | 20,000 | 0.142700 | Yes |
| 1 | 9,333 | 0.111083 | Yes |
| 2 | 10,333 | 0.122950 | Yes |

The selected epochs come from each seed's `best_validation_mmd.json`. The archived `run_config_used.yaml` retains `seed: 0` in all three folders, but `reproducibility.json` correctly records seeds 0, 1, and 2 and the corresponding launch commands.

## Held-out stability results

| Seed | Normalized score mean ± std | Median | Raw mean edges mean ± std | Dense graphs | Rollouts containing dense graph | Worst graph |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7764 ± 0.0565 | 0.7833 | 87.73 ± 28.47 | 95 / 10,500 (0.90%) | 86% | 4,821 edges |
| 1 | 0.7908 ± 0.0637 | 0.7833 | 58.89 ± 7.84 | 20 / 10,500 (0.19%) | 28% | 4,809 edges |
| 2 | 0.8380 ± 0.0462 | 0.8406 | 108.43 ± 35.05 | 156 / 10,499 (1.49%) | 96% | 4,817 edges |

One seed-2 draw was empty and therefore omitted by the legacy generation helper, giving 10,499 rather than 10,500 retained graphs.

## Validation stability results

| Seed | Normalized score mean ± std | Median | Raw mean edges mean ± std | Dense graph rate | Rollouts containing dense graph | Worst graph |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6606 ± 0.0835 | 0.6615 | 131.12 ± 65.35 | 1.73% | 100% | 4,808 edges |
| 1 | 0.6400 ± 0.0473 | 0.6340 | 54.99 ± 2.44 | 0.29% | 30% | 235 edges |
| 2 | 0.7129 ± 0.0466 | 0.7229 | 139.28 ± 42.28 | 2.69% | 100% | 4,790 edges |

## Interpretation

The generator is seed-dependent and not reliably stable.

- Seed 1 is clearly the most density-stable model. Its typical edge count is close to the real distribution, and its validation generation never produced the near-complete ~4,800-edge failure seen for seeds 0 and 2.
- Seed 0 has the best mean held-out structural score by a small margin, but this hides a serious heavy tail. Dense outliers occur in 86% of its held-out rollout sets, and its average edge count is substantially too high.
- Seed 2 is the least stable overall: it has the worst held-out structural score, the largest density bias, and dense outliers in 96% of held-out rollout sets.
- Seed 1 is not perfectly safe. Fourteen of its 50 held-out rollouts were affected by at least one dense graph, including a rare 4,809-edge near-complete failure. Its aggregate failure probability is much lower, however.
- The original best-checkpoint score does not adequately penalize density. Seed 1 had the best original selection score and happens to be the most stable, but seeds 0 and 2 were still saved as “best” despite severe dense tails.

## Recommendation

Among these three saved models, use seed 1. Do not treat any seed as fully stable without generation-time safeguards.

For future model selection, add raw-density penalties and repeated validation rollouts. At generation time, crop to the requested node count and reject or cap graphs exceeding an edge budget calibrated from the real training/validation distribution. Report both structural MMD and raw dense-tail statistics; largest-component MMD alone can conceal near-complete graph failures.

## Freshly recomputed requested criteria

These values were recomputed from new graphs generated from each saved `best_validation_mmd_model`; no previous metric JSON was used. All models received the same latent-random seed for validation and the same separate latent-random seed for test. Local GIN and third-party Random-GIN each used 10 newly initialized feature extractors. Generated and reference graph arrays are preserved under `fresh_recomputed/seed_*`.

Seeds 0 and 1 produced all 104 validation and 210 test graphs. Seed 2 produced all 104 validation graphs but one empty test draw was removed by the legacy generation helper, leaving 209 generated test graphs; the third-party evaluator matched the reference count to 209.

### Validation and local test GIN

| Seed | Val F1-PR ↑ | Val MMD RBF ↓ | Test F1-PR ↑ | Test MMD RBF ↓ | Precision ↑ | Recall ↑ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.882364 | 0.000250 | 0.935756 | 0.090163 | 0.967619 | 0.907143 |
| 1 | 0.882453 | 0.049828 | 0.941810 | 0.073418 | 0.953810 | 0.931429 |
| 2 | 0.873597 | 0.046333 | 0.961373 | 0.000715 | 0.946890 | 0.977619 |

### Test structural MMD

| Seed | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.042429 | 0.029670 | 0.016391 | 0.021645 | 0.035770 |
| 1 | 0.044003 | 0.030039 | 0.011760 | 0.024184 | 0.043626 |
| 2 | 0.043565 | 0.029113 | 0.022161 | 0.021042 | 0.032887 |

### Third-party Random-GIN

| Seed | Third-party F1-PR ↑ | Third-party MMD RBF ↓ | Third-party linear MMD trimmed ↓ |
| ---: | ---: | ---: | ---: |
| 0 | 0.866901 | 0.084491 | 18.612188 |
| 1 | 0.911043 | 0.067197 | 29.101385 |
| 2 | 0.911554 | 0.001315 | 3,445,981.484375 |

The fresh result again exposes generator randomness: seed 2 happened to score very well under RBF GIN metrics but produced an extreme third-party linear-MMD outlier. This agrees with its poor repeated-rollout density stability. No single regenerated set is sufficient to certify generator stability.

## Second fresh evaluation

A second full evaluation with generation seed `20260717` is preserved under `fresh_recomputed_repeat2/`. The repeat-to-repeat comparison is in `fresh_randomness_comparison.md`.

The largest change occurs for seed 0: its first fresh test set contained no dense graphs and had trimmed linear MMD `18.61`; its second contained five near-complete graphs and had trimmed linear MMD `5,042,080.94`. At the same time, RBF MMD improved from `0.09016` to `0.00107`, showing that favorable RBF MMD does not rule out catastrophic dense outliers. Seed 1 was the most repeatable across both evaluations.
