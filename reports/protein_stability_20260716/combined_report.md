# PROTEINS best-model stability analysis

Each saved best model was resampled with 10 validation rollouts and 50 held-out test rollouts. Dense means more edges than the real split mean plus three standard deviations. MMD uses largest connected components; density statistics below use raw generated graphs.

| Seed | Selected epoch | Validation score mean ± std | Test score mean ± std | Test mean edges / real | Test dense graphs | Test rollouts with ≥1 dense graph | Worst graph edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed_0 | 20000 | 0.6606 ± 0.0835 | 0.7764 ± 0.0565 | 87.7 / 54.6 | 0.90% | 86.00% | 4821 |
| seed_1 | 9333 | 0.6400 ± 0.0473 | 0.7908 ± 0.0637 | 58.9 / 54.6 | 0.19% | 28.00% | 4809 |
| seed_2 | 10333 | 0.7129 ± 0.0466 | 0.8380 ± 0.0462 | 108.4 / 54.6 | 1.49% | 96.00% | 4817 |

## Interpretation

- Seed 1 is the most stable: its mean edge count is closest to the reference and both dense-graph and dense-rollout rates are lowest.
- Seeds 0 and 2 have severe heavy tails. Their average dense-graph percentages look small only because each rollout contains many graphs; nearly every rollout contains at least one dense outlier.
- All three seeds produced at least one nearly complete padded graph (about 4,800 edges), so the decoder instability is present in every saved best model.
- The online best-checkpoint metric did not protect against this tail because it emphasized MMD/GNN realism and summarized only one generated set.
