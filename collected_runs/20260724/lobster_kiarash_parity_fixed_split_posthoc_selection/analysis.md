# Motif-derived Kiarash parity held-out evaluation

Every checkpoint was selected using validation graphs only. The combined winner manifest was written before any held-out graph was loaded. Each selected checkpoint then received 10 paired held-out prior rollouts.

Values below are means ± sample standard deviations across the three training-seed means; each seed mean contains all held-out rollouts.
All runs use the byte-identical held-out reference set with SHA-256 `def54eda43fc52c49c70c5d219fa996553314e99cd3975a3f29e9d5321ccfb62`.

| Condition | Degree | Clustering | Orbit | Spectral | Diameter | LCC nodes | Raw nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| lobster_kiarash_parity_kia40_2000_legacy | 0.00711 ± 0.00416 | 0.00002 ± 0.00002 | 0.04247 ± 0.00912 | 0.02877 ± 0.00232 | 0.21575 ± 0.05457 | 35.47500 ± 2.34693 | 42.58333 ± 3.19048 |
| lobster_kiarash_parity_kia40_2000_corrected | 0.01182 ± 0.00229 | 0.00000 ± 0.00000 | 0.04855 ± 0.00304 | 0.03580 ± 0.00188 | 0.30637 ± 0.01392 | 30.10667 ± 0.25325 | 38.65500 ± 0.56738 |
| lobster_kiarash_parity_plain1_1_legacy | 0.09068 ± 0.02258 | 0.50503 ± 0.08238 | 0.57881 ± 0.11998 | 0.09549 ± 0.01352 | 0.53033 ± 0.12531 | 37.15167 ± 3.54119 | 41.42000 ± 3.03861 |
| lobster_kiarash_parity_plain1_1_corrected | 0.06969 ± 0.02175 | 0.36683 ± 0.26913 | 0.58895 ± 0.09861 | 0.09035 ± 0.02006 | 0.58581 ± 0.06626 | 37.76833 ± 5.44161 | 42.15500 ± 4.37699 |
| GraphVAE-MM/Kiarash published control | 0.00990 | 0.00000 | 0.06988 | 0.03136 | 0.24844 | not reported | not reported |

The held-out reference contains 62.05 mean nodes.

The published control is a point estimate, so it is not used as if it had zero sampling uncertainty. Per-run and per-rollout values are in `heldout_rollouts.json` and `per_run_summary.csv`.

## Random-GIN

Values are means ± sample standard deviations across the three training-seed
means; every seed mean contains ten Random-GIN initializations.

| Condition | F1-PR ↑ | RBF MMD ↓ |
|---|---:|---:|
| lobster_kiarash_parity_kia40_2000_legacy | 0.97782 ± 0.03624 | 0.31214 ± 0.16706 |
| lobster_kiarash_parity_kia40_2000_corrected | 0.99061 ± 0.01412 | 0.36311 ± 0.08623 |
| lobster_kiarash_parity_plain1_1_legacy | 0.50944 ± 0.06807 | 0.02337 ± 0.02013 |
| lobster_kiarash_parity_plain1_1_corrected | 0.56637 ± 0.02974 | 0.11510 ± 0.14975 |
| GraphVAE-MM/Kiarash published control | 1.00001 | 0.44455 |

The low RBF values of the `[1,1]` conditions do not indicate useful parity:
their F1-PR and structural metrics collapse together.

## Decision

The legacy `40/2000` condition reproduces the published structural point
estimates: degree, orbit, spectral, and diameter MMD are all lower, while
clustering remains effectively zero. It also reaches a high Random-GIN F1-PR
and a lower RBF MMD point estimate.

Graph-size parity is not established. Legacy `40/2000` produces
42.58 ± 3.19 raw nodes and 35.48 ± 2.35 largest-component nodes against
62.05 reference nodes, a 31% raw-node shortfall. The published control did not
report node counts, and the older local GraphVAE-MM reproduction used a
different split/protocol. Therefore the predefined parity gate blocks the
semantically grouped hybrid. The next experiment should first rerun the exact
GraphVAE-MM control on this frozen split to establish its graph-size baseline;
the hybrid should not be launched from the current result.
