# Motif-derived Kiarash parity held-out evaluation

Every checkpoint was selected using validation graphs only. The combined winner manifest was written before any held-out graph was loaded. Each selected checkpoint then received 10 paired held-out prior rollouts.

Values below are means ± sample standard deviations across the three training-seed means; each seed mean contains all held-out rollouts.
All runs use the byte-identical held-out reference set with SHA-256 `def54eda43fc52c49c70c5d219fa996553314e99cd3975a3f29e9d5321ccfb62`.

| Condition | Degree | Clustering | Orbit | Spectral | Diameter | LCC nodes | Raw nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| lobster_graphvae_mm_fixed_split_matched1_legacy | 0.00650 ± 0.00107 | 0.00009 ± 0.00007 | 0.03826 ± 0.01059 | 0.02974 ± 0.00160 | 0.21747 ± 0.00962 | 35.83500 ± 0.32913 | 42.12667 ± 0.43596 |
| lobster_kiarash_parity_kia40_2000_legacy | 0.00711 ± 0.00416 | 0.00002 ± 0.00002 | 0.04247 ± 0.00912 | 0.02877 ± 0.00232 | 0.21575 ± 0.05457 | 35.47500 ± 2.34693 | 42.58333 ± 3.19048 |
| lobster_graphvae_mm_fixed_split_native40_legacy | 0.00756 ± 0.00168 | 0.00000 ± 0.00000 | 0.05831 ± 0.00634 | 0.03568 ± 0.00646 | 0.28278 ± 0.07155 | 32.15667 ± 4.60615 | 37.58167 ± 3.76322 |
| lobster_kiarash_parity_kia40_2000_feature40_legacy | 0.00575 ± 0.00120 | 0.00002 ± 0.00003 | 0.04586 ± 0.01496 | 0.03249 ± 0.00447 | 0.31254 ± 0.02896 | 32.57667 ± 3.94830 | 38.11333 ± 2.57990 |
| GraphVAE-MM/Kiarash published control | 0.00990 | 0.00000 | 0.06988 | 0.03136 | 0.24844 | not reported | not reported |

The held-out reference contains 62.05 mean nodes.

The published control is a point estimate, so it is not used as if it had zero sampling uncertainty. Per-run and per-rollout values are in `heldout_rollouts.json` and `per_run_summary.csv`.

## Parity decision

**Pass the implementation-parity gate and proceed to the low-weight semantic
hybrid.** The relevant comparison is motif-derived statistics against the
native `GlobalProperties.kernel` under otherwise matched training conditions,
not either implementation against a published point estimate from a different
run.

The table reports the seed-paired difference `motif - native` and a two-sided
95% Student-t interval over the three training seeds. Every interval includes
zero for the four preregistered gate metrics and both graph-size measurements.

| Feature weights | Degree | Clustering | Spectral | LCC nodes | Raw nodes |
|---|---:|---:|---:|---:|---:|
| 1/1 | +0.00060 [-0.01223, +0.01344] | -0.000070 [-0.000213, +0.000072] | -0.00097 [-0.00985, +0.00791] | -0.36 [-6.69, +5.97] | +0.46 [-8.23, +9.14] |
| 40/40 | -0.00181 [-0.00781, +0.00418] | +0.000016 [-0.000053, +0.000084] | -0.00318 [-0.01794, +0.01158] | +0.42 [-16.29, +17.13] | +0.53 [-10.56, +11.63] |

Orbit and diameter also have intervals containing zero; all paired results are
in `parity_gate.csv`. The exact tensor-level tests independently verify that
all eight motif-derived outputs and their calibrated-Gaussian loss match the
legacy implementation to `rtol=atol=1e-12`.

This is an operational parity conclusion, not a formal equivalence proof:
three training seeds produce wide intervals. Both implementations also
generate graphs smaller than the 62.05-node held-out reference (about 42 nodes
with feature weights 1/1 and 38 nodes with 40/40). The important result for
this gate is that the motif replacement does not cause additional graph-size
collapse relative to its matched native control.

## Matched RandomGIN F1-PR and RBF evaluation

The previously missing RandomGIN comparison was run post hoc on the frozen
rollout-0 graph collection from every checkpoint. Each run contains 20
generated and 20 byte-identical held-out reference graphs. The Kia-style
evaluator uses degree, clustering, and square-clustering node features and 10
untrained RandomGIN initializations with common seeds 0 through 9.

Values below are means ± sample standard deviations across the three training
seeds. Each training-seed value is itself the mean over the 10 common
RandomGIN initializations.

| Feature weights | Statistics implementation | F1-PR | RBF MMD |
|---|---|---:|---:|
| 1/1 | Native `GlobalProperties.kernel` | 0.98620 ± 0.01293 | 0.22935 ± 0.08816 |
| 1/1 | Motif-derived `kiarash_statistics` | 0.97782 ± 0.03624 | 0.31214 ± 0.16706 |
| 40/40 | Native `GlobalProperties.kernel` | 0.99303 ± 0.00557 | 0.26386 ± 0.14034 |
| 40/40 | Motif-derived `kiarash_statistics` | 0.99916 ± 0.00148 | 0.29817 ± 0.09237 |
| Published GraphVAE-MM/Kiarash | Historical point estimate | 1.00001 | 0.44455 |

The seed-paired difference is `motif - native`. The intervals are two-sided
95% Student-t intervals over the three training seeds.

| Feature weights | F1-PR difference | RBF MMD difference |
|---|---:|---:|
| 1/1 | -0.00838 [-0.09876, +0.08199] | +0.08279 [-0.48714, +0.65271] |
| 40/40 | +0.00612 [-0.00918, +0.02143] | +0.03431 [-0.24309, +0.31172] |

Every interval includes zero. Therefore the matched RandomGIN comparison also
passes the operational implementation-parity gate: there is no detected
F1-PR or RBF change attributable to replacing the native statistics with the
motif-derived bundle. This remains a low-power compatibility result rather
than a formal equivalence proof because there are only three training seeds.

The published `1.00001` is explained by the evaluator's `1e-5` offset in the
harmonic-mean formula when precision and recall both equal one. It is not the
score of a trained GNN. Optional TensorFlow FID/KID imports were unavailable
during this post-hoc run, but those metrics are outside this comparison; every
required RandomGIN F1-PR, precision, recall, RBF, and linear-MMD value was
produced. Per-run summaries are in
`random_gin_matched_rollout0_summary.csv`, and paired results are in
`random_gin_parity_gate.csv`. Each evaluated run also preserves its 10 raw
repeat values in `graph_realism_random_gin_kiarash_matched_rollout0.json`.
