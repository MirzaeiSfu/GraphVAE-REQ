# MUTAG/PTC motif integration validation

Date: 2026-07-16

## Dataset and database alignment

| Dataset | Graphs | Node states | Nodes | Directed edge rows | Missing reverse rows |
|---|---:|---:|---:|---:|---:|
| MUTAG | 188 | 7 | 3,371 | 7,442 | 0 |
| PTC | 344 | 19 | 8,792 | 17,862 | 0 |

The loaders use DGL `GINDataset(..., self_loop=False)` and import the
categorical `ndata['label']` state as the same 1-based `node_feature` stored in
MySQL. Graph classification labels are not motif features.

## Exact FactorBase sanity checks

The sanity configurations use `legacy_80_20` because sanity mode merges the
training and test partitions. This covers every database graph without an
omitted validation partition.

| Dataset | Loaded rules | Motif/value entries | Aggregated edge count | FactorBase `local_mult` match |
|---|---:|---:|---:|---|
| MUTAG | 3 | 24 | 7,442 | **True** |
| PTC | 3 | 85 | 17,862 | **True** |

## Motif-only gradient descent

The diagnostic optimized the actual GraphVAE AveEncoder, FC adjacency decoder,
and node-feature decoder using motif loss alone. No adjacency reconstruction,
KL, kernel, node-feature reconstruction, or edge-feature reconstruction term
was included. It used one real largest graph, calibrated-Gaussian motif loss,
Adam, learning rate `0.0003`, and 100 updates.

| Dataset | Nodes | Initial loss | Final loss | Initial encoder gradient norm | Initial adjacency gradient norm | Initial node gradient norm | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| MUTAG | 28 | 3.321124 | -0.676413 | 5.955900 | 0.975215 | 3.808232 | **Pass** |
| PTC | 109 | 4.112204 | -2.668340 | 3.504009 | 0.279084 | 1.260515 | **Pass** |

The encoder and both decoder branches received finite, nonzero gradients, and
motif loss decreased. Negative calibrated-Gaussian values are valid Gaussian
log-density objectives and do not indicate an error.

An additional stress test at learning rate `0.003` became non-finite after 22
MUTAG updates and 16 PTC updates. The actual experiment learning rate of
`0.0003` remained finite for all 100 updates, so the higher motif-only rate
should not be used.

The first parallel PTC full-path attempt hit a DGL CUDA illegal-memory error on
GPU 1. The same deterministic diagnostic completed successfully on GPU 0 with
synchronous CUDA error reporting; this was a device/runtime failure rather
than a motif-loss gradient failure.

## Artifacts

- Configs: `configs/cluster_tests/mutag_motif_sanity.yaml` and
  `configs/cluster_tests/ptc_motif_sanity.yaml`
- Motif caches: `cache_motifs/mutag_undir_feat.pkl` and
  `cache_motifs/ptc_undir_feat.pkl`
- Gradient diagnostic: `scripts/check_motif_gradient_descent.py`
- Runtime JSON: `runs/diagnostics/mutag_motif_gradient.json` and
  `runs/diagnostics/ptc_motif_gradient.json`
