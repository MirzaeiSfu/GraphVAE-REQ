# DeFoG seed-0 legacy structural Random-GIN preview

This report evaluates the supplied DeFoG generated collections with the same
legacy third-party Random-GIN mode used in
`FINAL_ALL_MOTIF_RESULTS_PRUNED_COUNT_DISTANCE.md`. Dataset node attributes are
ignored. The evaluator adds self-loops and constructs the Kia-style degree,
clustering, and square-clustering node channels.

The values below are means and population SDs across evaluator seeds 0--9.
They represent one DeFoG training seed, not three independent DeFoG training
seeds. Linear MMD is reported both as its ordinary mean and as the historical
10% trimmed mean.

## DeFoG results

| Dataset | Graphs (generated/reference) | F1-PR | Precision | Recall | MMD-RBF | MMD-linear | MMD-linear trimmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MUTAG | 39/39 | 0.985758 +/- 0.003796 | 0.992308 +/- 0.011750 | 0.979487 +/- 0.010256 | 0.050514 +/- 0.002114 | 1.637538 +/- 0.640277 | 1.637510 |
| PROTEINS | 210/210 | 0.969096 +/- 0.011174 | 0.988095 +/- 0.007752 | 0.950952 +/- 0.018201 | 0.023794 +/- 0.003291 | 1.487888 +/- 0.376006 | 1.397914 |

## MUTAG comparison to the historical GraphVAE table

| Metric | Better | DeFoG seed 0 | GraphVAE motif=False | GraphVAE true total | GraphVAE true full |
| --- | :---: | ---: | ---: | ---: | ---: |
| Third-party MMD-RBF | down | **0.050514 +/- 0.002114** | 0.148380 +/- 0.035971 | 0.137844 +/- 0.036011 | 0.172392 +/- 0.046839 |
| Third-party linear MMD, trimmed | down | **1.637510** | 996.173 +/- 1051.071 | 41.746 +/- 14.158 | 73.567 +/- 51.522 |
| Third-party precision | up | **0.992308 +/- 0.011750** | 0.479487 +/- 0.025641 | 0.793162 +/- 0.023825 | 0.749573 +/- 0.045605 |
| Third-party recall | up | 0.979487 +/- 0.010256 | **1.000000 +/- 0.000000** | 0.991453 +/- 0.010675 | 0.997436 +/- 0.004441 |
| Third-party F1-PR | up | **0.985758 +/- 0.003796** | 0.644294 +/- 0.026153 | 0.875365 +/- 0.019813 | 0.850675 +/- 0.030455 |

The canonical MUTAG reference has 20.128205 undirected edges per graph. The
saved DeFoG collection has 19.589744, an absolute difference of 0.538462.
The historical GraphVAE table reports edge-count errors of 2.786, 3.889, and
4.470 for motif=False, true-total, and true-full respectively.

## Comparability limits

- MUTAG DeFoG uses the same 39-graph, 20.128-edge reference as the GraphVAE
  motif=True columns. The historical motif=False reference differs.
- The GraphVAE values aggregate three training seeds, while DeFoG has only
  training seed 0. DeFoG's displayed SD is evaluator variation only.
- The supplied DeFoG artifact does not record its generation seed and its
  checkpoint is not proven to be selected solely by validation data.
- PROTEINS uses the historical 210-graph reference so generated/reference
  counts remain equal. The repaired canonical reference has 209 accepted
  graphs, and the main GraphVAE report intentionally omits third-party
  PROTEINS values for motif=True. The PROTEINS row is therefore standalone,
  not a completed controlled comparison.

## Reproduction

The unchanged `scripts/evaluate_graph_realism_batch.py` was run on GPU with
`--repeats 10 --seed 0 --max-graphs 1000`. Raw ignored artifacts and JSON are
under `runs/defog/structural_preview/`.

Source collection identities are recorded in
`reports/defog/artifact_inventory.yaml`. The topology-only NumPy conversion
hashes are:

- MUTAG generated: `5d73c9ba8e2b6d72c3b9166eecb58ab199b3e589c599698e018f2a3121932124`
- MUTAG reference: `1e537e34213ab6312395b2cd50556e07a59b3ca9c4b452cc3fc134bbfe676e19`
- PROTEINS generated: `44be208801c634a041bb2e19c7dca7c9ee87344e49cfbee2996d9e98253be520`
- PROTEINS reference: `10a855e4a0032ca5cfe3b70829dbe21375918323e425f65e1cb6f7a2219cf84b`
