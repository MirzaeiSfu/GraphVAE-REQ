# PTC complete verified experiment report

This 2026-09-17 copy gathers the current completed three-seed campaign. It is
kept separate from the still-running AIDs and MUTAG campaigns.

Verified 2026-09-14 against the archived JSON/YAML artifacts on
`cs-cl-09.cmpt.sfu.ca`. This is the current PTC source-of-truth report.

## Verification and comparability verdict

**Evaluation comparability: strong for the primary held-out structural and
structural-feature Random-GIN tables. Training comparability: suitable for a
model benchmark, but not a strict motif-only ablation.**

- All four methods have generator-training seeds 0, 1, and 2.
- All use the paper 70/10/20 split with split seed 123: 240 train, 34
  validation, and 70 test graphs.
- The matched Random-GIN files use the exact same serialized 70-graph test
  reference, SHA-256
  `aa0fa76e2a90491b7343aee12284e7127813317ef38ab52c8d849d74162558b9`.
- Each Random-GIN generator-seed result averages evaluator seeds 0 through 9;
  reported `±` values are sample SD across generator-training seeds.
- The three archived configs within each GraphVAE setting are identical after
  replacing only seed/path text; their normalized SHA-256 values are
  `7db521...` (false), `91b7bf...` (true total), and `d4562c...` (true full).
- The GraphVAE settings share the dataset, split, optimizer learning rate,
  encoder/decoder family, graph postprocessing, and evaluation graph counts.
  However, motif=False uses latent dimension 1024 and batch size 240, whereas
  motif=True uses latent dimension 128 and batch size 256. Database identity
  also changes from `ptc_undir_feat` to `ptc_multi`. Thus differences cannot be
  attributed solely to motif loss.
- True total and true full are directly comparable to one another: their
  archived configs differ in motif output mode only.
- DeFoG uses a different architecture/objective and a 1,000-epoch, batch-12
  schedule. It is a fair held-out model comparison on the frozen split, not an
  epoch-, update-, or architecture-matched ablation.
- PTC has one 19-way categorical node-label field and no edge attributes.
  GraphVAE's nominal motif=True edge-feature weight of 1 is inactive because
  no edge-attribute target exists.
- All three archived GraphVAE YAML families contain `directed: true`, despite
  the PTC database/evaluation protocol being undirected; DeFoG exports
  undirected graphs. Because the flag is shared by false/total/full it does not
  confound those GraphVAE variants, but the code path should be documented (or
  rerun with an explicitly undirected flag) before claiming a perfectly
  direction-matched DeFoG comparison.

### Random-GIN protocol names used in this report

1. **Structural-node-feature Random-GIN (primary):** node inputs are degree,
   clustering coefficient, and square clustering. These are derived from
   adjacency and are not the original 19-way PTC node labels.
2. **Constant-input Random-GIN (no node features):** every node receives the
   same scalar input. It measures topology only. GraphVAE and DeFoG artifacts
   use separate evaluator wrappers, so this is secondary evidence.
3. **Decoded-original-label Random-GIN:** uses the generated 19-way PTC node
   label. This is archived for DeFoG only; no identically frozen GraphVAE result
   exists, so cross-model cells are correctly left N/A.

### Motif-correlation support

- **The primary motif-correlation metric in this report is hard pruned-state
  TV.** It is the exact training-support comparison: it uses
  the retained 142 categorical states of the trained multi-atom path rule,
  sums counts across 240 generated graphs, normalizes once within the rule,
  and compares against the same 240 training-reference graphs. Motif=False,
  motif=True total, motif=True full, and DeFoG are all counted on this exact
  same retained state list and against the exact same reference collection.
- **Complete-state TV is not called motif correlation in the primary
  comparison.** It uses the same multi-atom rule structure but all 6,859
  possible states, including states pruned out of motif=True training. It is a
  supplementary generalization diagnostic, not an exact-training-state metric.

Generated 2026-09-13. This report consolidates GraphVAE motif=False, GraphVAE-REQ
motif=True total-count, GraphVAE-REQ motif=True full-matrix, and DeFoG results.
Unless stated otherwise, `±` is the sample standard deviation across independent
generator-training seeds 0, 1, and 2. Lower is better for MMD/error/TV metrics;
higher is better for precision, recall, and F1-PR.

## Experimental coverage

| Method | Training seeds | Status | Primary model |
|---|---:|---|---|
| GraphVAE motif=False | 0, 1, 2 | Complete | Best validation-MMD model |
| GraphVAE-REQ motif=True total count | 0, 1, 2 | Complete | Best validation-MMD model |
| GraphVAE-REQ motif=True full matrix | 0, 1, 2 | Complete | Best validation-MMD model |
| DeFoG | 0, 1, 2 | Complete | Minimum-validation-loss checkpoint |

## Dataset and split

- Dataset: PTC.
- Frozen split: paper 70/10/20, split seed 123.
- Training/validation/test graph counts: 240/34/70.
- GraphVAE loader seed: 123.
- Categorical node feature: one 19-way node-label field.
- Edge attributes: none. Consequently, an edge-feature reconstruction weight has
  no attributed edge target on PTC.
- Held-out evaluations use 70 generated and 70 test graphs.
- Exact-rule train-reference evaluations use 240 generated and 240 training graphs.
- Graph postprocessing uses adjacency threshold 0.5, removes self-loops and
  isolated nodes, and retains the deterministic largest connected component.

## Complete settings and hyperparameters

| Setting | Motif=False | Motif=True total | Motif=True full | DeFoG |
|---|---:|---:|---:|---:|
| Epochs | 20,000 | 20,000 | 20,000 | 1,000 |
| Batch size | 240 | 256 | 256 | 12 |
| Learning rate | 0.0003 | 0.0003 | 0.0003 | 0.0002 |
| Latent dimension | 1,024 | 128 | 128 | N/A |
| Encoder/decoder | AvePool/FC | AvePool/FC | AvePool/FC | 8-layer graph transformer |
| Kernel/BCE weight | 1 | 1 | 1 | Native DeFoG CE |
| KL weight | 1 | 1 | 1 | N/A |
| Node-feature weight | 1 | 1 | 1 | Native categorical CE |
| Edge-feature weight | 0 | 1 (inactive: no edge attrs) | 1 (inactive: no edge attrs) | No edge attributes |
| Motif weight | 0 | 0.1 | 0.1 | None |
| Syntactic-literal motif weight | 0 | 0.1 | 0.1 | None |
| Motif loss mode | Disabled | Calibrated Gaussian | Calibrated Gaussian | None |
| Motif output | N/A | Total count | Full matrix | N/A |
| FactorBase source | N/A | `_CP_smoothed` | `_CP_smoothed` | Evaluated post hoc |
| Database | `ptc_undir_feat` | `ptc_multi` | `ptc_multi` | Frozen PTC graph split |
| Pruning enabled | N/A | Yes | Yes | N/A |
| Prune score threshold | N/A | 0.0 | 0.0 | N/A |
| Maximum values per rule | N/A | 256 | 256 | N/A |
| Motif counting batch | N/A | 512 | 512 | Post-hoc exact counting |
| Full/retained combinations | N/A | 6,899/182 | 6,899/182 | Same evaluation support |
| Random-GIN repeats | 10 | 10 | 10 | 10 |

The motif=False and motif=True GraphVAE configurations are not a perfectly
controlled motif-only ablation: latent dimension, batch size, database identity,
and nominal edge-loss weight also differ. This limitation must be disclosed.

## Rules used during motif=True training

1. `edges(nodes0,nodes1)`
2. `edges(nodes1,nodes2)`
3. `node_feature(nodes0) AND edges(nodes0,nodes1) AND edges(nodes1,nodes2) AND node_feature(nodes1) AND node_feature(nodes2)`
4. `node_feature(nodes1)`
5. `node_feature(nodes2)`

The retained training selection contains 182 rows: two singleton edge rows, 142
retained categorical states of the multi-atom path rule, and two 19-state unary
node-label rules.

## Phase 1 — Structural metrics

| Metric | Motif=False | True total | True full | DeFoG | Best |
|---|---:|---:|---:|---:|---|
| Degree MMD | 0.046812 ± 0.003388 | 0.033791 ± 0.000779 | **0.009845 ± 0.002415** | 0.014457 ± 0.013596 | True full |
| Clustering MMD | 0.224453 ± 0.087998 | 0.186363 ± 0.025161 | **0.026486 ± 0.009725** | 0.092806 ± 0.127809 | True full |
| Orbit MMD | 0.028590 ± 0.011803 | 0.007098 ± 0.004106 | **0.000777 ± 0.000347** | 0.014863 ± 0.021857 | True full |
| Spectral MMD | 0.026251 ± 0.005187 | 0.015621 ± 0.001618 | **0.011404 ± 0.001299** | 0.015081 ± 0.005804 | True full |
| Diameter MMD | 0.137099 ± 0.047510 | 0.056952 ± 0.031155 | **0.048548 ± 0.028277** | 0.087853 ± 0.057889 | True full |
| Sparsity error | 5.564e-8 ± 3.260e-9 | 1.544e-7 ± 1.040e-7 | 1.039e-7 ± 2.368e-8 | **7.082e-9 ± 9.327e-9** | DeFoG |
| Triangle metric | 6.259e-6 ± 4.238e-6 | 8.695e-6 ± 8.549e-7 | **6.660e-7 ± 3.953e-7** | 3.405e-6 ± 5.515e-6 | True full |
| Mean-edge absolute error | 2.4095 ± 0.5818 | **2.0286 ± 1.6896** | 5.8810 ± 2.1798 | 2.6143 ± 2.7852 | True total |
| Generated mean edges | 25.2429 ± 1.4828 | 24.7905 ± 2.1718 | 20.6048 ± 2.1798 | 29.1000 ± 2.7852 | Descriptive |

The fixed held-out reference has 26.4857 mean edges.

## Phase 2 — Random-GIN without native node features

The primary table uses degree, clustering, and square-clustering derived from
adjacency. The constant-input topology-only ablation follows in this phase.

These use three topology-derived node channels (degree, clustering coefficient,
and square clustering), not the original 19-way PTC node label, and 70 test
graphs. Each per-training-seed score is itself the mean of ten evaluator
initializations. This distinction is important: this table is neither the
constant-input topology-only ablation below nor an original-node-label test.
The motif=False cells were recomputed on 2026-09-13 against the same frozen
70-graph reference used by motif=True and DeFoG; the old per-run references are
superseded for these Random-GIN comparisons.

| Metric | Motif=False | True total | True full | DeFoG | Best |
|---|---:|---:|---:|---:|---|
| F1-PR | 0.789586 ± 0.043412 | 0.812762 ± 0.021276 | **0.937449 ± 0.014804** | 0.892656 ± 0.094403 | True full |
| Precision | 0.691905 ± 0.059270 | 0.711429 ± 0.017379 | **0.896190 ± 0.020914** | 0.833810 ± 0.129916 | True full |
| Recall | 0.962857 ± 0.028140 | 0.976190 ± 0.018862 | **0.985714 ± 0.005714** | 0.972381 ± 0.031826 | True full |
| MMD-RBF | 0.112566 ± 0.018035 | 0.098803 ± 0.004247 | **0.060967 ± 0.017321** | 0.078266 ± 0.066604 | True full |
| Linear MMD mean | 140.6255 ± 105.0137 | 238.4962 ± 355.9339 | **23.8260 ± 31.2683** | 152.6463 ± 255.9313 | True full |
| Linear MMD median | 24.0383 ± 16.4178 | 12.1161 ± 4.9753 | **6.8330 ± 4.6179** | 23.9552 ± 36.6093 | True full |
| Linear MMD trimmed mean | 41.1115 ± 28.9212 | 28.5998 ± 15.6772 | **7.3651 ± 4.5380** | 51.6319 ± 83.2827 | True full |

### F1-PR by training seed

| Method | Seed 0 | Seed 1 | Seed 2 | Aggregate |
|---|---:|---:|---:|---:|
| Motif=False | 0.830716 | 0.793839 | 0.744205 | 0.789586 ± 0.043412 |
| True total | 0.794950 | 0.807014 | 0.836321 | 0.812762 ± 0.021276 |
| True full | 0.923626 | 0.935650 | 0.953071 | **0.937449 ± 0.014804** |
| DeFoG | 0.959217 | 0.934136 | 0.784615 | 0.892656 ± 0.094403 |

DeFoG seed 2 is substantially weaker. It remains included in the primary
aggregation and must not be silently removed.

### Constant-input topology-only ablation

This is the requested no-node-feature result. Every node is assigned the same
constant scalar input and no original PTC node label is supplied. Consequently,
the evaluator can distinguish graphs only through adjacency/message passing.
The generated and reference collections contain 70 graphs, and every
generator-training seed is evaluated over ten Random-GIN initializations.

| Metric | Motif=False | True total | True full | DeFoG |
|---|---:|---:|---:|---:|
| F1-PR | 0.841825 ± 0.035538 | 0.913783 ± 0.008277 | **0.977390 ± 0.012908** | 0.925015 ± 0.082789 |
| Precision | 0.748571 ± 0.041625 | 0.858571 ± 0.018571 | **0.963333 ± 0.018645** | 0.891429 ± 0.121655 |
| Recall | 0.968571 ± 0.029312 | 0.981905 ± 0.020817 | **0.992381 ± 0.007047** | 0.970476 ± 0.027005 |
| MMD-RBF | 0.082844 ± 0.007799 | 0.079498 ± 0.023859 | 0.067382 ± 0.021052 | **0.043710 ± 0.024622** |

### Topology-only results by generator-training seed

| Method | Seed 0 F1-PR | Seed 1 F1-PR | Seed 2 F1-PR | Aggregate F1-PR |
|---|---:|---:|---:|---:|
| Motif=False | 0.865300 | 0.859236 | 0.800938 | 0.841825 ± 0.035538 |
| True total | 0.915040 | 0.904949 | 0.921360 | 0.913783 ± 0.008277 |
| True full | 0.964133 | 0.978120 | 0.989917 | **0.977390 ± 0.012908** |
| DeFoG | 0.975545 | 0.970029 | 0.829472 | 0.925015 ± 0.082789 |

| Method | Seed 0 MMD-RBF | Seed 1 MMD-RBF | Seed 2 MMD-RBF | Aggregate MMD-RBF |
|---|---:|---:|---:|---:|
| Motif=False | 0.073960 | 0.088567 | 0.086004 | 0.082844 ± 0.007799 |
| True total | 0.064020 | 0.067499 | 0.106974 | 0.079498 ± 0.023859 |
| True full | 0.085813 | 0.071893 | 0.044440 | 0.067382 ± 0.021052 |
| DeFoG | 0.029010 | 0.029984 | 0.072135 | **0.043710 ± 0.024622** |

The GraphVAE values originate from the archived local adjacency-only evaluator;
the DeFoG values originate from its explicitly named `topology_control` mode.
Both suppress original node labels by using constant node inputs, but they are
separate evaluator implementations. Therefore this is a useful topology-only
comparison, while a future perfectly frozen cross-generator rerun would be the
strongest paper protocol.

## Phase 3 — Random-GIN with native node features

| Metric | DeFoG mean ± training-seed SD |
|---|---:|
| F1-PR | 0.892879 ± 0.028895 |
| Precision | 0.849048 ± 0.041829 |
| Recall | 0.944762 ± 0.019024 |
| MMD-RBF | 0.033019 ± 0.005299 |
| Linear MMD | 196.6861 ± 71.5530 |

No identically frozen decoded-node GraphVAE evaluation is archived, so those
cells are deliberately not fabricated.

## Phase 4 — Motif-correlation metrics on the identical pruned rule set

The primary metric is hard pruned-state TV on the exact training support.

Reference and generated collections both contain 240 graphs. Counts are summed
over each entire collection, normalized once across the 142 retained states of
the trained multi-atom path rule, and compared by TV.

For retained state `j`, let `C_gen(j) = sum_g c_gen(g,j)` and
`C_train(j) = sum_g c_train(g,j)`. Define
`p_gen(j) = C_gen(j) / sum_k C_gen(k)` and
`p_train(j) = C_train(j) / sum_k C_train(k)`, where both `j` and `k` range over
the identical 142-state pruned list. The reported error is
`TV = 0.5 * sum_j |p_gen(j) - p_train(j)|`. No per-graph averaging occurs before
normalization.

The eligible rule is
`node_feature(nodes0) AND edges(nodes0,nodes1) AND edges(nodes1,nodes2) AND node_feature(nodes1) AND node_feature(nodes2)`.

| Method | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| Motif=False | 0.143111 | 0.122318 | 0.127128 | 0.130852 ± 0.010885 |
| True total | 0.121472 | 0.112145 | 0.138020 | 0.123879 ± 0.013105 |
| True full | 0.072469 | **0.089040** | **0.103559** | 0.088356 ± 0.015557 |
| DeFoG | **0.051566** | 0.091106 | 0.105292 | **0.082654 ± 0.027842** |

True full improves over motif=False by 32.5%. DeFoG's aggregate is 6.5% lower
than true full, driven by DeFoG seed 0; true full beats DeFoG on seeds 1 and 2.

## Soft pruned-state TV

This uses GraphVAE decoder probabilities before hard discretization. DeFoG has
no comparable saved soft decoder output.

| Method | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| Motif=False | **0.105782** | **0.108908** | **0.114587** | **0.109759 ± 0.004464** |
| True total | 0.116505 | 0.119234 | 0.134284 | 0.123341 ± 0.009575 |
| True full | 0.130009 | 0.136078 | 0.149619 | 0.138569 ± 0.010039 |
| DeFoG | N/A | N/A | N/A | N/A |

The calibrated-Gaussian training objective is not normalized TV, explaining why
motif=True need not win this soft distribution metric.

## Supplementary unpruned complete-state distribution (not the primary motif-correlation metric)

This is deliberately excluded from the primary motif-correlation claim. It is
distinct from pruned-state TV because it evaluates the complete 6,899-state
FactorBase universe. The eligible multi-atom path rule contains 6,859 states.

| Method | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| Motif=False | 0.149207 | 0.143636 | 0.150217 | 0.147687 ± 0.003544 |
| True total | 0.143901 | 0.133083 | 0.163325 | 0.146770 ± 0.015324 |
| True full | 0.092486 | 0.117715 | 0.133861 | 0.114687 ± 0.020853 |
| DeFoG | **0.070306** | **0.090171** | **0.086112** | **0.082196 ± 0.010495** |

## Other exact retained-rule metrics

All hard metrics use the exact 182 retained rows and 240-graph train-reference
protocol unless noted.

| Metric | Motif=False | True total | True full | DeFoG | Best |
|---|---:|---:|---:|---:|---|
| Hard Gaussian NLL | -1.101128 ± 0.137274 | -1.106701 ± 0.135878 | **-1.159071 ± 0.107546** | -1.147343 ± 0.206609 | True full |
| Aggregate count RMSE | 398.145 ± 36.704 | 398.398 ± 118.949 | 693.306 ± 29.796 | **190.017 ± 156.988** | DeFoG |
| Aggregate count MAE | 94.119 ± 5.393 | 91.172 ± 20.190 | 139.018 ± 5.359 | **34.016 ± 22.461** | DeFoG |
| Mean-vector RMSE | 1.65894 ± 0.15293 | 1.65999 ± 0.49562 | 2.88878 ± 0.12415 | **0.79174 ± 0.65412** | DeFoG |
| Standardized count RMSE | 0.30662 ± 0.01839 | 0.32538 ± 0.12956 | 0.18515 ± 0.00664 | **0.08706 ± 0.02810** | DeFoG |
| Log1p count RMSE | 0.18517 ± 0.00065 | 0.18536 ± 0.02153 | 0.17166 ± 0.00644 | **0.08426 ± 0.02050** | DeFoG |
| Trimmed log1p-Wasserstein | 0.01789 ± 0.00038 | 0.01531 ± 0.00092 | 0.01338 ± 0.00116 | **0.01217 ± 0.00275** | DeFoG |
| Median log1p-Wasserstein | 0.00498 ± 0.00069 | 0.00498 ± 0.00069 | **0.00401 ± 0.00098** | 0.00472 ± 0.00099 | True full |

DeFoG Gaussian graph-wise residuals are node-count-sorted diagnostics rather
than reconstruction-paired quantities. Aggregate counts, normalized TV, and
Wasserstein summaries do not depend on graph ordering.

## Paper-ready conclusion

PTC provides strong evidence for full-matrix motif regularization on general
graph fidelity. True full beats motif=False and DeFoG on the primary held-out
Random-GIN F1-PR, precision, recall, MMD-RBF, and most structural metrics. It
also reduces exact hard pruned-state TV by 32.5% relative to motif=False and
beats DeFoG on that metric in two of three seeds. DeFoG retains a small 6.5%
aggregate advantage in hard pruned-state TV and a clearer advantage in complete-
state TV and raw count errors. True full's main failure mode is edge-count
under-generation.

## Consolidated archive layout

Archive host: `cs-cl-09.cmpt.sfu.ca`

Archive root: `/local-scratch2/mirzaei/PTC`

- `graphvae_motif_false/setting_01`: all motif=False runs, models, logs, configs,
  generated graphs, and evaluation artifacts.
- `graphvae_motif_true/ptc/{total_count,full_matrix}/seed_{0,1,2}`: all motif=True
  runs and selected models.
- `defog/frozen_campaign`: training runs, checkpoints, generated graphs, frozen
  reference collections, manifests, and Random-GIN outputs.
- `defog/full_metrics`: 240-graph generation, structural metrics, exact-rule
  counts, and combined machine-readable report.
- `evaluations/pruned_rule_metrics`: GraphVAE exact retained-rule evaluations.
- `evaluations/full_state_correlation`: complete-state correlation outputs.
- `evaluations/pruned_state_tv_hard`: hard train/test-reference pruned-state TV.
- `evaluations/pruned_state_tv_soft`: soft GraphVAE pruned-state TV.
- `reports`: previous PTC reports retained for provenance.
- `reproducibility`: scripts used for hard and soft pruned-state TV.

## Primary source artifacts

- Motif=False original: `/local-scratch2/new/gather/datasets/ptc/setting_01/seed_{0,1,2}`
- Motif=True original: `/local-scratch2/mirzaei/motif_true_clean_20260906/ptc/{total_count,full_matrix}/seed_{0,1,2}`
- DeFoG original: `/local-scratch2/mirzaei/defog_ptc_frozen_20260906/jobs/ptc/seed_{0,1,2}`
- Hard pruned TV: `/local-scratch2/mirzaei/ptc_pruned_state_tv_train_test_20260909`
- Soft pruned TV: `/local-scratch2/mirzaei/ptc_pruned_state_tv_soft_20260909`
- Full combined metrics: `/local-scratch2/mirzaei/defog_ptc_full_metrics_20260907`
