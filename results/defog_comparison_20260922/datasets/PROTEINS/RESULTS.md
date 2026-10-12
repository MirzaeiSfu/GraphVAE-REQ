# PROTEINS complete verified report: best Motif=True versus Motif=False versus DeFoG

## Corrected independent-seed update (2026-09-23)

This section supersedes the structural and RandomGIN tables below. It replaces the duplicated historical DeFoG generations with independent corrected outputs. Every method uses the same serialized 209-graph reference and 209 generated graphs per seed. RandomGIN uses 10 evaluator seeds; only topology-control and decoded-node modes are reported because the archived collections do not provide compatible native edge attributes.

| Method | Degree MMD | Clustering MMD | Orbit MMD | Spectral MMD | Diameter MMD | Triangle MMD | Sparsity MMD | Mean-edge error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Motif=False | 0.043587 | 0.029191 | 0.024905 | 0.022455 | 0.046686 | 1.146e-4 | 3.310e-9 | 13.199 |
| Motif=True full, alpha 0.03 | 0.023801 | 0.023050 | **0.003685** | 0.012978 | **0.023753** | **3.192e-5** | 1.451e-7 | 14.101 |
| Corrected DeFoG | **0.014637** | **0.018399** | 0.040145 | **0.007790** | 0.024753 | 7.700e-5 | **2.926e-9** | **4.203** |

| Mode | Method | F1-PR | Precision | Recall | MMD-RBF |
|---|---|---:|---:|---:|---:|
| Topology control | Motif=False | 0.904366 | 0.908772 | 0.901595 | 0.088519 |
| Topology control | Motif=True full | 0.953490 | 0.939553 | **0.968262** | 0.041992 |
| Topology control | Corrected DeFoG | **0.967054** | **0.986762** | 0.948485 | **0.033753** |
| Decoded node | Motif=False | 0.871237 | 0.949442 | 0.806061 | 0.052647 |
| Decoded node | Motif=True full | 0.932259 | 0.958852 | 0.908293 | 0.025910 |
| Decoded node | Corrected DeFoG | **0.959695** | **0.978150** | **0.942105** | **0.019958** |

Motif=True topology F1-PR rises from the historical `0.9186` to `0.9535`, but corrected DeFoG remains higher at `0.9671`. Motif=True wins topology recall, orbit MMD, triangle MMD, and slightly diameter MMD. The corrected independent DeFoG outputs remove the earlier duplicate-seed uncertainty.

This 2026-09-17 copy gathers the current completed three-seed campaign. It is
kept separate from the still-running AIDs and MUTAG campaigns.

Verified 2026-09-14. The primary GraphVAE-REQ result is the selected
full-matrix `alpha_motif=0.03` campaign, with three generator seeds. The
Random-GIN and structural comparisons use the fixed 209-graph reference and
209 generated graphs per method/seed. The primary motif-correlation metric is
restricted to the identical 19 retained training states across all five pruned
rules; historical 35/27-state complete-state results are not used for the
motif-correlation claim.

Fairness verdict: held-out evaluation is matched, but DeFoG is a different
architecture/objective. Its motif-correlation generation collection contains
210 graphs per seed versus 731 train-matched GraphVAE graphs, so normalized TV
is comparable in definition but has different sampling variance. Also, DeFoG
seeds 0 and 1 have byte-identical generated files; its three-seed SD is not a
clean independent-generation uncertainty estimate and must be disclosed.

Generated 2026-09-13. GraphVAE Motif=True uses the selected full-matrix motif
weight 0.03. Values are mean +/- sample SD over nominal generator seeds 0, 1,
and 2. Higher is better for F1-PR, precision, and recall. Lower is better for
MMD, error, count-distance, and total-variation metrics.

## Protocol

- Test metrics use the same fixed PROTEINS test adjacency collection.
- The empty reference entry is removed and all methods are deterministically
  matched to 209 reference and 209 generated graphs.
- Motif=False test graphs were freshly regenerated from its three saved
  best-validation checkpoints using this fixed node-count sequence.
- Random-GIN uses topology-derived node features and ten repeats per generator
  seed. The table aggregates the repeat means across generator seeds.
- Motif correlation uses the same 731 training graphs as reference and the
  exact 19 states across all 5 rules retained by Motif=True training pruning.

## Phase 1 — Structural metrics

| Metric | Motif=False | Motif=True full, alpha 0.03 | DeFoG | Best |
|---|---:|---:|---:|---|
| Degree MMD | 0.037520 +/- 0.005647 | 0.021855 +/- 0.003040 | **0.011490 +/- 0.006438** | DeFoG |
| Clustering MMD | 0.025883 +/- 0.000997 | 0.021611 +/- 0.001361 | **0.015506 +/- 0.002097** | DeFoG |
| Orbit MMD | 0.024463 +/- 0.004194 | **0.010732 +/- 0.002849** | 0.019636 +/- 0.013139 | Motif=True |
| Spectral MMD | 0.021577 +/- 0.001381 | 0.013285 +/- 0.002163 | **0.005697 +/- 0.000684** | DeFoG |
| Diameter MMD | 0.097364 +/- 0.010305 | 0.042071 +/- 0.021942 | **0.033008 +/- 0.010849** | DeFoG |
| Triangle MMD | 2.549e-4 +/- 9.206e-5 | **3.051e-5 +/- 2.545e-5** | 1.432e-4 +/- 2.287e-5 | Motif=True |
| Sparsity MMD | 5.550e-7 +/- 1.381e-7 | 3.439e-7 +/- 2.162e-7 | **1.384e-7 +/- 9.045e-9** | DeFoG |
| Generated mean edges | 34.336523 +/- 1.175713 | 38.843700 +/- 2.840036 | **58.103668 +/- 2.011057** | DeFoG |
| Reference mean edges | 60.698565 | 60.698565 | 60.698565 | -- |
| Mean-edge absolute error | 26.362041 +/- 1.175713 | 21.854864 +/- 2.840036 | **2.594896 +/- 2.011057** | DeFoG |

## Phase 2 — Random-GIN without native node features

This evaluator uses topology-derived channels only; it does not use the
categorical PROTEINS node label.

| Metric | Motif=False | Motif=True full, alpha 0.03 | DeFoG | Best |
|---|---:|---:|---:|---|
| F1-PR | 0.822022 +/- 0.032039 | 0.918386 +/- 0.022650 | **0.942607 +/- 0.012067** | DeFoG |
| Precision | 0.941946 +/- 0.002455 | **0.942743 +/- 0.007166** | 0.925199 +/- 0.013812 | Motif=True |
| Recall | 0.731100 +/- 0.051437 | 0.897129 +/- 0.049254 | **0.960925 +/- 0.010221** | DeFoG |
| MMD-RBF | 0.101706 +/- 0.009499 | 0.060668 +/- 0.005350 | **0.014566 +/- 0.000718** | DeFoG |
| Linear MMD mean | 10.651035 +/- 1.322440 | **8.470177 +/- 2.197207** | 9946.557803 +/- 4845.337788 | Motif=True |
| Linear MMD median | 10.877461 +/- 1.522373 | 8.867824 +/- 2.301717 | **1.359655 +/- 0.539198** | DeFoG |
| Linear MMD trimmed | 10.666796 +/- 1.334547 | **8.517218 +/- 2.197125** | 31.508558 +/- 15.340562 | Motif=True |

DeFoG's linear-MMD mean is dominated by a very large outlying repeat; its
median is much more favorable. This is why mean, median, and trimmed mean are
all retained.

## Phase 3 — Random-GIN with native node features

A matched three-way evaluation using the original categorical PROTEINS node
label is not archived for this selected `alpha_motif=0.03` campaign. It is
therefore reported as **N/A**, rather than mixing results from an older
checkpoint campaign or treating topology-derived channels as native features.

## Phase 4 — Motif-correlation metrics on the identical pruned rule set

The corrected primary metric sums counts across each graph collection,
normalizes the retained states within each of all 5 pruned rules, computes TV
per rule, and reports the macro-average. GraphVAE generated 731 train-matched
graphs per seed; DeFoG has 210 per seed.

| Method | Seed 0 | Seed 1 | Seed 2 | Mean +/- SD | Improvement vs false |
|---|---:|---:|---:|---:|---:|
| Motif=False | 0.026235 | 0.019699 | 0.019291 | 0.021742 +/- 0.003897 | -- |
| **Motif=True full, alpha 0.03** | **0.018732** | 0.023354 | **0.008935** | **0.017007 +/- 0.007363** | **21.78%** |
| DeFoG | 0.012013 | **0.012013** | 0.036805 | 0.020277 +/- 0.014314 | 6.74% |

Two rules contain one retained state and necessarily contribute zero TV. The
macro-average over the 3 informative multi-state rules is 0.036236 +/- 0.006495
for Motif=False, 0.028345 +/- 0.012271 for Motif=True, and 0.033795 +/- 0.023857
for DeFoG. DeFoG seeds 0 and 1 are byte-identical generated files.

The previously reported 35-state/27-state multi-atom values (0.124721,
0.071084, and 0.047639) are superseded for the requested all-pruned-rules
comparison. They remain archived as historical complete-state diagnostics.

Additional GraphVAE-only TV diagnostics:

| Metric | Motif=False | Motif=True full, alpha 0.03 | Better |
|---|---:|---:|---|
| Mean per-graph state TV | 0.127842 +/- 0.010162 | **0.064605 +/- 0.002993** | Motif=True |
| Paired-graph TV mean | **0.453708 +/- 0.012948** | 0.504546 +/- 0.015753 | Motif=False |

Paired-graph TV is secondary because graph index does not establish semantic
pairing between independently generated and reference graphs. It is not
defined for the 210-versus-731 DeFoG comparison.

## Complete GraphVAE motif-count diagnostics

These diagnostics require train-matched generated/reference graph pairs and
therefore are not defined for the archived DeFoG collection.

| View and metric | Motif=False | Motif=True full, alpha 0.03 | Better |
|---|---:|---:|---|
| Soft mean-vector count RMSE | **23.825367 +/- 0.778134** | 28.161523 +/- 1.538299 | False |
| Soft aggregate count RMSE | **17416.343245 +/- 568.816131** | 20586.073576 +/- 1124.496372 | False |
| Soft mean absolute difference | **9.201342 +/- 0.433451** | 12.248852 +/- 0.763699 | False |
| Soft aggregate mean absolute difference | **6726.181038 +/- 316.852397** | 8953.910979 +/- 558.263765 | False |
| Soft standardized RMSE | **0.168736 +/- 0.008954** | 0.247757 +/- 0.016828 | False |
| Soft relative distance | **0.388745 +/- 0.012696** | 0.459496 +/- 0.025100 | False |
| Soft log1p mean-vector RMSE | 0.433316 +/- 0.044707 | **0.418190 +/- 0.051135** | Motif=True |
| Soft Wasserstein mean | **0.200160 +/- 0.015617** | 0.280797 +/- 0.039959 | False |
| Soft Wasserstein median | **0.126421 +/- 0.016041** | 0.247384 +/- 0.059821 | False |
| Soft robust Wasserstein | **0.170640 +/- 0.016146** | 0.249852 +/- 0.037590 | False |
| Soft paired RMSE mean | **48.299183 +/- 0.815941** | 51.239973 +/- 0.529457 | False |
| Soft paired RMSE median | **35.051184 +/- 2.292423** | 37.244685 +/- 0.715997 | False |
| Soft paired RMSE SD | **42.469545 +/- 0.646460** | 44.491281 +/- 0.979450 | False |
| Soft top-1 squared-error share | 0.434320 +/- 0.010385 | 0.394542 +/- 0.014700 | diagnostic |
| Soft top-5 squared-error share | 0.985632 +/- 0.003268 | 0.953632 +/- 0.006651 | diagnostic |
| Hard mean-vector count RMSE | **17.066371 +/- 0.596277** | 20.342370 +/- 1.062551 | False |
| Hard aggregate count RMSE | **12475.517341 +/- 435.878364** | 14870.272404 +/- 776.724882 | False |
| Hard mean absolute difference | **7.651228 +/- 0.356189** | 9.944290 +/- 0.552094 | False |
| Hard aggregate mean absolute difference | **5593.047619 +/- 260.373868** | 7269.276190 +/- 403.580798 | False |
| Hard standardized RMSE | **0.253573 +/- 0.008737** | 0.323059 +/- 0.016702 | False |
| Hard relative distance | **0.420646 +/- 0.014697** | 0.501392 +/- 0.026189 | False |
| Hard log1p mean-vector RMSE | **0.441908 +/- 0.036369** | 0.477852 +/- 0.040652 | False |
| Hard Wasserstein mean | **0.192878 +/- 0.011068** | 0.270632 +/- 0.041348 | False |
| Hard Wasserstein median | **0.088127 +/- 0.016647** | 0.104251 +/- 0.018546 | False |
| Hard robust Wasserstein | **0.162181 +/- 0.012005** | 0.221270 +/- 0.035725 | False |
| Hard paired RMSE mean | **34.429272 +/- 0.569196** | 36.241620 +/- 0.278814 | False |
| Hard paired RMSE median | **24.917369 +/- 1.586087** | 26.668088 +/- 0.669487 | False |
| Hard paired RMSE SD | **29.858565 +/- 0.465024** | 31.314006 +/- 0.607797 | False |
| Hard top-1 squared-error share | 0.310212 +/- 0.010439 | 0.286093 +/- 0.015191 | diagnostic |
| Hard top-5 squared-error share | 0.969666 +/- 0.003487 | 0.927287 +/- 0.008165 | diagnostic |

This apparent tension is real: alpha 0.03 substantially improves normalized
joint-state correlation TV, while Motif=False is better on most absolute-count
distances. Correlation evaluates relative state proportions; count RMSE also
penalizes absolute motif-count scale and graph density.

## Conclusion

- DeFoG wins F1-PR, recall, RBF MMD, five of seven structural MMDs, and
  edge-count fidelity.
- Motif=True alpha 0.03 wins precision, orbit MMD, triangle MMD, and robust
  linear-MMD summaries. It has the best all-pruned-rules motif-correlation TV,
  improving by 21.78% over false.
- Motif=False wins most absolute motif-count diagnostics but is worst on the
  primary aggregate motif-correlation TV and most graph-quality metrics.

## Artifacts

- Comparison root: `/local-scratch2/mirzaei/proteins_3way_comparison_20260913`
- Motif=True full-state outputs: `/local-scratch2/mirzaei/proteins_motif003_fullstate_trainref_20260913`
- Motif=False full-state outputs: `/local-scratch2/mirzaei/proteins_false_fullstate_trainref_20260913`
- DeFoG full-state outputs: `/local-scratch2/mirzaei/proteins_defog_3seed_20260910/trainref_fullstate_20260913`
- Corrected all-pruned-rules outputs: `/local-scratch2/mirzaei/proteins_all_pruned_rules_correlation_20260913`
