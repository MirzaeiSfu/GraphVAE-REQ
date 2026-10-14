# DeFoG versus GraphVAE best-seed third-party Random-GIN

This report compares the supplied DeFoG training-seed-0 collection with the
best-F1 training seed from each GraphVAE condition:

- motif=False;
- motif=True with total-count motifs;
- motif=True with full-matrix motifs.

The comparison uses the same external third-party Random-GIN implementation,
10 evaluator initializations (seeds 0--9), and `k=5`. Two feature modes are
reported:

- **With features (`decoded_node`)**: adjacency plus MUTAG's 7-category or
  PROTEINS' 3-category categorical node representation. Edge features are not
  used.
- **Without dataset features (`topology_control`)**: adjacency with a constant
  node channel. This removes the dataset's categorical node information.

For each GraphVAE condition, the seed with the highest mean F1-PR in the given
feature mode is selected. Every other metric in that row comes from that same
seed; seeds are not independently cherry-picked per metric. Values are mean
`+/-` population SD across the 10 evaluator initializations.

## MUTAG -- with categorical node features

| Method | Selected training seed | F1-PR up | Precision up | Recall up | MMD-RBF down | MMD-linear down |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphVAE motif=False | 1 | 0.750287 +/- 0.064350 | 0.641026 +/- 0.093862 | 0.917949 +/- 0.022353 | 0.082832 +/- 0.011131 | 137.834321 +/- 349.791693 |
| GraphVAE motif=True total | 1 | 0.774934 +/- 0.038999 | 0.666667 +/- 0.059584 | 0.930769 +/- 0.028205 | 0.139216 +/- 0.013833 | 288.190524 +/- 693.254224 |
| GraphVAE motif=True full | 1 | 0.836002 +/- 0.028609 | 0.735897 +/- 0.041424 | **0.969231 +/- 0.015385** | 0.102563 +/- 0.007161 | 23.258702 +/- 17.583345 |
| **DeFoG** | **0** | **0.952008 +/- 0.011265** | **0.964103 +/- 0.028553** | 0.941026 +/- 0.016418 | **0.049155 +/- 0.000596** | **3.346058 +/- 0.881286** |

DeFoG has the highest F1-PR and precision and the lowest two MMD values. The
GraphVAE full-matrix seed has the highest recall. Relative to the strongest
GraphVAE F1-PR row (full matrix), DeFoG's F1-PR is about 13.9% higher.

## MUTAG -- without dataset node features

| Method | Selected training seed | F1-PR up | Precision up | Recall up | MMD-RBF down | MMD-linear down |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphVAE motif=False | 0 | 0.796040 +/- 0.036538 | 0.671795 +/- 0.048379 | 0.979487 +/- 0.019188 | 0.075557 +/- 0.004064 | 12.197342 +/- 4.583551 |
| GraphVAE motif=True total | 2 | 0.888206 +/- 0.079709 | 0.807692 +/- 0.123104 | **1.000000 +/- 0.000000** | 0.132191 +/- 0.031692 | 17.350779 +/- 3.898572 |
| GraphVAE motif=True full | 0 | 0.918390 +/- 0.062913 | 0.864103 +/- 0.110614 | 0.989744 +/- 0.012561 | 0.067148 +/- 0.019786 | 8.157674 +/- 2.620936 |
| **DeFoG** | **0** | **0.981792 +/- 0.011893** | **0.992308 +/- 0.011750** | 0.971795 +/- 0.021299 | **0.045321 +/- 0.005850** | **0.630524 +/- 0.290043** |

DeFoG has the highest F1-PR and precision and the lowest MMD values. GraphVAE
total-count has perfect recall. Relative to the strongest GraphVAE F1-PR row
(full matrix), DeFoG's F1-PR is about 6.9% higher.

## PROTEINS -- with categorical node features

| Method | Selected training seed | F1-PR up | Precision up | Recall up | MMD-RBF down | MMD-linear down |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphVAE motif=False | 2 | 0.922807 +/- 0.014873 | 0.947143 +/- 0.031619 | 0.900476 +/- 0.016000 | 0.043669 +/- 0.004040 | **2.915424 +/- 0.273549** |
| GraphVAE motif=True total | 0 | 0.937804 +/- 0.018934 | 0.957416 +/- 0.024062 | 0.919139 +/- 0.017960 | 0.045243 +/- 0.002648 | 7.084909 +/- 1.826497 |
| GraphVAE motif=True full | 1 | 0.921734 +/- 0.012615 | 0.959809 +/- 0.017928 | 0.887081 +/- 0.022463 | 0.024364 +/- 0.002476 | 7.237224 +/- 7.106248 |
| **DeFoG** | **0** | **0.965866 +/- 0.006155** | **0.986667 +/- 0.011625** | **0.946190 +/- 0.014762** | **0.022352 +/- 0.001650** | 3.244922 +/- 0.636469 |

DeFoG has the highest F1-PR, precision, and recall and the lowest MMD-RBF.
GraphVAE motif=False has a slightly lower linear MMD. Relative to the strongest
GraphVAE F1-PR row (total count), DeFoG's F1-PR is about 3.0% higher.

## PROTEINS -- without dataset node features

| Method | Selected training seed | F1-PR up | Precision up | Recall up | MMD-RBF down | MMD-linear down |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphVAE motif=False | 2 | 0.939089 +/- 0.025913 | 0.916667 +/- 0.032103 | 0.963333 +/- 0.030717 | 0.084005 +/- 0.016129 | 2.891218 +/- 0.592613 |
| GraphVAE motif=True total | 0 | 0.939643 +/- 0.025180 | 0.894737 +/- 0.047172 | **0.990909 +/- 0.005434** | 0.071778 +/- 0.013148 | 8.153193 +/- 1.542170 |
| GraphVAE motif=True full | 1 | 0.908136 +/- 0.034059 | 0.882297 +/- 0.066684 | 0.938756 +/- 0.012255 | 0.044387 +/- 0.010070 | 5.749703 +/- 1.045480 |
| **DeFoG** | **0** | **0.963059 +/- 0.009935** | **0.979048 +/- 0.012989** | 0.947619 +/- 0.009283 | **0.025491 +/- 0.002512** | **0.572796 +/- 0.141098** |

DeFoG has the highest F1-PR and precision and the lowest MMD values. GraphVAE
total-count has the highest recall. Relative to the strongest GraphVAE F1-PR
row (total count), DeFoG's F1-PR is about 2.5% higher.

## Important limitations

1. **Best-seed selection is descriptive.** Selecting a seed by test F1-PR is
   not valid validation-only model selection. This report is an explicitly
   requested best-observed-seed comparison, not the primary statistical
   estimate. The primary result should average three training seeds.
2. **DeFoG has only one supplied training seed.** Selecting the best of three
   GraphVAE seeds actually gives GraphVAE a selection advantage, but it does
   not substitute for DeFoG training-seed variance.
3. **MUTAG reference mismatch.** GraphVAE motif=False uses loader-state `None`
   and split fingerprint `318497fe...`; motif=True and the canonical DeFoG
   evaluation use loader seed 123 and fingerprint `9f44527a...`.
4. **PROTEINS reference mismatch.** GraphVAE motif=False and the saved DeFoG
   preview use 210-graph historical collections. The repaired motif=True
   package accepts 209 graphs and has fingerprint `5e903aeb...`.
5. **Saved DeFoG provenance is incomplete.** Its training seed is 0, but its
   generation seed was not recorded and the supplied checkpoint is not proven
   to be selected using validation data only.
6. **Feature-mode meaning.** `topology_control` here is the newer explicit
   constant-node-channel ablation. It is not the older Kia evaluator that
   constructs degree, clustering, and square-clustering channels.

## Sources

- GraphVAE per-seed feature-aware results:
  `/local-scratch2/mirzaei/node_feature_evaluation_20260901/`
- GraphVAE aggregate report:
  `/local-scratch2/mirzaei/node_feature_evaluation_20260901/NODE_FEATURE_RANDOM_GIN_RESULTS.md`
- DeFoG raw preview results:
  `runs/defog/preview_random_gin/{mutag,proteins}/evaluation.json`
- Frozen comparison protocol:
  `baselines/defog/frozen_eval/manifest.yaml`
