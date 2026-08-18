# PROTEINS attributed Random-GIN comparison

## Scope

This report evaluates the archived `best_validation_mmd_model` checkpoints
with the feature-aware DGL Random-GIN evaluator.

- Dataset split: paper 70/10/20 split with split seed 123
- Test graphs: 210
- Training seeds: 0, 1, and 2
- Generation seed: 12345 for every checkpoint
- Random-GIN initializations: 10 per checkpoint, base seed 0
- Precision/recall neighbourhood: 5
- Node feature dimension: 3 categorical channels
- Edge feature dimension: 0

PROTEINS has no edge attributes in this cache, so the primary attributed mode
is `decoded_node`. Values below are the mean and sample standard deviation
across the three training-seed means.

All nine evaluations used logically identical reference graphs. The saved DGL
serialization bytes differ between files, but their ordered topology and
feature tensors have the same SHA-256 digest:
`975d0b9c001a6b60be9b68d24623a5bc7326f87a9ab5850949ffff2c8c7c5a0d`.

## Aggregate results

Higher F1-PR, precision, and recall are better. Lower MMD-RBF is better.

| Model | Mode | F1-PR | Precision | Recall | MMD-RBF |
| --- | --- | ---: | ---: | ---: | ---: |
| GraphVAE | topology control | 0.9123 ± 0.0446 | 0.9130 ± 0.0038 | 0.9152 ± 0.0870 | 0.08804 ± 0.01304 |
| GraphVAE | decoded node | 0.8924 ± 0.0444 | 0.9451 ± 0.0146 | 0.8475 ± 0.0686 | 0.05030 ± 0.00971 |
| GraphVAE + motif, constant temperature | topology control | 0.9364 ± 0.0116 | 0.9173 ± 0.0099 | 0.9575 ± 0.0169 | 0.02610 ± 0.03074 |
| GraphVAE + motif, constant temperature | decoded node | 0.9232 ± 0.0284 | 0.9498 ± 0.0122 | 0.8997 ± 0.0433 | 0.02586 ± 0.01226 |
| GraphVAE + motif, annealed temperature | topology control | 0.9418 ± 0.0037 | 0.9230 ± 0.0093 | 0.9629 ± 0.0102 | 0.02661 ± 0.02984 |
| GraphVAE + motif, annealed temperature | decoded node | **0.9256 ± 0.0262** | **0.9548 ± 0.0073** | 0.8995 ± 0.0436 | 0.03378 ± 0.01311 |

The lowest attributed MMD-RBF is **0.02586 ± 0.01226** from the
constant-temperature motif variant.

## Primary attributed results by training seed

| Model | Seed | Decoded-node F1-PR | Decoded-node MMD-RBF |
| --- | ---: | ---: | ---: |
| GraphVAE | 0 | 0.84147 | 0.06144 |
| GraphVAE | 1 | 0.91293 | 0.04578 |
| GraphVAE | 2 | 0.92281 | 0.04367 |
| GraphVAE + motif, constant temperature | 0 | 0.89154 | 0.01208 |
| GraphVAE + motif, constant temperature | 1 | 0.94647 | 0.02996 |
| GraphVAE + motif, constant temperature | 2 | 0.93166 | 0.03555 |
| GraphVAE + motif, annealed temperature | 0 | 0.89616 | 0.04837 |
| GraphVAE + motif, annealed temperature | 1 | 0.94647 | 0.02996 |
| GraphVAE + motif, annealed temperature | 2 | 0.93414 | 0.02300 |

The constant- and annealed-temperature seed-1 archives contain the exact same
checkpoint (SHA-256
`7272692f8decc9553817b304413cb14c6cf80c824d35c9f9ad99c99f7d0e9591`).
Their identical seed-1 results are therefore not independent evidence about
the temperature choice.

## Paired differences from GraphVAE

The following differences pair models by training-seed number:

| Motif variant | Attributed F1-PR difference | Attributed MMD-RBF difference |
| --- | ---: | ---: |
| Constant temperature minus GraphVAE | +0.03082 ± 0.02074 | -0.02443 ± 0.02193 |
| Annealed temperature minus GraphVAE | +0.03319 ± 0.02168 | -0.01652 ± 0.00385 |

Both motif variants have better absolute attributed-node results than
GraphVAE: F1-PR is higher and MMD-RBF is lower for every aggregate comparison.
With only three training seeds, these values should be treated as descriptive
rather than a formal significance claim.

## Does adding decoded node information help beyond topology?

| Model | Decoded-node F1 minus topology F1 | Topology MMD-RBF minus decoded-node MMD-RBF |
| --- | ---: | ---: |
| GraphVAE | -0.01994 ± 0.00399 | +0.03774 ± 0.00523 |
| Motif, constant temperature | -0.01320 ± 0.02056 | +0.00024 ± 0.02732 |
| Motif, annealed temperature | -0.01621 ± 0.02734 | -0.00717 ± 0.03468 |

This ablation is important:

- The motif checkpoints improve both topology-control and attributed-node
  scores relative to GraphVAE.
- Adding decoded node labels lowers F1-PR relative to topology control for all
  three model groups.
- GraphVAE's decoded nodes substantially improve MMD-RBF over topology, while
  the motif variants show no stable additional MMD-RBF benefit from node
  labels.

Therefore, these results support the claim that motif training improves the
overall attributed graph distribution, but they do **not** yet show that the
advantage comes specifically from better decoded node labels rather than
better topology or the shared latent representation.

Random-GIN linear MMD is not used for the conclusion because several motif
runs produce extreme linear-kernel outliers, consistent with the previously
observed PROTEINS instability.

## GraphVAE-MM availability

An exhaustive search under `/local-scratch2` found no trained pure
`GraphVAE-MM` checkpoint for PROTEINS. The system contains only older
`GraphVAE-MM + motif` settings 09, 10, and 11, each representing a different
motif rule configuration rather than a clean MM baseline. They are not
substituted for the missing baseline in this comparison.

To complete the requested three-way comparison, a pure PROTEINS GraphVAE-MM
checkpoint trained on this same split and feature encoding is still required.

