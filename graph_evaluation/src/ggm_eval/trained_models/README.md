# Bundled GraphCL-GIN checkpoints

This directory contains the frozen encoders produced by the documented
seven-dataset GraphCL campaign. They are package data for
`evaluate_with_trained_gnns` and `ggm-eval evaluate-trained`; callers should
resolve them through those interfaces rather than hard-coding paths.

Each dataset has three independently initialized checkpoints (seeds 0, 1,
and 2). All encoders use the released upstream `GConv` implementation, three
GIN layers, hidden width 32, orthogonal initialization, the released
Lipschitz limiter, and 100 GraphCL epochs. The input and edge dimensions vary
with the dataset's exported feature schema.

| Dataset | Feature mode | Node/edge dims | Training graphs | Split |
| --- | --- | ---: | ---: | --- |
| AIDS | `decoded_node_edge` | 59 / 3 | 1,600 | legacy 80/20 |
| ENZYMES | `decoded_node` | 115 / 0 | 480 | legacy 80/20 |
| MUTAG | `decoded_node` | 7 / 0 | 150 | legacy 80/20 |
| PROTEINS | `decoded_node` | 3 / 0 | 731 | paper 70/10/20 |
| PTC | `decoded_node` | 19 / 0 | 275 | legacy 80/20 |
| QM9 | `decoded_node_edge` | 9 / 3 | 8,000 | legacy 80/20 |
| ogbg-molbbbp | `decoded_node_edge` | 39 / 9 | 1,413 | paper 70/10/20 |

QM9 was deterministically capped at 10,000 source molecules before its split;
all other datasets used the full retained source collection.

`manifest.json` records the training split identity and digest, graph counts,
model configuration, upstream Git revision, final loss, file size, and
SHA-256 digest for every checkpoint. The resolver verifies every selected
file before evaluation.

The research source is intentionally not copied here. Evaluation still needs
an external checkout of
`hamed1375/Self-Supervised-Models-for-GGM-Evaluation` at commit
`fb6bc26237eb21d7617fd41b22b4bb26ab29bf95`; the existing runner validates
that checkout before use.

These checkpoints are dataset- and feature-schema-specific. They must not be
used on graphs whose node/edge channel meanings differ, even when tensor
dimensions happen to match. The worker enforces the declared dataset and
feature-schema identities of training, generated, and reference artifacts.
