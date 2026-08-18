# Attributed Random-GIN Evaluation

The primary `decoded_node_edge` result consumes adjacency plus node and edge attributes decoded from the same latent sample. No degree, clustering, square-clustering, or other hand-made attributes are used.

- Run: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/00_graphvae/seed_2`
- Checkpoint: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/00_graphvae/seed_2/best_validation_mmd_model`
- Reference split: `test`
- Accepted graphs: `210`
- Node feature dimension: `3`
- Edge feature dimension: `0`
- Random-GIN repeats: `10`
- Primary mode: `decoded_node`
- Primary attributed F1-PR: `0.922807 ± 0.014873`

Higher is better for F1-PR, precision, and recall. Lower is better for MMD.

| Mode | F1-PR | Precision | Recall | MMD-RBF | MMD-linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| topology_control | 0.939089 ± 0.025913 | 0.916667 ± 0.032103 | 0.963333 ± 0.030717 | 0.084005 ± 0.016129 | 2.891218 ± 0.592613 |
| decoded_node | 0.922807 ± 0.014873 | 0.947143 ± 0.031619 | 0.900476 ± 0.016000 | 0.043669 ± 0.004040 | 2.915424 ± 0.273549 |

## Interpretation

- `topology_control`: adjacency only, represented by a fixed node constant and zero edge attributes in the full feature dimensions.
- `decoded_node`: adjacency plus decoded/reference node attributes; edge attributes are zeroed.
- `decoded_edge`: adjacency plus decoded/reference edge attributes; node attributes are fixed constants.
- `decoded_node_edge`: adjacency plus both decoded/reference attribute types; this is the primary attributed result.
