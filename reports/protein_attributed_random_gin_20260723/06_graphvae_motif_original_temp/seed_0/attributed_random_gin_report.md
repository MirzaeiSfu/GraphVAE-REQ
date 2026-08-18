# Attributed Random-GIN Evaluation

The primary `decoded_node_edge` result consumes adjacency plus node and edge attributes decoded from the same latent sample. No degree, clustering, square-clustering, or other hand-made attributes are used.

- Run: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/06_graphvae_motif_original_temp/seed_0`
- Checkpoint: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/06_graphvae_motif_original_temp/seed_0/best_validation_mmd_model`
- Reference split: `test`
- Accepted graphs: `210`
- Node feature dimension: `3`
- Edge feature dimension: `0`
- Random-GIN repeats: `10`
- Primary mode: `decoded_node`
- Primary attributed F1-PR: `0.896158 ± 0.025972`

Higher is better for F1-PR, precision, and recall. Lower is better for MMD.

| Mode | F1-PR | Precision | Recall | MMD-RBF | MMD-linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| topology_control | 0.941795 ± 0.011319 | 0.912857 ± 0.019524 | 0.973333 ± 0.022956 | 0.011640 ± 0.012085 | 29841839.851892 ± 88289352.073765 |
| decoded_node | 0.896158 ± 0.025972 | 0.948571 ± 0.024169 | 0.850000 ± 0.036962 | 0.048371 ± 0.004798 | 532.997589 ± 229.243648 |

## Interpretation

- `topology_control`: adjacency only, represented by a fixed node constant and zero edge attributes in the full feature dimensions.
- `decoded_node`: adjacency plus decoded/reference node attributes; edge attributes are zeroed.
- `decoded_edge`: adjacency plus decoded/reference edge attributes; node attributes are fixed constants.
- `decoded_node_edge`: adjacency plus both decoded/reference attribute types; this is the primary attributed result.
