# Attributed Random-GIN Evaluation

The primary `decoded_node_edge` result consumes adjacency plus node and edge attributes decoded from the same latent sample. No degree, clustering, square-clustering, or other hand-made attributes are used.

- Run: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_1`
- Checkpoint: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_1/best_validation_mmd_model`
- Reference split: `test`
- Accepted graphs: `210`
- Node feature dimension: `3`
- Edge feature dimension: `0`
- Random-GIN repeats: `10`
- Primary mode: `decoded_node`
- Primary attributed F1-PR: `0.946469 ± 0.014987`

Higher is better for F1-PR, precision, and recall. Lower is better for MMD.

| Mode | F1-PR | Precision | Recall | MMD-RBF | MMD-linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| topology_control | 0.938056 ± 0.015046 | 0.925238 ± 0.034736 | 0.952857 ± 0.026552 | 0.060973 ± 0.014781 | 111.406547 ± 265.656065 |
| decoded_node | 0.946469 ± 0.014987 | 0.962857 ± 0.034127 | 0.931905 ± 0.022941 | 0.029963 ± 0.006301 | 13.040283 ± 22.714060 |

## Interpretation

- `topology_control`: adjacency only, represented by a fixed node constant and zero edge attributes in the full feature dimensions.
- `decoded_node`: adjacency plus decoded/reference node attributes; edge attributes are zeroed.
- `decoded_edge`: adjacency plus decoded/reference edge attributes; node attributes are fixed constants.
- `decoded_node_edge`: adjacency plus both decoded/reference attribute types; this is the primary attributed result.
