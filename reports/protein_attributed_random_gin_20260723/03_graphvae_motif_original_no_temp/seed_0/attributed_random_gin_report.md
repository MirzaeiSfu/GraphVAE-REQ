# Attributed Random-GIN Evaluation

The primary `decoded_node_edge` result consumes adjacency plus node and edge attributes decoded from the same latent sample. No degree, clustering, square-clustering, or other hand-made attributes are used.

- Run: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_0`
- Checkpoint: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_0/best_validation_mmd_model`
- Reference split: `test`
- Accepted graphs: `210`
- Node feature dimension: `3`
- Edge feature dimension: `0`
- Random-GIN repeats: `10`
- Primary mode: `decoded_node`
- Primary attributed F1-PR: `0.891540 ± 0.013086`

Higher is better for F1-PR, precision, and recall. Lower is better for MMD.

| Mode | F1-PR | Precision | Recall | MMD-RBF | MMD-linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| topology_control | 0.924047 ± 0.024631 | 0.906190 ± 0.031590 | 0.943333 ± 0.029466 | 0.002930 ± 0.002826 | 496402974.817969 ± 1468675462.361937 |
| decoded_node | 0.891540 ± 0.013086 | 0.938571 ± 0.034762 | 0.850476 ± 0.025305 | 0.012079 ± 0.009976 | 2542145.366748 ± 7381100.839271 |

## Interpretation

- `topology_control`: adjacency only, represented by a fixed node constant and zero edge attributes in the full feature dimensions.
- `decoded_node`: adjacency plus decoded/reference node attributes; edge attributes are zeroed.
- `decoded_edge`: adjacency plus decoded/reference edge attributes; node attributes are fixed constants.
- `decoded_node_edge`: adjacency plus both decoded/reference attribute types; this is the primary attributed result.
