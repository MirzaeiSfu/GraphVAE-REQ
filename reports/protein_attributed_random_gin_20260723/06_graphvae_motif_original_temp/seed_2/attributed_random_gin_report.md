# Attributed Random-GIN Evaluation

The primary `decoded_node_edge` result consumes adjacency plus node and edge attributes decoded from the same latent sample. No degree, clustering, square-clustering, or other hand-made attributes are used.

- Run: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/06_graphvae_motif_original_temp/seed_2`
- Checkpoint: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/06_graphvae_motif_original_temp/seed_2/best_validation_mmd_model`
- Reference split: `test`
- Accepted graphs: `210`
- Node feature dimension: `3`
- Edge feature dimension: `0`
- Random-GIN repeats: `10`
- Primary mode: `decoded_node`
- Primary attributed F1-PR: `0.934144 ± 0.030186`

Higher is better for F1-PR, precision, and recall. Lower is better for MMD.

| Mode | F1-PR | Precision | Recall | MMD-RBF | MMD-linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| topology_control | 0.945554 ± 0.022328 | 0.930952 ± 0.021951 | 0.962381 ± 0.045895 | 0.007217 ± 0.007880 | 128030333.072461 ± 378799181.390930 |
| decoded_node | 0.934144 ± 0.030186 | 0.952857 ± 0.027059 | 0.916667 ± 0.039684 | 0.022997 ± 0.013399 | 2404843.008545 ± 7193614.295722 |

## Interpretation

- `topology_control`: adjacency only, represented by a fixed node constant and zero edge attributes in the full feature dimensions.
- `decoded_node`: adjacency plus decoded/reference node attributes; edge attributes are zeroed.
- `decoded_edge`: adjacency plus decoded/reference edge attributes; node attributes are fixed constants.
- `decoded_node_edge`: adjacency plus both decoded/reference attribute types; this is the primary attributed result.
