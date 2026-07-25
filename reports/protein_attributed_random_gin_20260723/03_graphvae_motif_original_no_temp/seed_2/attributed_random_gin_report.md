# Attributed Random-GIN Evaluation

The primary `decoded_node_edge` result consumes adjacency plus node and edge attributes decoded from the same latent sample. No degree, clustering, square-clustering, or other hand-made attributes are used.

- Run: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_2`
- Checkpoint: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_2/best_validation_mmd_model`
- Reference split: `test`
- Accepted graphs: `210`
- Node feature dimension: `3`
- Edge feature dimension: `0`
- Random-GIN repeats: `10`
- Primary mode: `decoded_node`
- Primary attributed F1-PR: `0.931659 ± 0.027677`

Higher is better for F1-PR, precision, and recall. Lower is better for MMD.

| Mode | F1-PR | Precision | Recall | MMD-RBF | MMD-linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| topology_control | 0.947169 ± 0.020594 | 0.920476 ± 0.028734 | 0.976190 ± 0.025995 | 0.014394 ± 0.014857 | 31897858.685559 ± 94372367.633157 |
| decoded_node | 0.931659 ± 0.027677 | 0.948095 ± 0.031905 | 0.916667 ± 0.037024 | 0.035545 ± 0.016023 | 2380768.206293 ± 7137209.401575 |

## Interpretation

- `topology_control`: adjacency only, represented by a fixed node constant and zero edge attributes in the full feature dimensions.
- `decoded_node`: adjacency plus decoded/reference node attributes; edge attributes are zeroed.
- `decoded_edge`: adjacency plus decoded/reference edge attributes; node attributes are fixed constants.
- `decoded_node_edge`: adjacency plus both decoded/reference attribute types; this is the primary attributed result.
