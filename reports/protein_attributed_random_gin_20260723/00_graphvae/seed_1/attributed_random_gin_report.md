# Attributed Random-GIN Evaluation

The primary `decoded_node_edge` result consumes adjacency plus node and edge attributes decoded from the same latent sample. No degree, clustering, square-clustering, or other hand-made attributes are used.

- Run: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/00_graphvae/seed_1`
- Checkpoint: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/00_graphvae/seed_1/best_validation_mmd_model`
- Reference split: `test`
- Accepted graphs: `210`
- Node feature dimension: `3`
- Edge feature dimension: `0`
- Random-GIN repeats: `10`
- Primary mode: `decoded_node`
- Primary attributed F1-PR: `0.912929 ± 0.014335`

Higher is better for F1-PR, precision, and recall. Lower is better for MMD.

| Mode | F1-PR | Precision | Recall | MMD-RBF | MMD-linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| topology_control | 0.937132 ± 0.030326 | 0.909048 ± 0.041810 | 0.967619 ± 0.023015 | 0.077495 ± 0.008739 | 2.491367 ± 0.475619 |
| decoded_node | 0.912929 ± 0.014335 | 0.958571 ± 0.027689 | 0.871905 ± 0.014038 | 0.045775 ± 0.004589 | 2.777145 ± 0.409113 |

## Interpretation

- `topology_control`: adjacency only, represented by a fixed node constant and zero edge attributes in the full feature dimensions.
- `decoded_node`: adjacency plus decoded/reference node attributes; edge attributes are zeroed.
- `decoded_edge`: adjacency plus decoded/reference edge attributes; node attributes are fixed constants.
- `decoded_node_edge`: adjacency plus both decoded/reference attribute types; this is the primary attributed result.
