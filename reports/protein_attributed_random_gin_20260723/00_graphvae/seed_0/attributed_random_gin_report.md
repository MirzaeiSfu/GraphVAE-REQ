# Attributed Random-GIN Evaluation

The primary `decoded_node_edge` result consumes adjacency plus node and edge attributes decoded from the same latent sample. No degree, clustering, square-clustering, or other hand-made attributes are used.

- Run: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/00_graphvae/seed_0`
- Checkpoint: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/00_graphvae/seed_0/best_validation_mmd_model`
- Reference split: `test`
- Accepted graphs: `210`
- Node feature dimension: `3`
- Edge feature dimension: `0`
- Random-GIN repeats: `10`
- Primary mode: `decoded_node`
- Primary attributed F1-PR: `0.841474 ± 0.021078`

Higher is better for F1-PR, precision, and recall. Lower is better for MMD.

| Mode | F1-PR | Precision | Recall | MMD-RBF | MMD-linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| topology_control | 0.860809 ± 0.038366 | 0.913333 ± 0.036502 | 0.814762 ± 0.046679 | 0.102621 ± 0.010939 | 3.973701 ± 0.746431 |
| decoded_node | 0.841474 ± 0.021078 | 0.929524 ± 0.046404 | 0.770000 ± 0.020871 | 0.061444 ± 0.005914 | 4.141047 ± 0.415765 |

## Interpretation

- `topology_control`: adjacency only, represented by a fixed node constant and zero edge attributes in the full feature dimensions.
- `decoded_node`: adjacency plus decoded/reference node attributes; edge attributes are zeroed.
- `decoded_edge`: adjacency plus decoded/reference edge attributes; node attributes are fixed constants.
- `decoded_node_edge`: adjacency plus both decoded/reference attribute types; this is the primary attributed result.
