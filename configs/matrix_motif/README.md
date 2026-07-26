# Motif statistic representations

The counting algorithm always returns one canonical padded full-matrix tensor
and its natural-shape mask. The loss layer derives one of six representations
from that shared result:

| Mode | Tensor shape | Definition |
| --- | --- | --- |
| `full_matrix` | `(graphs, motifs, N_max, N_max)` | Every valid entry of the final motif matrix-chain result. |
| `row_column_marginals` | `(graphs, motifs, 2, N_max)` | Row and column marginals for `NxN`; only the node-sized marginal for `Nx1`/`1xN`; one scalar channel for `1x1`. |
| `marginal_histogram` | `(graphs, motifs, 2, bins)` | Permutation-invariant soft histograms of the valid row/column marginals. |
| `degree_histogram` | `(graphs, motifs, N_max)` | GraphVAE-MM's integer-centered triangular soft histogram of row sums; restricted to natural `N_max x N_max` motif matrices. |
| `kiarash_statistics` | heterogeneous bundle | GraphVAE-MM's separate `P^1,...,P^5`, in/out degree histograms, and total-triangle scalar, derived from exactly one natural `N_max x N_max` unit-edge motif. |
| `total_count` | `(graphs, motifs)` | Sum of every final matrix entry; this is the original non-matrix motif count. |

The old names `matrix` and `count` remain accepted aliases for `full_matrix`
and `total_count`.

`motif_output_mode` and `motif_loss_mode` are backward-compatible defaults for
both motif groups. Override either group independently with:

- `non_literal_motif_output_mode` / `non_literal_motif_loss_mode` for original
  relational motifs;
- `syntactic_literal_motif_output_mode` /
  `syntactic_literal_motif_loss_mode` for injected literal motifs.
- `unit_relation_motif_output_mode` / `unit_relation_motif_loss_mode` for an
  optional separate group of positive bare binary-relation motifs.

For example, this uses full matrices for original motifs and shape-aware
marginals for literal motifs while counting the motif matrices only once:

```yaml
motif:
  syntactic_literal_rule_mode: both
  non_literal_motif_output_mode: full_matrix
  non_literal_motif_loss_mode: calibrated_gaussian
  syntactic_literal_motif_output_mode: row_column_marginals
  syntactic_literal_motif_loss_mode: calibrated_gaussian
```

Leaving a group-specific setting `null` inherits its global setting. Each group
loss is averaged over the motifs in that group and the groups are composed
directly with their explicit weights:

```text
L = alpha_original * L_original + alpha_literal * L_literal
    + alpha_unit_relation * L_unit_relation
```

Consequently, the relative influence of a group is controlled by its alpha and
does not shrink merely because that group contains fewer motif rules. See the
runnable partial configuration in `group_specific_representation_example.yaml`.

The unit-relation group is opt-in. When enabled, its mask is removed from the
original/non-literal group so no motif is supervised twice. With
`protect_unit_relation_motifs_from_pruning: true`, global pruning still applies
to every other rule while positive cached value rows are restored for bare
binary relations. Explicit false/complement rows are not assigned to the
degree group. This treats the cache as an external rule source and does not
modify it.

`degree_histogram` is designed for this group. For a unit relation such as
`edges(nodes0,nodes1)`, its full motif count matrix is the relation adjacency
matrix. The representation sums each row and applies the same bin centers,
width `0.1`, and triangular membership function as the repository's existing
GraphVAE-MM degree statistic. The runnable controlled experiment is
`lobster_matrix_full_original_unit_relation_degree_histogram_currentdb_constant.yaml`;
its inventory is `unit_relation_degree_experiment_manifest.csv`.

`kiarash_statistics` is the exact composite parity representation for the
same protected unit-edge motif. It returns the eight tensors in
`GlobalProperties.kernel` order:

```text
P^1, P^2, P^3, P^4, P^5,
in_degree_histogram, out_degree_histogram, total_number_of_triangles
```

Here `P = D^-1 M_edge`, using the same row normalization and degree bins as
GraphVAE-MM. The bundle also preserves `GlobalProperties`' unconventional
labels: its `in_degree_dist` is built from row sums and its `out_degree_dist`
from column sums. For strict code parity, the triangle statistic first zeros
the diagonal and computes `trace(M^3)/6`; this is
`M - diag(diag(M))`, not literal subtraction of the identity for a soft
decoded diagonal. Each of the eight tensors receives its own RMSE-calibrated
sigma and Gaussian loss, and the eight losses are summed exactly like
`OptimizerVAE`.

An optional `alpha_unit_relation_edge_count_loss` adds a separately calibrated
graph-level statistic derived from the same protected matrix:

```text
C_E(M) = 0.5 * sum_{i != j} M_ij
```

The factor of one half counts symmetric LOBSTER adjacencies as undirected
edges, and the diagonal is excluded. This auxiliary composes with
`kiarash_statistics`; it does not add a handcrafted decoder feature or count
the motif a second time.

The four parity configurations cross legacy/corrected reparameterization with
adjacency-BCE/KL weights `40/2000` and `1/1`. In all four, the Kiarash bundle
has weight `1` and the relational/literal groups have weight `0`. Their
inventory is `kiarash_parity_experiment_manifest.csv`. Zero-weight groups are
projected out before motif counting, so these controls count only the one
unit-edge matrix while hybrids retain every nonzero-weight group.
They also set `dataset_loader_seed: 0`. This decouples the loader's aligned
pre-split shuffle from the model/training seed; `split_seed: 123` therefore
produces identical train, validation, and held-out sets for all three training
seeds. Leaving `dataset_loader_seed` unset preserves legacy global-RNG
behavior.

The strict parity gate additionally uses a matched 2x2 control because native
GraphVAE-MM gives the decoded node and edge feature losses weights `40/40`,
whereas the initial motif-derived parity runs used `1/1`:

| Statistics implementation | Feature weights 1/1 | Feature weights 40/40 |
| --- | --- | --- |
| `GlobalProperties.kernel` | `lobster_graphvae_mm_fixed_split_matched1_legacy.yaml` | `lobster_graphvae_mm_fixed_split_native40_legacy.yaml` |
| motif `kiarash_statistics` | `lobster_kiarash_parity_kia40_2000_legacy.yaml` | `lobster_kiarash_parity_kia40_2000_feature40_legacy.yaml` |

All four cells use the same fixed split, legacy reparameterization, `40/2000`
adjacency-BCE/KL weights, three training seeds, five candidate checkpoints,
ten validation rollouts for checkpoint selection, and ten frozen held-out
rollouts. This isolates the statistics implementation from feature weighting.
The full protocol is in `kiarash_control_factorial_manifest.csv`.

Post-training, use `select_lobster_checkpoints_per_run.py` with ten validation
rollouts to freeze one checkpoint independently for every training run. Merge
those validation-only manifests with
`scripts/evaluate_lobster_frozen_selections.py`; the evaluator writes the
combined frozen decision before opening held-out graphs, then evaluates every
selected checkpoint with ten paired prior rollouts and reports both raw and
largest-component graph sizes. Pass one `--selection-json` argument for each
worker selection manifest. `--runs-root` is repeatable so a fixed evaluation
can combine a reused seed-0 root with corrected seed-1/seed-2 reruns.
For a custom experiment matrix, repeat `--condition` in the desired report
order; the evaluator then verifies the complete three-seed Cartesian matrix.

All structured modes use Kia-MM-style calibrated Gaussian NLL. Full matrices
receive one RMSE-calibrated sigma per motif. Row and column marginals receive
separate sigmas, like GraphVAE-MM's in-degree and out-degree statistics; their
valid directional losses are averaged back to one loss per motif. Histogram
channels follow the same rule. Motifs are averaged within each group before the
weighted group sum. The validity mask excludes artificial padding and redundant
scalar marginals.

For `marginal_histogram`, bins are calibrated once from the real targets and
then reused for every reconstruction. Histograms operate on `log1p` marginal
counts and use differentiable sigmoid boundaries. The first/last bins include
underflow/overflow, so reconstructed values outside the observed range still
contribute. Configure them with:

```yaml
motif:
  motif_output_mode: marginal_histogram
  motif_histogram_num_bins: 16
  motif_histogram_smoothing: 0.25
  motif_loss_mode: calibrated_gaussian
```

## Full-matrix experiments

`lobster_graphvae_matrix_motifs.yaml` enables `motif_output_mode: matrix`.
The canonical counter output stored for training is:

- `motif_full_matrices`: `(graphs, motifs, N_max, N_max)`
- `motif_full_matrix_mask`: `(motifs, N_max, N_max)`

The mask marks the original cells of results whose natural shape is `1x1`,
`1xN`, `Nx1`, or `NxN`; all other cells are zero padding. Real-graph matrices
are stored on CPU, while reconstructed matrices remain differentiable on the
training device.

Full-matrix mode applies Kia's GraphVAE-MM calibrated Gaussian objective. Each motif
matrix is treated as one independent graph statistic, analogous to one of
`P^1,...,P^5`:

1. Compute one minibatch RMSE `sigma_u` over all valid entries of motif `u`.
2. Average the Gaussian negative log-likelihood over those entries.
3. Average the independently normalized losses over motifs.

The validity mask excludes only artificial bottom/right padding. The global
motif average keeps the objective scale stable when pruning changes the number
of selected motifs. With separate regular/literal weights, the weighted losses
are still divided by the total motif count rather than averaging each group
independently.

Run the configuration with:

```bash
python main.py --config configs/matrix_motif/lobster_graphvae_matrix_motifs.yaml
```

Canonical full-matrix counting can require substantially more memory than the
old early scalar reduction, even when the selected loss later uses marginals or
total counts. Real targets remain on CPU and only the current training batch is
transferred to the training device. Use a smaller `motif_batch_size` when
needed.

## Matrix-motif weight sweep

The eight `lobster_matrix_*.yaml` files form a controlled Cartesian sweep over:

- adjacency BCE / KL weights: plain GraphVAE `1/1` or Kia LOBSTER `40/2000`;
- globally averaged matrix-motif weight: `0.1` or `1.0`;
- motif temperature: constant `1.0`, or annealed `1.0 -> 0.5` after the first
  half of training.

Every sweep configuration uses matrix output, original FactorBase rules only,
the calibrated Gaussian objective, node/edge weights `1/1`, and the old-v1
LOBSTER data/cache setup used by the posthoc scalar-count sweep. Constant runs
disable the temperature guard. Annealed runs use the standard guard ratio
`2.0` with relax/sharpen factors `1.05/0.995`.

`motif_batch_size: 200` matches the configured training batch cap. The current
paper split has 70 training graphs, so all 70 are processed together. A full
matrix forward/backward pass with targets shaped `(70, 174, 98, 98)` was
verified on a 24 GiB TITAN RTX without an out-of-memory error.

The generated inventory is `matrix_weight_sweep_manifest.csv`. Regenerate the
configs and manifest with:

```bash
python scripts/generate_lobster_matrix_motif_sweep.py
```
