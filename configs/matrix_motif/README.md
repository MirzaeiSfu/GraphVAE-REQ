# Motif statistic representations

The counting algorithm always returns one canonical padded full-matrix tensor
and its natural-shape mask. The loss layer derives one of four representations
from that shared result:

| Mode | Tensor shape | Definition |
| --- | --- | --- |
| `full_matrix` | `(graphs, motifs, N_max, N_max)` | Every valid entry of the final motif matrix-chain result. |
| `row_column_marginals` | `(graphs, motifs, 2, N_max)` | Row and column marginals for `NxN`; only the node-sized marginal for `Nx1`/`1xN`; one scalar channel for `1x1`. |
| `marginal_histogram` | `(graphs, motifs, 2, bins)` | Permutation-invariant soft histograms of the valid row/column marginals. |
| `total_count` | `(graphs, motifs)` | Sum of every final matrix entry; this is the original non-matrix motif count. |

The old names `matrix` and `count` remain accepted aliases for `full_matrix`
and `total_count`.

`motif_output_mode` and `motif_loss_mode` are backward-compatible defaults for
both motif groups. Override either group independently with:

- `non_literal_motif_output_mode` / `non_literal_motif_loss_mode` for original
  relational motifs;
- `syntactic_literal_motif_output_mode` /
  `syntactic_literal_motif_loss_mode` for injected literal motifs.

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
```

Consequently, the relative influence of a group is controlled by its alpha and
does not shrink merely because that group contains fewer motif rules. See the
runnable partial configuration in `group_specific_representation_example.yaml`.

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
