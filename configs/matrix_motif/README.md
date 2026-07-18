# Matrix-valued motif output

`lobster_graphvae_matrix_motifs.yaml` enables `motif_output_mode: matrix`.
Instead of reducing each final motif matrix chain to one scalar count, the
counter returns:

- `motif_matrices`: `(graphs, motifs, N_max, N_max)`
- `motif_matrix_mask`: `(motifs, N_max, N_max)`

The mask marks the original cells of results whose natural shape is `1x1`,
`1xN`, `Nx1`, or `NxN`; all other cells are zero padding. Real-graph matrices
are stored on CPU, while reconstructed matrices remain differentiable on the
training device.

Matrix mode applies Kia's GraphVAE-MM calibrated Gaussian objective. Each motif
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

Matrix output can require substantially more memory than scalar counts. The
configuration therefore uses a smaller motif batch size.

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
