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
