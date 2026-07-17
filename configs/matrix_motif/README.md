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

The matrix loss is intentionally disabled in `main.py`. Search for
`TODO(matrix-motif-loss)` to define it using `observed_motif_matrices`,
`current_recon_matrices`, and `current_recon_matrix_mask`.

Run the configuration with:

```bash
python main.py --config configs/matrix_motif/lobster_graphvae_matrix_motifs.yaml
```

Matrix output can require substantially more memory than scalar counts. The
configuration therefore uses a smaller motif batch size.
