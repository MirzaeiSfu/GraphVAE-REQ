# Table 2 Grid Reproduction

This folder contains opt-in reproduction settings for the Grid rows of Table 2 in `Kia paper with appendix.pdf`.

## Important split distinction

The `50/50 split` row in Table 2 is an ideal/reference score, not the training split for GraphVAE. The paper describes model training/evaluation separately as `70%` train, `10%` validation, and `20%` test.

## 1. Compute the `50/50 split` reference row

```bash
python scripts/reproduce_table2_grid.py \
  --mode ideal-50-50 \
  --output-dir runs/table2_reproduction/grid_50_50
```

This writes:

- `runs/table2_reproduction/grid_50_50/metrics.json`
- `runs/table2_reproduction/grid_50_50/table2_grid_reproduction.md`

## 2. Train GraphVAE with the paper-style split

```bash
python main.py --config configs/reproduce_table2/grid_graphvae_table2.yaml
```

This config is isolated from default runs:

- It sets `split_mode: paper_70_10_20`.
- It sets `bfs_strategy: legacy_first_component` to match the original paper code path.
- It writes outputs to `runs/table2_reproduction/grid_graphvae`.
- It writes dataset cache files to `runs/table2_reproduction/dataset_cache`.
- It does not change the default `legacy_80_20` split or `all_components` BFS behavior.

## 3. Compare the generated GraphVAE result with Table 2

After training finishes, run:

```bash
python scripts/reproduce_table2_grid.py \
  --mode evaluate-generated \
  --generated runs/table2_reproduction/grid_graphvae/Single_comp_generatedGraphs_adj_final_eval.npy \
  --test-graphs runs/table2_reproduction/grid_graphvae/testGraphs_adj_.npy \
  --output-dir runs/table2_reproduction/grid_graphvae_eval
```

Use `--test-graphs` when available so the comparison uses the exact test graphs saved by the run.

## Motif Variant

To train the same Grid / GraphVAE Table 2 setup with motif-count loss added:

```bash
python main.py --config configs/reproduce_table2/grid_graphvae_table2_motif.yaml
```

This keeps the Table 2 reproduction split, BFS strategy, VAE latent mode, epochs, learning rate, and batch size. The changed training weights are node `10`, edge `10`, motif `1`, and adjacency reconstruction `0.01`. The motif DB points at the live Grid FactorBase snapshot on this machine: `grid_undir_feat_snap_7a58e6`.

The readable run tag is `grid-table2-graphvae-motif-v1`. At startup, `main.py` writes `RUN_TAG.txt`, `REPRODUCE.md`, `reproducibility.json`, `run_config_used.yaml`, `git_status.txt`, and `git_diff.patch` into the run folder.

## Notes

- The reproduction path is opt-in. Existing configs and default CLI behavior still use the legacy split.
- The script computes the statistics-based Table 2 metrics only: degree, clustering, orbit, spectral, and diameter MMD.
- The script does not compute Table 1 GNN-based metrics (`MMD RBF`, `F1 PR`).
