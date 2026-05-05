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

The readable run label is `grid-table2-graphvae-motif-v1`. At startup, `main.py` writes `RUN_LABEL.txt`, `REPRODUCE.md`, `reproducibility.json`, `run_config_used.yaml`, `git_status.txt`, and `git_diff.patch` into the run folder.

## Best Validation MMD Checkpoint

To save the checkpoint with the best validation MMD and use it for final test generation, add:

```bash
python main.py --config configs/reproduce_table2/grid_graphvae_table2_motif_best_mmd.yaml
```

The default validation score is a normalized mean over the Table 2 MMD metrics: degree, clustering, orbit, spectral, and diameter. When enabled, the run folder contains `best_validation_mmd_model` and `best_validation_mmd.json`, and `Single_comp_generatedGraphs_adj_final_eval.npy` is generated from that best checkpoint.

Existing configs explicitly set `keep_best_validation_mmd: false`, so old runs still use the final epoch unless the flag is enabled from the command line or changed in the config.

The dedicated best-MMD config writes to `runs/table2_reproduction/grid_graphvae_motif_best_mmd`, so it does not overwrite the previous motif run in `runs/table2_reproduction/grid_graphvae_motif`.

By default, checkpoint selection uses `best_validation_mmd_metric: normalized_table2`: each metric is divided by the Grid GraphVAE paper value before averaging, so large-scale metrics such as orbit do not dominate just because of their numeric scale. Other supported modes are `raw_mean`, `degree`, `clustering`, `orbit`, `spectral`, and `diameter`.

## Edge-Count Loss and Resampling Selection

To train the Grid / GraphVAE Table 2 motif setup with the additional edge-count loss and cheap periodic checkpoint saving:

```bash
python main.py --config configs/reproduce_table2/grid_graphvae_table2_motif_edge_count_best_mmd.yaml
```

This writes to `runs/table2_reproduction/grid_graphvae_motif_edge_count_best_mmd`, so the earlier motif and best-MMD runs are not overwritten. The config keeps the motif weights at node `10`, edge `10`, motif `1`, adjacency reconstruction `0.01`, and adds `edge_count_loss: true` with `alpha_edge_count: 0.1`.

Training still evaluates and saves checkpoints only at the existing validation cadence, `Vis_step: 1000`. The separate post-training resampling script then evaluates saved checkpoints across multiple generations:

```bash
python scripts/resample_grid_checkpoints.py \
  --config configs/reproduce_table2/grid_graphvae_table2_motif_edge_count_best_mmd.yaml \
  --run-dir runs/table2_reproduction/grid_graphvae_motif_edge_count_best_mmd \
  --samples 10 \
  --dense-definition twice_mean
```

The script writes `resampling_eval/resampling_metrics.json` and `resampling_eval/resampling_report.md` under the run folder. It selects the checkpoint by median normalized validation MMD across repeated generations. Table 2 MMD scores use the largest connected component of each generated graph for compatibility with the original evaluation path, while the report also includes raw generated graph edge counts and dense-outlier rates before largest-component filtering. Dense-rate selection penalties are optional through `--dense-penalty-weight`; the default is `0.0`, so dense rates are reported without changing the selection rule. When a penalty is enabled, it uses the raw validation dense rate.

Dense graph definitions are selected with `--dense-definition`:

- `twice_mean`: edge count is greater than `2 * mean(edge_count)` in the reference split.
- `mean_plus_3std`: edge count is greater than `mean(edge_count) + 3 * std(edge_count)` in the reference split.
- `max_reference`: edge count is greater than the maximum edge count in the reference split.

For leakage control, checkpoint selection uses validation metrics and validation dense rates only. Test metrics, test edge-count summaries, and test dense rates are final reporting fields after the checkpoint has already been selected; they should not be used to tune weights or choose a checkpoint.

## Notes

- The reproduction path is opt-in. Existing configs and default CLI behavior still use the legacy split.
- The script computes the statistics-based Table 2 metrics only: degree, clustering, orbit, spectral, and diameter MMD.
- The script does not compute Table 1 GNN-based metrics (`MMD RBF`, `F1 PR`).
