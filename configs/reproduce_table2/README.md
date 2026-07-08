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

## 4. Batch GNN-Based Graph Realism Evaluation

For Kia-style Random-GIN graph realism metrics on already-saved `VGAREQ` outputs,
use the batch evaluator:

```bash
python scripts/evaluate_graph_realism_batch.py \
  --root-dir runs/table2_reproduction
```

This script scans recursively for run directories containing:

- `Single_comp_generatedGraphs_adj_final_eval.npy`
- `testGraphs_adj_.npy`

For each matching run directory it writes:

- `graph_realism_random_gin.json` inside the run directory

It also writes one batch summary CSV:

- `runs/table2_reproduction/graph_realism_batch_summary.csv`

This path is intentionally post-hoc. It lets us re-run GNN evaluation on old
saved graph sets after changing the evaluator, without retraining or
regenerating graphs.

## Motif Variant

Before distributing motif-loss training jobs to GPU workers, prepare the motif
rule cache once on the machine that has the FactorBase/MySQL databases:

```bash
python main.py \
  --config configs/reproduce_table2/grid_table2_graphvae_motif.yaml \
  --prepare_motif_cache_only true
```

This initializes `cache_motifs/<database_name>_allRules.pkl` from the configured
FactorBase database and exits before dataset loading, model creation, or
training. Afterward, copy `cache_motifs/*.pkl` to the worker machines. Workers
do not need MySQL for training as long as the required motif pickle exists.

To train the same Grid / GraphVAE Table 2 setup with motif-count loss added:

```bash
python main.py --config configs/reproduce_table2/grid_graphvae_table2_motif.yaml
```

This keeps the Table 2 reproduction split, BFS strategy, VAE latent mode, epochs, learning rate, and batch size. The changed training weights are node `10`, edge `10`, motif `1`, and adjacency reconstruction `0.01`. The motif DB points at the live Grid FactorBase snapshot on this machine: `grid_undir_feat_snap_7a58e6`.

The readable run label is `grid-table2-graphvae-motif-v1`. At startup, `main.py` writes `RUN_LABEL.txt`, `REPRODUCE.md`, `reproducibility.json`, `run_config_used.yaml`, `git_status.txt`, and `git_diff.patch` into the run folder.

## Controlled Baseline / GraphVAE-MM / Motif Configs

Kia's GraphVAE-MM stats loss uses the internal `alpha` vector in `main.py`.
For `GRID` and `TRIANGULAR_GRID`, the GraphVAE-MM vector is
`[1,1,1,1,1,1,1,1,50,2000]`; for `LOBSTER` it is
`[1,1,1,1,1,1,1,1,40,2000]`. The first eight entries are the graph-statistic
loss weights, so each graph-statistic loss has weight `1`.

The motif configs use `motif_loss_mode: calibrated_gaussian` and keep each
model's base reconstruction/KL weighting unchanged. Plain GraphVAE/kipf motif
runs use `alpha_node_feat: 1.0`, `alpha_edge_feat: 1.0`, and
`alpha_motif_loss: 0.1`. GraphVAE-MM motif runs use stronger feature decoder
supervision, `alpha_node_feat: 40.0` and `alpha_edge_feat: 40.0`, while
`alpha_motif_loss: 1.0` matches the graph-statistic loss scale.

Grid:

```bash
python main.py --config configs/reproduce_table2/grid_table2_graphvae_baseline.yaml
python main.py --config configs/reproduce_table2/grid_table2_graphvae_mm.yaml
python main.py --config configs/reproduce_table2/grid_table2_graphvae_motif.yaml
```

Triangular Grid:

```bash
python main.py --config configs/reproduce_table2/triangular_grid_table2_graphvae_baseline.yaml
python main.py --config configs/reproduce_table2/triangular_grid_table2_graphvae_mm.yaml
python main.py --config configs/reproduce_table2/triangular_grid_table2_graphvae_motif.yaml
```

Lobster:

```bash
python main.py --config configs/reproduce_table2/lobster_table2_graphvae_baseline.yaml
python main.py --config configs/reproduce_table2/lobster_table2_graphvae_mm.yaml
python main.py --config configs/reproduce_table2/lobster_table2_graphvae_motif.yaml
```

For generated outputs from the GraphVAE-MM configs, compare against the paper's
`GraphVAE-MM` row:

```bash
python scripts/reproduce_table2_grid.py \
  --dataset GRID \
  --mode evaluate-generated \
  --paper-row GraphVAE-MM \
  --row-label grid_graphvae_mm \
  --generated runs/table2_reproduction/grid_graphvae_mm/Single_comp_generatedGraphs_adj_final_eval.npy \
  --test-graphs runs/table2_reproduction/grid_graphvae_mm/testGraphs_adj_.npy \
  --output-dir runs/table2_reproduction/grid_graphvae_mm_eval
```

For motif outputs, use `--paper-row GraphVAE-MM` when the question is
"does motif loss replace the stats loss?", or `--paper-row GraphVAE` when the
question is "how much does the motif run improve over the plain baseline?".

## Best Validation MMD Checkpoint

To save the checkpoint with the best validation MMD and use it for final test generation, add:

```bash
python main.py --config configs/reproduce_table2/grid_graphvae_table2_motif_best_mmd.yaml
```

The default validation score can now include both paper metric families. Use `best_validation_mmd_metric: normalized_table2_table3` to average Table 2 metrics normalized by the dataset's GraphVAE paper row (degree, clustering, orbit, spectral, diameter) together with Table 3 terms (`mmd_rbf` normalized by the dataset's GraphVAE-MM paper row and `(1 - f1_pr) / 0.05`). Normalized MMD-style denominators are floored at `1e-3`, and normalized score components are capped at `10.0`, so a tiny paper value cannot dominate checkpoint selection. When enabled, the run folder contains `best_validation_mmd_model` and `best_validation_mmd.json`, and `Single_comp_generatedGraphs_adj_final_eval.npy` is generated from that best checkpoint.

Existing configs explicitly set `keep_best_validation_mmd: false`, so old runs still use the final epoch unless the flag is enabled from the command line or changed in the config.

The dedicated best-MMD config writes to `runs/table2_reproduction/grid_graphvae_motif_best_mmd`, so it does not overwrite the previous motif run in `runs/table2_reproduction/grid_graphvae_motif`.

The supported checkpoint-selection modes are `normalized_table2`, `normalized_table2_table3`, `raw_mean`, `raw_mean_table2_table3`, `table3`, `degree`, `clustering`, `orbit`, `spectral`, `diameter`, `mmd_rbf`, and `f1_pr`. For `f1_pr`, the internal score is `1 - f1_pr` because lower validation scores are treated as better.

After final test generation, the run folder also contains `final_table2_metrics.json`, `final_table3_metrics.json`, and `final_metrics_summary.json`. Table 2 is parsed from the final generated-vs-test structural evaluation. Table 3 includes the local generated-vs-test GNN metrics and, when `third_party_eval: true`, the third-party Random-GIN JSON payload.

## Resampling Selection

For the Grid / GraphVAE Table 2 motif setup, the best-MMD config enables cheap
periodic checkpoint saving:

```bash
python main.py --config configs/reproduce_table2/grid_graphvae_table2_motif_best_mmd.yaml
```

Training still evaluates and saves checkpoints only at the existing validation cadence, `Vis_step: 1000`. The separate post-training resampling script then evaluates saved checkpoints across multiple generations:

```bash
python scripts/resample_grid_checkpoints.py \
  --config configs/reproduce_table2/grid_graphvae_table2_motif_best_mmd.yaml \
  --run-dir runs/table2_reproduction/grid_graphvae_motif_best_mmd \
  --samples 10 \
  --dense-definition twice_mean
```

The script writes `resampling_eval/resampling_metrics.json` and `resampling_eval/resampling_report.md` under the run folder. It selects the checkpoint by median normalized validation MMD across repeated generations. Table 2 MMD scores use the largest connected component of each generated graph for compatibility with the original evaluation path, while the report also includes raw generated graph edge counts and dense-outlier rates before largest-component filtering. Dense-rate selection penalties are optional through `--dense-penalty-weight`; the default is `0.0`, so dense rates are reported without changing the selection rule. When a penalty is enabled, it uses the raw validation dense rate.

Dense graph definitions are selected with `--dense-definition`:

- `twice_mean`: edge count is greater than `2 * mean(edge_count)` in the reference split.
- `mean_plus_3std`: edge count is greater than `mean(edge_count) + 3 * std(edge_count)` in the reference split.
- `max_reference`: edge count is greater than the maximum edge count in the reference split.

For leakage control, checkpoint selection uses validation metrics and validation dense rates only. Test metrics, generated graph edge-count summaries, and test dense rates are final reporting fields after the checkpoint has already been selected; they should not be used to tune weights or choose a checkpoint.

## Notes

- The reproduction path is opt-in. Existing configs and default CLI behavior still use the legacy split.
- The script computes the statistics-based Table 2 metrics only: degree, clustering, orbit, spectral, and diameter MMD.
- The script does not compute Table 1 GNN-based metrics (`MMD RBF`, `F1 PR`).
- Our local statistics evaluator for Table 2 lives in `stat_rnn.py` and `eval/`; it does not depend on `third_party/ggmeval`.
- The vendored `third_party/ggmeval` folder is kept for Kia-style GNN-based graph realism evaluation.
- Within that folder, `RuleEval.py` is the narrow runner for the VGAE `SaveSamples(...)` output format, while `eval_all_in_dir_2023.py` is the broader directory-based runner used for GraphVAE-style saved graph sets.
