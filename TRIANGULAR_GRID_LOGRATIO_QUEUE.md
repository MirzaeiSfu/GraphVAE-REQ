# Triangular Grid V2_undir: abs_log_ratio follow-up queue

Currently running: `runs/triangular_grid_v2_undir_alpha01_20260709/` — settings
03, 05, 06 on `triangular_grid_V2_undir`, `motif_loss_mode=calibrated_gaussian`,
`alpha_motif_loss=0.01`, `alpha_syntactic_literal_motif_loss=0.01`,
`rule_prune_method=2 tau=0.15 min_support_frac=0.004` (118 total motif
columns), batch 200, on cs-cl-13 GPU 0 / cs-cl-17 GPU 0 / cs-cl-17 GPU 1.

## Follow-up (launch once ALL THREE of the above finish or crash)

Same three settings, same database, same pruning, same alpha, but
`--motif_loss_mode abs_log_ratio` instead of the default calibrated_gaussian.
New run-root so it doesn't mix with the calibrated_gaussian results:
`runs/triangular_grid_v2_undir_logratio_20260709/`

```
cd /localhome/mirzaei/ali/GraphVAE-REQ
bash scripts/cluster_run_schedule.sh \
  --repo-paths CLUSTER_REPO_PATHS.txt \
  --schedule CLUSTER_GPU_CONFIGS_TRIANGULAR_V2_UNDIR.txt \
  --python-paths CLUSTER_MICRO_PYTHON_PATHS.txt \
  --date-prefix 20260709 \
  --run-root runs/triangular_grid_v2_undir_logratio_20260709 \
  -- --database_name triangular_grid_V2_undir --train_batch_size 200 \
     --rule_prune true --rule_prune_method 2 --rule_prune_tau 0.15 \
     --rule_prune_min_support_frac 0.004 --alpha_motif_loss 0.01 \
     --alpha_syntactic_literal_motif_loss 0.01 \
     --motif_loss_mode abs_log_ratio
```

## Monitoring procedure

Check tmux sessions for the three calibrated_gaussian jobs:
- `20260709_triangular_grid_table2_03_graphvae_motif_original_no_temp__cs-cl-13_gpu0`
- `20260709_triangular_grid_table2_05_graphvae_motif_both_no_temp__cs-cl-17_gpu0`
- `20260709_triangular_grid_table2_06_graphvae_motif_original_temp__cs-cl-17_gpu1`

Once all three no longer have a live tmux session (finished or crashed), run
the launch command above, then verify all three new sessions come up and
don't immediately OOM (same VRAM risk as before -- 118 columns used ~15-16GB
last time, should be safe, but check anyway).

Status: NOT YET LAUNCHED. Waiting for the calibrated_gaussian batch.
