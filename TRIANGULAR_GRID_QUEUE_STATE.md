# Triangular Grid Table 2 — heavy-job queue state

Run-root: `runs/triangular_grid_table2_ruleprune_m2_moderate_20260707/`
Args for every job: `--train_batch_size 200 --rule_prune true --rule_prune_method 2 --rule_prune_tau 0.3 --rule_prune_min_support_frac 0.01`

Under moderate pruning, 6 of the 11 configs need a 24 GiB GPU (everything using
motif mode `original` or `both`: 03, 05, 06, 08, 09, 11). Only 3 such GPUs
exist (cs-cl-13 GPU 0, cs-cl-17 GPU 0, cs-cl-17 GPU 1), so 3 heavy jobs run at
a time per slot, queued in this order:

| Slot | Now running | Queued next | Status |
|---|---|---|---|
| cs-cl-13 GPU 0 | 09 (graphvae_mm_motif_original_no_temp) | 03 (graphvae_motif_original_no_temp) | 09 running |
| cs-cl-17 GPU 0 | 08 (graphvae_motif_both_temp) | (none) | 05 OOM'd at epoch 8590; 08 backfilled and running |
| cs-cl-17 GPU 1 | 11 (graphvae_mm_motif_both_no_temp) | 06 (graphvae_motif_original_temp) | 11 running |

Note (2026-07-08 00:xx): main.py on cs-cl-09/13/16/17/19/26 was found reverted to
a pre-rule_prune_method version (0 matches vs controller's 5) -- something
external re-synced older code onto those shared paths after the original
distribution. Re-distributed the correct code to all hosts; if a future
backfill hits "unrecognized arguments: --rule_prune_method ...", re-run
`cluster_distribute_code.sh` before relaunching.

Light jobs (01, 02, 04, 07, 10) run concurrently on other hosts and need no
queueing: cs-cl-18:0=01, cs-cl-16:0=02, cs-cl-26:0=04, cs-cl-26:1=07,
cs-cl-19:0=10.

## Backfill procedure (repeat until all three "queued next" cells are empty)

For each slot above, check whether its "now running" tmux session
(`20260707_<config-name>__<host>_gpu<n>`) still exists.

If it does not (job finished or crashed):
1. Update `CLUSTER_GPU_CONFIGS_TRIANGULAR_GRID_TABLE2.txt`: replace that
   slot's row with the "queued next" config.
2. Re-run `cluster_run_schedule.sh` with the same repo-paths/python-paths/
   date-prefix/run-root/extra-args as above (existing rows harmlessly
   report "tmux session already exists" and are skipped).
3. Update this table: move "queued next" into "now running", clear
   "queued next" for that slot (nothing left to queue after that).

Once all three slots show no "queued next" entry and their current job has
also finished, all 11 configs have been launched and this file is done.
