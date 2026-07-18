# Lobster generation-mode stability diagnostic

- checkpoint: `collected_runs/20260629/cluster_smoke_grid_motif/lobster_table2_02_graphvae_mm__cs-cl-09_gpu1/model_9999_1`
- rollouts: 50
- calibrated threshold: 0.142980
- fixed edge budget: 45

| Mode | rollout mean edges ± std | all-graph edge range |
|---|---:|---:|
| full_threshold_0p5 | 40.75 ± 5.22 | 4–139 |
| cropped_threshold_0p5 | 33.55 ± 3.58 | 4–139 |
| cropped_calibrated | 45.40 ± 4.10 | 11–167 |
| cropped_fixed_budget | 45.00 ± 0.00 | 45–45 |
