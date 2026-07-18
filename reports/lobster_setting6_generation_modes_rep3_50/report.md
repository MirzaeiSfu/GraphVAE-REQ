# Lobster generation-mode stability diagnostic

- checkpoint: `collected_runs/20260708/lobster_setting6_exact_no_guard/lobster_setting6_exact_no_guard_rep3__cs-cl-09_gpu1/best_validation_mmd_model`
- rollouts: 50
- calibrated threshold: 0.999990
- fixed edge budget: 45

| Mode | rollout mean edges ± std | all-graph edge range |
|---|---:|---:|
| full_threshold_0p5 | 271.57 ± 211.66 | 5–4748 |
| cropped_threshold_0p5 | 133.07 ± 109.21 | 5–4555 |
| cropped_calibrated | 45.40 ± 70.26 | 0–4477 |
| cropped_fixed_budget | 45.00 ± 0.00 | 45–45 |
