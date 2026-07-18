# GraphVAE Checkpoint Resampling

Lower is better for all MMD metrics and normalized scores.

- config: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_2/run_config_used.yaml`
- run_dir: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_2`
- samples_per_checkpoint_split: `{'validation': 10, 'test': 50}`
- dense_definition: `mean_plus_3std`
- dense_edge_threshold: `{'validation': 164.52742337184597, 'test': 179.68291095822195}`
- selection_split: `validation`
- dense_penalty_weight: `0.0`
- selected_checkpoint: `best_validation_mmd_model`

MMD scores use largest connected components of generated graphs for Table 2 compatibility. Raw dense statistics are computed before largest-component filtering.

Checkpoint selection uses validation only. Test metrics and test dense rates are reported after selection.

## Score Summary

| Checkpoint | Split | Median | Mean | Std | Worst | LCC Median Mean Edges | LCC Worst Max Edges | LCC Dense Rate | Raw Median Mean Edges | Raw Worst Max Edges | Raw Dense Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best_validation_mmd_model | validation | 0.722858 | 0.712861 | 0.046620 | 0.762874 | 137.83 | 4790.00 | 2.69% | 143.38 | 4790.00 | 2.69% |
| best_validation_mmd_model | test | 0.840556 | 0.838043 | 0.046209 | 0.969076 | 92.03 | 4817.00 | 1.48% | 98.26 | 4817.00 | 1.49% |

## Selection Candidates

| Checkpoint | Median Validation MMD | Validation LCC Dense Rate | Validation Raw Dense Rate | Selection Score |
| --- | ---: | ---: | ---: | ---: |
| best_validation_mmd_model | 0.722858 | 2.69% | 2.69% | 0.722858 |

## Selected Final Test Summary

Selected by validation: `best_validation_mmd_model`

| Median | Mean | Std | Worst | LCC Dense Rate | Raw Dense Rate | Raw Worst Max Edges |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.840556 | 0.838043 | 0.046209 | 0.969076 | 1.48% | 1.49% | 4817.00 |

## Metric Summary

### best_validation_mmd_model / validation

| Metric | Median | Mean | Std | Worst |
| --- | ---: | ---: | ---: | ---: |
| degree | 0.037096 | 0.037347 | 0.003392 | 0.042649 |
| clustering | 0.030163 | 0.030448 | 0.002088 | 0.032995 |
| orbit | 0.006055 | 0.006729 | 0.001958 | 0.010431 |
| spectral | 0.020602 | 0.020272 | 0.001734 | 0.022540 |
| diameter | 0.022876 | 0.024491 | 0.008751 | 0.043975 |

### best_validation_mmd_model / test

| Metric | Median | Mean | Std | Worst |
| --- | ---: | ---: | ---: | ---: |
| degree | 0.045113 | 0.045037 | 0.003095 | 0.054535 |
| clustering | 0.028981 | 0.028746 | 0.001671 | 0.032232 |
| orbit | 0.011944 | 0.012523 | 0.003895 | 0.022535 |
| spectral | 0.022963 | 0.022842 | 0.001708 | 0.026380 |
| diameter | 0.033675 | 0.034204 | 0.007165 | 0.052977 |
