# GraphVAE Checkpoint Resampling

Lower is better for all MMD metrics and normalized scores.

- config: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_0/run_config_used.yaml`
- run_dir: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_0`
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
| best_validation_mmd_model | validation | 0.661506 | 0.660574 | 0.083535 | 0.838652 | 91.84 | 4808.00 | 1.73% | 97.71 | 4808.00 | 1.73% |
| best_validation_mmd_model | test | 0.783258 | 0.776416 | 0.056477 | 0.891683 | 70.33 | 4821.00 | 0.90% | 75.65 | 4821.00 | 0.90% |

## Selection Candidates

| Checkpoint | Median Validation MMD | Validation LCC Dense Rate | Validation Raw Dense Rate | Selection Score |
| --- | ---: | ---: | ---: | ---: |
| best_validation_mmd_model | 0.661506 | 1.73% | 1.73% | 0.661506 |

## Selected Final Test Summary

Selected by validation: `best_validation_mmd_model`

| Median | Mean | Std | Worst | LCC Dense Rate | Raw Dense Rate | Raw Worst Max Edges |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.783258 | 0.776416 | 0.056477 | 0.891683 | 0.90% | 0.90% | 4821.00 |

## Metric Summary

### best_validation_mmd_model / validation

| Metric | Median | Mean | Std | Worst |
| --- | ---: | ---: | ---: | ---: |
| degree | 0.030812 | 0.030766 | 0.005430 | 0.041400 |
| clustering | 0.029223 | 0.030093 | 0.002436 | 0.035460 |
| orbit | 0.013762 | 0.013471 | 0.002275 | 0.017637 |
| spectral | 0.018923 | 0.019040 | 0.001887 | 0.022543 |
| diameter | 0.034386 | 0.032995 | 0.010093 | 0.047408 |

### best_validation_mmd_model / test

| Metric | Median | Mean | Std | Worst |
| --- | ---: | ---: | ---: | ---: |
| degree | 0.039557 | 0.039834 | 0.003532 | 0.048366 |
| clustering | 0.027828 | 0.027788 | 0.001669 | 0.031300 |
| orbit | 0.014668 | 0.016045 | 0.004625 | 0.030944 |
| spectral | 0.020878 | 0.020601 | 0.001781 | 0.025434 |
| diameter | 0.038892 | 0.039900 | 0.010308 | 0.066278 |
