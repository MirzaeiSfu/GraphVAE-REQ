# GraphVAE Checkpoint Resampling

Lower is better for all MMD metrics and normalized scores.

- config: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_1/run_config_used.yaml`
- run_dir: `/local-scratch2/new/protein_best_models_20260715/proteins_table2/03_graphvae_motif_original_no_temp/seed_1`
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
| best_validation_mmd_model | validation | 0.633969 | 0.639954 | 0.047301 | 0.705738 | 49.16 | 235.00 | 0.29% | 55.02 | 235.00 | 0.29% |
| best_validation_mmd_model | test | 0.783265 | 0.790759 | 0.063677 | 0.914254 | 50.10 | 4809.00 | 0.18% | 56.19 | 4809.00 | 0.19% |

## Selection Candidates

| Checkpoint | Median Validation MMD | Validation LCC Dense Rate | Validation Raw Dense Rate | Selection Score |
| --- | ---: | ---: | ---: | ---: |
| best_validation_mmd_model | 0.633969 | 0.29% | 0.29% | 0.633969 |

## Selected Final Test Summary

Selected by validation: `best_validation_mmd_model`

| Median | Mean | Std | Worst | LCC Dense Rate | Raw Dense Rate | Raw Worst Max Edges |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.783265 | 0.790759 | 0.063677 | 0.914254 | 0.18% | 0.19% | 4809.00 |

## Metric Summary

### best_validation_mmd_model / validation

| Metric | Median | Mean | Std | Worst |
| --- | ---: | ---: | ---: | ---: |
| degree | 0.032834 | 0.031855 | 0.003269 | 0.036737 |
| clustering | 0.029337 | 0.029514 | 0.001700 | 0.034002 |
| orbit | 0.009910 | 0.010042 | 0.002506 | 0.015092 |
| spectral | 0.018495 | 0.018058 | 0.001342 | 0.020193 |
| diameter | 0.026448 | 0.026600 | 0.008395 | 0.043627 |

### best_validation_mmd_model / test

| Metric | Median | Mean | Std | Worst |
| --- | ---: | ---: | ---: | ---: |
| degree | 0.040635 | 0.040952 | 0.003986 | 0.050387 |
| clustering | 0.027000 | 0.027255 | 0.001767 | 0.031041 |
| orbit | 0.013953 | 0.014106 | 0.003776 | 0.023445 |
| spectral | 0.021402 | 0.021420 | 0.001696 | 0.025320 |
| diameter | 0.036910 | 0.038146 | 0.007821 | 0.052258 |
