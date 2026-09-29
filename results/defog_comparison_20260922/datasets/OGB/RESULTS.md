# OGB: results and experiment archive

Updated 2026-09-23

Archive host: cs-cl-19. Root: `/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB`.

LGD is outside this comparison. The OGB GraphVAE and DeFoG artifacts are archived, and a topology-only common-reference evaluation completed on 2026-09-23.

## Result summary

Two different generated-graph artifacts have been evaluated. Their numbers must not be mixed into one comparison.

### Historical GraphVAE final-evaluation outputs

These results use `Single_comp_generatedGraphs_adj_final_eval.npy` generated from the saved best GraphVAE checkpoints. Motif=True setting 03 beat the available Motif=False setting 01 results on F1-PR, but setting 01 had only two completed seeds.

| Setting | Motif | Completed seeds | Test F1-PR | Third-party F1-PR | Degree MMD | Clustering MMD | Orbit MMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | False | 2 | 0.7259 +/- 0.0600 | 0.7447 +/- 0.0471 | 0.0132 +/- 0.0004 | 0.2310 +/- 0.0646 | 0.0023 +/- 0.0024 |
| 03, no temperature annealing | True | 3 | **0.7757 +/- 0.0258** | **0.8007 +/- 0.0103** | **0.0125 +/- 0.0020** | **0.1818 +/- 0.0422** | **0.0012 +/- 0.0007** |
| 06, temperature annealing | True | 2 | 0.7726 +/- 0.0590 | 0.7880 +/- 0.0572 | 0.0112 +/- 0.0060 | 0.2007 +/- 0.0178 | 0.0011 +/- 0.0006 |

### Common topology-only terminal-chain evaluation

This completed evaluation uses three seeds per method, the exact same serialized reference, 289 graphs per collection after nonempty/largest-component cleanup, and 10 RandomGIN evaluator seeds. GraphVAE inputs come from chain 65 of `terminal_chain_binary_matrices.npz`; DeFoG inputs come from its archived generated DGL graphs.

| Method | Seeds | F1-PR | Precision | Recall | MMD-RBF |
|---|---:|---:|---:|---:|---:|
| Motif=False, setting 01 | 3 | 0.661598 | 0.941061 | 0.535294 | 0.850591 |
| Motif=True, setting 03 | 3 | 0.577909 | 0.936678 | 0.479239 | 0.898314 |
| DeFoG | 3 | **0.968970** | **0.961246** | **0.976932** | **0.010217** |

| Method | Degree MMD | Clustering MMD | Orbit MMD | Spectral MMD | Diameter MMD | Triangle MMD | Sparsity MMD | Mean-edge error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Motif=False, setting 01 | 0.098846 | 0.005023 | 0.029593 | 0.151699 | 0.755938 | 1.450e-6 | 3.672e-6 | 19.7647 |
| Motif=True, setting 03 | 0.086423 | 0.005644 | 0.028582 | 0.150930 | 0.798809 | 1.280e-6 | 3.711e-6 | 19.5479 |
| DeFoG | **0.000580** | **0.000294** | **0.000052** | **0.005757** | **0.036438** | **9.960e-9** | **5.194e-9** | **1.2214** |

DeFoG wins every metric in the common terminal-chain comparison. This does **not** establish a controlled `0.7757` versus `0.968970` comparison: `0.7757` belongs to the historical GraphVAE final-evaluation outputs, whereas `0.968970` belongs to the new common terminal-chain protocol. A controlled final-output comparison still requires reevaluating historical GraphVAE `final_eval.npy` outputs and DeFoG together with the same reference, cleanup, graph count, and evaluator seeds. No retraining is required when those saved outputs are available.

## Transfer status

Tmux on system 19: `archive_ogb_20260921`. Log: [transfer.log](manifests/transfer.log). `TRANSFER_COMPLETE` means the listed archive transfers succeeded; it does not mean running model training or LGD evaluation has completed.

Original files are copied and retained. Environments, tar archives, git objects, and selected redundant raw datasets inside code snapshots are excluded. Invalid/old campaigns retain their original names and are not pooled into a new aggregate.

## Existing result reports

The linked historical reports preserve their dates, protocols, seeds, hyperparameters and original comparison tables. Results from different campaigns are not silently combined.

## LGD evaluation

Each seed uses its completed diffusion checkpoint, original encoder and frozen split. Structural metrics and topology Random-GIN use the existing benchmark evaluator (10 initializations, evaluator seed 0). Native-node and node/edge-feature GIN are computed where native features exist. Motif TV uses retained multi-atom training states; counts are summed across graphs and normalized within each rule. Test and training references are reported separately. Training-reference TV uses the same generated sample set as test-reference TV; it is not a matched-collection-size replication.

Errors remain explicitly recorded in each metrics.json; absent results are not zeros.
No LGD metric files are included because LGD was excluded from the requested comparison.

## Full source and destination inventory

| Original host | Original artifact | Archive location |
| --- | --- | --- |
| cs-cl-13 | `/local-scratch/localhome/mirzaei/defog_ogbg_3seed_20260920` | `sources/cs-cl-13/defog_ogbg_3seed_20260920` |
| cs-cl-13 | `/localhome/mirzaei/defog_ogbg_3seed_20260920` | `sources/cs-cl-13/defog_ogbg_3seed_20260920` |
| cs-cl-17 | `/local-scratch/localhome/mirzaei/defog_ogbg_3seed_20260920` | `sources/cs-cl-17/defog_ogbg_3seed_20260920` |
| cs-cl-17 | `/local-scratch/localhome/mirzaei/ogb_vram_20260712_083826` | `sources/cs-cl-17/ogb_vram_20260712_083826` |
| cs-cl-17 | `/local-scratch/localhome/mirzaei/ogb_table2_20260712` | `sources/cs-cl-17/ogb_table2_20260712` |
| cs-cl-17 | `/local-scratch/localhome/mirzaei/ogb_vram_benchmark` | `sources/cs-cl-17/ogb_vram_benchmark` |
| cs-cl-17 | `/localhome/mirzaei/defog_ogbg_3seed_20260920` | `sources/cs-cl-17/defog_ogbg_3seed_20260920` |
| cs-cl-17 | `/localhome/mirzaei/ogb_vram_20260712_083826` | `sources/cs-cl-17/ogb_vram_20260712_083826` |
| cs-cl-17 | `/localhome/mirzaei/ogb_table2_20260712` | `sources/cs-cl-17/ogb_table2_20260712` |
| cs-cl-17 | `/localhome/mirzaei/ogb_vram_benchmark` | `sources/cs-cl-17/ogb_vram_benchmark` |
| cs-cl-18 | `/local-scratch2/mirzaei/defog_ogbg_3seed_20260920` | `sources/cs-cl-18/defog_ogbg_3seed_20260920` |
| cs-cl-19 | `/local-scratch2/mirzaei/defog_ogbg_3seed_20260920` | `sources/cs-cl-19/defog_ogbg_3seed_20260920` |
| cs-cl-18 | `/local-scratch2/new/gather/datasets/ogb` | `sources/gather` |
