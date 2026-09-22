# OGB: results and experiment archive

Updated 2026-09-22T08:28:22

Archive host: cs-cl-19. Root: `/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB`.

No completed OGB LGD checkpoint found; LGD evaluation skipped. Available GraphVAE/DeFoG artifacts are being archived.

## Transfer status

Tmux on system 19: `archive_ogb_20260921`. Log: [transfer.log](manifests/transfer.log). `TRANSFER_COMPLETE` means the listed archive transfers succeeded; it does not mean running model training or LGD evaluation has completed.

Original files are copied and retained. Environments, tar archives, git objects, and selected redundant raw datasets inside code snapshots are excluded. Invalid/old campaigns retain their original names and are not pooled into a new aggregate.

## Existing result reports

The linked historical reports preserve their dates, protocols, seeds, hyperparameters and original comparison tables. Results from different campaigns are not silently combined.

## LGD evaluation

Each seed uses its completed diffusion checkpoint, original encoder and frozen split. Structural metrics and topology Random-GIN use the existing benchmark evaluator (10 initializations, evaluator seed 0). Native-node and node/edge-feature GIN are computed where native features exist. Motif TV uses retained multi-atom training states; counts are summed across graphs and normalized within each rule. Test and training references are reported separately. Training-reference TV uses the same generated sample set as test-reference TV; it is not a matched-collection-size replication.

Errors remain explicitly recorded in each metrics.json; absent results are not zeros.
No completed LGD metric files have reached the archive yet.

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
