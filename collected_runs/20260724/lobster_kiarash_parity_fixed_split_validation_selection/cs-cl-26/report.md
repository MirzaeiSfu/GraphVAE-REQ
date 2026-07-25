# Per-run Lobster posthoc checkpoint selection

Every checkpoint was ranked using validation graphs only. All per-run winners were frozen before any held-out reference was loaded.

| Run | Selected checkpoint | Validation median | Std | Dense rate | Selection score |
|---|---|---:|---:|---:|---:|
| seed_1/lobster_kiarash_parity_plain1_1_corrected__cs-cl-26_gpu0/seed_1 | periodic_epoch_08000.pt | 1.275878 | 0.229695 | 14.00% | 2.091249 |
| seed_2/lobster_kiarash_parity_kia40_2000_legacy__cs-cl-26_gpu1/seed_2 | periodic_epoch_08000.pt | 0.184446 | 0.109743 | 0.00% | 0.233915 |

## Global validation winner

`seed_2/lobster_kiarash_parity_kia40_2000_legacy__cs-cl-26_gpu1/seed_2/periodic_epoch_08000.pt`
