# LOBSTER GraphVAE Original Temp Guard Sweep Analysis

Generated CSV: `reports/lobster_graphvae_original_temp_guard_sweep_comparison_20260708.csv`

Lower normalized score is better. The score averages degree, clustering, orbit, spectral, diameter, MMD RBF, and F1-PR error after normalization by the same denominator family used for validation selection. Table 3 values use third-party Random GIN metrics when available.

## Ranking

| Rank | Source | Setting | Model | Score | Degree | Clustering | Orbit | Spectral | Diameter | MMD RBF | MMD Linear | Precision | Recall | F1-PR | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | lobster_docx_20260708 | 02 | GraphVAE-MM | 0.305408602812 | 0.0014 | 0 | 0.014 | 0.0119 | 0.099 | 0.1103 | 17.3809 | 1 | 1 | 1 |  |
| 2 | lobster_docx_20260708 | 09 | GraphVAE-MM motif original | 0.396766088442 | 0.0032 | 0 | 0.0078 | 0.0181 | 0.1287 | 0.1396 | 18.4406 | 1 | 1 | 1 |  |
| 3 | lobster_docx_20260708 | 06 | GraphVAE motif original temp | 0.465120983015 | 0.0296 | 0.0381 | 0.0268 | 0.0331 | 0.0228 | 0.1999 | 13.1825 | 1 | 1 | 1 | Source metrics CSV says this was a post-hoc eval on a saved best-validation generated graph sample; the source run crashed before normal finalization. |
| 4 | lobster_docx_20260708 | 10 | GraphVAE-MM motif literals | 0.843228083381 | 0.0045 | 0 | 0.0211 | 0.0337 | 0.2889 | 0.2745 | 43.9827 | 1 | 0.98 | 0.9898 |  |
| 5 | loss_weight_sweep_docx_20260708 | coarse_01 | GraphVAE+Motif both no-temp | 1.01288801778 | 0.0381 | 0.3933 | 0.0405 | 0.0415 | 0.231 | 0.2197 | 43.382 | 0.895 | 0.99 | 0.9375 |  |
| 6 | lobster_docx_20260708 | 03 | GraphVAE motif original | 1.11338138353 | 0.0286 | 0.1782 | 0.0493 | 0.0447 | 0.2221 | 0.2897 | 85.9737 | 0.85 | 1 | 0.9175 |  |
| 7 | lobster_docx_20260708 | 11 | GraphVAE-MM motif both | 1.14569098974 | 0.0041 | 0 | 0.0578 | 0.0383 | 0.3559 | 0.4371 | 70.3934 | 1 | 1 | 1 |  |
| 8 | lobster_docx_20260708 | 05 | GraphVAE motif both | 1.25326479718 | 0.0344 | 0.0991 | 0.0519 | 0.0578 | 0.2725 | 0.4238 | 83.5399 | 0.935 | 1 | 0.9654 |  |
| 9 | loss_weight_sweep_docx_20260708 | coarse_02 | GraphVAE+Motif both no-temp | 1.28664788869 | 0.0273 | 0.0897 | 0.0217 | 0.0485 | 0.3809 | 0.4353 | 72.0375 | 0.97 | 1 | 0.9841 |  |
| 10 | loss_weight_sweep_docx_20260708 | coarse_06 | GraphVAE+Motif both no-temp | 1.28982685233 | 0.0296 | 0.0836 | 0.0191 | 0.0446 | 0.2618 | 0.4717 | 55.3462 | 0.91 | 1 | 0.9522 |  |
| 11 | loss_weight_sweep_docx_20260708 | coarse_08 | GraphVAE+Motif both no-temp | 1.40231063823 | 0.0516 | 0.3485 | 0.0874 | 0.0802 | 0.4593 | 0.045 | 13140000000 | 0.74 | 1 | 0.8485 |  |
| 12 | lobster_docx_20260708 | 08 | GraphVAE motif both temp | 1.42132603734 | 0.0306 | 0.0427 | 0.0806 | 0.0729 | 0.2819 | 0.48 | 66.2002 | 0.905 | 1 | 0.9495 |  |

## Guarded Sweep Takeaways

- Best evaluated guarded run: `n1_m0p1` (alpha_node/edge=1.0, alpha_motif=0.1), score 1.63812639088, F1-PR 0.808833903542, MMD RBF 0.0799462080002.
- Next guarded candidate: `n0p3_m0p1` with score 1.75113125977, F1-PR 0.785389959568, MMD RBF 0.0152356505394.
- Next guarded candidate: `n0p3_m0p03` with score 1.9786357894, F1-PR 0.930308770297, MMD RBF 0.583551970124.
- `n1_m0p03` is not final-test comparable: missing final test eval; training log contains NaN loss; crashed during plotting with StopIteration. Its best validation score was 0.479105216525 at epoch 6999.

## Source Notes

- Requested DOCX files `lobster.docx` and `lobster_loss_weight_sweep_results_20260708.docx` are now parsed directly from `reports/`. The older 20260630 CSV exports are intentionally not used because they do not match these later DOCX reports.
- Guarded sweep rows come from `final_metrics_summary.json`, `best_validation_mmd.json`, and `manifest.csv` under the 20260708 sweep paths.
