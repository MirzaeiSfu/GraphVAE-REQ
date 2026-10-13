# Evaluation scripts behind the paper numbers

The paper's table numbers were computed after training, by scripts that were never committed at the time. They are
collected here, unchanged, so every number can be traced to the script that produced it. **Retraining does not use
this folder** (see [`scripts/retrain/`](../scripts/retrain/)); it is only needed to recompute the evaluations.

## Layout

Each script keeps its original path on the machine it came from:

| Folder | Original location |
|---|---|
| `cs-cl-17/<path>` | `/localhome/mirzaei/<path>` on cs-cl-17 |
| `cs-cl-18/<path>` | `/local-scratch2/mirzaei/<path>` on cs-cl-18 |
| `cs-cl-18-home/<path>` | `/localhome/mirzaei/<path>` on cs-cl-18 |
| `cs-cl-19/<path>` | `/local-scratch2/mirzaei/<path>` on cs-cl-19 |
| `dependencies/` | GraphVAE-REQ code that the scripts import (see below) |

Only `.py`, `.sh`, `.slurm` and `.sbatch` files were collected. Outputs are on the external drive attached to cs-cl-18
(`/media/mirzaei/backup/`).

## Which script computed each table number

| Dataset | Method | Structural MMD | Random-GIN F1-PR |
|---|---|---|---|
| GRID, TRIANGULAR_GRID, LOBSTER | GraphVAE, GraphVAE+RG | `main.py` at the end of training (`final_table2_metrics.json` in each run folder) | `cs-cl-18-home/evaluate_paper_s_20260925.py` |
| GRID, TRIANGULAR_GRID | DeFoG | `cs-cl-17/run_synthetic_common_eval_20260923.sh`, `cs-cl-17/evaluate_topology_gin_20260923.py` | `cs-cl-18-home/evaluate_paper_s_20260925.py` |
| LOBSTER | DeFoG | `cs-cl-18/defog_synthetic_evaluation_20260906/evaluate_defog_synthetic_20260906.py` | `cs-cl-18-home/evaluate_paper_s_20260925.py` |
| PTC | GraphVAE | `cs-cl-18-home/reevaluate_ptc_structural_20260925.py` | `cs-cl-18-home/evaluate_paper_s_20260925.py`; topology features: `cs-cl-18/prepare_ptc_matched_feature_eval.py`, `cs-cl-18/run_ptc_matched_false_eval.sh` |
| PTC | GraphVAE+RG | `main.py` at the end of training | `cs-cl-18-home/evaluate_paper_s_20260925.py` |
| PTC | DeFoG | `cs-cl-18-home/evaluate_defog_ptc_rule_metrics_20260907.py`, `cs-cl-18-home/build_ptc_defog_comparison_20260907.py`, `cs-cl-18/ptc_rule_metrics_20260906/` | `cs-cl-18-home/evaluate_paper_s_20260925.py` |
| PROTEINS | GraphVAE, GraphVAE+RG | `cs-cl-18/proteins_3way_comparison_20260913/evaluate_structural_arrays.py` | `cs-cl-18-home/evaluate_proteins_baselines_20260922.sh`, `cs-cl-18-home/proteins_feature_evaluator_20260922.py` |
| PROTEINS | DeFoG | `cs-cl-17/finish_proteins_corrected_eval_20260922.sh` | same script |
| QM9 | GraphVAE, GraphVAE+RG, DeFoG | `cs-cl-18/qm9_common_eval_20260917/run_common_structural.sh`, which calls `cs-cl-18/aids_common_eval_10k_20260917/scripts/evaluate_dgl_common_metrics.py` | `cs-cl-18/qm9_common_eval_20260917/run_graphvae_attributed_eval.sh` |
| GRID, LOBSTER, PTC, PROTEINS, QM9 | LGD | `cs-cl-18/lgd_common_eval_20260925/scripts/add_dataset.sh` (runs `fetch_lgd.sh`, `sanity.sh`, `run_lgd_eval.sh`) | same script |

`cs-cl-18-home/audit_paper_results_20260924.py` rechecked every table value against its source file and wrote
`NUMERICAL_AUDIT.json` (cs-cl-18: `/local-scratch2/mirzaei/PAPER_RESULTS_AUDIT_20260924/`), which lists the source
file and checksum behind each of the 126 checked values. Compare a recomputed number against that file.

## `dependencies/`: the GraphVAE-REQ code the scripts import

Some scripts import GraphVAE-REQ modules from a specific copy of the repository. Those copies are kept whole, so they
contain their own `main.py`; they are never used for training.

| Folder | Was on cs-cl-18 | Used by |
|---|---|---|
| (none needed) | `/local-scratch2/mirzaei/fb/GraphVAE-REQ`, identical to commit `9f01785` | `evaluate_paper_s_20260925.py`, `reevaluate_ptc_structural_20260925.py`, the PROTEINS evaluation scripts |
| `dependencies/new__GraphVAE-REQ-terminal-motif-mmd/` | `/local-scratch2/new/GraphVAE-REQ-terminal-motif-mmd` (commit `8e7d157` plus 6 changed files) | `run_synthetic_common_eval_20260923.sh`, LGD `run_lgd_eval.sh` and `sanity.sh` |
| `dependencies/mirzaei__aids_common_eval_10k_20260917__source__GraphVAE-REQ/` | `/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ` (`9f01785` plus changed `main.py` and `data.py`) | LGD `lgd_common.py`, `run_lgd_eval.sh`, `sanity.sh` |

A few other folders here (`cs-cl-18/graphvae-smoothed-pruned-20260817/GraphVAE-REQ/`,
`cs-cl-19/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ/`, ...) are also code copies from other campaigns
(AIDS, MUTAG), kept for the record.

## Note

Several scripts were edited while their campaign was running, so an output can come from an earlier version of the
same script. Only the LGD scripts kept a backup, on cs-cl-18 (`/local-scratch2/mirzaei/lgd_common_eval_20260925/scripts.bak_20260925_1840/`). To
confirm a number, rerun the evaluation and compare with `NUMERICAL_AUDIT.json`.
