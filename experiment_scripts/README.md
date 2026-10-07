# Experiment scripts recovered from the lab machines

These scripts were never in git. They ran, evaluated and reported the experiments, and until now existed only on
the lab disks. Each file is kept at its original path:

- `cs-cl-18/<path>` was `/local-scratch2/mirzaei/<path>` on cs-cl-18
- `cs-cl-19/<path>` was `/local-scratch2/mirzaei/<path>` on cs-cl-19
- `cs-cl-17/<path>` was `/localhome/mirzaei/<path>` on cs-cl-17

Only `.py`, `.sh`, `.slurm` and `.sbatch` files are included, and only those whose content is not already in the
repository history. Left out: vendored Python packages, upstream DeFoG/LGD code (see `third_party/` on branch
`baselines-code-and-provenance`), backups and shell snapshots. The outputs these scripts produced are on the
external drive attached to cs-cl-18.

Notable folders behind the paper numbers:

| Folder | What it does |
|---|---|
| `cs-cl-18/aids_common_eval_10k_20260917/` | 17 files |
| `cs-cl-18/qm9_common_eval_20260917/` | 14 files |
| `cs-cl-18/lgd_common_eval_20260925/` | 17 files |
| `cs-cl-18/defog_synthetic_evaluation_20260906/` | 2 files |
| `cs-cl-18/proteins_3way_comparison_20260913/` | 4 files |
| `cs-cl-18/rule_mmd_20260923/` | 9 files |
| `cs-cl-19/rule_mmd_20260923/` | 3 files |
| `cs-cl-17/rule_mmd_20260923/` | 3 files |
| `cs-cl-18/ptc_rule_metrics_20260906/` | 3 files |
| `cs-cl-18/PAPER_REVISION_GRAPHVAE_FULL_20260922/` | 1 files |
| `cs-cl-18/mutag_edgefeat_campaign_20260910/` | 3 files |
| `cs-cl-19/mutag_edgefeat_campaign_20260910/` | 5 files |

`cs-cl-19/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ/` holds the uncommitted `main.py` and `data.py` used by 9 MUTAG/AIDS runs (see `reports/experiment_code_provenance.csv`).
