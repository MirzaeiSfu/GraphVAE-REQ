# Which code produced each experiment

This file maps every result in the paper's two main tables (structural MMD and Random-GIN F1-PR) to the
exact code that produced it. Commit hashes link to this repository unless marked as DeFoG or LGD.

Many runs were launched from copies of the repository that had no `.git` folder, so they did not record a commit.
For those runs the code was identified by comparing the saved run settings and code files with the git history.
The **Status** column says how each entry was established:

| Status | Meaning |
|---|---|
| **Recorded** | The run saved its commit at launch (`reproducibility.json` or `job_record.json`). |
| **Verified** | The run's launch folder was found, and its code files are byte-identical to the commit shown. For GRID, triangular grid and LOBSTER this is the `fb/GraphVAE-REQ` folder used on cs-cl-16, 17, 18 and 19, which equals `9f01785` (that commit was made the morning after the first runs started). |
| **Inferred** | `main.py` in the commit shown has exactly the run's options. Other modules were not checked. |
| **Recovered** | The code was not committed when the run was launched. It was committed later, byte-identical to the run's launch folder, on branch `recovered/paper-code-20260904`. See [Recovered code](#recovered-code). |

The exact configuration of every run listed here is in [`configs/paper_runs/`](configs/paper_runs/).

Run folders refer to `EXPERIMENT_ARCHIVE_20260921` unless another location is given. A copy of that archive, and of
everything listed here, is on the external drive attached to cs-cl-18 (`/media/mirzaei/backup/`).

## Paper tables

### GRID

| Method | Seeds | Code | Status | Run folders |
|---|---|---|---|---|
| GraphVAE | 0, 1, 2 | [`9f01785`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/9f01785b4) | Verified | `GRID/experiments/graphvae/motif_false/` |
| GraphVAE+RG (full matrix, λ=0.1) | 0, 1, 2 | [`9f01785`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/9f01785b4) | Verified | `GRID/experiments/graphvae/motif_true_full_matrix/` |
| DeFoG | 0, 4, 5 | DeFoG [`c631697`](https://github.com/MirzaeiSfu/defog/commit/c631697b9cd5a2474d22ba12de33943c6b49e53e) | Recorded | `GRID/experiments/defog/` |
| LGD | 0, 1, 2 | LGD `f597e1d` + local changes | Recorded (campaign) | Solar `~/LGD_FIX_20260924/results/` |

### Triangular grid

| Method | Seeds | Code | Status | Run folders |
|---|---|---|---|---|
| GraphVAE | 0, 1, 2 | [`9f01785`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/9f01785b4) | Verified | `TRIANGULAR_GRID/experiments/graphvae/motif_false/` |
| GraphVAE+RG (full matrix, λ=0.1) | 0, 1, 2 | [`9f01785`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/9f01785b4) | Verified | `TRIANGULAR_GRID/experiments/graphvae/motif_true_full_matrix/` |
| DeFoG | 0, 1, 3 | DeFoG [`c631697`](https://github.com/MirzaeiSfu/defog/commit/c631697b9cd5a2474d22ba12de33943c6b49e53e) | Recorded | `TRIANGULAR_GRID/experiments/defog/` |
| LGD | — | not reported in the paper | | |

### LOBSTER

| Method | Seeds | Code | Status | Run folders |
|---|---|---|---|---|
| GraphVAE | 0, 1, 2 | [`9f01785`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/9f01785b4) | Verified | `LOBSTER/experiments/graphvae/motif_false/` |
| GraphVAE+RG (full matrix, λ=0.1) | 0, 1, 2 | [`9f01785`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/9f01785b4) | Verified | `LOBSTER/experiments/graphvae/motif_true_full_matrix/` |
| DeFoG | 0, 1, 2 | DeFoG [`c631697`](https://github.com/MirzaeiSfu/defog/commit/c631697b9cd5a2474d22ba12de33943c6b49e53e) | Recorded | `LOBSTER/experiments/defog/` |
| LGD | 0, 1, 2 | LGD `f597e1d` + local changes | Recorded (campaign) | `LGD_FIX_20260924/` on the lab machines |

### PTC

| Method | Seeds | Code | Status | Run folders |
|---|---|---|---|---|
| GraphVAE | 0, 1, 2 | [`c10bff5`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/c10bff58b) | Recorded (working tree had no code changes) | cs-cl-18 `/local-scratch2/new/gather/datasets/ptc/setting_01/` |
| GraphVAE+RG (full matrix, λ=0.1) | 0 | [`f75f761`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/f75f761fd) (code A) | Recovered (launch folder committed afterwards; byte-identical) | cs-cl-18 `/local-scratch2/mirzaei/motif_true_clean_20260906/ptc/full_matrix/seed_0/` |
| GraphVAE+RG (full matrix, λ=0.1) | 1, 2 | [`af56479`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/af56479bb) (code B) | Recovered (launch folder committed afterwards; byte-identical) | cs-cl-18 `/local-scratch2/mirzaei/motif_true_clean_20260906/ptc/full_matrix/seed_{1,2}/` |
| DeFoG | 0, 1, 2 | DeFoG [`c631697`](https://github.com/MirzaeiSfu/defog/commit/c631697b9cd5a2474d22ba12de33943c6b49e53e) | Recorded | `PTC/sources/cs-cl-18/defog_ptc_frozen_20260906/jobs/ptc/`; metrics in cs-cl-18 `defog_ptc_full_metrics_20260907/` |
| LGD | 0, 1, 2 | LGD `f597e1d` + local changes | Recorded (campaign) | `LGD_FIX_20260924/` on the lab machines |

### PROTEINS

| Method | Seeds | Code | Status | Run folders |
|---|---|---|---|---|
| GraphVAE | 0, 1, 2 | [`2a80b0a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/2a80b0aa0) | Inferred (the runs predate this commit, so the code was committed afterwards) | `PROTEINS/graphvae_motif_false/setting_01/` |
| GraphVAE+RG (full matrix, λ=0.03) | 0, 1, 2 | [`af56479`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/af56479bb) (code B) | Recovered (launch folder committed afterwards; byte-identical) | `PROTEINS/graphvae_motif_true/alpha_003_full_matrix/`; Solar `~/proteins_full_matrix_motif003_20260910/` |
| DeFoG | 0, 1, 2 | DeFoG [`c631697`](https://github.com/MirzaeiSfu/defog/commit/c631697b9cd5a2474d22ba12de33943c6b49e53e) | Inferred (the DeFoG commit pinned for every campaign; these runs did not record it) | `PROTEINS/defog/corrected_generation_20260922/` |
| LGD | 0, 1, 2 | LGD `f597e1d` + local changes | Recorded (campaign) | Solar `~/LGD_FIX_20260924/results/` |

### QM9

| Method | Seeds | Code | Status | Run folders |
|---|---|---|---|---|
| GraphVAE (250 epochs) | 0, 1, 2 | [`af56479`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/af56479bb) (code B) | Recovered (launch folder committed afterwards; byte-identical) | Solar `~/qm9_motif_false_fast250_20260915/runs/`; final models in `QM9/sources/cs-cl-18/qm9_common_eval_20260917/graphvae/false/` |
| GraphVAE+RG (full matrix, λ=0.1, 250 epochs) | 0, 1, 2 | [`af56479`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/af56479bb) (code B) | Recovered (launch folder committed afterwards; byte-identical) | Solar `~/qm9_full_fast250_cp_smoothed_top10_20260914/runs/full_matrix/`; final models in `QM9/sources/cs-cl-18/qm9_common_eval_20260917/graphvae/true_full/` |
| DeFoG | 0, 1, 2 | DeFoG [`c631697`](https://github.com/MirzaeiSfu/defog/commit/c631697b9cd5a2474d22ba12de33943c6b49e53e) + 3 local files | Verified (launch folder found) | cs-cl-17 `qm9_defog_seedfixed250_batch128_20260916/`, code from `qm9_defog_lab24_20260914/work/source/` |
| LGD | 0, 1, 2 | LGD `f597e1d` + local changes | Recorded (campaign) | Solar `~/LGD_FIX_20260924/results/` |

## Recovered code

Two versions of the code behind paper results were never committed when the runs were launched. They were committed
afterwards from the runs' launch folders, on branch
[`recovered/paper-code-20260904`](https://github.com/MirzaeiSfu/GraphVAE-REQ/tree/recovered/paper-code-20260904),
on top of [`9f01785`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/9f01785b4). Every code file (`.py`, `.sh`,
`.yaml`, `.sbatch`) in each commit is byte-identical to its launch folder, and each `main.py` has exactly the options
the runs recorded.

| Commit | Used for | Changes | Taken from |
|---|---|---|---|
| [`f75f761`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/f75f761fd) (code A, written 2026-09-04 09:38) | PTC GraphVAE+RG seed 0 | `main.py` (adds `--motif_prune_score_threshold`), `motif_counting/motif_counter.py`, `loss_weight_utils.py`, plus 37 configs and launch scripts | cs-cl-17 `/var/tmp/mirzaei_archive/ali/GraphVAE-REQ-kia-motif-20260904-cl17/` |
| [`af56479`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/af56479bb) (code B, written 2026-09-04 15:30) | PTC GraphVAE+RG seeds 1–2, PROTEINS GraphVAE+RG, QM9 GraphVAE and GraphVAE+RG | code A plus `main.py` (adds `--resume_from_latest_checkpoint`), `data.py`, and the PTC/AIDS launch and monitoring scripts | Solar `/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/` |

To rerun one of these experiments, check out the commit and use the `command` saved in the run's
`reproducibility.json`.

## Baselines

- **DeFoG**: fork [`MirzaeiSfu/defog`](https://github.com/MirzaeiSfu/defog), branch `feat/frozen-graphvae-benchmark`,
  commit `c631697`. The QM9 runs also used uncommitted changes to `src/main.py`, `src/datasets/qm9_dataset.py` and
  `src/analysis/ggmeval/evaluation/models/gin/gin.py`.
- **LGD**: upstream [`zhouc20/LatentGraphDiffusion`](https://github.com/zhouc20/LatentGraphDiffusion) commit `f597e1d`
  plus local changes, from the `LGD_FIX_20260924` campaign, where the encoder checkpoint is chosen by validation
  `loss_recon` and never epoch 0.
- The code for both baselines, with the local changes, is in `third_party/defog/` and `third_party/lgd/` on branch
  [`baselines-code-and-provenance`](https://github.com/MirzaeiSfu/GraphVAE-REQ/tree/baselines-code-and-provenance).

## Other experiments

All 346 runs in `EXPERIMENT_ARCHIVE_20260921`, including AIDS, MUTAG, OGB and runs not in the paper, are listed one
per row in
[`reports/experiment_code_provenance.csv`](https://github.com/MirzaeiSfu/GraphVAE-REQ/blob/baselines-code-and-provenance/reports/experiment_code_provenance.csv)
on branch `baselines-code-and-provenance`, with the method used for each entry.

## Evaluation code

The paper numbers were computed by evaluation scripts that are not in git:
`aids_common_eval_10k_20260917/scripts/`, `qm9_common_eval_20260917/`, `lgd_common_eval_20260925/scripts/`,
`defog_synthetic_evaluation_20260906/`, `proteins_3way_comparison_20260913/` and `rule_mmd_20260923/`, all on cs-cl-18.
Copies are on the external drive. `PAPER_RESULTS_AUDIT_20260924/NUMERICAL_AUDIT.json` on cs-cl-18 lists the source
file and checksum behind each of the 126 checked table values.
