# Experiment code provenance

Generated 2026-10-05 from `EXPERIMENT_ARCHIVE_20260921` (copy on the cs-cl-18 external drive, `/media/mirzaei/backup/`). One row per archived run is in [`experiment_code_provenance.csv`](experiment_code_provenance.csv).

## Summary

- **GraphVAE-REQ runs:** 177
  - 32 recorded their commit at launch (exact).
  - 93 were launched from copies of the repo without `.git`, so no commit was recorded. Their commit is inferred (see Method).
  - **52 ran code that was never committed.** The code still exists on lab disks; see [Code that is not in git](#code-that-is-not-in-git).
- **DeFoG runs:** 129, all on the [`MirzaeiSfu/defog`](https://github.com/MirzaeiSfu/defog/commit/c631697b9cd5a2474d22ba12de33943c6b49e53e) fork at `c631697`. That code, plus 3 uncommitted QM9/GRID changes, is in [`third_party/defog/`](../third_party/defog/).
- **LGD runs in the archive:** 40, all from the old `LGD_3SEED_CAMPAIGN_20260920` (encoder-selection bug). The fixed `LGD_FIX_20260924` runs behind the paper's LGD numbers are not in this archive. Their code is in [`third_party/lgd/`](../third_party/lgd/).

## Method

1. **Exact:** `reproducibility.json` in the run folder has a `git_commit`. If the working tree was dirty, the run folder also has `git_diff.patch`.
2. **Inferred:** without a recorded commit, the run's argument names (from `reproducibility.json`) are compared with the `argparse` options of `main.py` in every commit on every branch. The newest matching commit made before the run started is reported, along with the full range of matching commits. This pins `main.py` exactly but does not verify other modules (`motif_counting/`, `data.py`), so treat it as the most likely commit, not a proof.
3. **Not in git:** the run's argument set matches no commit, but it does match a `main.py` found on disk. That copy is listed as the code location.

## Per dataset

### AIDS

| Model / method | Runs | Code commit | Confidence |
|---|---:|---|---|
| DeFoG | 16 | `c631697 (campaign pin, not recorded per run)` | inferred from campaign |
| GraphVAE (motif=False) | 1 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE (motif=False) | 2 | `NOT IN GIT` | exact argparse match to an on-disk copy |
| GraphVAE (motif=False) | 3 | [`3fb44e6b6`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/3fb44e6b6) | exact (recorded at launch) |
| GraphVAE+RG motif=True (mode option not yet in code) (lambda=0.1) | 3 | [`3fb44e6b6`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/3fb44e6b6) | exact (recorded at launch) |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 2 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 7 | `NOT IN GIT` | exact argparse match to an on-disk copy |
| GraphVAE+RG motif=True full_matrix (lambda=0.15) | 1 | `NOT IN GIT` | exact argparse match to an on-disk copy |
| GraphVAE+RG motif=True total_count (lambda=0.1) | 7 | `NOT IN GIT` | exact argparse match to an on-disk copy |
| LGD (LGD_3SEED_CAMPAIGN_20260920) | 6 | `zhouc20/LatentGraphDiffusion f597e1d + local changes` | exact (campaign repo state) |

### GRID

| Model / method | Runs | Code commit | Confidence |
|---|---:|---|---|
| DeFoG | 6 | [`c631697b9`](https://github.com/MirzaeiSfu/defog/commit/c631697b9) | exact (job_record.json) |
| GraphVAE (motif=False) | 3 | [`90ee5d6bd`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/90ee5d6bd) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 1 | [`90ee5d6bd`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/90ee5d6bd) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 1 | [`613fe9e7d`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/613fe9e7d) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 1 | [`b0348ce15`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b0348ce15) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True total_count (lambda=0.1) | 1 | [`90ee5d6bd`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/90ee5d6bd) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True total_count (lambda=0.1) | 2 | [`b0348ce15`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b0348ce15) | inferred: newest commit before the run with an identical main.py argument set |

### LOBSTER

| Model / method | Runs | Code commit | Confidence |
|---|---:|---|---|
| DeFoG | 6 | [`c631697b9`](https://github.com/MirzaeiSfu/defog/commit/c631697b9) | exact (job_record.json) |
| GraphVAE (motif=False) | 3 | [`90ee5d6bd`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/90ee5d6bd) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 2 | [`90ee5d6bd`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/90ee5d6bd) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 1 | [`613fe9e7d`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/613fe9e7d) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True total_count (lambda=0.1) | 2 | [`90ee5d6bd`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/90ee5d6bd) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True total_count (lambda=0.1) | 1 | [`613fe9e7d`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/613fe9e7d) | inferred: newest commit before the run with an identical main.py argument set |
| LGD (LGD_3SEED_CAMPAIGN_20260920) | 8 | `zhouc20/LatentGraphDiffusion f597e1d + local changes` | exact (campaign repo state) |

### MUTAG

| Model / method | Runs | Code commit | Confidence |
|---|---:|---|---|
| DeFoG | 20 | `c631697 (campaign pin, not recorded per run)` | inferred from campaign |
| GraphVAE (motif=False) | 4 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE (motif=False) | 3 | [`c10bff58b`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/c10bff58b) | exact (recorded at launch) |
| GraphVAE+RG motif=True (mode option not yet in code) (lambda=0.1) | 3 | [`c10bff58b`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/c10bff58b) | exact (recorded at launch) |
| GraphVAE+RG motif=True full_matrix (lambda=0.01) | 3 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.05) | 3 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 7 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 2 | [`90ee5d6bd`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/90ee5d6bd) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 2 | [`9f01785b4`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/9f01785b4) | inferred (low): run predates every commit with this main.py, so the code was committed after the run; first such commit shown |
| GraphVAE+RG motif=True full_matrix (lambda=0.15) | 3 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.15) | 4 | `NOT IN GIT` | exact argparse match to an on-disk copy |
| GraphVAE+RG motif=True full_matrix (lambda=0.2) | 3 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.25) | 3 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.3) | 3 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.75) | 1 | `NOT IN GIT` | exact argparse match to an on-disk copy |
| GraphVAE+RG motif=True full_matrix (lambda=1.5) | 1 | `NOT IN GIT` | exact argparse match to an on-disk copy |
| LGD (LGD_3SEED_CAMPAIGN_20260920) | 9 | `zhouc20/LatentGraphDiffusion f597e1d + local changes` | exact (campaign repo state) |

### OGB

| Model / method | Runs | Code commit | Confidence |
|---|---:|---|---|
| DeFoG | 17 | `c631697 (campaign pin, not recorded per run)` | inferred from campaign |
| GraphVAE (motif=False) | 1 | [`2a80b0aa0`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/2a80b0aa0) | inferred (low): run predates every commit with this main.py, so the code was committed after the run; first such commit shown |
| GraphVAE (motif=False) | 3 | [`2a80b0aa0`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/2a80b0aa0) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True (mode option not yet in code) (lambda=0.0) | 5 | [`2a80b0aa0`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/2a80b0aa0) | inferred (low): run predates every commit with this main.py, so the code was committed after the run; first such commit shown |
| GraphVAE+RG motif=True (mode option not yet in code) (lambda=0.1) | 13 | [`2a80b0aa0`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/2a80b0aa0) | inferred: newest commit before the run with an identical main.py argument set |

### PROTEINS

| Model / method | Runs | Code commit | Confidence |
|---|---:|---|---|
| DeFoG | 15 | `c631697 (campaign pin, not recorded per run)` | inferred from campaign |
| GraphVAE (motif=False) | 3 | [`2a80b0aa0`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/2a80b0aa0) | inferred (low): run predates every commit with this main.py, so the code was committed after the run; first such commit shown |
| GraphVAE+RG motif=True full_matrix (lambda=0.03) | 3 | `NOT IN GIT` | exact argparse match to an on-disk copy |
| GraphVAE+RG motif=True full_matrix (lambda=0.2) | 2 | [`b7268544a`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b7268544a) | inferred: newest commit before the run with an identical main.py argument set |

### PTC

| Model / method | Runs | Code commit | Confidence |
|---|---:|---|---|
| DeFoG | 18 | [`c631697b9`](https://github.com/MirzaeiSfu/defog/commit/c631697b9) | exact (job_record.json) |
| DeFoG | 10 | `c631697 (campaign pin, not recorded per run)` | inferred from campaign |
| GraphVAE (motif=False) | 9 | [`c10bff58b`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/c10bff58b) | exact (recorded at launch) |
| GraphVAE+RG motif=True (mode option not yet in code) (lambda=0.1) | 3 | [`c10bff58b`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/c10bff58b) | exact (recorded at launch) |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 13 | `NOT IN GIT` | exact argparse match to an on-disk copy |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 2 | [`946146f4d`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/946146f4d) | exact (recorded at launch) |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 1 | [`90ee5d6bd`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/90ee5d6bd) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True total_count (lambda=0.1) | 13 | `NOT IN GIT` | exact argparse match to an on-disk copy |
| GraphVAE+RG motif=True total_count (lambda=0.1) | 2 | [`9f01785b4`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/9f01785b4) | inferred (low): run predates every commit with this main.py, so the code was committed after the run; first such commit shown |
| LGD (LGD_3SEED_CAMPAIGN_20260920) | 9 | `zhouc20/LatentGraphDiffusion f597e1d + local changes` | exact (campaign repo state) |

### QM9

| Model / method | Runs | Code commit | Confidence |
|---|---:|---|---|
| DeFoG | 15 | `c631697 (campaign pin, not recorded per run)` | inferred from campaign |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 5 | [`946146f4d`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/946146f4d) | exact (recorded at launch) |
| GraphVAE+RG motif=True total_count (lambda=0.1) | 1 | [`946146f4d`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/946146f4d) | exact (recorded at launch) |

### TRIANGULAR_GRID

| Model / method | Runs | Code commit | Confidence |
|---|---:|---|---|
| DeFoG | 6 | [`c631697b9`](https://github.com/MirzaeiSfu/defog/commit/c631697b9) | exact (job_record.json) |
| GraphVAE (motif=False) | 3 | [`90ee5d6bd`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/90ee5d6bd) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 1 | [`613fe9e7d`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/613fe9e7d) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True full_matrix (lambda=0.1) | 2 | [`b0348ce15`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b0348ce15) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True total_count (lambda=0.1) | 1 | [`90ee5d6bd`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/90ee5d6bd) | inferred: newest commit before the run with an identical main.py argument set |
| GraphVAE+RG motif=True total_count (lambda=0.1) | 2 | [`b0348ce15`](https://github.com/MirzaeiSfu/GraphVAE-REQ/commit/b0348ce15) | inferred: newest commit before the run with an identical main.py argument set |
| LGD (LGD_3SEED_CAMPAIGN_20260920) | 8 | `zhouc20/LatentGraphDiffusion f597e1d + local changes` | exact (campaign repo state) |

## Code that is not in git

These GraphVAE-REQ runs used `main.py` options that exist in no commit on any branch. Exact copies were found on disk:

| Argument fingerprint | Runs | What the code adds | Where it is |
|---|---:|---|---|
| `643ccccbb9` | 24 | adds --motif_prune_score_threshold | cs-cl-17:/var/tmp/mirzaei_archive/ali/GraphVAE-REQ-kia-motif-20260904-cl17/ (main.py, motif_counting/motif_counter.py dated 2026-09-04 09:38) |
| `e1a5cc9dfa` | 19 | adds --motif_prune_score_threshold and --resume_from_latest_checkpoint | cs-cl-13:/localhome/mirzaei/solar_patch_work/ (main.py 2026-09-04 15:17, data.py 15:19) laid over the GraphVAE-REQ-kia-motif-20260904-cl17 tree |
| `9c2290e55f` | 9 | adds --alpha_degree_distribution_loss and --alpha_edge_density_loss | cs-cl-18 and cs-cl-19:/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ/ (identical main.py also in cs-cl-18 aids_common_eval_10k_20260917/source/GraphVAE-REQ/) |
| `493cf42979` | 6 | adds --motif_prune_max_total_values | commit 3fb44e6 plus the git_diff.patch saved in each run folder; full tree also in cs-cl-18 and cs-cl-19:/local-scratch2/new/deploy_alpha005_20260720/GraphVAE-REQ/ |

Affected runs, by label:

- `643ccccbb9`: AIDS ptc-aids-motif-AIDS-full_matrix-m0.1-s0; AIDS ptc-aids-motif-AIDS-full_matrix-m0.1-s1; AIDS ptc-aids-motif-AIDS-full_matrix-m0.1-s2; AIDS ptc-aids-motif-AIDS-total_count-m0.1-s0; AIDS ptc-aids-motif-AIDS-total_count-m0.1-s1; AIDS ptc-aids-motif-AIDS-total_count-m0.1-s2; AIDS ptc-aids-motif-PTC-full_matrix-m0.1-s0; AIDS ptc-aids-motif-PTC-total_count-m0.1-s0; AIDS ptc-cl17-full_matrix-m0.1-s0; AIDS ptc-cl17-total_count-m0.1-s0; PTC ptc-aids-motif-AIDS-full_matrix-m0.1-s0; PTC ptc-aids-motif-AIDS-full_matrix-m0.1-s1; PTC ptc-aids-motif-AIDS-full_matrix-m0.1-s2; PTC ptc-aids-motif-AIDS-total_count-m0.1-s0; PTC ptc-aids-motif-AIDS-total_count-m0.1-s1; PTC ptc-aids-motif-AIDS-total_count-m0.1-s2; PTC ptc-aids-motif-PTC-full_matrix-m0.1-s0; PTC ptc-aids-motif-PTC-total_count-m0.1-s0; PTC ptc-cl17-full_matrix-m0.1-s0 (x3); PTC ptc-cl17-total_count-m0.1-s0 (x3)
- `e1a5cc9dfa`: AIDS ptc-aids-motif-PTC-full_matrix-m0.1-s1; AIDS ptc-aids-motif-PTC-full_matrix-m0.1-s2; AIDS ptc-aids-motif-PTC-total_count-m0.1-s1; AIDS ptc-aids-motif-PTC-total_count-m0.1-s2; PROTEINS proteins-full-matrix-motif003-seed-0; PROTEINS proteins-full-matrix-motif003-seed-1; PROTEINS proteins-full-matrix-motif003-seed-2; PTC ptc-aids-motif-PTC-full_matrix-m0.1-s1 (x3); PTC ptc-aids-motif-PTC-full_matrix-m0.1-s2 (x3); PTC ptc-aids-motif-PTC-total_count-m0.1-s1 (x3); PTC ptc-aids-motif-PTC-total_count-m0.1-s2 (x3)
- `9c2290e55f`: AIDS aids-graphvae-false-seed-1; AIDS aids-graphvae-false-seed-2; AIDS mutag-edgefeat-full-matrix-20k; MUTAG mutag-edgefeat-full-cpsmoothed-motif15-seed0; MUTAG mutag-edgefeat-full-matrix-20k (x2); MUTAG mutag-edgefeat-full-matrix-topology-aware-20k (x3)
- `493cf42979`: AIDS solar-aids-setting01-seed0; AIDS solar-aids-setting01-seed1; AIDS solar-aids-setting01-seed2; AIDS solar-aids-setting03-seed0; AIDS solar-aids-setting03-seed1; AIDS solar-aids-setting03-seed2

To put this code under version control, commit each copy as its own branch (for example `recovered/code-<name>`) and then regenerate this report.

## Baselines

- **DeFoG**: fork `MirzaeiSfu/defog`, branch `feat/frozen-graphvae-benchmark`, commit `c631697` (2026-09-01, *Add frozen GraphVAE benchmark adapter*). The frozen-benchmark runs record `defog_commit` in `job_record.json`. The QM9 campaigns (`qm9_defog_campaign_20260914`, `qm9_defog_lab24_20260914`, `QM9_FULL_TEST_20260925`) and `GRID_DEFOG_SEED6_20260924` ran with 3 uncommitted files, which are included in `third_party/defog/`: `src/main.py`, `src/datasets/qm9_dataset.py`, `src/analysis/ggmeval/evaluation/models/gin/gin.py`.
- **LGD**: upstream `zhouc20/LatentGraphDiffusion` commit `f597e1d` (2024-12-17) plus local changes (GraphVAE-REQ dataset export and loader, TU generation and sampling, configs). `third_party/lgd/` holds the fixed `LGD_FIX_20260924` state: encoder checkpoint chosen by validation `loss_recon`, never epoch 0. The old `LGD_3SEED_CAMPAIGN_20260920` differs in `sample_tu.py` and `select_best_encoder_ckpt.py`; those versions are in `third_party/lgd/legacy_LGD_3SEED_CAMPAIGN_20260920/`.

## Fixed LGD runs (not in the archive)

The LGD numbers in the paper come from `LGD_FIX_20260924`, which ran after the archive was made. Code: `third_party/lgd/` (upstream `f597e1d` plus the fixed local changes). Run folders, now also copied to `POST_ARCHIVE_RESULTS_20261005/` on the external drive:

| Where | Datasets / seeds |
|---|---|
| Solar `~/LGD_FIX_20260924/results/` | GRID 0-2, PROTEINS 0-2, QM9 0-2, TRIANGULAR_GRID b16 seed 1 and b32 seeds 0/2 |
| cs-cl-18, cs-cl-19, cs-cl-09, cs-cl-16 `LGD_FIX_20260924/` | PTC, LOBSTER, TRIANGULAR_GRID seeds 0-5 |
| cs-cl-18 `lgd_common_eval_20260925*/` | common-reference evaluation of all of the above |

## Evaluation code

The paper's common-reference evaluation numbers were produced by scripts in campaign folders such as `aids_common_eval_10k_20260917/scripts/`, `qm9_common_eval_20260917/`, `lgd_common_eval_20260925/scripts/` and `rule_mmd_20260923/`, which are not in git. Copies are on the external drive under `POST_ARCHIVE_RESULTS_20261005/`.

