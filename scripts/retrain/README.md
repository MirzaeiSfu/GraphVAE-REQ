# Retraining the paper experiments

These scripts retrain any of the 69 runs behind the paper tables (GRID, TRIANGULAR_GRID, LOBSTER, PTC, PROTEINS and
QM9; GraphVAE, GraphVAE+RG, DeFoG and LGD; 3 seeds each). Each run is retrained from a fresh clone of this repository,
on the system it originally ran on, from its YAML config in `configs/paper_runs/`.

| File | Role |
|---|---|
| `launch.py` | Run on **cs-cl-18**. Clones the repository on the run's original system, copies the data that is not in git, and starts the run. |
| `run_experiment.py` | Runs **on the target system**, inside the clone. Starts one experiment from its YAML config. |
| `retrain_plan.yaml` | For every run: its config files, the data it needs, and its original system (machine, GPU, Slurm settings). |
| `make_retrain_plan.py` | Regenerates `retrain_plan.yaml`. |

## Quick start

```bash
ssh mirzaei@cs-cl-18.cmpt.sfu.ca
git clone https://github.com/MirzaeiSfu/GraphVAE-REQ.git && cd GraphVAE-REQ

python3 scripts/retrain/launch.py PTC/graphvae_rg/seed_1 --dry-run   # show every step, change nothing
python3 scripts/retrain/launch.py PTC/graphvae_rg/seed_1             # one run
python3 scripts/retrain/launch.py PTC/                               # every PTC run
python3 scripts/retrain/launch.py all --host cs-cl-18                # all 69 (--host only for PROTEINS GraphVAE)
```

`launch.py` needs `python3` with PyYAML; it runs fine with the system Python on cs-cl-18. Run IDs are the keys under
`runs:` in `retrain_plan.yaml`, in the form `<DATASET>/<method>/seed_N` with method `graphvae` (motif off),
`graphvae_rg` (motif on), `defog` or `lgd`.

## What a launch does

1. **Clones the code** at `--ref` (default: tag `paper-reproduction-v1`) into `<workspace>/GraphVAE-REQ_<commit>` on
   the original system, or reuses that clone. The commit is resolved once, so every run in a launch uses the same
   code.
2. **Copies the data** that run needs from the reproduction package
   (`cs-cl-18:/media/mirzaei/backup/REPRODUCTION_PACKAGE_20261008`) to `<workspace>/data/`: dataset and motif caches,
   frozen DeFoG splits, LGD datasets.
3. **Starts the run** with `run_experiment.py`, using the Python environment the original run used:
   - **Lab machine:** tmux session `retrain_<DATASET>_<method>_seed_N`, on a free GPU (or `--gpu N`).
   - **Solar:** an sbatch job with the original partition, `--nodelist` node, GPU type, CPUs, memory and time limit.

| System | Workspace |
|---|---|
| cs-cl-18, cs-cl-19 | `/local-scratch2/mirzaei/RETRAIN` |
| cs-cl-16 | `/local-scratch/mirzaei/RETRAIN` |
| cs-cl-17 | `/local-scratch/localhome/mirzaei/RETRAIN` |
| Solar | `/project/cs-schulte-lab/ali/RETRAIN` |

Outputs: `<workspace>/outputs/<DATASET>/<method>/seed_N/`. Logs: `<workspace>/logs/`. Each launch is recorded in
`~/retrain_launch_log.tsv` on the machine you launched from.

## What each run executes

- **GraphVAE:** `python main.py --config configs/paper_runs/<id>.yaml`, plus output, cache and data paths. To check a
  rerun's settings, compare its `reproducibility.json` with `configs/paper_runs/<id>.effective.yaml`.
- **DeFoG:** `python src/main.py <overrides>` in `third_party/defog/`, or for QM9 and PROTEINS in a copy with
  `variants/qm9.patch` or `variants/proteins.patch` applied (built in the run's output folder, and every file checked
  against the paper's code before the run starts), with the overrides in
  `configs/paper_runs/<id>/training_overrides.yaml`, then `generation_overrides.yaml`. Generation uses the checkpoint
  the original run used (best validation loss, or the last one for PROTEINS). To generate from an existing
  checkpoint without training, run `run_experiment.py <id> --ckpt <file> ...`.
- **LGD:** `pretrain.py --cfg encoder.yaml`, then `select_best_encoder_ckpt.py` (lowest validation `loss_recon`), then
  `train_diffusion.py --cfg diffusion.yaml`, then `sample_tu.py --cfg diffusion.yaml`.

## Will a retrain match the paper?

| Method | Result |
|---|---|
| GraphVAE, GraphVAE+RG | **Identical**, when run uninterrupted on the same GPU model with the same environment. Checked by retraining GRID and TRIANGULAR_GRID GraphVAE+RG (seeds 0–2) on their original machines: identical losses for all 20,000 epochs, and identical models, generated graphs and metrics. |
| DeFoG | **Generation: identical. Training: close, not guaranteed identical.** Generating from the paper's checkpoints with `run_experiment.py <id> --ckpt ...` reproduced the paper's graphs for GRID seed 0 (byte-identical) and PROTEINS seed 0 (identical graphs and metrics, using the PROTEINS version built from `variants/proteins.patch`). The QM9 version built from `variants/qm9.patch` starts from exactly the paper run's initial weights (same checksum, `d9c0a758...`, for seed 0). DeFoG training sets the seed but not PyTorch's deterministic mode, and the original runs kept no per-epoch log to compare with (a GRID training run takes about 2.5 days), so a retrained model may differ slightly. |
| LGD | **Close, not identical.** LGD is not deterministic on GPU: two runs with the same code, config, seed and GPU differ from the second epoch on (training loss 13.5422 vs 13.5446). A full retrain of PTC seed 0 ran through all four steps, but picked a different encoder checkpoint (epoch 24 vs 74) and its quick sample check differs from the paper run's (degree MMD 0.053 vs 0.032, clustering 0.072 vs 0.110). |

## Notes

- The original machine of the PROTEINS GraphVAE runs is not recorded; pass `--host`.
- PTC GraphVAE+RG seeds 1 and 2 were preempted on Solar and resumed (they started on cs-venus-08, A100, and finished
  on cs-venus-13, A40). A retrain runs uninterrupted on cs-venus-13, so it will be close to the paper but not identical.
- Give each run its own GPU. Two GraphVAE runs on one 11 GB GPU ran out of memory in the final evaluation. `launch.py`
  picks a GPU with no running processes when there is one.
- On Solar, the end-of-training Random-GIN evaluation has occasionally frozen. If a job's log stops changing and its
  GPU sits at 0%, cancel it and rerun only the evaluation.
- If the final evaluation fails (e.g. out of memory), training results are already saved. Rerun only the evaluation
  with the command printed in the log after "Running third-party graph realism evaluation:".
