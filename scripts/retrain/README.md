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
   - **Lab machine:** tmux session `retrain_<DATASET>_<method>_seed_N`, on the least-used GPU (or `--gpu N`).
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
- **DeFoG:** `python src/main.py <overrides>` in the run's DeFoG folder, using the overrides in
  `configs/paper_runs/<id>/training_overrides.yaml`, then the generation overrides. Pass `--ckpt` to
  `run_experiment.py` to generate from an existing checkpoint.
- **LGD:** `pretrain.py --cfg encoder.yaml`, then `select_best_encoder_ckpt.py` (lowest validation `loss_recon`), then
  `train_diffusion.py --cfg diffusion.yaml`, then `sample_tu.py --cfg diffusion.yaml`.

## Tests (2026-10-11)

Each kind of run was launched with `launch.py` from a fresh clone on its original lab machine:

- **GraphVAE** (PTC GraphVAE+RG seed 0, cs-cl-17): the first 100 epochs logged the same losses as the original code
  (code A, `f75f761`) run with its original command on the same GPU model, on every logged field.
- **LGD** (PTC seed 0, cs-cl-18): the encoder trained all 200 epochs. Its saved config equals the paper run's except
  for the data and output paths.
- **DeFoG** (GRID seed 0, cs-cl-18): training started. Its resolved Hydra config equals the paper run's except for
  `dataset.root` (the data path).

The test runs were stopped after these checks; they were not trained to the end.

## Notes

- The original system of the PROTEINS GraphVAE runs is not recorded; pass `--host`.
- PTC GraphVAE+RG seeds 1–2 originally started on cs-venus-08 (A100), were preempted, and finished on cs-venus-13
  (A40). The plan uses cs-venus-13.
- A run reproduces the paper bit for bit when it runs uninterrupted on the same GPU model with the same environment.
  A run that was preempted and resumed (PTC GraphVAE+RG seed 1) gives close but different numbers.
- On Solar, the end-of-training Random-GIN evaluation has occasionally frozen. If a job's log stops changing and its
  GPU sits at 0%, cancel it and rerun only the evaluation.
