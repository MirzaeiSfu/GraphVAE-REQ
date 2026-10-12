# Paper run configurations

The exact configuration of every run behind the paper tables (structural MMD and Random-GIN F1-PR), one file per
seed. Which commit each run used is in [`EXPERIMENT_COMMITS.md`](../../EXPERIMENT_COMMITS.md).

```
paper_runs/<DATASET>/
├── graphvae/seed_N.yaml        GraphVAE (motif loss off)
├── graphvae_rg/seed_N.yaml     GraphVAE+RG (full matrix)
├── defog/seed_N/               DeFoG Hydra configs: training_config.yaml, training_overrides.yaml,
│                               generation_config.yaml, generation_overrides.yaml
└── lgd/seed_N/                 LGD encoder.yaml, diffusion.yaml, SOURCES.txt
```

Datasets: GRID, TRIANGULAR_GRID, LOBSTER, PTC, PROTEINS and QM9. LGD is not reported for TRIANGULAR_GRID. Seeds are
the ones in the paper; DeFoG uses seeds 0/4/5 on GRID and 0/1/3 on TRIANGULAR_GRID.

## GraphVAE and GraphVAE+RG

These files are **not** copies of the runs' `run_config_used.yaml`. That file copies only the base config the run
started from and misses command-line overrides, so it is wrong for many runs. For example, it shows seed 0 for
seeds 1 and 2, and lambda=0.1 for PROTEINS GraphVAE+RG, which used 0.03.

Each file here is built from the run's `reproducibility.json` `args`, the values actually in effect. The header of
each file gives the code commit, the source run folder and the original launch command. To rerun:

```bash
cp configs/paper_runs/<DATASET>/<method>/seed_N.yaml /tmp/run.yaml   # keep a copy before checking out
git checkout <commit from the file header>
python main.py --config /tmp/run.yaml
```

The paths in each file (`data_dir`, `dataset_cache_dir`, `graph_save_path`, `motif_cache_dir`) are those of the
original machine and may need changing.

## DeFoG and LGD

These are copied unchanged from the configs the runs saved:

- **DeFoG:** Hydra `.hydra/` files. Code: fork `MirzaeiSfu/defog`, commit `c631697`. For QM9 one Hydra run covered
  both training and generation, so its files are named `training_and_generation_*`.
- **LGD:** yacs `config.yaml` files from the `LGD_FIX_20260924` campaign. Code: `third_party/lgd/` on branch
  `baselines-code-and-provenance`. For PTC and QM9 the encoder was trained in the earlier `LGD_3SEED_CAMPAIGN_20260920`
  campaign and reused. `SOURCES.txt` in each seed folder says where each file came from.
