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

Each seed has two files:

| File | Use |
|---|---|
| `seed_N.yaml` | **Input config.** Run it with `main.py` on `main`: `python main.py --config configs/paper_runs/<DATASET>/<method>/seed_N.yaml` |
| `seed_N.effective.yaml` | **Record.** Every setting the original paper run ended up with (from its `reproducibility.json`), plus the original commit and launch command in the header. Use it to check a rerun, not as input. |

`seed_N.yaml` merges the run's original base config file (from the exact code version it ran with) with its
original command-line flags. Command-line values are kept as strings, so `main.py` converts them exactly as it did on
the command line. The effective record can't be used as input, because `main.py` writes some derived values back
into the settings it records.

Paths in the input files (`graph_save_path`, `dataset_cache_dir`, `motif_cache_dir`, and `data_dir` / `PATH` where
present) are the original machine's. Point them at your own folders on the command line, after `--config`:

```bash
python main.py --config configs/paper_runs/PTC/graphvae_rg/seed_0.yaml \
  --graph_save_path /my/out/PTC/graphvae_rg/seed_0 \
  --dataset_cache_dir /my/cache/PTC --motif_cache_dir /my/motif_cache
```

To retrain a run on its original machine (clone, data copy and start), use
[`scripts/retrain/launch.py`](../../scripts/retrain/).

### Checks

- **Settings.** All 36 input files load, and the settings `main.py` records match the paper run's
  `seed_N.effective.yaml` exactly (93 settings each). Options added to `main.py` after a run take their defaults, which
  keep the old behaviour.
- **First 100 epochs.** For each code version, the original commit with its original command and `main` with the
  input file logged identical losses for 100 epochs on the same GPU model: `9f01785` (GRID GraphVAE+RG), `f75f761`
  (PTC GraphVAE+RG), `c10bff5` (PTC GraphVAE) and `2a80b0a` (PROTEINS GraphVAE).
- **Longer runs.** PTC GraphVAE, 2,010 epochs: the same validation scores, selected checkpoints and best model
  (byte-identical).
- **Complete retrains.** GRID and TRIANGULAR_GRID GraphVAE+RG, seeds 0–2, retrained with `launch.py` on their original
  machines: all 20,000 epochs logged identical losses, and the best and final models, the generated graphs and every
  metric (`best_validation_mmd.json`, `graph_realism_random_gin.json`, `final_table2/3_metrics.json`) are identical to
  the paper runs.

## DeFoG and LGD

These are copied unchanged from the configs the runs saved:

- **DeFoG:** Hydra files (`.hydra/config.yaml` and `overrides.yaml`) of each training and generation run. For QM9 one
  Hydra run covered both, so its files are named `training_and_generation_*`. The code is `third_party/defog/`; QM9
  and PROTEINS ran a modified version, stored as a patch in `third_party/defog/variants/`.
- **LGD:** yacs `config.yaml` files from the `LGD_FIX_20260924` campaign; code in `third_party/lgd/`. For PTC and QM9
  the encoder was trained in the earlier `LGD_3SEED_CAMPAIGN_20260920` campaign and reused (its training code is
  identical). `SOURCES.txt` in each seed folder says where each file came from.
