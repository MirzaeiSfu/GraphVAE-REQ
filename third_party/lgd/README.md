# LGD: Latent Graph Diffusion (baseline)

This is the LGD code behind the paper's LGD numbers (the `LGD_FIX_20260924` campaign).

- **Upstream:** [`zhouc20/LatentGraphDiffusion`](https://github.com/zhouc20/LatentGraphDiffusion) by Cai Zhou.
  MIT License (see `LICENSE`). The upstream README is `UPSTREAM_README.md`.
- **Base commit:** `f597e1dbd5a0ab4b9f8acc0ad2bdc0642bc42f59` (2024-12-17).
- **Local changes:** everything that differs from `f597e1d`. Datasets (`datasets/graphvae_req/*/*.pt`), logs and
  `__pycache__` are left out; only `metadata.json` is kept for each dataset.

## What was changed

- **Data:** `prepare_graphvae_req_data.py`, `prepare_all_graphvae_req.sh`, `reexport_tu_graphvae_req.py`,
  `lgd/loader/graphvae_req_dataset.py`, `lgd/asset/graphvae_req_export.py`, `GRAPHVAE_REQ_DATASETS.md`. These export
  the same frozen train/val/test splits used by GraphVAE-REQ and DeFoG.
- **Generation and sampling:** `lgd/train/tu_generation.py`, `sample_tu.py`, `check_generated.py`.
- **Configs:** `cfg/GraphVAEReq-*.yaml`, `cfg/AIDS-*.yaml`, `cfg/ENZYMES-*.yaml`.
- **Launchers:** `run_lgd_seed.sh`, `run_lgd_fixed.sh`, `run_lgd_fixed_v2.sh`, `solar_lgd_heavy.sbatch`.
- **Environment:** `env.yaml`, `env_lgd.yaml`.
- **Changed upstream files:** `pretrain.py`, `train_diffusion.py`, and several modules under `lgd/` (config,
  loader, model, ddpm, encoder, transform).

## The encoder-selection fix

`select_best_encoder_ckpt.py` picks the encoder checkpoint by the lowest validation `loss_recon`, never epoch 0, and
records the choice in `SELECTION.json`. The earlier campaign (`LGD_3SEED_CAMPAIGN_20260920`) picked by validation
`loss`. That number is the error of a graph-level head that is never trained (`graph_factor: 0`), so the minimum was
always epoch 0, an untrained encoder. Diffusion is retrained on the selected encoder, and every run ends with a
sampling check (`GATE.json`).

The old campaign's versions of the two files that changed are kept in
[`legacy_LGD_3SEED_CAMPAIGN_20260920/`](legacy_LGD_3SEED_CAMPAIGN_20260920/), because the LGD runs in
`EXPERIMENT_ARCHIVE_20260921` come from that campaign.
