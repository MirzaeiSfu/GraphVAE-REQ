## Kiarash GraphVAE configs

`grid_graphvae.yaml` is the repo's plain GraphVAE baseline for `GRID`.

`grid_graphvae_table3_nearest.yaml` is the closest local run config for the
paper's Grid Table 3 setting while staying on the current `GRID` dataset path:

- dataset: `GRID`
- split: `legacy_80_20`
- BFS: `legacy_first_component`
- model: base `GraphVAE`
- losses: kernel/BCE/KL plus node/edge feature decoder supervision
- motif and adjacency add-on losses disabled
- runtime: `micro` environment with CUDA-enabled DGL

Run it with:

```bash
python main.py --config configs/kiarash_graphvae/grid_graphvae_table3_nearest.yaml
```
