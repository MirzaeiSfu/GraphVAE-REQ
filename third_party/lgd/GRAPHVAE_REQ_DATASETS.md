# GraphVAE-REQ datasets in LatentGraphDiffusion

This fork supports topology plus any number of categorical node and edge
feature fields.  It uses the exact frozen GraphVAE-REQ splits, so generated
results can be evaluated against GraphVAE-REQ and DeFoG without independently
reshuffling the source dataset.

## Representation and losses

- `x[:, j]` is categorical node field `j`. Each field has an independent
  embedding and decoder head.
- `edge_attr[:, 0]` is binary adjacency (`0` no edge, `1` edge).
- `edge_attr[:, j > 0]` is native edge field `j - 1`. Class `0` is reserved
  for absent dense pairs; real categories start at `1`.
- Encoder pretraining averages cross-entropy across node heads. The adjacency
  head is trained on all dense pairs. Native edge heads are trained only on
  pairs whose adjacency target is present, preventing the many non-edges from
  overwhelming bond/edge-type learning.
- The diffusion model operates on the resulting node/edge latents. Sampling
  decodes topology, node fields, and edge fields and writes them to NetworkX
  as `categorical_features`.
- All datasets below are undirected. Sampling averages `(u,v)` and `(v,u)`
  logits before discretization and keeps isolated nodes, so graph size is not
  silently changed after generation.
- Unconditional sampling draws graph-size templates from the training split
  and generates exactly as many graphs as the held-out test split. It does not
  copy held-out test graph sizes into the generated collection.

## Prepare the exact frozen datasets

Run from this repository. Change `--output-dir` if desired; it must match
`dataset.dir` in the YAML or the CLI override used for training.

```bash
python prepare_graphvae_req_data.py \
  --cache /local-scratch2/mirzaei/Abdolreza/harvest/cs-cl-19/local-scratch2__mirzaei__fb__GraphVAE-REQ/runs/count_distance_3seed/grid/dataset_cache/GRID_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-default.pkl \
  --dataset GRID --topology-only

python prepare_graphvae_req_data.py \
  --cache /local-scratch2/mirzaei/Abdolreza/harvest/cs-cl-16/local-scratch__localhome__mirzaei__fb__GraphVAE-REQ/runs/count_distance_3seed/lobster/dataset_cache/LOBSTER_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-lobster-optimal_v2.pkl \
  --dataset LOBSTER --topology-only

python prepare_graphvae_req_data.py \
  --cache /local-scratch2/mirzaei/countv5_tri_cache/TRIANGULAR_GRID_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-default.pkl \
  --dataset TRIANGULAR_GRID --topology-only

python prepare_graphvae_req_data.py \
  --cache /local-scratch2/mirzaei/ptc_rule_metrics_20260906/cache/PTC_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-gin-node-label-v2.pkl \
  --dataset PTC

python prepare_graphvae_req_data.py \
  --cache /local-scratch2/mirzaei/aids_common_eval_10k_20260917/cache/dataset/AIDS_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-all_components_features-tu-quantile8-maxall.pkl \
  --dataset AIDS

python prepare_graphvae_req_data.py \
  --indexed-raw-dir /local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG/data/mutag_edgefeat/raw \
  --raw-prefix mutag --dataset MUTAG

python prepare_graphvae_req_data.py \
  --cache /local-scratch2/mirzaei/qm9_common_eval_20260917/cache/dataset/QM9_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-default.pkl \
  --dataset QM9

python prepare_graphvae_req_data.py \
  --cache /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/count_distance_3seed/proteins/dataset_cache/PROTEINS_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-default.pkl \
  --dataset PROTEINS
```

The MUTAG command intentionally uses the frozen indexed tensors because its
edge-aware GraphVAE run disabled dataset-cache writing. This preserves the
131/18/39 split and includes the bond/edge label. Do not substitute the older
node-only MUTAG cache when comparing edge-aware models.

The synthetic commands intentionally pass `--topology-only`, matching the
featureless GRID/LOBSTER/TRIANGULAR_GRID protocol used in the final reports.
Remove that flag only for a separate attributed-synthetic experiment.

Each prepared directory contains `train.pt`, `val.pt`, `test.pt`, and
`metadata.json`. Inspect `metadata.json` before training; it records split
counts, feature names/cardinalities, source path, and whether topology-only
mode was used.

## Train one dataset

Pretrain the attributed encoder/decoder first:

```bash
python pretrain.py --cfg cfg/GraphVAEReq-encoder.yaml --repeat 3 \
  dataset.name MUTAG \
  dataset.dir "$PWD/datasets/graphvae_req" \
  accelerator cuda:0
```

Then point the diffusion config at the selected encoder checkpoint:

```bash
python train_diffusion.py --cfg cfg/GraphVAEReq-diffusion.yaml --repeat 3 \
  dataset.name MUTAG \
  dataset.dir "$PWD/datasets/graphvae_req" \
  diffusion.first_stage_config /absolute/path/to/encoder.ckpt \
  accelerator cuda:0
```

Use seeds 0, 1, and 2 for the paper comparison. Keep the prepared dataset
directory unchanged between methods. Batch size and epoch/step budget should
be set per dataset and then held identical across the three seeds.

## Sample and export

```bash
python sample_tu.py \
  --cfg cfg/GraphVAEReq-diffusion.yaml \
  --run-id 0 \
  --output-dir /absolute/path/to/samples \
  dataset.name MUTAG \
  dataset.dir "$PWD/datasets/graphvae_req" \
  diffusion.first_stage_config /absolute/path/to/encoder.ckpt
```

The sampling script saves raw NetworkX graphs even when GraphVAE-REQ's
optional `ggm_eval` package is absent. With `ggm_eval` installed, it also
exports train/reference/generated PyG collections containing all decoded node
and edge fields and runs the common evaluator.

## Environment expectations

The original repository requirements are still required. In particular,
GraphGym needs `yacs` and the training entrypoints need PyTorch Lightning.
This fork does not install or modify the environment.
