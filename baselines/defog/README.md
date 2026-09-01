# DeFoG evaluation baseline

DeFoG remains an independent sibling repository so its history and dependency
environment stay intact. The maintained fork is
[`MirzaeiSfu/defog`](https://github.com/MirzaeiSfu/defog); the frozen integration
commit is `474f9405bdcdddc2d96cfedc3305172dffbe8fbd`.

`protocol.yaml` is the single source of truth for conditions shared with
GraphVAE. It freezes the PROTEINS source/filter, 70/10/20 split, exact split
digest, feature schema, generation count/seed, validation-only checkpoint
selection, normalization rules, and evaluator settings. Architecture and
optimizer settings remain in each model's native config because they are not
equivalent across model families.

## Repositories

```text
/local-scratch2/mirzaei/Abdolreza/
├── DeFoG/
└── GraphVAE-REQ/
```

Use separate environments. Install GraphVAE-REQ's evaluation package into the
DeFoG environment without importing GraphVAE's model code:

```bash
python -m pip install -e /local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/graph_evaluation
```

## Verify the frozen inputs

Run this before generation and again before evaluation:

```bash
/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python \
  baselines/defog/verify_protocol.py
```

The verifier rejects wrong commits, dirty repositories, changed native configs,
an inconsistent split, and incompatible graph artifacts.

## Generate with DeFoG

Run from `DeFoG/src` in the DeFoG environment. Replace the checkpoint path:

```bash
python main.py \
  +experiment=proteins \
  dataset=proteins \
  general.test_only=/absolute/path/to/checkpoint.ckpt \
  general.final_model_samples_to_generate=210 \
  general.save_samples=true \
  general.wandb=disabled \
  train.seed=12345
```

The integration commit writes `generated_graphs.pt` and its JSON manifest.
Copy or link them under the ignored `runs/defog/proteins/` directory. Keep the
checkpoint and raw generated files out of Git, and complete
`artifact_manifest.example.yaml` with their SHA-256 identities.

## Evaluate

Use the frozen GraphVAE held-out collection, not a separately reconstructed
test set:

```bash
/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python \
  baselines/defog/verify_protocol.py \
  --generated runs/defog/proteins/generated_graphs.pt \
  --reference runs/defog/proteins/real_test_graphs.pt

ggm-eval evaluate-trained \
  --dataset PROTEINS \
  --generated runs/defog/proteins/generated_graphs.pt \
  --reference runs/defog/proteins/real_test_graphs.pt \
  --seeds 0 1 2 \
  --max-graphs 210 \
  --output-dir reports/defog/proteins
```

Commit only the protocol, verifier, completed provenance manifest, tests, and
small JSON/CSV/Markdown reports.
