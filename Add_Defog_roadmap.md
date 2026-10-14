# Add DeFoG Roadmap

## Goal

Compare DeFoG and GraphVAE on exactly the same frozen graphs and external
Random-GIN evaluator. Keep DeFoG in its own Git repository and keep large
checkpoints/generated artifacts under ignored `runs/defog/` directories.

## Repositories

- GraphVAE integration branch: `feat/defog-fair-benchmark`
- DeFoG fork: <https://github.com/MirzaeiSfu/defog>
- DeFoG benchmark branch: `feat/frozen-graphvae-benchmark`
- Pinned DeFoG commit: `c631697b9cd5a2474d22ba12de33943c6b49e53e`
- Local DeFoG campaign clone: `runs/defog/source`

The fork retains the original DeFoG history. Do not copy the DeFoG source tree
into the GraphVAE repository.

## Frozen comparison

The source of truth is `baselines/defog/frozen_eval/manifest.yaml`. It freezes:

- MUTAG, PROTEINS, GRID, LOBSTER, and TRIANGULAR_GRID;
- exact serialized train/validation/test graphs and hashes;
- node-feature schemas and graph postprocessing;
- training seeds 0, 1, 2 and generation seed 12345;
- validation-only checkpoint selection and exact generated counts;
- legacy third-party Random-GIN seeds 0–9, k=5, architecture, and metrics;
- aggregation within training seed, then mean and sample SD across seeds.

PTC, QM9, and ENZYMES are excluded. Original and LinkCorr motif universes are
evaluation views of the same synthetic graphs, not separate DeFoG training
runs.

## Current status

- [x] Preserve and publish the original DeFoG history and ZIP changes.
- [x] Create a separate DeFoG Python environment.
- [x] Export all five exact GraphVAE split collections to safe tensor files.
- [x] Pin split, collection, source-cache, and evaluator-code hashes.
- [x] Add a generic DeFoG frozen-split loader.
- [x] Add validation-loss best-checkpoint selection.
- [x] Add seeded strict generation with accepted/rejected/attempt counts.
- [x] Add fail-closed verification, Random-GIN execution, and aggregation.
- [x] Pass the focused test suite and GPU training/generation smoke tests.
- [ ] Run all 15 DeFoG jobs (five datasets times three training seeds).
- [ ] Verify every generated collection and collect it centrally.
- [ ] Run 10 Random-GIN initializations for each collection.
- [ ] Aggregate and publish compact JSON/CSV/Markdown reports.
- [ ] Rerun GraphVAE on this canonical package where historical splits differ.
- [ ] Commit/push reports and merge both feature branches after review.

## Resume procedure

1. Read this file and `baselines/defog/frozen_eval/README.md`.
2. Run `git status --short --branch` in GraphVAE-REQ and in
   `runs/defog/source`.
3. Check worker `job_record.json` files and logs; a result is usable only when
   its record says `complete` and the strict verifier passes.
4. Resume or reassign failed jobs without changing the frozen manifest.
5. After all seeds finish, run `run_random_gin.py`, then `aggregate.py`.

Do not use the older supplied DeFoG checkpoints as primary results: they cover
only one training seed and do not prove validation-only model selection or the
required generation seed.
