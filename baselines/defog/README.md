# DeFoG evaluation baseline

DeFoG remains an independent repository so its upstream history and Python
environment remain isolated. The maintained fork is
[`MirzaeiSfu/defog`](https://github.com/MirzaeiSfu/defog). The fair benchmark
adapter is pinned to commit
`c631697b9cd5a2474d22ba12de33943c6b49e53e` on branch
`feat/frozen-graphvae-benchmark`.

Use [`frozen_eval/README.md`](frozen_eval/README.md) for the five-dataset
campaign. Its manifest, exact GraphVAE split artifacts, strict verifier,
worker runner, external Random-GIN runner, and two-level aggregator supersede
the earlier PROTEINS-only example in `protocol.yaml`.

Large checkpoints, generated graphs, and worker logs belong under the ignored
`runs/defog/` tree. Commit only code, frozen hashes/provenance, tests, and compact
reports.
