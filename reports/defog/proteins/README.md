# DeFoG PROTEINS evaluation

The supplied DeFoG collection was evaluated against the frozen 210-graph
PROTEINS test split defined by `baselines/defog/protocol.yaml`. Both collections
passed the protocol verifier before evaluation.

- DeFoG checkpoint: `protein.ckpt`, epoch 1297, training seed 0
- DeFoG collection digest: `a5d6b0b8b95aa21e4e4b70ae076b80b567684137e64de93a1467a2a966f8b672`
- Reference collection digest: `b2d3b6d8d68441bb209310e0543c768514f8242a2296348c66ee397541ab31d9`
- Evaluator: three bundled PROTEINS GraphCL-GIN checkpoints, seeds 0, 1, 2
- Upstream evaluator revision: `fb6bc26237eb21d7617fd41b22b4bb26ab29bf95`
- Feature mode: decoded node attributes
- Nearest-neighbour parameter: 5

The aggregate result is in `evaluation.md`; complete per-checkpoint metadata is
in `evaluation.json` and the `checkpoint_*` directories.

The generation seed was not embedded in the supplied graph file. The DeFoG
checkpoint records training seed 0 and DeFoG seeds each invocation from that
configuration, but the original generation command was not supplied. Treat
this as a provenance limitation when reporting the result; regenerate from the
checkpoint with an explicitly recorded command and seed for a final archival
benchmark.

Large graph and checkpoint files remain under ignored `runs/defog/` and are
not part of Git. Their file and logical collection hashes are recorded in
`reports/defog/artifact_inventory.yaml`.
