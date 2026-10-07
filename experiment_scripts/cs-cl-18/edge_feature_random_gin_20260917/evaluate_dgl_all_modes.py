#!/usr/bin/env python3
"""Run all four RandomGIN feature modes on aligned DGL collections."""
import argparse
import json
import sys
from pathlib import Path

import dgl
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--generated", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--repeats", type=int, default=10)
args = parser.parse_args()

sys.path.insert(0, str(args.repo.resolve()))
from eval.attributed_gin import evaluate_dgl_feature_modes

generated, _ = dgl.load_graphs(str(args.generated))
reference, _ = dgl.load_graphs(str(args.reference))
count = min(len(generated), len(reference))
generated, reference = generated[:count], reference[:count]
evaluation = evaluate_dgl_feature_modes(
    generated,
    reference,
    modes=("topology_control", "decoded_node", "decoded_edge", "decoded_node_edge"),
    repeats=args.repeats,
    seed=0,
    nearest_k=5,
    device=torch.device(args.device),
)
payload = {
    "schema_version": "all-random-gin-feature-modes-v1",
    "label": args.label,
    "training_seed": args.seed,
    "graph_count": count,
    "generated": str(args.generated.resolve()),
    "reference": str(args.reference.resolve()),
    "evaluator_seeds": list(range(args.repeats)),
    "evaluation": evaluation,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(args.output)
