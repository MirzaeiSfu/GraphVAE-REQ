#!/usr/bin/env python3
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
parser.add_argument("--device", default="cpu")
parser.add_argument("--repeats", type=int, default=10)
args = parser.parse_args()

sys.path.insert(0, str(args.repo.resolve()))
from eval.attributed_gin import evaluate_dgl_feature_modes

generated, _ = dgl.load_graphs(str(args.generated))
reference, _ = dgl.load_graphs(str(args.reference))
count = min(len(generated), len(reference))
evaluation = evaluate_dgl_feature_modes(
    generated[:count],
    reference[:count],
    modes=("topology_control",),
    repeats=args.repeats,
    seed=0,
    nearest_k=5,
    device=torch.device(args.device),
)
payload = {
    "schema_version": "topology-random-gin-v1",
    "label": args.label,
    "training_seed": args.seed,
    "graph_count": count,
    "evaluator_seeds": list(range(args.repeats)),
    "evaluation": evaluation,
}
args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(args.output)
