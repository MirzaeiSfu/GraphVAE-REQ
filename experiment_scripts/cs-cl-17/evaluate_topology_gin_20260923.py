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
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))
from eval.attributed_gin import evaluate_dgl_feature_modes

generated, _ = dgl.load_graphs(str(args.generated))
reference, _ = dgl.load_graphs(str(args.reference))
if len(generated) != len(reference):
    raise ValueError("Generated and reference collection sizes differ")
evaluation = evaluate_dgl_feature_modes(
    generated,
    reference,
    modes=("topology_control",),
    repeats=10,
    seed=0,
    nearest_k=5,
    device=torch.device(args.device),
)
payload = {
    "schema_version": "common-topology-random-gin-v1",
    "label": args.label,
    "training_seed": args.seed,
    "graph_count": len(generated),
    "generated": str(args.generated.resolve()),
    "reference": str(args.reference.resolve()),
    "evaluation": evaluation,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(args.output)
