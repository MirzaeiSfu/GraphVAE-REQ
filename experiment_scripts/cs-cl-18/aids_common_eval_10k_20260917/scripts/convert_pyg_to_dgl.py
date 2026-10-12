#!/usr/bin/env python3
"""Convert the feature-bearing PyG collection exported by DeFoG to DGL."""
import argparse
import json
import torch
import dgl


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()
try:
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(args.input, map_location="cpu")
graphs = payload.get("graphs", payload) if isinstance(payload, dict) else payload
converted = []
for item in graphs:
    def field(name, default=None):
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    edge_index = torch.as_tensor(field("edge_index"), dtype=torch.long)
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=int(field("num_nodes")))
    graph.ndata["attr"] = torch.as_tensor(field("x"), dtype=torch.float32)
    edge_attr = field("edge_attr")
    if edge_attr is not None:
        graph.edata["attr"] = torch.as_tensor(edge_attr, dtype=torch.float32)
    converted.append(graph)
dgl.save_graphs(args.output, converted)
meta = {
    "generator": "DeFoG", "training_seed": args.seed,
    "graph_count": len(converted),
    "node_feature_dim": int(converted[0].ndata["attr"].shape[1]),
    "edge_feature_dim": int(converted[0].edata["attr"].shape[1]) if "attr" in converted[0].edata else 0,
    "input": args.input, "output": args.output,
}
with open(args.metadata, "w") as handle:
    json.dump(meta, handle, indent=2, sort_keys=True)
print(json.dumps(meta, indent=2))
