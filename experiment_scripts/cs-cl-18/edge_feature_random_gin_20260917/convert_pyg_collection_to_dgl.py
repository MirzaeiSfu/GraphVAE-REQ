#!/usr/bin/env python3
"""Convert the repository's feature-bearing PyG-dictionary format to DGL."""
import argparse
import torch
import dgl

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
try:
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(args.input, map_location="cpu")
items = payload.get("graphs", payload) if isinstance(payload, dict) else payload
graphs = []
for item in items:
    get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
    edge_index = torch.as_tensor(get("edge_index"), dtype=torch.long)
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=int(get("num_nodes")))
    graph.ndata["attr"] = torch.as_tensor(get("x"), dtype=torch.float32)
    edge_attr = get("edge_attr")
    if edge_attr is not None:
        graph.edata["attr"] = torch.as_tensor(edge_attr, dtype=torch.float32)
    graphs.append(graph)
dgl.save_graphs(args.output, graphs)
print(f"{args.output}: {len(graphs)} graphs")
