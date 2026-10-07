#!/usr/bin/env python3
"""Normalize DeFoG's MUTAG export to the GraphVAE interchange schema."""

from pathlib import Path

import torch
from torch_geometric.data import Data

from graph_evaluation.src.ggm_eval.io import save_pyg_collection


source = Path(__file__).with_name("generated_graphs.pt")
target = Path(__file__).with_name("generated_graphs_normalized.pt")
payload = torch.load(source, map_location="cpu")

graphs = []
for graph in payload["graphs"]:
    edge_attr = graph.get("edge_attr")
    if edge_attr is None or edge_attr.ndim != 2 or edge_attr.shape[1] != 5:
        raise ValueError(f"Expected DeFoG MUTAG edge_attr [E,5], got {None if edge_attr is None else edge_attr.shape}")
    # Channel zero is DeFoG's absent-edge state. Exported edge_index contains
    # present edges only, while MUTAG represents its four bond types directly.
    graphs.append(Data(
        x=graph["x"],
        edge_index=graph["edge_index"],
        edge_attr=edge_attr[:, 1:].contiguous(),
        num_nodes=int(graph["num_nodes"]),
    ))

metadata = dict(payload.get("metadata") or {})
metadata["feature_schema"] = "gin-node-label-v2|export=decoded_node_edge"
metadata["feature_mode"] = "decoded_node_edge"
save_pyg_collection(target, graphs, metadata=metadata, normalize=True)
print(target)
