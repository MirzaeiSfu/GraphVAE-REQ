#!/usr/bin/env python3
"""Normalize raw DeFoG QM9 samples to the common GraphVAE DGL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dgl
import networkx as nx
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize(item):
    n = int(item["num_nodes"])
    edge_index = torch.as_tensor(item["edge_index"], dtype=torch.long)
    x5 = torch.as_tensor(item["x"], dtype=torch.float32)
    original_edge_attr = torch.as_tensor(item["edge_attr"], dtype=torch.float32)

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(n))
    nx_graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
    if not nx_graph.number_of_nodes():
        raise ValueError("empty graph after isolate removal")
    largest = sorted(max(nx.connected_components(nx_graph), key=len))
    old_to_new = {old: new for new, old in enumerate(largest)}

    kept_edges, kept_attrs, aromatic = [], [], 0
    seen = set()
    for edge_number, (u, v) in enumerate(edge_index.t().tolist()):
        if u not in old_to_new or v not in old_to_new or u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair in seen:
            continue
        seen.add(pair)
        label = int(original_edge_attr[edge_number].argmax().item())
        common = torch.zeros(3, dtype=torch.float32)
        if label in (1, 2, 3):
            common[label - 1] = 1.0
        elif label == 4:
            # QM9's training cache contains no aromatic bond state. Preserve
            # the topological edge but leave its three in-support channels at
            # zero; motif evaluation reports this out-of-support count.
            aromatic += 1
        else:
            raise ValueError(f"unexpected present-edge class {label}")
        kept_edges.append((old_to_new[pair[0]], old_to_new[pair[1]]))
        kept_attrs.append(common)

    selected_x5 = x5[largest]
    atom_labels = selected_x5.argmax(dim=1)
    node_attr = torch.zeros((len(largest), 9), dtype=torch.float32)
    node_attr[:, :5] = selected_x5[:, :5]
    for node in range(len(largest)):
        hydrogen_neighbours = sum(
            1
            for neighbour in nx_graph.neighbors(largest[node])
            if int(x5[neighbour].argmax().item()) == 0
        )
        node_attr[node, 5 + min(hydrogen_neighbours, 3)] = 1.0

    sources, targets = [], []
    directed_attrs = []
    for (u, v), attr in zip(kept_edges, kept_attrs):
        sources.extend((u, v))
        targets.extend((v, u))
        directed_attrs.extend((attr, attr.clone()))
    graph = dgl.graph((sources, targets), num_nodes=len(largest))
    graph.ndata["attr"] = node_attr
    graph.edata["attr"] = (
        torch.stack(directed_attrs)
        if directed_attrs
        else torch.zeros((0, 3), dtype=torch.float32)
    )
    return graph, aromatic


def main():
    args = parse_args()
    payload = load(args.input)
    items = payload["graphs"] if isinstance(payload, dict) else payload
    graphs, failures, aromatic_edges = [], [], 0
    for index, item in enumerate(items):
        try:
            graph, aromatic = normalize(item)
            graphs.append(graph)
            aromatic_edges += aromatic
        except Exception as exc:
            failures.append({"index": index, "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(args.output), graphs)
    metadata = {
        "generator": "DeFoG",
        "training_seed": args.seed,
        "input": str(args.input.resolve()),
        "input_graph_count": len(items),
        "accepted_graph_count": len(graphs),
        "rejected_graph_count": len(failures),
        "failures": failures,
        "out_of_training_support_aromatic_undirected_edges": aromatic_edges,
        "node_schema": "atom_type[5] + derived num_h[4]",
        "edge_schema": "single/double/triple[3]; aromatic retained topologically with zero attribute row",
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
