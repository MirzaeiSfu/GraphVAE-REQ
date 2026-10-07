#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch

root = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo / "scripts"))
from evaluate_best_per_chain_gin_metrics import terminal_adjacencies_to_graphs


def dgl_topologies(path):
    graphs, _ = dgl.load_graphs(str(path))
    result = []
    for graph in graphs:
        converted = nx.Graph(dgl.to_networkx(graph, node_attrs=[], edge_attrs=[]))
        converted.remove_edges_from(nx.selfloop_edges(converted))
        converted.remove_nodes_from(list(nx.isolates(converted)))
        if converted.number_of_nodes() and not nx.is_connected(converted):
            largest = max(nx.connected_components(converted), key=len)
            converted = converted.subgraph(largest).copy()
        if converted.number_of_nodes():
            result.append(nx.convert_node_labels_to_integers(converted))
    return result


collections = {}
provenance = {}
for method, setting in (("false", "setting_01"), ("true_full", "setting_03")):
    for seed in range(3):
        source = root / "raw" / method / f"seed_{seed}"
        best = json.loads((source / "best_chain_random_gin_metrics.json").read_text())
        chain = int(best["chain_index"])
        with np.load(source / "terminal_chain_binary_matrices.npz") as archive:
            selected = archive["selected_chain_indices"].astype(int).tolist()
            position = selected.index(chain)
            adjacency = archive["binary_terminal_matrices"][:, position]
        graphs, cleanup = terminal_adjacencies_to_graphs(adjacency)
        key = f"{method}/seed_{seed}"
        collections[key] = graphs
        provenance[key] = {"chain_index": chain, "cleanup": cleanup, "source_count": len(adjacency), "usable_count": len(graphs)}

for seed in range(3):
    key = f"defog/seed_{seed}"
    graphs = dgl_topologies(root / "raw" / "defog" / f"seed_{seed}" / "generated.bin")
    collections[key] = graphs
    provenance[key] = {"source_count": 405, "usable_count": len(graphs)}

reference = dgl_topologies(root / "raw" / "reference.bin")
count = min([len(reference)] + [len(value) for value in collections.values()])
if count <= 0:
    raise RuntimeError("No common nonempty OGB graph count")

def save(graphs, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = []
    for graph in graphs[:count]:
        item = dgl.from_networkx(graph)
        item.ndata["attr"] = torch.ones((item.num_nodes(), 1), dtype=torch.float32)
        converted.append(item)
    dgl.save_graphs(str(path), converted)

save(reference, root / "aligned" / "reference.bin")
for key, graphs in collections.items():
    save(graphs, root / "aligned" / key / "generated.bin")
manifest = {
    "schema_version": "ogb-common-topology-alignment-v1",
    "common_graph_count": count,
    "reference_usable_count": len(reference),
    "selection": "first N nonempty largest-component graphs, N=min across all methods and seeds",
    "collections": provenance,
}
(root / "alignment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
