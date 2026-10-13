#!/usr/bin/env python3
"""Evaluate GraphVAE structural metrics for two adjacency-array collections."""

import argparse
import json
import os
import sys

import networkx as nx
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--reference", required=True)
parser.add_argument("--generated", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

sys.path.insert(0, args.repo)
os.chdir(args.repo)
from stat_rnn import (  # noqa: E402
    MMD_diam,
    MMD_triangles,
    clustering_stats,
    degree_stats,
    orbit_stats_all,
    sparsity_stats_all,
    spectral_stats,
)

def load_graphs(path):
    graphs = []
    for adjacency in np.load(path, allow_pickle=True):
        graph = nx.from_numpy_array(adjacency)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph.remove_nodes_from(list(nx.isolates(graph)))
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        if components:
            graphs.append(nx.Graph(graph.subgraph(components[0])))
    return graphs


reference = load_graphs(args.reference)
generated = load_graphs(args.generated)
matched_count = min(len(reference), len(generated))
reference = reference[:matched_count]
generated = generated[:matched_count]
for graph in reference + generated:
    graph.remove_edges_from(nx.selfloop_edges(graph))

sparsity, reference_edges, generated_edges = sparsity_stats_all(reference, generated)
result = {
    "reference_graph_count": len(reference),
    "generated_graph_count": len(generated),
    "degree": float(degree_stats(reference, generated)),
    "clustering": float(clustering_stats(reference, generated)),
    "orbit": float(orbit_stats_all(reference, generated)),
    "spectral": float(spectral_stats(reference, generated)),
    "diameter": float(MMD_diam(reference, generated)),
    "triangle": float(MMD_triangles(reference, generated)),
    "sparsity": float(sparsity),
    "reference_edge_count": float(reference_edges),
    "generated_edge_count": float(generated_edges),
    "edge_count_absolute_error": float(abs(reference_edges - generated_edges)),
}
with open(args.output, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(result, sort_keys=True))
