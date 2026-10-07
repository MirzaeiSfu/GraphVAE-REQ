#!/usr/bin/env python3
"""Generate GraphVAE adjacency arrays matched to a fixed reference's node counts."""

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dataset-cache", required=True)
parser.add_argument("--reference", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

sys.path.insert(0, args.repo)
from scripts.evaluate_motif_count_distance_correlation import (  # noqa: E402
    build_decoders,
    configure_rng,
    load_pickle,
    load_state_dict,
    load_yaml,
)

_, config = load_yaml(Path(args.config))
cache = load_pickle(Path(args.dataset_cache))
train_dataset = cache["list_graphs"]
state = load_state_dict(Path(args.checkpoint))
device = torch.device("cpu")

configure_rng(args.seed)
decoder, _, _, graph_dim = build_decoders(state, config, train_dataset, device)
decoder.eval()
configure_rng(args.seed)

reference = np.load(args.reference, allow_pickle=True)
node_counts = [int(np.asarray(adjacency).shape[0]) for adjacency in reference]
generated = []
for start in range(0, len(node_counts), args.batch_size):
    counts = node_counts[start : start + args.batch_size]
    latent = torch.randn(len(counts), graph_dim, dtype=torch.float32)
    with torch.no_grad():
        probabilities = torch.sigmoid(decoder(latent)).cpu().numpy()
    for probability, node_count in zip(probabilities, counts):
        adjacency = probability[:node_count, :node_count] > 0.5
        adjacency = np.logical_or(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, False)
        graph = nx.from_numpy_array(adjacency.astype(np.float32))
        graph.remove_nodes_from(list(nx.isolates(graph)))
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        if components:
            graph = nx.Graph(graph.subgraph(components[0]))
            adjacency = nx.to_numpy_array(graph, dtype=np.float32)
        else:
            adjacency = np.zeros((0, 0), dtype=np.float32)
        generated.append(adjacency)

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
np.save(output, np.asarray(generated, dtype=object), allow_pickle=True)
print(f"wrote {len(generated)} graphs to {output}")
