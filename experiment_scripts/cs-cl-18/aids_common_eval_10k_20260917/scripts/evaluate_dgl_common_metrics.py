#!/usr/bin/env python3
"""Evaluate aligned DGL collections with QM9 structural and Random-GIN metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import dgl
import networkx as nx
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--skip-random-gin", action="store_true")
    return parser.parse_args()


def topology(graph):
    converted = dgl.to_networkx(graph, node_attrs=[], edge_attrs=[])
    simple = nx.Graph(converted)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    simple.remove_nodes_from(list(nx.isolates(simple)))
    if simple.number_of_nodes() and not nx.is_connected(simple):
        largest = max(nx.connected_components(simple), key=len)
        simple = simple.subgraph(largest).copy()
    return nx.convert_node_labels_to_integers(simple)


def main():
    args = parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from eval.attributed_gin import evaluate_dgl_feature_modes
    from stat_rnn import (
        MMD_diam,
        MMD_triangles,
        clustering_stats,
        degree_stats,
        orbit_stats_all,
        sparsity_stats_all,
        spectral_stats,
    )

    generated, _ = dgl.load_graphs(str(args.generated))
    reference, _ = dgl.load_graphs(str(args.reference))
    count = min(len(generated), len(reference))
    generated = generated[:count]
    reference = reference[:count]
    generated_nx = [topology(graph) for graph in generated]
    reference_nx = [topology(graph) for graph in reference]

    structural = {
        "degree": float(degree_stats(reference_nx, generated_nx)),
        "clustering": float(clustering_stats(reference_nx, generated_nx)),
        "orbit": float(orbit_stats_all(reference_nx, generated_nx)),
        "spectral": float(spectral_stats(reference_nx, generated_nx)),
        "diameter": float(MMD_diam(reference_nx, generated_nx)),
        "triangle": float(MMD_triangles(reference_nx, generated_nx)),
    }
    sparsity, reference_edges, generated_edges = sparsity_stats_all(
        reference_nx, generated_nx
    )
    structural.update(
        {
            "sparsity": float(sparsity),
            "reference_edge_count": float(reference_edges),
            "generated_edge_count": float(generated_edges),
            "edge_count_absolute_error": float(abs(reference_edges - generated_edges)),
        }
    )

    random_gin = None
    if not args.skip_random_gin:
        random_gin = evaluate_dgl_feature_modes(
            generated,
            reference,
            modes=("topology_control", "decoded_node"),
            repeats=args.repeats,
            seed=0,
            nearest_k=5,
            device=torch.device(args.device),
        )
    payload = {
        "schema_version": "qm9-common-evaluation-v1",
        "label": args.label,
        "training_seed": args.seed,
        "graph_count": count,
        "generated": str(args.generated.resolve()),
        "reference": str(args.reference.resolve()),
        "structural_mmd": structural,
        "random_gin": random_gin,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
