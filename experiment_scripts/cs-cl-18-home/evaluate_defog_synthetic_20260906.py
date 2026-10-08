#!/usr/bin/env python3
"""Evaluate frozen synthetic DeFoG samples with GraphVAE-compatible metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("grid", "lobster", "triangular_grid", "ptc"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_safe_pyg(path: Path):
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "ggm-eval-pyg-tensors":
        raise ValueError(f"Unexpected collection format in {path}")
    graphs = []
    for index, item in enumerate(payload["graphs"]):
        n = int(item["num_nodes"])
        edge_index = torch.as_tensor(item["edge_index"], dtype=torch.long)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        if list(nx.isolates(graph)):
            raise ValueError(f"{path} graph {index} contains isolates after frozen normalization")
        if not nx.is_connected(graph):
            raise ValueError(f"{path} graph {index} is disconnected after frozen normalization")
        graphs.append(graph)
    return graphs, payload.get("metadata", {}), payload.get("collection_sha256")


def canonical_graph_digest(graphs):
    digest = hashlib.sha256()
    for graph in graphs:
        digest.update(struct.pack(">Q", graph.number_of_nodes()))
        edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges())
        digest.update(struct.pack(">Q", len(edges)))
        for u, v in edges:
            digest.update(struct.pack(">QQ", u, v))
    return digest.hexdigest()


def edge_path_state_counts(graphs):
    """Count FactorBase's E(0,1),E(1,2) states, including repeated variables."""
    totals = np.zeros(4, dtype=np.float64)  # FF, TT, FT, TF: cache row order.
    per_graph = []
    for graph in graphs:
        n = graph.number_of_nodes()
        degrees = np.asarray([degree for _, degree in graph.degree()], dtype=np.int64)
        tt = int(np.dot(degrees, degrees))
        mixed = int(np.dot(degrees, n - degrees))
        ff = int(n ** 3 - tt - 2 * mixed)
        row = np.asarray([ff, tt, mixed, mixed], dtype=np.float64)
        if int(row.sum()) != n ** 3:
            raise AssertionError("Edge-path state partition does not sum to n^3")
        totals += row
        per_graph.append(row.tolist())
    return totals, per_graph


def state_distribution_comparison(reference_graphs, generated_graphs):
    ref_counts, ref_per_graph = edge_path_state_counts(reference_graphs)
    gen_counts, gen_per_graph = edge_path_state_counts(generated_graphs)
    ref_distribution = ref_counts / ref_counts.sum()
    gen_distribution = gen_counts / gen_counts.sum()
    difference = gen_distribution - ref_distribution
    return {
        "rule": ["edges(nodes0,nodes1)", "edges(nodes1,nodes2)"],
        "state_order": ["FF", "TT", "FT", "TF"],
        "reference_aggregate_counts": ref_counts.tolist(),
        "generated_aggregate_counts": gen_counts.tolist(),
        "reference_distribution": ref_distribution.tolist(),
        "generated_distribution": gen_distribution.tolist(),
        "distribution_difference": difference.tolist(),
        "total_variation": float(0.5 * np.abs(difference).sum()),
        "reference_per_graph_counts": ref_per_graph,
        "generated_per_graph_counts": gen_per_graph,
        "aggregation": "count each graph exactly, sum state counts, then normalize once",
    }


def main():
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    from stat_rnn import (
        MMD_diam,
        MMD_triangles,
        clustering_stats,
        degree_stats,
        orbit_stats_all,
        sparsity_stats_all,
        spectral_stats,
    )
    from scripts.evaluate_graph_realism_batch import evaluate_graph_collections

    dataset_root = artifact_root / args.dataset
    reference_path = dataset_root / "real_test_graphs.pt"
    generated_path = dataset_root / "generated" / f"seed_{args.seed}" / "generated_graphs.pt"
    reference_graphs, reference_metadata, reference_serialized_digest = load_safe_pyg(reference_path)
    generated_graphs, generated_metadata, generated_serialized_digest = load_safe_pyg(generated_path)
    if len(reference_graphs) != len(generated_graphs):
        raise ValueError("Frozen generated/reference graph counts differ")

    structural = {
        "degree": float(degree_stats(reference_graphs, generated_graphs)),
        "clustering": float(clustering_stats(reference_graphs, generated_graphs)),
        "orbit": float(orbit_stats_all(reference_graphs, generated_graphs)),
        "spectral": float(spectral_stats(reference_graphs, generated_graphs)),
        "diameter": float(MMD_diam(reference_graphs, generated_graphs)),
        "triangle": float(MMD_triangles(reference_graphs, generated_graphs)),
    }
    sparsity, reference_edge_mean, generated_edge_mean = sparsity_stats_all(
        reference_graphs, generated_graphs
    )
    structural.update(
        {
            "sparsity": float(sparsity),
            "reference_edge_count": float(reference_edge_mean),
            "generated_edge_count": float(generated_edge_mean),
            "edge_count_absolute_error": float(abs(generated_edge_mean - reference_edge_mean)),
        }
    )
    random_gin = evaluate_graph_collections(
        generated_graphs=generated_graphs,
        reference_graphs=reference_graphs,
        repeats=10,
        seed=0,
        device=torch.device(args.device),
        use_structural_features=True,
    )
    payload = {
        "schema_version": "defog-synthetic-graphvae-compatible-evaluation-v1",
        "dataset": args.dataset.upper(),
        "training_seed": args.seed,
        "generated_path": str(generated_path),
        "reference_path": str(reference_path),
        "generated_graph_count": len(generated_graphs),
        "reference_graph_count": len(reference_graphs),
        "generated_metadata": generated_metadata,
        "reference_metadata": reference_metadata,
        "generated_serialized_sha256": generated_serialized_digest,
        "reference_serialized_sha256": reference_serialized_digest,
        "generated_canonical_graph_sha256": canonical_graph_digest(generated_graphs),
        "reference_canonical_graph_sha256": canonical_graph_digest(reference_graphs),
        "structural_mmd": structural,
        "third_party_random_gin_structural_features": random_gin,
        "linkcorr_full_state_distribution": state_distribution_comparison(
            reference_graphs, generated_graphs
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
