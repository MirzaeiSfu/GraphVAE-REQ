#!/usr/bin/env python3
"""Mirror Kia's original 50/50 Random-GIN evaluation as closely as possible.

This script is intentionally narrow. It reproduces the logic in
``third_party/ggmeval/eval_all_in_dir_5050.py`` on a raw dataset list instead
of hard-coded baseline directories, so we can test whether the Grid Table 3
`50/50 split` mismatch is caused by split/order semantics rather than the core
Random-GIN evaluator.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GGMEVAL_ROOT = REPO_ROOT / "third_party" / "ggmeval"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GGMEVAL_ROOT))

from data import list_graph_loader  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from evaluation.models.gin import gin as gin_module  # noqa: E402


if "ndata" not in inspect.signature(dgl.batch).parameters:
    _orig_dgl_batch = dgl.batch

    def _compat_dgl_batch(graphs, ndata=None, edata=None):
        return _orig_dgl_batch(graphs)

    dgl.batch = _compat_dgl_batch

if len(inspect.signature(gin_module.expand_as_pair).parameters) == 1:
    def _compat_expand_as_pair(feat, graph):
        return feat, feat

    gin_module.expand_as_pair = _compat_expand_as_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="GRID", help="Dataset name for list_graph_loader.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed for the 50/50 split. The upstream helper used an unseeded shuffle.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of Random-GIN initializations to average.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for the GIN evaluator.",
    )
    parser.add_argument(
        "--no-structural-features",
        action="store_true",
        help="Disable Kia-style structural node features (degree, clustering, square clustering).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to runs/table3_reproduction/<dataset>_kia_5050_seed<seed>.json",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def self_loop_graphs(graphs: list[nx.Graph]) -> list[nx.Graph]:
    for graph in graphs:
        graph.add_edges_from((node_id, node_id) for node_id in range(graph.number_of_nodes()))
    return graphs


def preprocess_like_upstream(graphs: list[nx.Graph]) -> list[nx.Graph]:
    cleaned: list[nx.Graph] = []
    for graph in graphs:
        graph = nx.Graph(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph.remove_nodes_from(list(nx.isolates(graph)))
        if nx.is_empty(graph):
            continue
        largest_component = max(nx.connected_components(graph), key=len)
        graph = nx.Graph(graph.subgraph(largest_component))
        cleaned.append(graph)
    return self_loop_graphs(cleaned)


def to_dgl_like_upstream(graph: nx.Graph, use_structural_features: bool) -> dgl.DGLGraph:
    graph = nx.Graph(graph)
    if use_structural_features:
        cluster_attr = nx.clustering(graph)
        degree_attr = dict(graph.degree())
        orbit_like_attr = nx.square_clustering(graph)
        node_attr = {}
        for node_id, degree_value in degree_attr.items():
            node_attr[node_id] = np.array(
                [degree_value + 0.0, cluster_attr[node_id] + 0.0, orbit_like_attr[node_id] + 0.0],
                dtype=np.float32,
            )
        nx.set_node_attributes(graph, node_attr, "attr")
        if hasattr(dgl, "from_networkx"):
            return dgl.from_networkx(graph, node_attrs=["attr"])

        dgl_graph = dgl.DGLGraph(graph)
        node_ids = list(graph.nodes())
        features = np.asarray([graph.nodes[node_id]["attr"] for node_id in node_ids], dtype=np.float32)
        dgl_graph.ndata["attr"] = torch.from_numpy(features)
        return dgl_graph
    return dgl.DGLGraph(graph)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def item_to_nx_graph(item) -> nx.Graph:
    if isinstance(item, nx.Graph):
        return nx.Graph(item)
    if sp.issparse(item):
        if hasattr(nx, "from_scipy_sparse_array"):
            return nx.from_scipy_sparse_array(item)
        return nx.from_scipy_sparse_matrix(item)
    return nx.from_numpy_array(np.asarray(item))


def main() -> int:
    args = parse_args()
    use_structural_features = not args.no_structural_features
    device = resolve_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)

    items = list_graph_loader(args.dataset, return_labels=True)[0]
    graphs = [item_to_nx_graph(item) for item in items]

    shuffled = list(graphs)
    random.shuffle(shuffled)
    shuffled = preprocess_like_upstream(shuffled)

    midpoint = len(shuffled) // 2
    generated = shuffled[:midpoint]
    reference = shuffled[midpoint:]

    generated_dgl = [to_dgl_like_upstream(graph, use_structural_features) for graph in generated]
    reference_dgl = [to_dgl_like_upstream(graph, use_structural_features) for graph in reference]

    f1_values: list[float] = []
    mmd_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []

    input_dim = 3 if use_structural_features else 1
    for repeat_index in range(args.repeats):
        repeat_seed = args.seed + repeat_index
        random.seed(repeat_seed)
        np.random.seed(repeat_seed)
        torch.manual_seed(repeat_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(repeat_seed)

        evaluator = Evaluator(input_dim=input_dim, device=device)
        result = evaluator.evaluate_all(generated_dgl, reference_dgl)
        f1_values.append(float(result["f1_pr"]))
        mmd_values.append(float(result["mmd_rbf"]))
        precision_values.append(float(result["precision"]))
        recall_values.append(float(result["recall"]))

    payload = {
        "metadata": {
            "dataset": args.dataset,
            "seed": args.seed,
            "repeats": args.repeats,
            "device": str(device),
            "structural_features": use_structural_features,
            "num_generated_graphs": len(generated),
            "num_reference_graphs": len(reference),
            "logic": "mirrors third_party/ggmeval/eval_all_in_dir_5050.py on raw list_graph_loader output",
        },
        "metrics": {
            "f1_pr": summarize(f1_values),
            "mmd_rbf": summarize(mmd_values),
            "precision": summarize(precision_values),
            "recall": summarize(recall_values),
        },
        "raw_metrics": {
            "f1_pr": f1_values,
            "mmd_rbf": mmd_values,
            "precision": precision_values,
            "recall": recall_values,
        },
    }

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (
            REPO_ROOT
            / "runs"
            / "table3_reproduction"
            / f"{args.dataset.lower()}_kia_5050_seed{args.seed}.json"
        ).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"\nSaved JSON to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
