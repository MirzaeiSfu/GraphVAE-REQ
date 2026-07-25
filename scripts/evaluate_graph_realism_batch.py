#!/usr/bin/env python3
"""Batch GNN-based graph realism evaluation for saved VGAREQ runs.

This script evaluates already-saved generated graph sets against their saved
reference test graphs using the vendored Random-GIN metrics from
``third_party/ggmeval``. It is intentionally post-hoc and batch-oriented so we
can recompute metrics on old runs without retraining or regenerating graphs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GGMEVAL_ROOT = REPO_ROOT / "third_party" / "ggmeval"
sys.path.insert(0, str(GGMEVAL_ROOT))

from evaluation.evaluator import Evaluator  # noqa: E402


DEFAULT_GENERATED_FILENAME = "Single_comp_generatedGraphs_adj_final_eval.npy"
DEFAULT_REFERENCE_FILENAME = "testGraphs_adj_.npy"
DEFAULT_JSON_FILENAME = "graph_realism_random_gin.json"
DEFAULT_SUMMARY_FILENAME = "graph_realism_batch_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate saved VGAREQ generated graphs with the vendored "
            "Random-GIN graph realism metrics."
        )
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help=(
            "Specific run directory to evaluate. Can be passed multiple times. "
            "Each run directory must contain the generated and reference files."
        ),
    )
    parser.add_argument(
        "--root-dir",
        action="append",
        default=[],
        help=(
            "Root directory to scan recursively for run directories containing "
            "the generated and reference files. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--generated-filename",
        default=DEFAULT_GENERATED_FILENAME,
        help=f"Generated graph filename to look for. Default: {DEFAULT_GENERATED_FILENAME}",
    )
    parser.add_argument(
        "--reference-filename",
        default=DEFAULT_REFERENCE_FILENAME,
        help=f"Reference graph filename to look for. Default: {DEFAULT_REFERENCE_FILENAME}",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help=(
            "Where to write the batch summary CSV. Default: "
            "<root-dir>/graph_realism_batch_summary.csv when a single root-dir "
            "is provided, otherwise ./graph_realism_batch_summary.csv."
        ),
    )
    parser.add_argument(
        "--json-filename",
        default=DEFAULT_JSON_FILENAME,
        help=(
            "Per-run JSON filename to write inside each evaluated run directory. "
            f"Default: {DEFAULT_JSON_FILENAME}"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of random GIN initializations to average. Default: 10",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=1000,
        help="Maximum number of generated and reference graphs to evaluate. Default: 1000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used for deterministic graph subsampling and repeat seeding.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for the GIN evaluator. Default: auto",
    )
    parser.add_argument(
        "--no-structural-features",
        action="store_true",
        help=(
            "Disable the Kia-style structural node features "
            "(degree, clustering, square clustering)."
        ),
    )
    args = parser.parse_args()
    if not args.run_dir and not args.root_dir:
        args.root_dir = ["runs"]
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def discover_run_dirs(
    run_dirs: Iterable[str],
    root_dirs: Iterable[str],
    generated_filename: str,
    reference_filename: str,
) -> list[Path]:
    discovered: set[Path] = set()

    for raw_path in run_dirs:
        run_dir = Path(raw_path).expanduser().resolve()
        generated_path = run_dir / generated_filename
        reference_path = run_dir / reference_filename
        if generated_path.is_file() and reference_path.is_file():
            discovered.add(run_dir)

    for raw_root in root_dirs:
        root_dir = Path(raw_root).expanduser().resolve()
        if not root_dir.exists():
            continue

        if (root_dir / generated_filename).is_file() and (root_dir / reference_filename).is_file():
            discovered.add(root_dir)

        for generated_path in root_dir.rglob(generated_filename):
            run_dir = generated_path.parent
            if (run_dir / reference_filename).is_file():
                discovered.add(run_dir)

    return sorted(discovered)


def load_graph_items(path: Path) -> list:
    with path.open("rb") as handle:
        items = np.load(handle, allow_pickle=True)
    return list(items)


def item_to_graph(item) -> nx.Graph:
    if isinstance(item, nx.Graph):
        graph = nx.Graph(item)
    elif sp.issparse(item):
        if hasattr(nx, "from_scipy_sparse_array"):
            graph = nx.from_scipy_sparse_array(item)
        else:
            graph = nx.from_scipy_sparse_matrix(item)
    else:
        graph = nx.from_numpy_array(np.asarray(item))

    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() == 0:
        return graph
    largest_component = max(nx.connected_components(graph), key=len)
    return nx.Graph(graph.subgraph(largest_component))


def preprocess_graphs(items: list, max_graphs: int, seed: int, shuffle: bool) -> list[nx.Graph]:
    graphs = [item_to_graph(item) for item in items]
    graphs = [graph for graph in graphs if not nx.is_empty(graph)]
    if max_graphs > 0:
        graphs = graphs[:max_graphs]
    if shuffle:
        shuffled_graphs = list(graphs)
        random.Random(seed).shuffle(shuffled_graphs)
        return shuffled_graphs
    return graphs


def add_self_loops(graph: nx.Graph) -> nx.Graph:
    graph = nx.Graph(graph)
    graph.add_edges_from((node_id, node_id) for node_id in range(graph.number_of_nodes()))
    return graph


def to_dgl_graph(graph: nx.Graph, use_structural_features: bool) -> dgl.DGLGraph:
    graph = add_self_loops(graph)
    if not use_structural_features:
        return dgl.from_networkx(graph)

    degree_attr = dict(graph.degree())
    clustering_attr = nx.clustering(graph)
    orbit_like_attr = nx.square_clustering(graph)
    node_attr = {}
    for node_id, degree_value in degree_attr.items():
        node_attr[node_id] = np.array(
            [
                degree_value + 0.0,
                clustering_attr[node_id] + 0.0,
                orbit_like_attr[node_id] + 0.0,
            ],
            dtype=np.float32,
        )

    nx.set_node_attributes(graph, node_attr, "attr")
    return dgl.from_networkx(graph, node_attrs=["attr"])


def summarize_metric(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def summarize_mmd_linear(values: list[float], trim_fraction: float = 0.1) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    sorted_array = np.sort(array)
    trim_count = int(np.floor(len(sorted_array) * trim_fraction))
    if trim_count > 0 and 2 * trim_count < len(sorted_array):
        trimmed_array = sorted_array[trim_count:-trim_count]
    else:
        trimmed_array = sorted_array
    median = float(np.median(sorted_array))
    trimmed_mean = float(trimmed_array.mean())
    mean = float(array.mean())
    return {
        "mean": mean,
        "std": float(array.std()),
        "median": median,
        "trimmed_mean": trimmed_mean,
        "trim_fraction": float(trim_fraction),
        "min": float(sorted_array[0]),
        "max": float(sorted_array[-1]),
        "max_to_median_ratio": float(sorted_array[-1] / max(median, 1e-12)),
        "mean_to_median_ratio": float(mean / max(median, 1e-12)),
    }


def evaluate_graph_collections(
    generated_graphs: list[nx.Graph],
    reference_graphs: list[nx.Graph],
    repeats: int,
    seed: int,
    device: torch.device,
    use_structural_features: bool,
) -> dict:
    generated_graphs = list(generated_graphs)
    reference_graphs = list(reference_graphs)
    generated_graphs = generated_graphs[: len(reference_graphs)]
    reference_graphs = reference_graphs[: len(generated_graphs)]

    if not generated_graphs or not reference_graphs:
        raise ValueError("No non-empty graphs remained after preprocessing.")

    generated_dgl = [to_dgl_graph(graph, use_structural_features) for graph in generated_graphs]
    reference_dgl = [to_dgl_graph(graph, use_structural_features) for graph in reference_graphs]

    f1_values: list[float] = []
    mmd_rbf_values: list[float] = []
    mmd_linear_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []

    input_dim = 3 if use_structural_features else 1
    for repeat_index in range(repeats):
        repeat_seed = seed + repeat_index
        random.seed(repeat_seed)
        np.random.seed(repeat_seed)
        torch.manual_seed(repeat_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(repeat_seed)

        evaluator = Evaluator(input_dim=input_dim, device=device)
        result = evaluator.evaluate_all(generated_dgl, reference_dgl)
        required_keys = ("f1_pr", "mmd_rbf", "mmd_linear", "precision", "recall")
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            present_keys = sorted(result.keys())
            raise ValueError(
                "Evaluator did not return the required Random-GIN metrics. "
                f"Missing keys: {missing_keys}. Present keys: {present_keys}. "
                f"generated_graphs={len(generated_graphs)}, reference_graphs={len(reference_graphs)}"
            )
        f1_values.append(float(result["f1_pr"]))
        mmd_rbf_values.append(float(result["mmd_rbf"]))
        mmd_linear_values.append(float(result["mmd_linear"]))
        precision_values.append(float(result["precision"]))
        recall_values.append(float(result["recall"]))

    return {
        "num_generated_graphs": len(generated_graphs),
        "num_reference_graphs": len(reference_graphs),
        "repeats": repeats,
        "seed": seed,
        "device": str(device),
        "structural_features": use_structural_features,
        "metrics": {
            "f1_pr": summarize_metric(f1_values),
            "mmd_rbf": summarize_metric(mmd_rbf_values),
            "mmd_linear": summarize_mmd_linear(mmd_linear_values),
            "precision": summarize_metric(precision_values),
            "recall": summarize_metric(recall_values),
        },
        "raw_metrics": {
            "f1_pr": f1_values,
            "mmd_rbf": mmd_rbf_values,
            "mmd_linear": mmd_linear_values,
            "precision": precision_values,
            "recall": recall_values,
        },
    }


def evaluate_run(
    run_dir: Path,
    generated_filename: str,
    reference_filename: str,
    repeats: int,
    max_graphs: int,
    seed: int,
    device: torch.device,
    use_structural_features: bool,
) -> dict:
    generated_path = run_dir / generated_filename
    reference_path = run_dir / reference_filename

    generated_graphs = preprocess_graphs(
        load_graph_items(generated_path),
        max_graphs=max_graphs,
        seed=seed,
        shuffle=True,
    )
    reference_graphs = preprocess_graphs(
        load_graph_items(reference_path),
        max_graphs=max_graphs,
        seed=seed,
        shuffle=False,
    )

    result = evaluate_graph_collections(
        generated_graphs=generated_graphs,
        reference_graphs=reference_graphs,
        repeats=repeats,
        seed=seed,
        device=device,
        use_structural_features=use_structural_features,
    )
    result.update({
        "run_dir": str(run_dir),
        "generated_filename": generated_filename,
        "reference_filename": reference_filename,
        "max_graphs": max_graphs,
    })
    return result


def summary_row(result: dict) -> dict[str, object]:
    return {
        "run_dir": result["run_dir"],
        "num_generated_graphs": result["num_generated_graphs"],
        "num_reference_graphs": result["num_reference_graphs"],
        "repeats": result["repeats"],
        "structural_features": result["structural_features"],
        "f1_pr_mean": result["metrics"]["f1_pr"]["mean"],
        "f1_pr_std": result["metrics"]["f1_pr"]["std"],
        "mmd_rbf_mean": result["metrics"]["mmd_rbf"]["mean"],
        "mmd_rbf_std": result["metrics"]["mmd_rbf"]["std"],
        "mmd_linear_mean": result["metrics"]["mmd_linear"]["mean"],
        "mmd_linear_std": result["metrics"]["mmd_linear"]["std"],
        "mmd_linear_median": result["metrics"]["mmd_linear"]["median"],
        "mmd_linear_trimmed_mean": result["metrics"]["mmd_linear"]["trimmed_mean"],
        "mmd_linear_min": result["metrics"]["mmd_linear"]["min"],
        "mmd_linear_max": result["metrics"]["mmd_linear"]["max"],
        "mmd_linear_max_to_median_ratio": result["metrics"]["mmd_linear"]["max_to_median_ratio"],
        "mmd_linear_mean_to_median_ratio": result["metrics"]["mmd_linear"]["mean_to_median_ratio"],
        "precision_mean": result["metrics"]["precision"]["mean"],
        "precision_std": result["metrics"]["precision"]["std"],
        "recall_mean": result["metrics"]["recall"]["mean"],
        "recall_std": result["metrics"]["recall"]["std"],
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "run_dir",
        "num_generated_graphs",
        "num_reference_graphs",
        "repeats",
        "structural_features",
        "f1_pr_mean",
        "f1_pr_std",
        "mmd_rbf_mean",
        "mmd_rbf_std",
        "mmd_linear_mean",
        "mmd_linear_std",
        "mmd_linear_median",
        "mmd_linear_trimmed_mean",
        "mmd_linear_min",
        "mmd_linear_max",
        "mmd_linear_max_to_median_ratio",
        "mmd_linear_mean_to_median_ratio",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def default_summary_csv(args: argparse.Namespace) -> Path:
    if args.summary_csv:
        return Path(args.summary_csv).expanduser().resolve()
    if len(args.root_dir) == 1 and not args.run_dir:
        return Path(args.root_dir[0]).expanduser().resolve() / DEFAULT_SUMMARY_FILENAME
    return (REPO_ROOT / DEFAULT_SUMMARY_FILENAME).resolve()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    use_structural_features = not args.no_structural_features

    run_dirs = discover_run_dirs(
        run_dirs=args.run_dir,
        root_dirs=args.root_dir,
        generated_filename=args.generated_filename,
        reference_filename=args.reference_filename,
    )
    if not run_dirs:
        print("No run directories found with the requested generated/reference filenames.", file=sys.stderr)
        return 1

    results: list[dict] = []
    summary_rows: list[dict[str, object]] = []
    summary_csv_path = default_summary_csv(args)

    for run_dir in run_dirs:
        result = evaluate_run(
            run_dir=run_dir,
            generated_filename=args.generated_filename,
            reference_filename=args.reference_filename,
            repeats=args.repeats,
            max_graphs=args.max_graphs,
            seed=args.seed,
            device=device,
            use_structural_features=use_structural_features,
        )
        results.append(result)
        summary_rows.append(summary_row(result))
        write_json(run_dir / args.json_filename, result)
        print(
            f"{run_dir}: "
            f"f1_pr={result['metrics']['f1_pr']['mean']:.6f}, "
            f"mmd_rbf={result['metrics']['mmd_rbf']['mean']:.6f}, "
            f"mmd_linear={result['metrics']['mmd_linear']['mean']:.6f}, "
            f"mmd_linear_median={result['metrics']['mmd_linear']['median']:.6f}, "
            f"mmd_linear_trimmed={result['metrics']['mmd_linear']['trimmed_mean']:.6f}, "
            f"precision={result['metrics']['precision']['mean']:.6f}, "
            f"recall={result['metrics']['recall']['mean']:.6f}"
        )

    write_summary_csv(summary_csv_path, summary_rows)
    print(f"Wrote batch summary CSV to {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
