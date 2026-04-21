#!/usr/bin/env python3
"""Reproduce Kia paper Table 2 metrics for the Grid dataset.

This script is intentionally separate from the training entrypoint. It can:

1. Compute the Table 2 `50/50 split` ideal/reference row.
2. Compare a saved generated-graphs `.npy` file against the paper-style Grid
   test split for the `GraphVAE` row.

Outputs are written under `runs/table2_reproduction/...` by default.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# stat_rnn's ORCA wrapper uses relative paths like ./eval/orca/orca.
os.chdir(REPO_ROOT)

from data import list_graph_loader  # noqa: E402
from stat_rnn import (  # noqa: E402
    MMD_diam,
    clustering_stats,
    degree_stats,
    orbit_stats_all,
    spectral_stats,
)


PAPER_TABLE2_GRID = {
    "50/50 split": {
        "degree": 1e-5,
        "clustering": 0.0,
        "orbit": 2e-5,
        "spectral": 0.004,
        "diameter": 0.014,
    },
    "GraphVAE": {
        "degree": 0.062,
        "clustering": 0.055,
        "orbit": 0.515,
        "spectral": 0.018,
        "diameter": 0.143,
    },
}


@contextmanager
def preserve_orca_tmp():
    tmp_path = REPO_ROOT / "eval" / "orca" / "tmp.txt"
    existed = tmp_path.exists()
    original = tmp_path.read_bytes() if existed else None
    try:
        yield
    finally:
        if existed:
            tmp_path.write_bytes(original)
        elif tmp_path.exists():
            tmp_path.unlink()


def adjacency_to_graph(adj) -> nx.Graph:
    if isinstance(adj, nx.Graph):
        graph = nx.Graph(adj)
    elif hasattr(adj, "toarray"):
        graph = nx.from_numpy_array(adj.toarray())
    else:
        graph = nx.from_numpy_array(np.asarray(adj))

    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def largest_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph
    return nx.Graph(graph.subgraph(max(nx.connected_components(graph), key=len)))


def to_graphs(items, keep_largest_component: bool = False) -> list[nx.Graph]:
    graphs = []
    for item in items:
        graph = adjacency_to_graph(item)
        if keep_largest_component:
            graph = largest_component(graph)
        if graph.number_of_nodes() > 0:
            graphs.append(graph)
    return graphs


def load_grid_adjacencies(max_graphs: int | None = None):
    list_adj = list_graph_loader("GRID", return_labels=True)[0]
    if max_graphs is not None:
        list_adj = list_adj[:max_graphs]
    return list_adj


def deterministic_shuffle(items, seed: int):
    shuffled = list(items)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def split_50_50(items, seed: int):
    shuffled = deterministic_shuffle(items, seed)
    midpoint = len(shuffled) // 2
    return shuffled[:midpoint], shuffled[midpoint:]


def split_paper_70_10_20(items, seed: int):
    shuffled = deterministic_shuffle(items, seed)
    n = len(shuffled)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def load_npy_graphs(path: Path, keep_largest_component: bool) -> list[nx.Graph]:
    arrays = np.load(path, allow_pickle=True)
    return to_graphs(arrays, keep_largest_component=keep_largest_component)


def compute_table2_metrics(reference_graphs: list[nx.Graph], generated_graphs: list[nx.Graph]):
    return {
        "degree": float(degree_stats(reference_graphs, generated_graphs)),
        "clustering": float(clustering_stats(reference_graphs, generated_graphs)),
        "orbit": float(orbit_stats_all(reference_graphs, generated_graphs)),
        "spectral": float(spectral_stats(reference_graphs, generated_graphs)),
        "diameter": float(MMD_diam(reference_graphs, generated_graphs)),
    }


def compare_to_paper(row_name: str, current: dict[str, float]):
    paper = PAPER_TABLE2_GRID[row_name]
    comparison = {}
    for metric, paper_value in paper.items():
        current_value = current[metric]
        comparison[metric] = {
            "paper": paper_value,
            "current": current_value,
            "difference": current_value - paper_value,
        }
    return comparison


def format_value(value: float) -> str:
    if abs(value) < 1e-4 and value != 0:
        return f"{value:.6e}"
    return f"{value:.6f}"


def write_outputs(output_dir: Path, rows: dict[str, dict], metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata,
        "paper_table2_grid": PAPER_TABLE2_GRID,
        "rows": rows,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Table 2 Grid Reproduction",
        "",
        "Lower is better for all MMD metrics.",
        "",
        "| Row | Metric | Paper | Current | Difference |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row_name, comparison in rows.items():
        for metric, values in comparison.items():
            lines.append(
                "| {row} | {metric} | {paper} | {current} | {diff} |".format(
                    row=row_name,
                    metric=metric,
                    paper=format_value(values["paper"]),
                    current=format_value(values["current"]),
                    diff=format_value(values["difference"]),
                )
            )

    lines.extend([
        "",
        "## Metadata",
        "",
    ])
    for key, value in metadata.items():
        lines.append(f"- `{key}`: `{value}`")

    table_path = output_dir / "table2_grid_reproduction.md"
    table_path.write_text("\n".join(lines) + "\n")
    return metrics_path, table_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["ideal-50-50", "evaluate-generated", "all"],
        default="ideal-50-50",
        help="Which Table 2 row(s) to compute.",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=None,
        help="Saved generated graph .npy file for the GraphVAE row.",
    )
    parser.add_argument(
        "--test-graphs",
        type=Path,
        default=None,
        help="Optional saved test graph .npy file. If omitted, the paper-style test split is regenerated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/table2_reproduction/grid_metrics"),
        help="Directory for metrics.json and table2_grid_reproduction.md.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Deterministic split seed. The current codebase uses 123 for data_split.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional smoke-test limit. Do not use for paper numbers.",
    )
    args = parser.parse_args()

    if args.mode in {"evaluate-generated", "all"} and args.generated is None:
        raise SystemExit("--generated is required for evaluate-generated/all mode.")

    rows = {}
    metadata = {
        "dataset": "GRID",
        "seed": args.seed,
        "max_graphs": args.max_graphs,
        "mode": args.mode,
    }

    grid_adjs = load_grid_adjacencies(max_graphs=args.max_graphs)

    with preserve_orca_tmp():
        if args.mode in {"ideal-50-50", "all"}:
            left, right = split_50_50(grid_adjs, seed=args.seed)
            left_graphs = to_graphs(left)
            right_graphs = to_graphs(right)
            current = compute_table2_metrics(left_graphs, right_graphs)
            rows["50/50 split"] = compare_to_paper("50/50 split", current)
            metadata["ideal_50_50_counts"] = f"{len(left_graphs)}/{len(right_graphs)}"

        if args.mode in {"evaluate-generated", "all"}:
            generated_graphs = load_npy_graphs(args.generated, keep_largest_component=True)
            if args.test_graphs is not None:
                test_graphs = load_npy_graphs(args.test_graphs, keep_largest_component=False)
                metadata["test_source"] = str(args.test_graphs)
            else:
                _, _, test_adjs = split_paper_70_10_20(grid_adjs, seed=args.seed)
                test_graphs = to_graphs(test_adjs)
                metadata["test_source"] = "regenerated paper_70_10_20 split"

            current = compute_table2_metrics(test_graphs, generated_graphs)
            rows["GraphVAE"] = compare_to_paper("GraphVAE", current)
            metadata["generated_source"] = str(args.generated)
            metadata["graphvae_counts"] = f"{len(test_graphs)}/{len(generated_graphs)}"

    metrics_path, table_path = write_outputs(args.output_dir, rows, metadata)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
