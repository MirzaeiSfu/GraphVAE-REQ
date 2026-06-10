#!/usr/bin/env python3
"""Regenerate Kia paper 50/50 results across all core Table 2/3 datasets.

This script computes the ideal/reference `50/50 split` rows for both:

1. Table 2 statistics-based metrics:
   degree, clustering, orbit, spectral, diameter
2. Table 3 GNN-based metrics:
   MMD RBF, F1 PR

It uses the stricter Kia-style 50/50 GNN evaluation semantics from
``reproduce_table3_kia_5050.py`` and writes one combined JSON + Markdown report.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch
from dgl.data.utils import load_graphs


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from data import list_graph_loader  # noqa: E402
from reproduce_table2_grid import compute_table2_metrics, locked_orca_tmp  # noqa: E402
from reproduce_table3 import PAPER_TABLE3_BY_DATASET  # noqa: E402
from reproduce_table3_kia_5050 import (  # noqa: E402
    Evaluator,
    item_to_nx_graph,
    preprocess_like_upstream,
    resolve_device,
    summarize,
    to_dgl_like_upstream,
)


DATASET_ORDER = (
    "TRIANGULAR_GRID",
    "LOBSTER",
    "GRID",
    "ogbg-molbbbp",
    "PROTEINS",
)

DATASET_DISPLAY_NAMES = {
    "TRIANGULAR_GRID": "Triangle Grid",
    "LOBSTER": "Lobster",
    "GRID": "Grid",
    "ogbg-molbbbp": "ogbg-molbbbp",
    "PROTEINS": "Protein",
}

PAPER_TABLE2_50_50 = {
    "TRIANGULAR_GRID": {
        "degree": 3e-5,
        "clustering": 0.002,
        "orbit": 8e-5,
        "spectral": 0.004,
        "diameter": 0.014,
    },
    "LOBSTER": {
        "degree": 0.002,
        "clustering": 0.0,
        "orbit": 0.002,
        "spectral": 0.005,
        "diameter": 0.032,
    },
    "GRID": {
        "degree": 1e-5,
        "clustering": 0.0,
        "orbit": 2e-5,
        "spectral": 0.004,
        "diameter": 0.014,
    },
    "ogbg-molbbbp": {
        "degree": 2e-4,
        "clustering": 2e-5,
        "orbit": 9e-5,
        "spectral": 5e-4,
        "diameter": 0.002,
    },
    "PROTEINS": {
        "degree": 4e-5,
        "clustering": 0.004,
        "orbit": 5e-4,
        "spectral": 4e-4,
        "diameter": 0.003,
    },
}

TABLE2_METRIC_ORDER = ("degree", "clustering", "orbit", "spectral", "diameter")
TABLE2_METRIC_LABELS = {
    "degree": "Deg",
    "clustering": "Clus",
    "orbit": "Orbit",
    "spectral": "Spect",
    "diameter": "Diam",
}
TABLE3_METRIC_ORDER = ("mmd_rbf", "f1_pr")
TABLE3_METRIC_LABELS = {
    "mmd_rbf": "MMD RBF",
    "f1_pr": "F1 PR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASET_ORDER),
        help="Datasets to evaluate. Default: all five Kia Table 2/3 datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Deterministic 50/50 split seed. Default: 123.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of Random-GIN initializations to average for Table 3. Default: 10.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for the GIN evaluator. Default: auto.",
    )
    parser.add_argument(
        "--no-structural-features",
        action="store_true",
        help="Disable Kia-style structural node features for Table 3.",
    )
    parser.add_argument(
        "--ogbg-source",
        type=Path,
        default=None,
        help=(
            "Optional ogbg-molbbbp source graph file or DGL processed graph file. "
            "If omitted, auto-detect from local repo cache."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/5050_reproduction/all_datasets"),
        help="Directory for the combined JSON and Markdown report.",
    )
    return parser.parse_args()


def canonicalize_dataset_name(dataset: str) -> str:
    normalized = dataset.strip()
    upper = normalized.upper()
    if upper == "TRIANGLE_GRID":
        return "TRIANGULAR_GRID"
    if upper == "PROTEIN":
        return "PROTEINS"
    if normalized.lower() == "ogbg-molbbbp":
        return "ogbg-molbbbp"
    return upper


def format_value(value: float) -> str:
    if abs(value) < 1e-4 and value != 0:
        return f"{value:.6e}"
    return f"{value:.6f}"


def clamp_percent(value: float) -> float:
    return float(min(100.0, max(0.0, value)))


def load_ogbg_graphs_from_dgl(processed_path: Path) -> list[nx.Graph]:
    graphs, _ = load_graphs(str(processed_path))
    nx_graphs: list[nx.Graph] = []
    for graph in graphs:
        src, dst = graph.edges()
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(graph.num_nodes()))
        nx_graph.add_edges_from(zip(src.tolist(), dst.tolist()))
        nx_graphs.append(nx_graph)
    return nx_graphs


def load_dataset_graphs(dataset: str, seed: int, ogbg_source: Path | None) -> list[nx.Graph]:
    if dataset == "ogbg-molbbbp":
        candidate_paths = []
        if ogbg_source is not None:
            candidate_paths.append(ogbg_source.expanduser().resolve())
        candidate_paths.append(
            (REPO_ROOT / "data_raw" / "ogb" / "ogbg_molbbbp" / "processed" / "dgl_data_processed").resolve()
        )

        for candidate in candidate_paths:
            if not candidate.exists():
                continue
            if candidate.suffix == ".npy":
                arrays = np.load(candidate, allow_pickle=True)
                return [item_to_nx_graph(item) for item in arrays]
            return load_ogbg_graphs_from_dgl(candidate)

        raise FileNotFoundError(
            "Could not find ogbg-molbbbp source graphs. Pass --ogbg-source or ensure "
            "data_raw/ogb/ogbg_molbbbp/processed/dgl_data_processed exists."
        )

    random.seed(seed)
    np.random.seed(seed)
    items = list_graph_loader(dataset, return_labels=True)[0]
    return [item_to_nx_graph(item) for item in items]


def split_50_50(items: list[nx.Graph], seed: int) -> tuple[list[nx.Graph], list[nx.Graph]]:
    shuffled = list(items)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    return shuffled[:midpoint], shuffled[midpoint:]


def compute_table2_50_50(graphs: list[nx.Graph], seed: int) -> dict[str, float]:
    left_graphs, right_graphs = split_50_50(graphs, seed=seed)
    with locked_orca_tmp():
        return compute_table2_metrics(left_graphs, right_graphs)


def compare_table2_to_paper(dataset: str, current_metrics: dict[str, float]) -> dict[str, dict[str, float]]:
    paper_metrics = PAPER_TABLE2_50_50[dataset]
    return {
        metric_name: {
            "paper": paper_metrics[metric_name],
            "current": current_metrics[metric_name],
            "difference": current_metrics[metric_name] - paper_metrics[metric_name],
        }
        for metric_name in TABLE2_METRIC_ORDER
    }


def compute_table3_50_50(
    dataset: str,
    graphs: list[nx.Graph],
    seed: int,
    repeats: int,
    device: torch.device,
    use_structural_features: bool,
) -> dict:
    shuffled = list(graphs)
    random.Random(seed).shuffle(shuffled)
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
    for repeat_index in range(repeats):
        repeat_seed = seed + repeat_index
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

    current_metrics = {
        "f1_pr": summarize(f1_values),
        "mmd_rbf": summarize(mmd_values),
        "precision": summarize(precision_values),
        "recall": summarize(recall_values),
    }
    f1_percent_mean = clamp_percent(current_metrics["f1_pr"]["mean"] * 100.0)
    f1_percent_std = clamp_percent(current_metrics["f1_pr"]["std"] * 100.0)
    paper_metrics = PAPER_TABLE3_BY_DATASET[dataset]["50/50 split"]
    comparison = {
        "mmd_rbf": {
            "paper_mean": paper_metrics["mmd_rbf"]["mean"],
            "paper_std": paper_metrics["mmd_rbf"]["std"],
            "current_mean": current_metrics["mmd_rbf"]["mean"],
            "current_std": current_metrics["mmd_rbf"]["std"],
            "difference_mean": current_metrics["mmd_rbf"]["mean"] - paper_metrics["mmd_rbf"]["mean"],
        },
        "f1_pr": {
            "paper_mean": paper_metrics["f1_pr"]["mean"],
            "paper_std": paper_metrics["f1_pr"]["std"],
            "current_mean": f1_percent_mean,
            "current_std": f1_percent_std,
            "difference_mean": f1_percent_mean - paper_metrics["f1_pr"]["mean"],
        },
    }

    return {
        "metrics": current_metrics,
        "comparison": comparison,
        "metadata": {
            "num_generated_graphs": len(generated),
            "num_reference_graphs": len(reference),
            "repeats": repeats,
            "structural_features": use_structural_features,
            "device": str(device),
            "logic": "Kia-style 50/50 evaluator",
        },
    }


def build_report(dataset_results: dict, metadata: dict) -> str:
    lines = [
        "# 50/50 Paper Regeneration",
        "",
        "This report recomputes the ideal/reference `50/50 split` rows for Kia's",
        "five core graph-generation datasets.",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Table 3 MMD RBF | Paper | Diff | Table 3 F1 PR | Paper | Diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for dataset in DATASET_ORDER:
        if dataset not in dataset_results:
            continue
        display = DATASET_DISPLAY_NAMES[dataset]
        table3 = dataset_results[dataset]["table3"]["comparison"]
        lines.append(
            "| {dataset} | {mmd_cur} | {mmd_paper} | {mmd_diff} | {f1_cur} | {f1_paper} | {f1_diff} |".format(
                dataset=display,
                mmd_cur=format_value(table3["mmd_rbf"]["current_mean"]),
                mmd_paper=format_value(table3["mmd_rbf"]["paper_mean"]),
                mmd_diff=format_value(table3["mmd_rbf"]["difference_mean"]),
                f1_cur=format_value(table3["f1_pr"]["current_mean"]),
                f1_paper=format_value(table3["f1_pr"]["paper_mean"]),
                f1_diff=format_value(table3["f1_pr"]["difference_mean"]),
            )
        )

    lines.extend([
        "",
        "## Per-Dataset Details",
        "",
    ])

    for dataset in DATASET_ORDER:
        if dataset not in dataset_results:
            continue
        display = DATASET_DISPLAY_NAMES[dataset]
        result = dataset_results[dataset]
        lines.extend([
            f"### {display}",
            "",
            "#### Table 3",
            "",
            "| Metric | Current Mean | Current Std | Paper Mean | Paper Std | Difference |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for metric_name in TABLE3_METRIC_ORDER:
            comp = result["table3"]["comparison"][metric_name]
            lines.append(
                "| {metric} | {current_mean} | {current_std} | {paper_mean} | {paper_std} | {diff} |".format(
                    metric=TABLE3_METRIC_LABELS[metric_name],
                    current_mean=format_value(comp["current_mean"]),
                    current_std=format_value(comp["current_std"]),
                    paper_mean=format_value(comp["paper_mean"]),
                    paper_std=format_value(comp["paper_std"]),
                    diff=format_value(comp["difference_mean"]),
                )
            )

        lines.extend([
            "",
            "#### Table 2",
            "",
            "| Metric | Current | Paper | Difference |",
            "| --- | ---: | ---: | ---: |",
        ])
        for metric_name in TABLE2_METRIC_ORDER:
            comp = result["table2"]["comparison"][metric_name]
            lines.append(
                "| {metric} | {current} | {paper} | {diff} |".format(
                    metric=TABLE2_METRIC_LABELS[metric_name],
                    current=format_value(comp["current"]),
                    paper=format_value(comp["paper"]),
                    diff=format_value(comp["difference"]),
                )
            )

        lines.extend([
            "",
            "Counts:",
            f"- raw graphs: `{result['dataset_metadata']['num_raw_graphs']}`",
            f"- Table 3 post-preprocess 50/50 counts: `{result['table3']['metadata']['num_generated_graphs']}/{result['table3']['metadata']['num_reference_graphs']}`",
            "",
        ])

    lines.extend([
        "## Run Metadata",
        "",
    ])
    for key, value in metadata.items():
        lines.append(f"- `{key}`: `{value}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    datasets = [canonicalize_dataset_name(dataset) for dataset in args.datasets]
    unsupported = [dataset for dataset in datasets if dataset not in DATASET_ORDER]
    if unsupported:
        raise SystemExit(
            "Unsupported datasets: " + ", ".join(unsupported) +
            ". Supported: " + ", ".join(DATASET_ORDER)
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    use_structural_features = not args.no_structural_features

    dataset_results = {}
    for dataset in datasets:
        graphs = load_dataset_graphs(dataset, seed=args.seed, ogbg_source=args.ogbg_source)
        table2_current = compute_table2_50_50(graphs, seed=args.seed)
        table3_current = compute_table3_50_50(
            dataset=dataset,
            graphs=graphs,
            seed=args.seed,
            repeats=args.repeats,
            device=device,
            use_structural_features=use_structural_features,
        )

        dataset_results[dataset] = {
            "dataset_metadata": {
                "num_raw_graphs": len(graphs),
            },
            "table2": {
                "current": table2_current,
                "comparison": compare_table2_to_paper(dataset, table2_current),
            },
            "table3": table3_current,
        }

    metadata = {
        "datasets": datasets,
        "seed": args.seed,
        "repeats": args.repeats,
        "device": str(device),
        "structural_features": use_structural_features,
        "ogbg_source": (
            str(args.ogbg_source.expanduser().resolve()) if args.ogbg_source is not None else "auto-detect"
        ),
    }

    payload = {
        "metadata": metadata,
        "paper_table2_50_50": PAPER_TABLE2_50_50,
        "paper_table3_50_50": {
            dataset: PAPER_TABLE3_BY_DATASET[dataset]["50/50 split"] for dataset in DATASET_ORDER
        },
        "datasets": dataset_results,
    }

    json_path = output_dir / "metrics.json"
    md_path = output_dir / "regenerate_50_50_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(build_report(dataset_results, metadata))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
