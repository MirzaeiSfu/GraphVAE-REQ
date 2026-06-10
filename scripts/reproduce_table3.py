#!/usr/bin/env python3
"""Reproduce Kia paper Table 3 GNN-based metrics for supported datasets.

This script complements the Table 2 statistics-based reproduction path.
It reuses the vendored Random-GIN evaluator to:

1. Compute the Table 3 `50/50 split` ideal/reference row.
2. Compare a saved generated-graphs `.npy` file against saved reference
   test graphs for a paper row such as `GraphVAE-MM`.

Outputs are written under `runs/table3_reproduction/...` by default.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from data import list_graph_loader  # noqa: E402
from evaluate_graph_realism_batch import (  # noqa: E402
    DEFAULT_GENERATED_FILENAME,
    DEFAULT_REFERENCE_FILENAME,
    evaluate_graph_collections,
    load_graph_items,
    preprocess_graphs,
    resolve_device,
)


PAPER_ROW_ORDER = (
    "50/50 split",
    "GraphVAE-MM",
    "GraphRNN-S",
    "GraphRNN",
    "GRAN",
    "BiGG",
)
TABLE3_METRIC_ORDER = ("mmd_rbf", "f1_pr")
TABLE3_METRIC_LABELS = {
    "mmd_rbf": "MMD RBF",
    "f1_pr": "F1 PR",
}
TABLE3_METRIC_SCALES = {
    "mmd_rbf": 1.0,
    "f1_pr": 100.0,
}
DATASET_DISPLAY_NAMES = {
    "TRIANGULAR_GRID": "Triangle Grid",
    "LOBSTER": "Lobster",
    "GRID": "Grid",
    "ogbg-molbbbp": "ogbg-molbbbp",
    "PROTEINS": "Protein",
}
DATASET_ALIASES = {
    "triangle_grid": "TRIANGULAR_GRID",
    "triangular_grid": "TRIANGULAR_GRID",
    "lobster": "LOBSTER",
    "grid": "GRID",
    "ogbg-molbbbp": "ogbg-molbbbp",
    "protein": "PROTEINS",
    "proteins": "PROTEINS",
}

PAPER_TABLE3_BY_DATASET = {
    "TRIANGULAR_GRID": {
        "50/50 split": {
            "mmd_rbf": {"mean": 0.03, "std": 0.00},
            "f1_pr": {"mean": 98.58, "std": 0.00},
        },
        "GraphVAE-MM": {
            "mmd_rbf": {"mean": 0.17, "std": 0.01},
            "f1_pr": {"mean": 83.58, "std": 5.50},
        },
        "GraphRNN-S": {
            "mmd_rbf": {"mean": 0.72, "std": 0.17},
            "f1_pr": {"mean": 33.68, "std": 19.44},
        },
        "GraphRNN": {
            "mmd_rbf": {"mean": 0.64, "std": 0.11},
            "f1_pr": {"mean": 25.80, "std": 11.75},
        },
        "GRAN": {
            "mmd_rbf": {"mean": 0.88, "std": 0.09},
            "f1_pr": {"mean": 23.71, "std": 9.72},
        },
        "BiGG": {
            "mmd_rbf": {"mean": 0.41, "std": 0.13},
            "f1_pr": {"mean": 62.08, "std": 0.14},
        },
    },
    "LOBSTER": {
        "50/50 split": {
            "mmd_rbf": {"mean": 0.04, "std": 0.00},
            "f1_pr": {"mean": 98.58, "std": 0.00},
        },
        "GraphVAE-MM": {
            "mmd_rbf": {"mean": 0.10, "std": 0.00},
            "f1_pr": {"mean": 100.00, "std": 0.00},
        },
        "GraphRNN-S": {
            "mmd_rbf": {"mean": 0.98, "std": 0.13},
            "f1_pr": {"mean": 58.72, "std": 7.55},
        },
        "GraphRNN": {
            "mmd_rbf": {"mean": 0.87, "std": 0.04},
            "f1_pr": {"mean": 61.97, "std": 0.00},
        },
        "GRAN": {
            "mmd_rbf": {"mean": 0.24, "std": 0.04},
            "f1_pr": {"mean": 50.53, "std": 12.12},
        },
        "BiGG": {
            "mmd_rbf": {"mean": 0.12, "std": 0.00},
            "f1_pr": {"mean": 99.74, "std": 0.76},
        },
    },
    "GRID": {
        "50/50 split": {
            "mmd_rbf": {"mean": 0.009, "std": 0.00},
            "f1_pr": {"mean": 98.70, "std": 0.00},
        },
        "GraphVAE-MM": {
            "mmd_rbf": {"mean": 0.13, "std": 0.01},
            "f1_pr": {"mean": 97.09, "std": 6.33},
        },
        "GraphRNN-S": {
            "mmd_rbf": {"mean": 0.79, "std": 0.08},
            "f1_pr": {"mean": 71.18, "std": 2.36},
        },
        "GraphRNN": {
            "mmd_rbf": {"mean": 0.99, "std": 0.03},
            "f1_pr": {"mean": 13.22, "std": 0.05},
        },
        "GRAN": {
            "mmd_rbf": {"mean": 0.40, "std": 0.00},
            "f1_pr": {"mean": 78.73, "std": 0.02},
        },
        "BiGG": {
            "mmd_rbf": {"mean": 0.35, "std": 0.00},
            "f1_pr": {"mean": 92.43, "std": 0.00},
        },
    },
    "ogbg-molbbbp": {
        "50/50 split": {
            "mmd_rbf": {"mean": 0.002, "std": 0.00},
            "f1_pr": {"mean": 98.07, "std": 0.00},
        },
        "GraphVAE-MM": {
            "mmd_rbf": {"mean": 0.02, "std": 0.01},
            "f1_pr": {"mean": 93.78, "std": 1.33},
        },
        "GraphRNN-S": {
            "mmd_rbf": {"mean": 0.48, "std": 0.02},
            "f1_pr": {"mean": 81.41, "std": 0.71},
        },
        "GraphRNN": {
            "mmd_rbf": {"mean": 1.45, "std": 0.19},
            "f1_pr": {"mean": 98.94, "std": 0.56},
        },
        "GRAN": {
            "mmd_rbf": {"mean": 0.39, "std": 0.07},
            "f1_pr": {"mean": 94.06, "std": 2.60},
        },
        "BiGG": {
            "mmd_rbf": {"mean": 0.04, "std": 0.00},
            "f1_pr": {"mean": 96.16, "std": 0.31},
        },
    },
    "PROTEINS": {
        "50/50 split": {
            "mmd_rbf": {"mean": 0.04, "std": 0.00},
            "f1_pr": {"mean": 98.67, "std": 1.11},
        },
        "GraphVAE-MM": {
            "mmd_rbf": {"mean": 0.03, "std": 0.01},
            "f1_pr": {"mean": 90.78, "std": 3.76},
        },
        "GraphRNN-S": {
            "mmd_rbf": {"mean": 0.28, "std": 0.26},
            "f1_pr": {"mean": 72.36, "std": 27.63},
        },
        "GraphRNN": {
            "mmd_rbf": {"mean": 0.32, "std": 0.14},
            "f1_pr": {"mean": 93.94, "std": 0.56},
        },
        "GRAN": {
            "mmd_rbf": {"mean": 0.07, "std": 0.00},
            "f1_pr": {"mean": 98.05, "std": 0.76},
        },
        "BiGG": {
            "mmd_rbf": {"mean": 0.15, "std": 0.00},
            "f1_pr": {"mean": 98.11, "std": 0.62},
        },
    },
}


def canonicalize_dataset_name(raw_dataset: str) -> str:
    normalized = str(raw_dataset).strip()
    return DATASET_ALIASES.get(normalized.lower(), normalized)


def dataset_display_name(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES[dataset]


def dataset_slug(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="GRID",
        help=(
            "Dataset/Table 3 block to evaluate. Supported values: "
            + ", ".join(sorted(PAPER_TABLE3_BY_DATASET))
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["ideal-50-50", "evaluate-generated", "all"],
        default="ideal-50-50",
        help="Which Table 3 row(s) to compute.",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=None,
        help="Saved generated graph .npy file for the current row.",
    )
    parser.add_argument(
        "--test-graphs",
        type=Path,
        default=None,
        help=(
            "Saved reference/test graph .npy file. "
            f"Defaults to {DEFAULT_REFERENCE_FILENAME} inside --run-dir."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional run directory containing the generated/reference graph files. "
            f"When provided, defaults are {DEFAULT_GENERATED_FILENAME} and "
            f"{DEFAULT_REFERENCE_FILENAME}."
        ),
    )
    parser.add_argument(
        "--source-graphs",
        type=Path,
        default=None,
        help=(
            "Optional saved graph collection used for the ideal 50/50 row. "
            "This is useful for datasets such as ogbg-molbbbp when the raw loader "
            "is unavailable in the local environment."
        ),
    )
    parser.add_argument(
        "--paper-row",
        choices=PAPER_ROW_ORDER,
        default="GraphVAE-MM",
        help="Which paper row to compare the generated graphs against.",
    )
    parser.add_argument(
        "--row-label",
        default="Current",
        help="Label to use for the evaluated generated-graphs row in the report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for metrics.json and table3_<dataset>_reproduction.md. "
            "Default: runs/table3_reproduction/<dataset>."
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
        help="Maximum number of generated/reference graphs to evaluate. Default: 1000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for deterministic splits and evaluator seeding.",
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
    return parser.parse_args()


def deterministic_shuffle(items, seed: int):
    shuffled = list(items)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def split_50_50(items, seed: int):
    shuffled = deterministic_shuffle(items, seed)
    midpoint = len(shuffled) // 2
    return shuffled[:midpoint], shuffled[midpoint:]


def load_dataset_items(dataset: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return list_graph_loader(dataset, return_labels=True)[0]


def load_source_items(dataset: str, source_graphs: Path | None, seed: int):
    if source_graphs is not None:
        return load_graph_items(source_graphs.expanduser().resolve())
    if dataset == "ogbg-molbbbp":
        raise ValueError(
            "Dataset ogbg-molbbbp needs --source-graphs for ideal-50-50 on this repo, "
            "because the raw list_graph_loader path is not enabled locally."
        )
    return load_dataset_items(dataset, seed)


def scale_metric_summary_for_table3(metric_name: str, metric_summary: dict[str, float]) -> dict[str, float]:
    scale = TABLE3_METRIC_SCALES.get(metric_name, 1.0)
    return {
        "mean": metric_summary["mean"] * scale,
        "std": metric_summary["std"] * scale,
    }


def scale_metrics_for_table3(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric_name: scale_metric_summary_for_table3(metric_name, metric_summary)
        for metric_name, metric_summary in metrics.items()
    }


def compare_to_paper(dataset: str, paper_row: str, current_metrics: dict[str, dict[str, float]]):
    paper_metrics = PAPER_TABLE3_BY_DATASET[dataset][paper_row]
    scaled_current_metrics = scale_metrics_for_table3(current_metrics)
    comparison = {}
    for metric_name in TABLE3_METRIC_ORDER:
        current_summary = scaled_current_metrics[metric_name]
        paper_summary = paper_metrics[metric_name]
        comparison[metric_name] = {
            "paper_mean": paper_summary["mean"],
            "paper_std": paper_summary["std"],
            "current_mean": current_summary["mean"],
            "current_std": current_summary["std"],
            "difference_mean": current_summary["mean"] - paper_summary["mean"],
            "difference_std": current_summary["std"] - paper_summary["std"],
        }
    return comparison


def format_value(value: float) -> str:
    if abs(value) < 1e-4 and value != 0:
        return f"{value:.6e}"
    return f"{value:.6f}"


def format_summary(metric_summary: dict[str, float]) -> str:
    return f"{format_value(metric_summary['mean'])} +/- {format_value(metric_summary['std'])}"


def write_outputs(
    output_dir: Path,
    dataset: str,
    metadata: dict,
    current_rows: dict[str, dict],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata,
        "paper_table3": PAPER_TABLE3_BY_DATASET[dataset],
        "rows": current_rows,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# Table 3 {dataset_display_name(dataset)} Reproduction",
        "",
        "Lower is better for `MMD RBF`; higher is better for `F1 PR`.",
        "For paper comparability, `F1 PR` is reported as a percentage here.",
        "",
        "## Current vs Paper",
        "",
        "| Current Row | Paper Row | Metric | Paper Mean | Paper Std | Current Mean | Current Std | Difference |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row_label, row_payload in current_rows.items():
        for metric_name in TABLE3_METRIC_ORDER:
            values = row_payload["comparison"][metric_name]
            lines.append(
                "| {row_label} | {paper_row} | {metric} | {paper_mean} | {paper_std} | "
                "{current_mean} | {current_std} | {difference} |".format(
                    row_label=row_label,
                    paper_row=row_payload["paper_row"],
                    metric=TABLE3_METRIC_LABELS[metric_name],
                    paper_mean=format_value(values["paper_mean"]),
                    paper_std=format_value(values["paper_std"]),
                    current_mean=format_value(values["current_mean"]),
                    current_std=format_value(values["current_std"]),
                    difference=format_value(values["difference_mean"]),
                )
            )

    lines.extend([
        "",
        "## Paper Reference Rows",
        "",
        "| Paper Row | MMD RBF | F1 PR |",
        "| --- | ---: | ---: |",
    ])
    for paper_row in PAPER_ROW_ORDER:
        paper_metrics = PAPER_TABLE3_BY_DATASET[dataset][paper_row]
        lines.append(
            "| {paper_row} | {mmd_rbf} | {f1_pr} |".format(
                paper_row=paper_row,
                mmd_rbf=format_summary(paper_metrics["mmd_rbf"]),
                f1_pr=format_summary(paper_metrics["f1_pr"]),
            )
        )

    lines.extend([
        "",
        "## Metadata",
        "",
    ])
    for key, value in metadata.items():
        lines.append(f"- `{key}`: `{value}`")

    table_path = output_dir / f"table3_{dataset_slug(dataset)}_reproduction.md"
    table_path.write_text("\n".join(lines) + "\n")
    return metrics_path, table_path


def resolve_generated_paths(args: argparse.Namespace):
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        generated_path = (run_dir / DEFAULT_GENERATED_FILENAME) if args.generated is None else args.generated
        test_graphs_path = (run_dir / DEFAULT_REFERENCE_FILENAME) if args.test_graphs is None else args.test_graphs
        return run_dir, generated_path.expanduser().resolve(), test_graphs_path.expanduser().resolve()

    if args.generated is None or args.test_graphs is None:
        raise ValueError("--generated and --test-graphs are required unless --run-dir is provided.")

    return None, args.generated.expanduser().resolve(), args.test_graphs.expanduser().resolve()


def evaluate_ideal_row(
    dataset: str,
    source_graphs: Path | None,
    repeats: int,
    max_graphs: int,
    seed: int,
    device,
    use_structural_features: bool,
):
    items = load_source_items(dataset, source_graphs=source_graphs, seed=seed)
    left_items, right_items = split_50_50(items, seed)
    left_graphs = preprocess_graphs(left_items, max_graphs=max_graphs, seed=seed, shuffle=False)
    right_graphs = preprocess_graphs(right_items, max_graphs=max_graphs, seed=seed, shuffle=False)
    result = evaluate_graph_collections(
        generated_graphs=left_graphs,
        reference_graphs=right_graphs,
        repeats=repeats,
        seed=seed,
        device=device,
        use_structural_features=use_structural_features,
    )
    result["source_graphs"] = str(source_graphs.expanduser().resolve()) if source_graphs else "list_graph_loader"
    return result


def evaluate_generated_row(
    generated_path: Path,
    test_graphs_path: Path,
    repeats: int,
    max_graphs: int,
    seed: int,
    device,
    use_structural_features: bool,
):
    generated_graphs = preprocess_graphs(
        load_graph_items(generated_path),
        max_graphs=max_graphs,
        seed=seed,
        shuffle=True,
    )
    test_graphs = preprocess_graphs(
        load_graph_items(test_graphs_path),
        max_graphs=max_graphs,
        seed=seed,
        shuffle=False,
    )
    result = evaluate_graph_collections(
        generated_graphs=generated_graphs,
        reference_graphs=test_graphs,
        repeats=repeats,
        seed=seed,
        device=device,
        use_structural_features=use_structural_features,
    )
    result["generated_path"] = str(generated_path)
    result["test_graphs_path"] = str(test_graphs_path)
    return result


def main() -> int:
    args = parse_args()
    dataset = canonicalize_dataset_name(args.dataset)
    if dataset not in PAPER_TABLE3_BY_DATASET:
        supported = ", ".join(sorted(PAPER_TABLE3_BY_DATASET))
        raise ValueError(f"Unsupported dataset {args.dataset!r}. Supported values: {supported}")

    device = resolve_device(args.device)
    use_structural_features = not args.no_structural_features
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (REPO_ROOT / "runs" / "table3_reproduction" / dataset_slug(dataset)).resolve()
    )

    metadata = {
        "dataset": dataset,
        "dataset_display_name": dataset_display_name(dataset),
        "mode": args.mode,
        "seed": args.seed,
        "repeats": args.repeats,
        "max_graphs": args.max_graphs,
        "device": str(device),
        "structural_features": use_structural_features,
        "paper_row": args.paper_row,
    }
    current_rows = {}

    if args.mode in {"ideal-50-50", "all"}:
        ideal_result = evaluate_ideal_row(
            dataset=dataset,
            source_graphs=args.source_graphs,
            repeats=args.repeats,
            max_graphs=args.max_graphs,
            seed=args.seed,
            device=device,
            use_structural_features=use_structural_features,
        )
        current_rows["50/50 split"] = {
            "paper_row": "50/50 split",
            "metrics": ideal_result["metrics"],
            "table3_metrics": scale_metrics_for_table3(ideal_result["metrics"]),
            "raw_metrics": ideal_result["raw_metrics"],
            "comparison": compare_to_paper(dataset, "50/50 split", ideal_result["metrics"]),
            "details": ideal_result,
        }
        metadata["ideal_source_graphs"] = ideal_result["source_graphs"]

    if args.mode in {"evaluate-generated", "all"}:
        run_dir, generated_path, test_graphs_path = resolve_generated_paths(args)
        generated_result = evaluate_generated_row(
            generated_path=generated_path,
            test_graphs_path=test_graphs_path,
            repeats=args.repeats,
            max_graphs=args.max_graphs,
            seed=args.seed,
            device=device,
            use_structural_features=use_structural_features,
        )
        current_rows[args.row_label] = {
            "paper_row": args.paper_row,
            "metrics": generated_result["metrics"],
            "table3_metrics": scale_metrics_for_table3(generated_result["metrics"]),
            "raw_metrics": generated_result["raw_metrics"],
            "comparison": compare_to_paper(dataset, args.paper_row, generated_result["metrics"]),
            "details": generated_result,
        }
        metadata["run_dir"] = str(run_dir) if run_dir else None
        metadata["generated"] = str(generated_path)
        metadata["test_graphs"] = str(test_graphs_path)

    metrics_path, table_path = write_outputs(
        output_dir=output_dir,
        dataset=dataset,
        metadata=metadata,
        current_rows=current_rows,
    )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
