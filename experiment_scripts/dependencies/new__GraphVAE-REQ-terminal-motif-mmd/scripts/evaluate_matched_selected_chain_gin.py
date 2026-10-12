#!/usr/bin/env python3
"""Random-GIN and aggregate metrics for matched PTC/OGB terminal chains."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stat_rnn import ggm_gnn_quality_stats
from terminal_motif_evaluation import terminal_adjacencies_to_graphs, write_json


OUTPUT_NAME = "matched_selected_chain_random_gin.json"
AGGREGATE_NAME = "matched_selected_terminal_chain_results.json"
CSV_NAME = "matched_selected_terminal_chain_results.csv"
REPORT_NAME = "MATCHED_SELECTED_TERMINAL_CHAIN_RESULTS.md"
DEFAULT_SELECTED_CHAINS = {
    "AIDS": 2,
    "MUTAG": 9,
    "PROTEINS": 2,
    "PTC": 29,
    "ogbg-molbbbp": 65,
}
SETTING_SELECTED_CHAINS = {
    ("ENZYMES", "setting_01"): 67,
    ("ENZYMES", "setting_03"): 99,
    ("ENZYMES", "setting_03_top150_alpha001"): 149,
}
STRUCTURAL_METRICS = (
    "degree",
    "clustering",
    "sparsity",
    "orbit",
    "spectral",
    "triangle",
    "diameter",
)


def selected_chain_index(dataset: str, setting: str) -> int:
    setting_key = (dataset, setting)
    if setting_key in SETTING_SELECTED_CHAINS:
        return SETTING_SELECTED_CHAINS[setting_key]
    if dataset in DEFAULT_SELECTED_CHAINS:
        return DEFAULT_SELECTED_CHAINS[dataset]
    raise KeyError(f"No selected chain configured for {dataset}/{setting}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather-root", required=True, type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Evaluate matching seed-level runs without writing the root-level "
            "aggregate. This permits safe parallel evaluation; run once more "
            "with --aggregate-only after all workers finish."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def graphs_at_chain(archive_path: Path, chain_index: int):
    with np.load(archive_path, allow_pickle=False) as archive:
        selected = archive["selected_chain_indices"].astype(int).tolist()
        position = selected.index(chain_index)
        matrices = archive["binary_terminal_matrices"][:, position]
    return terminal_adjacencies_to_graphs(matrices)[0]


def evaluate_run(
    matched_path: Path,
    *,
    device: str,
    repetitions: int,
    overwrite: bool,
) -> Dict[str, Any]:
    matched = load_json(matched_path)
    chain_index = selected_chain_index(
        matched["dataset"], matched["setting"]
    )
    run_dir = matched_path.parent.parent
    output_path = matched_path.parent / OUTPUT_NAME
    if output_path.exists() and not overwrite:
        existing = load_json(output_path)
        if (
            existing["chain_index"] == chain_index
            and existing["random_gin_repetitions"] == repetitions
        ):
            print(f"[skip] {run_dir}")
            return existing

    generated_graphs = graphs_at_chain(
        matched_path.parent / "terminal_chain_binary_matrices.npz",
        chain_index,
    )
    reference_graphs = graphs_at_chain(
        Path(matched["reference_archive"]),
        chain_index,
    )
    seed = int(matched["training_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    os.environ["GRAPHVAE_GGM_GNN_DEVICE"] = device
    metrics = ggm_gnn_quality_stats(
        reference_graphs,
        generated_graphs,
        runs=repetitions,
    )
    if metrics.get("error") is not None:
        raise RuntimeError(metrics["error"])
    source_chain = next(
        chain for chain in matched["chains"]
        if int(chain["chain_index"]) == chain_index
    )
    payload = {
        "metric_family": "matched_independent_terminal_chain_random_gin",
        "run_dir": str(run_dir),
        "dataset": matched["dataset"],
        "setting": matched["setting"],
        "training_seed": seed,
        "chain_index": chain_index,
        "rule": source_chain["rule"],
        "states": source_chain["states"],
        "value": source_chain["value"],
        "comparison": "generated chain j versus real test chain j",
        "combined_with_other_chains": False,
        "binary_before_metrics": True,
        "generated_graph_count": len(generated_graphs),
        "reference_graph_count": len(reference_graphs),
        "random_gin_repetitions": repetitions,
        "metrics": metrics,
    }
    write_json(output_path, payload)
    print(
        f"[done] {matched['dataset']} {matched['setting']} seed={seed} "
        f"chain={chain_index} MMD-RBF={metrics['mmd_rbf']:.6f} "
        f"F1-PR={metrics['f1_pr']:.6f}"
    )
    return payload


def sample_stat(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(root: Path) -> Dict[str, Any]:
    structural_paths = sorted(
        root.rglob(
            "terminal_motif_mmd/matched_per_chain_structural_mmd.json"
        )
    )
    gin_by_run = {
        str(load_json(path)["run_dir"]): load_json(path)
        for path in root.rglob(f"terminal_motif_mmd/{OUTPUT_NAME}")
    }
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for path in structural_paths:
        matched = load_json(path)
        try:
            chain_index = selected_chain_index(
                matched["dataset"], matched["setting"]
            )
        except KeyError:
            continue
        chain = next(
            chain for chain in matched["chains"]
            if int(chain["chain_index"]) == chain_index
        )
        gin = gin_by_run.get(matched["run_dir"])
        if gin is None:
            continue
        grouped[(matched["dataset"], matched["setting"])].append(
            {
                "seed": int(matched["training_seed"]),
                "chain": chain,
                "gin": gin,
            }
        )

    settings = []
    for (dataset, setting), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["seed"])
        representative = rows[0]["chain"]
        structural = {
            metric: sample_stat(
                [float(row["chain"]["metrics"][metric]) for row in rows]
            )
            for metric in STRUCTURAL_METRICS
        }
        gin = {
            metric: sample_stat(
                [float(row["gin"]["metrics"][metric]) for row in rows]
            )
            for metric in ("mmd_rbf", "f1_pr", "precision", "recall")
        }
        settings.append(
            {
                "dataset": dataset,
                "setting": setting,
                "chain_index": selected_chain_index(dataset, setting),
                "rule": representative["rule"],
                "states": representative["states"],
                "value": representative["value"],
                "seeds": [row["seed"] for row in rows],
                "generated_nonempty_count": sample_stat(
                    [
                        float(row["chain"]["generated_nonempty_count"])
                        for row in rows
                    ]
                ),
                "reference_nonempty_count": sample_stat(
                    [
                        float(row["chain"]["reference_nonempty_count"])
                        for row in rows
                    ]
                ),
                "generated_edge_count": sample_stat(
                    [
                        float(row["chain"]["generated_edge_count"])
                        for row in rows
                    ]
                ),
                "reference_edge_count": sample_stat(
                    [
                        float(row["chain"]["reference_edge_count"])
                        for row in rows
                    ]
                ),
                "structural_metrics": structural,
                "random_gin_metrics": gin,
                "per_seed": rows,
            }
        )
    return {
        "metric_family": "matched_independent_terminal_chain",
        "comparison": "generated chain j versus real test chain j",
        "combination_across_chains": "none",
        "binarization_threshold": 0.5,
        "random_gin_repetitions_per_seed": 10,
        "aggregation": (
            "mean and sample standard deviation across training seeds"
        ),
        "run_count": sum(len(rows) for rows in grouped.values()),
        "setting_count": len(settings),
        "settings": settings,
    }


def fmt(stat: Dict[str, float], digits: int = 5) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def write_outputs(root: Path, payload: Dict[str, Any]) -> None:
    write_json(root / AGGREGATE_NAME, payload)
    fields = [
        "dataset",
        "setting",
        "chain_index",
        "seeds",
        "generated_edges_mean",
        "generated_edges_std",
        "reference_edges_mean",
        "reference_edges_std",
        *[
            f"{metric}_{suffix}"
            for metric in STRUCTURAL_METRICS
            for suffix in ("mean", "std")
        ],
        *[
            f"{metric}_{suffix}"
            for metric in ("mmd_rbf", "f1_pr", "precision", "recall")
            for suffix in ("mean", "std")
        ],
    ]
    with (root / CSV_NAME).open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for setting in payload["settings"]:
            row = {
                "dataset": setting["dataset"],
                "setting": setting["setting"],
                "chain_index": setting["chain_index"],
                "seeds": json.dumps(setting["seeds"]),
                "generated_edges_mean": setting["generated_edge_count"]["mean"],
                "generated_edges_std": setting["generated_edge_count"]["std"],
                "reference_edges_mean": setting["reference_edge_count"]["mean"],
                "reference_edges_std": setting["reference_edge_count"]["std"],
            }
            for metric in STRUCTURAL_METRICS:
                row[f"{metric}_mean"] = setting["structural_metrics"][metric][
                    "mean"
                ]
                row[f"{metric}_std"] = setting["structural_metrics"][metric][
                    "std"
                ]
            for metric in ("mmd_rbf", "f1_pr", "precision", "recall"):
                row[f"{metric}_mean"] = setting["random_gin_metrics"][metric][
                    "mean"
                ]
                row[f"{metric}_std"] = setting["random_gin_metrics"][metric][
                    "std"
                ]
            writer.writerow(row)

    lines = [
        "# Matched terminal-chain evaluation",
        "",
        "Each generated terminal matrix is compared with the same rule/value "
        "terminal matrix computed on the attributed real test graphs. Both "
        "sides are binarized at 0.5 and cleaned identically. Chains are never "
        "combined. Random-GIN values use 10 repetitions per seed; all ± values "
        "are sample standard deviations across training seeds.",
        "",
        "| Dataset | Setting | Chain | Seeds | Gen/ref edges | Degree ↓ | "
        "Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ | MMD-RBF ↓ | "
        "F1-PR ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in payload["settings"]:
        structural = setting["structural_metrics"]
        gin = setting["random_gin_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    setting["dataset"],
                    setting["setting"],
                    str(setting["chain_index"]),
                    ",".join(map(str, setting["seeds"])),
                    (
                        f"{setting['generated_edge_count']['mean']:.2f}/"
                        f"{setting['reference_edge_count']['mean']:.2f}"
                    ),
                    fmt(structural["degree"]),
                    fmt(structural["clustering"]),
                    fmt(structural["orbit"]),
                    fmt(structural["spectral"]),
                    fmt(structural["diameter"]),
                    fmt(gin["mmd_rbf"]),
                    fmt(gin["f1_pr"], 4),
                ]
            )
            + " |"
        )
    lines.append("")
    (root / REPORT_NAME).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cli = parse_args()
    root = cli.gather_root.expanduser().resolve()
    os.chdir(REPO_ROOT)
    if not cli.aggregate_only:
        requested = (
            {dataset.lower() for dataset in cli.dataset}
            if cli.dataset
            else None
        )
        paths = sorted(
            root.rglob(
                "terminal_motif_mmd/matched_per_chain_structural_mmd.json"
            )
        )
        paths = [
            path
            for path in paths
            if (
                (
                    requested is None
                    or load_json(path)["dataset"].lower() in requested
                )
            )
        ]
        print(f"Evaluating {len(paths)} matched seed-level chains.")
        failures = []
        for path in paths:
            try:
                evaluate_run(
                    path,
                    device=cli.device,
                    repetitions=cli.runs,
                    overwrite=cli.overwrite,
                )
            except Exception as exc:
                failures.append({"path": str(path), "error": repr(exc)})
                print(f"[error] {path}: {exc}", file=sys.stderr)
        if failures:
            write_json(root / "matched_selected_chain_failures.json", failures)
            return 1
        if cli.evaluate_only:
            return 0
    payload = aggregate(root)
    write_outputs(root, payload)
    print(
        f"Aggregated {payload['run_count']} runs / "
        f"{payload['setting_count']} settings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
