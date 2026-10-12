#!/usr/bin/env python3
"""Random-GIN MMD-RBF and F1-PR for the reported best independent chains."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stat_rnn import ggm_gnn_quality_stats
from terminal_motif_evaluation import (
    reference_adjacencies_to_graphs,
    terminal_adjacencies_to_graphs,
    write_json,
)


RUN_OUTPUT_NAME = "best_chain_random_gin_metrics.json"
AGGREGATE_JSON_NAME = "best_per_chain_random_gin_aggregated.json"
AGGREGATE_CSV_NAME = "best_per_chain_random_gin_aggregated.csv"
REPORT_NAME = "BEST_PER_CHAIN_RANDOM_GIN_RESULTS.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather-root", required=True, type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def best_chain_map(root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    aggregate = load_json(root / "per_chain_terminal_motif_mmd_aggregated.json")
    return {
        (setting["dataset"], setting["setting"]): setting[
            "best_complete_coverage_chain"
        ]
        for setting in aggregate["settings"]
    }


def discover_run_results(root: Path, datasets: List[str] | None) -> List[Path]:
    paths = sorted(
        root.rglob("terminal_motif_mmd/per_chain_structural_mmd.json")
    )
    if datasets:
        requested = {dataset.lower() for dataset in datasets}
        paths = [
            path
            for path in paths
            if load_json(path)["dataset"].lower() in requested
        ]
    return paths


def evaluate_run(
    per_chain_path: Path,
    *,
    best_chains: Dict[Tuple[str, str], Dict[str, Any]],
    device: str,
    repetitions: int,
    overwrite: bool,
) -> Dict[str, Any]:
    run_dir = per_chain_path.parent.parent
    output_path = per_chain_path.parent / RUN_OUTPUT_NAME
    per_chain = load_json(per_chain_path)
    key = (per_chain["dataset"], per_chain["setting"])
    best = best_chains[key]
    chain_index = int(best["chain_index"])

    if output_path.exists() and not overwrite:
        existing = load_json(output_path)
        if (
            int(existing["chain_index"]) == chain_index
            and int(existing["random_gin_repetitions"]) == repetitions
        ):
            print(f"[skip] {run_dir}")
            return existing

    archive_path = per_chain_path.parent / "terminal_chain_binary_matrices.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        selected = archive["selected_chain_indices"].astype(int).tolist()
        position = selected.index(chain_index)
        adjacencies = archive["binary_terminal_matrices"][:, position]
    references = np.load(run_dir / "testGraphs_adj_.npy", allow_pickle=True)
    generated_graphs, cleanup = terminal_adjacencies_to_graphs(adjacencies)
    reference_graphs = reference_adjacencies_to_graphs(references)

    evaluation_seed = int(per_chain["training_seed"])
    random.seed(evaluation_seed)
    np.random.seed(evaluation_seed)
    torch.manual_seed(evaluation_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(evaluation_seed)
    os.environ["GRAPHVAE_GGM_GNN_DEVICE"] = device
    metrics = ggm_gnn_quality_stats(
        reference_graphs,
        generated_graphs,
        runs=repetitions,
    )
    if metrics.get("error") is not None:
        raise RuntimeError(metrics["error"])

    payload = {
        "metric_family": "independent_terminal_chain_random_gin",
        "run_dir": str(run_dir),
        "dataset": per_chain["dataset"],
        "setting": per_chain["setting"],
        "training_seed": per_chain["training_seed"],
        "generation_seed": per_chain["generation_seed"],
        "checkpoint": per_chain["checkpoint"],
        "chain_index": chain_index,
        "rule": best["rule"],
        "states": best["states"],
        "value": best["value"],
        "combined_with_other_chains": False,
        "binary_before_metrics": True,
        "random_gin_repetitions": repetitions,
        "evaluation_seed": evaluation_seed,
        "reference_graph_count": len(reference_graphs),
        "generated_graph_count": len(generated_graphs),
        "graph_cleanup": cleanup,
        "metrics": metrics,
    }
    write_json(output_path, payload)
    print(
        f"[done] {per_chain['dataset']} {per_chain['setting']} "
        f"seed={per_chain['training_seed']} chain={chain_index} "
        f"MMD-RBF={metrics['mmd_rbf']:.6f} F1-PR={metrics['f1_pr']:.6f}"
    )
    return payload


def sample_mean_std(values: List[float]) -> Tuple[float, float]:
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def aggregate_results(root: Path) -> Dict[str, Any]:
    results = [
        load_json(path)
        for path in sorted(
            root.rglob(f"terminal_motif_mmd/{RUN_OUTPUT_NAME}")
        )
    ]
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault((result["dataset"], result["setting"]), []).append(
            result
        )

    settings = []
    for (dataset, setting), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["training_seed"]))
        metrics: Dict[str, Dict[str, float]] = {}
        for name in ("mmd_rbf", "f1_pr", "precision", "recall"):
            mean, std = sample_mean_std(
                [float(row["metrics"][name]) for row in rows]
            )
            metrics[name] = {"mean": mean, "std": std}
        settings.append(
            {
                "dataset": dataset,
                "setting": setting,
                "chain_index": rows[0]["chain_index"],
                "rule": rows[0]["rule"],
                "states": rows[0]["states"],
                "value": rows[0]["value"],
                "seeds": [int(row["training_seed"]) for row in rows],
                "random_gin_repetitions_per_seed": rows[0][
                    "random_gin_repetitions"
                ],
                "metrics": metrics,
                "per_seed": [
                    {
                        "seed": int(row["training_seed"]),
                        "generated_graph_count": row["generated_graph_count"],
                        "metrics": row["metrics"],
                    }
                    for row in rows
                ],
            }
        )
    return {
        "metric_family": "independent_terminal_chain_random_gin",
        "combination_across_chains": "none",
        "aggregation": (
            "mean and sample standard deviation across training-seed means; "
            "each seed mean uses 10 Random-GIN repetitions"
        ),
        "run_count": len(results),
        "setting_count": len(settings),
        "settings": settings,
    }


def fmt(metric: Dict[str, float], digits: int) -> str:
    return f"{metric['mean']:.{digits}f} ± {metric['std']:.{digits}f}"


def write_aggregate_outputs(root: Path, payload: Dict[str, Any]) -> None:
    write_json(root / AGGREGATE_JSON_NAME, payload)
    with (root / AGGREGATE_CSV_NAME).open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "dataset",
            "setting",
            "chain_index",
            "seeds",
            "mmd_rbf_mean",
            "mmd_rbf_std",
            "f1_pr_mean",
            "f1_pr_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for setting in payload["settings"]:
            row = {
                "dataset": setting["dataset"],
                "setting": setting["setting"],
                "chain_index": setting["chain_index"],
                "seeds": json.dumps(setting["seeds"]),
            }
            for metric in ("mmd_rbf", "f1_pr", "precision", "recall"):
                row[f"{metric}_mean"] = setting["metrics"][metric]["mean"]
                row[f"{metric}_std"] = setting["metrics"][metric]["std"]
            writer.writerow(row)

    lines = [
        "# Random-GIN results for best independent terminal chains",
        "",
        "Every row uses the same independently evaluated chain selected in the "
        "structural-MMD summary. No matrices from different chains are "
        "combined. Each seed value is averaged over 10 Random-GIN feature "
        "extractors; displayed ± values are sample standard deviations across "
        "training seeds.",
        "",
        "| Dataset | Setting | Chain | Seeds | MMD-RBF ↓ | F1-PR ↑ | "
        "Precision ↑ | Recall ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in payload["settings"]:
        metrics = setting["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    setting["dataset"],
                    setting["setting"],
                    str(setting["chain_index"]),
                    ",".join(map(str, setting["seeds"])),
                    fmt(metrics["mmd_rbf"], 5),
                    fmt(metrics["f1_pr"], 4),
                    fmt(metrics["precision"], 4),
                    fmt(metrics["recall"], 4),
                ]
            )
            + " |"
        )
    lines.append("")
    (root / REPORT_NAME).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cli = parse_args()
    if cli.runs < 1:
        raise ValueError("--runs must be positive.")
    root = cli.gather_root.expanduser().resolve()
    os.chdir(REPO_ROOT)
    if not cli.aggregate_only:
        best_chains = best_chain_map(root)
        files = discover_run_results(root, cli.dataset)
        print(f"Evaluating {len(files)} seed-level best chains.")
        failures = []
        for path in files:
            try:
                evaluate_run(
                    path,
                    best_chains=best_chains,
                    device=cli.device,
                    repetitions=cli.runs,
                    overwrite=cli.overwrite,
                )
            except Exception as exc:
                failures.append({"path": str(path), "error": repr(exc)})
                print(f"[error] {path}: {exc}", file=sys.stderr)
        if failures:
            write_json(root / "best_per_chain_random_gin_failures.json", failures)
            return 1

    payload = aggregate_results(root)
    write_aggregate_outputs(root, payload)
    print(
        f"Aggregated {payload['run_count']} runs / "
        f"{payload['setting_count']} settings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
