#!/usr/bin/env python3
"""Evaluate frozen per-run LOBSTER checkpoint selections on held-out graphs.

Checkpoint choices must already have been made from validation-only selection
manifests produced by ``select_lobster_checkpoints_per_run.py``. This script
merges those winners, freezes their local paths before loading any held-out
data, and evaluates every selected checkpoint with repeated prior rollouts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import select_lobster_checkpoints as selector  # noqa: E402


METRICS = ("degree", "clustering", "orbit", "spectral", "diameter")
CONDITION_ORDER = (
    "lobster_kiarash_parity_kia40_2000_legacy",
    "lobster_kiarash_parity_kia40_2000_corrected",
    "lobster_kiarash_parity_plain1_1_legacy",
    "lobster_kiarash_parity_plain1_1_corrected",
)
GRAPHVAE_MM_BASELINE = {
    "degree": 0.00990,
    "clustering": 0.00000,
    "orbit": 0.06988,
    "spectral": 0.03136,
    "diameter": 0.24844,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-json",
        type=Path,
        action="append",
        required=True,
        help="Validation-only selection manifest; repeat as needed.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        action="append",
        required=True,
        help="Collected run root; repeat when reused and rerun seeds differ.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        help=(
            "Expected condition in report order; repeat for custom experiment "
            "matrices. Defaults to the four motif-parity conditions."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=12)
    parser.add_argument("--test-rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=21260724)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--generated-filename",
        default="Single_comp_generatedGraphs_adj_kiarash_parity_rollout0.npy",
    )
    parser.add_argument(
        "--model-filename",
        default="kiarash_parity_frozen_validation_model.pt",
    )
    args = parser.parse_args()
    if args.expected_runs <= 0:
        parser.error("--expected-runs must be positive")
    if args.test_rollouts <= 0:
        parser.error("--test-rollouts must be positive")
    if args.condition and len(set(args.condition)) != len(args.condition):
        parser.error("--condition values must be unique")
    for label in ("generated_filename", "model_filename"):
        value = getattr(args, label)
        if Path(value).name != value or value in {"", ".", ".."}:
            parser.error(f"--{label.replace('_', '-')} must be a plain filename")
    return args


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_summary(values, *, sample_std: bool = False) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sequence")
    ddof = 1 if sample_std and array.size > 1 else 0
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=ddof)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def generated_graph_arrays(graphs: list[nx.Graph]) -> np.ndarray:
    matrices = [nx.to_numpy_array(graph, dtype=np.int8) for graph in graphs]
    result = np.empty(len(matrices), dtype=object)
    result[:] = matrices
    return result


def graph_density(graph: nx.Graph) -> float:
    node_count = graph.number_of_nodes()
    if node_count < 2:
        return 0.0
    return float(nx.density(graph))


def resolve_run_dir(winner: dict, runs_roots: list[Path]) -> Path:
    relative_matches = [
        (runs_root / winner["run"]).resolve()
        for runs_root in runs_roots
        if (runs_root / winner["run"]).is_dir()
    ]
    if len(relative_matches) > 1:
        artifact_parts = Path(winner["artifact_dir"]).parts
        source_root_name = next(
            (
                runs_root.name
                for runs_root in runs_roots
                if runs_root.name in artifact_parts
            ),
            None,
        )
        source_matches = [
            match
            for match in relative_matches
            if source_root_name is not None
            and source_root_name in match.parts
        ]
        if len(source_matches) == 1:
            return source_matches[0]
    if len(relative_matches) == 1:
        return relative_matches[0]
    if len(relative_matches) > 1:
        raise RuntimeError(
            "Frozen winner resolves under multiple run roots: "
            f"run={winner['run']!r}, matches={relative_matches}"
        )

    artifact_dir = Path(winner["artifact_dir"]).expanduser()
    if artifact_dir.is_dir():
        return artifact_dir.resolve()

    checkpoint_name = winner["checkpoint"]
    matches = [
        path.parent.resolve()
        for runs_root in runs_roots
        for path in runs_root.rglob(checkpoint_name)
        if path.parent.name.startswith("seed_")
        and winner["run"].split("/")[-1] == path.parent.name
    ]
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    raise FileNotFoundError(
        "Could not map frozen winner to exactly one local run directory: "
        f"run={winner['run']!r}, checkpoint={checkpoint_name!r}, "
        f"matches={len(unique_matches)}"
    )


def condition_and_seed(run_dir: Path) -> tuple[str, int]:
    if not run_dir.name.startswith("seed_"):
        raise ValueError(f"Expected run directory named seed_N: {run_dir}")
    seed = int(run_dir.name[len("seed_"):])
    job_name = run_dir.parent.name
    condition = job_name.split("__", maxsplit=1)[0]
    return condition, seed


def load_frozen_winners(
    selection_paths: list[Path],
    runs_roots: list[Path],
    expected_runs: int,
    condition_order: tuple[str, ...] = CONDITION_ORDER,
) -> list[dict]:
    resolved_by_run = {}
    for selection_path in selection_paths:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        for winner in payload["winners"]:
            run_dir = resolve_run_dir(winner, runs_roots)
            condition, seed = condition_and_seed(run_dir)
            checkpoint_path = run_dir / winner["checkpoint"]
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            local_winner = {
                "condition": condition,
                "seed": seed,
                "run_dir": str(run_dir),
                "checkpoint": winner["checkpoint"],
                "checkpoint_path": str(checkpoint_path.resolve()),
                "selection_score": winner["selection_score"],
                "validation": winner["validation"],
                "source_selection_json": str(selection_path.resolve()),
            }
            key = (condition, seed)
            if key in resolved_by_run:
                raise ValueError(f"Duplicate frozen winner for {key}")
            resolved_by_run[key] = local_winner

    winners = sorted(
        resolved_by_run.values(),
        key=lambda row: (
            condition_order.index(row["condition"])
            if row["condition"] in condition_order
            else len(condition_order),
            row["seed"],
        ),
    )
    if len(winners) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} frozen per-run winners, found {len(winners)}"
        )
    expected_keys = {
        (condition, seed)
        for condition in condition_order
        for seed in range(3)
    }
    actual_keys = {(row["condition"], row["seed"]) for row in winners}
    if expected_runs == len(expected_keys) and actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            f"Frozen parity matrix is incomplete: missing={missing}, extra={extra}"
        )
    return winners


def evaluate_winner(
    winner: dict,
    *,
    device: torch.device,
    latent_dim: int,
    test_rollouts: int,
    seed: int,
    generated_filename: str,
    model_filename: str,
) -> dict:
    run_dir = Path(winner["run_dir"])
    reference_path = run_dir / "heldoutTestGraphs_adj_.npy"
    heldout_items = np.load(reference_path, allow_pickle=True)
    references = selector.to_graphs(
        heldout_items,
        keep_largest_component=False,
    )
    if not references:
        raise ValueError(f"Held-out reference set is empty: {reference_path}")

    decoder = selector.load_decoder(
        Path(winner["checkpoint_path"]),
        device,
        latent_dim,
    )
    reference_nodes = [graph.number_of_nodes() for graph in references]
    reference_edges = [graph.number_of_edges() for graph in references]
    reference_densities = [graph_density(graph) for graph in references]
    denominators = selector.PAPER_TABLE2_BY_DATASET["LOBSTER"]["GraphVAE"]
    rollouts = []
    for rollout_index in range(test_rollouts):
        rollout_seed = seed + rollout_index
        raw_graphs, largest_components = selector.generate(
            decoder,
            len(references),
            latent_dim,
            device,
            rollout_seed,
        )
        metrics = selector.compute_table2_metrics(
            references,
            largest_components,
        )
        normalized_mmd = float(
            np.mean(
                [
                    metrics[metric] / denominators[metric]
                    for metric in METRICS
                ]
            )
        )
        rollout = {
            "rollout": rollout_index,
            "seed": rollout_seed,
            "metrics": metrics,
            "normalized_mmd": normalized_mmd,
            "raw_nodes": numeric_summary(
                [graph.number_of_nodes() for graph in raw_graphs]
            ),
            "lcc_nodes": numeric_summary(
                [graph.number_of_nodes() for graph in largest_components]
            ),
            "raw_edges": numeric_summary(
                [graph.number_of_edges() for graph in raw_graphs]
            ),
            "lcc_edges": numeric_summary(
                [graph.number_of_edges() for graph in largest_components]
            ),
            "lcc_density": numeric_summary(
                [graph_density(graph) for graph in largest_components]
            ),
        }
        rollouts.append(rollout)
        if rollout_index == 0:
            generated_path = run_dir / generated_filename
            with generated_path.open("wb") as handle:
                np.save(
                    handle,
                    generated_graph_arrays(largest_components),
                    allow_pickle=True,
                )

    model_copy = run_dir / model_filename
    shutil.copy2(winner["checkpoint_path"], model_copy)
    result = {
        **winner,
        "reference_path": str(reference_path.resolve()),
        "generated_rollout0_path": str((run_dir / generated_filename).resolve()),
        "materialized_model": str(model_copy.resolve()),
        "reference_nodes": numeric_summary(reference_nodes),
        "reference_edges": numeric_summary(reference_edges),
        "reference_density": numeric_summary(reference_densities),
        "rollouts": rollouts,
        "test_summary": {
            "normalized_mmd": numeric_summary(
                [row["normalized_mmd"] for row in rollouts]
            ),
            "metrics": {
                metric: numeric_summary(
                    [row["metrics"][metric] for row in rollouts]
                )
                for metric in METRICS
            },
            "raw_nodes": numeric_summary(
                [row["raw_nodes"]["mean"] for row in rollouts]
            ),
            "lcc_nodes": numeric_summary(
                [row["lcc_nodes"]["mean"] for row in rollouts]
            ),
            "raw_edges": numeric_summary(
                [row["raw_edges"]["mean"] for row in rollouts]
            ),
            "lcc_edges": numeric_summary(
                [row["lcc_edges"]["mean"] for row in rollouts]
            ),
            "lcc_density": numeric_summary(
                [row["lcc_density"]["mean"] for row in rollouts]
            ),
        },
    }
    del decoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def aggregate_conditions(run_results: list[dict]) -> dict:
    by_condition = defaultdict(list)
    for result in run_results:
        by_condition[result["condition"]].append(result)

    conditions = {}
    for condition, rows in by_condition.items():
        rows = sorted(rows, key=lambda row: row["seed"])
        metric_seed_means = {
            metric: [
                row["test_summary"]["metrics"][metric]["mean"]
                for row in rows
            ]
            for metric in METRICS
        }
        conditions[condition] = {
            "training_seeds": [row["seed"] for row in rows],
            "metrics_across_seed_means": {
                metric: numeric_summary(values, sample_std=True)
                for metric, values in metric_seed_means.items()
            },
            "metrics_across_all_rollouts": {
                metric: numeric_summary(
                    [
                        rollout["metrics"][metric]
                        for row in rows
                        for rollout in row["rollouts"]
                    ]
                )
                for metric in METRICS
            },
            "lcc_nodes_across_seed_means": numeric_summary(
                [row["test_summary"]["lcc_nodes"]["mean"] for row in rows],
                sample_std=True,
            ),
            "raw_nodes_across_seed_means": numeric_summary(
                [row["test_summary"]["raw_nodes"]["mean"] for row in rows],
                sample_std=True,
            ),
            "lcc_edges_across_seed_means": numeric_summary(
                [row["test_summary"]["lcc_edges"]["mean"] for row in rows],
                sample_std=True,
            ),
            "reference_nodes": rows[0]["reference_nodes"],
            "reference_edges": rows[0]["reference_edges"],
        }
    return conditions


def write_run_csv(path: Path, run_results: list[dict]) -> None:
    fieldnames = [
        "condition",
        "seed",
        "checkpoint",
        "validation_selection_score",
        *[f"{metric}_mean" for metric in METRICS],
        *[f"{metric}_std" for metric in METRICS],
        "lcc_nodes_mean",
        "lcc_nodes_std",
        "raw_nodes_mean",
        "raw_nodes_std",
        "lcc_edges_mean",
        "lcc_edges_std",
        "reference_nodes_mean",
        "reference_edges_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for result in run_results:
            row = {
                "condition": result["condition"],
                "seed": result["seed"],
                "checkpoint": result["checkpoint"],
                "validation_selection_score": result["selection_score"],
                "lcc_nodes_mean": result["test_summary"]["lcc_nodes"]["mean"],
                "lcc_nodes_std": result["test_summary"]["lcc_nodes"]["std"],
                "raw_nodes_mean": result["test_summary"]["raw_nodes"]["mean"],
                "raw_nodes_std": result["test_summary"]["raw_nodes"]["std"],
                "lcc_edges_mean": result["test_summary"]["lcc_edges"]["mean"],
                "lcc_edges_std": result["test_summary"]["lcc_edges"]["std"],
                "reference_nodes_mean": result["reference_nodes"]["mean"],
                "reference_edges_mean": result["reference_edges"]["mean"],
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = result["test_summary"]["metrics"][metric][
                    "mean"
                ]
                row[f"{metric}_std"] = result["test_summary"]["metrics"][metric][
                    "std"
                ]
            writer.writerow(row)


def format_mean_std(summary: dict) -> str:
    return f"{summary['mean']:.5f} ± {summary['std']:.5f}"


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Motif-derived Kiarash parity held-out evaluation",
        "",
        (
            "Every checkpoint was selected using validation graphs only. The "
            "combined winner manifest was written before any held-out graph "
            "was loaded. Each selected checkpoint then received "
            f"{payload['test_rollouts']} paired held-out prior rollouts."
        ),
        "",
        (
            "Values below are means ± sample standard deviations across the "
            "three training-seed means; each seed mean contains all held-out "
            "rollouts."
        ),
        (
            "All runs use the byte-identical held-out reference set with "
            f"SHA-256 `{payload['heldout_reference_sha256']}`."
        ),
        "",
        (
            "| Condition | Degree | Clustering | Orbit | Spectral | Diameter "
            "| LCC nodes | Raw nodes |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    conditions = payload["condition_summary"]
    condition_order = payload.get("condition_order", CONDITION_ORDER)
    for condition in condition_order:
        summary = conditions[condition]
        metrics = summary["metrics_across_seed_means"]
        lines.append(
            f"| {condition} | "
            + " | ".join(format_mean_std(metrics[metric]) for metric in METRICS)
            + f" | {format_mean_std(summary['lcc_nodes_across_seed_means'])} |"
            + f" {format_mean_std(summary['raw_nodes_across_seed_means'])} |"
        )
    lines.append(
        "| GraphVAE-MM/Kiarash published control | "
        + " | ".join(f"{GRAPHVAE_MM_BASELINE[metric]:.5f}" for metric in METRICS)
        + " | not reported | not reported |"
    )
    reference_nodes = conditions[condition_order[0]]["reference_nodes"]
    lines += [
        "",
        f"The held-out reference contains {reference_nodes['mean']:.2f} mean nodes.",
        "",
        "The published control is a point estimate, so it is not used as if it "
        "had zero sampling uncertainty. Per-run and per-rollout values are in "
        "`heldout_rollouts.json` and `per_run_summary.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    runs_roots = [path.expanduser().resolve() for path in args.runs_root]
    condition_order = tuple(args.condition or CONDITION_ORDER)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    winners = load_frozen_winners(
        args.selection_json,
        runs_roots,
        args.expected_runs,
        condition_order,
    )

    frozen_payload = {
        "selection_scope": "one_validation-selected_checkpoint_per_run",
        "selection_frozen_before_heldout_load": True,
        "selection_jsons": [
            str(path.expanduser().resolve()) for path in args.selection_json
        ],
        "runs_roots": [str(path) for path in runs_roots],
        "winners": winners,
    }
    write_json(output_dir / "frozen_selections.json", frozen_payload)

    # Selection is now frozen, so held-out artifacts may be inspected. Require
    # every training seed and condition to use the exact same reference file.
    reference_hashes = {
        sha256_file(Path(winner["run_dir"]) / "heldoutTestGraphs_adj_.npy")
        for winner in winners
    }
    if len(reference_hashes) != 1:
        raise RuntimeError(
            "Frozen runs do not share one identical held-out reference set: "
            f"{sorted(reference_hashes)}"
        )
    heldout_reference_sha256 = next(iter(reference_hashes))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_results = []
    with selector.locked_orca_tmp():
        for index, winner in enumerate(winners, start=1):
            print(
                f"[heldout {index}/{len(winners)}] "
                f"{winner['condition']} seed={winner['seed']} "
                f"{winner['checkpoint']}",
                flush=True,
            )
            run_results.append(
                evaluate_winner(
                    winner,
                    device=device,
                    latent_dim=args.latent_dim,
                    test_rollouts=args.test_rollouts,
                    seed=args.seed,
                    generated_filename=args.generated_filename,
                    model_filename=args.model_filename,
                )
            )
            write_json(
                output_dir / "heldout_rollouts.partial.json",
                {"runs": run_results},
            )

    payload = {
        **frozen_payload,
        "device": str(device),
        "test_rollouts": args.test_rollouts,
        "test_seed": args.seed,
        "paired_rollout_seeds_across_checkpoints": True,
        "condition_order": list(condition_order),
        "heldout_reference_sha256": heldout_reference_sha256,
        "metrics": list(METRICS),
        "runs": run_results,
        "condition_summary": aggregate_conditions(run_results),
        "graphvae_mm_baseline": GRAPHVAE_MM_BASELINE,
    }
    write_json(output_dir / "heldout_rollouts.json", payload)
    write_run_csv(output_dir / "per_run_summary.csv", run_results)
    write_report(output_dir / "analysis.md", payload)
    partial_path = output_dir / "heldout_rollouts.partial.json"
    if partial_path.exists():
        partial_path.unlink()
    return payload


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
