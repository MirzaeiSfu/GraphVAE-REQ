#!/usr/bin/env python3
"""Evaluate every saved terminal motif rule/value chain independently."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from per_chain_terminal_motif_evaluation import (
    compute_feature_cache,
    evaluate_signature_chain,
    prepare_generated_adjacency,
    prepare_reference_adjacency,
    reference_metric_context,
)
from terminal_motif_evaluation import write_json


OUTPUT_NAME = "per_chain_structural_mmd.json"
CSV_NAME = "per_chain_structural_mmd.csv"
METRIC_NAMES = (
    "degree",
    "clustering",
    "sparsity",
    "orbit",
    "spectral",
    "triangle",
    "diameter",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a separate structural MMD for every binarized terminal "
            "motif rule/value chain. No matrices are combined."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", action="append", type=Path)
    source.add_argument("--gather-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Limit gather-root discovery to one or more dataset directory names.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


def discover_runs(cli: argparse.Namespace) -> List[Path]:
    if cli.run_dir:
        runs = [path.expanduser().resolve() for path in cli.run_dir]
    else:
        runs = sorted(
            path.parent.parent
            for path in cli.gather_root.expanduser().resolve().rglob(
                "terminal_motif_mmd/terminal_chain_binary_matrices.npz"
            )
        )
    if cli.dataset:
        requested = {name.lower() for name in cli.dataset}
        runs = [
            run
            for run in runs
            if any(
                part.lower() in requested
                for part in run.parts
            )
        ]
    return runs[: cli.max_runs] if cli.max_runs is not None else runs


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _edge_context(
    reference_signatures: Sequence[bytes], cache: Dict[bytes, Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "samples": [
            float(cache[signature]["edge_count"])
            for signature in reference_signatures
        ]
    }


def write_chain_csv(path: Path, payload: Dict[str, Any]) -> None:
    fields = [
        "dataset",
        "setting",
        "training_seed",
        "chain_index",
        "rule_index",
        "value_index",
        "rule",
        "states",
        "value",
        "status",
        "input_sample_count",
        "nonempty_sample_count",
        "nonempty_fraction",
        *METRIC_NAMES,
        "reference_edge_count",
        "generated_edge_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for chain in payload["chains"]:
            row = {
                "dataset": payload["dataset"],
                "setting": payload["setting"],
                "training_seed": payload["training_seed"],
                "chain_index": chain["chain_index"],
                "rule_index": chain["rule_index"],
                "value_index": chain["value_index"],
                "rule": json.dumps(chain["rule"], sort_keys=True),
                "states": json.dumps(chain["states"], sort_keys=True),
                "value": json.dumps(chain["value"], sort_keys=True),
                "status": chain["status"],
                "input_sample_count": chain["input_sample_count"],
                "nonempty_sample_count": chain["nonempty_sample_count"],
                "nonempty_fraction": chain["nonempty_fraction"],
                "reference_edge_count": chain.get("reference_edge_count"),
                "generated_edge_count": chain.get("generated_edge_count"),
            }
            row.update(chain["metrics"])
            writer.writerow(row)


def evaluate_run(run_dir: Path, cli: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    output_dir = run_dir / "terminal_motif_mmd"
    output_path = output_dir / OUTPUT_NAME
    if output_path.exists() and not cli.overwrite:
        print(f"[skip] {run_dir}")
        return load_json(output_path)

    archive_path = output_dir / "terminal_chain_binary_matrices.npz"
    metadata_path = output_dir / "chain_metadata.json"
    union_result_path = output_dir / "terminal_motif_mmd.json"
    reference_path = run_dir / "testGraphs_adj_.npy"
    for required in (
        archive_path,
        metadata_path,
        union_result_path,
        reference_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    prior_result = load_json(union_result_path)
    all_metadata = load_json(metadata_path)
    metadata_by_index = {
        int(entry["chain_index"]): entry for entry in all_metadata
    }
    references = np.load(reference_path, allow_pickle=True)
    with np.load(archive_path, allow_pickle=False) as archive:
        binary = archive["binary_terminal_matrices"]
        selected_indices = archive["selected_chain_indices"].astype(int)

    if binary.ndim != 4:
        raise ValueError(f"Expected B,C,N,N archive, got {binary.shape}.")
    if binary.shape[1] != len(selected_indices):
        raise ValueError("Selected-chain metadata does not match archive.")
    if not np.all((binary == 0) | (binary == 1)):
        raise ValueError(f"Archive is not binary: {archive_path}")

    prepared_references = [
        prepared
        for adjacency in references
        if (prepared := prepare_reference_adjacency(adjacency)) is not None
    ]
    chain_signatures: List[List[bytes]] = []
    prepared_generated = []
    for chain_position in range(binary.shape[1]):
        signatures: List[bytes] = []
        for adjacency in binary[:, chain_position]:
            prepared = prepare_generated_adjacency(adjacency)
            if prepared is not None:
                signatures.append(prepared.signature)
                prepared_generated.append(prepared)
        chain_signatures.append(signatures)

    cache = compute_feature_cache(
        [*prepared_references, *prepared_generated],
        workers=cli.workers,
    )
    reference_signatures = [
        prepared.signature for prepared in prepared_references
    ]
    reference_context = reference_metric_context(reference_signatures, cache)
    reference_context["edge_count"] = _edge_context(
        reference_signatures, cache
    )

    chains: List[Dict[str, Any]] = []
    for chain_position, (chain_index, signatures) in enumerate(
        zip(selected_indices.tolist(), chain_signatures)
    ):
        metadata = metadata_by_index[chain_index]
        chain_result: Dict[str, Any] = {
            "archive_position": chain_position,
            "chain_index": chain_index,
            "rule_index": metadata["rule_index"],
            "value_index": metadata["value_index"],
            "rule": metadata["rule"],
            "states": metadata["states"],
            "value": metadata["value"],
            "rule_source": metadata.get("rule_source"),
            "terminal_shape": metadata["terminal_shape"],
            "status": "ok" if signatures else "empty_terminal_chain",
            "input_sample_count": int(binary.shape[0]),
            "nonempty_sample_count": len(signatures),
            "nonempty_fraction": len(signatures) / int(binary.shape[0]),
            "binary_before_metrics": True,
            "combined_with_other_chains": False,
        }
        if signatures:
            chain_result.update(
                evaluate_signature_chain(
                    signatures,
                    reference_context=reference_context,
                    cache=cache,
                )
            )
        else:
            chain_result.update(
                {
                    "metrics": {name: None for name in METRIC_NAMES},
                    "metric_generated_sample_counts": {
                        name: 0 for name in METRIC_NAMES
                    },
                    "reference_edge_count": float(
                        np.mean(reference_context["edge_count"]["samples"])
                    ),
                    "generated_edge_count": None,
                }
            )
        chains.append(chain_result)
        print(
            f"[chain] {run_dir.name} {chain_position + 1}/{binary.shape[1]} "
            f"nonempty={len(signatures)}"
        )

    payload = {
        "metric_family": "independent_binarized_terminal_chain_structural_mmd",
        "run_dir": str(run_dir),
        "dataset": prior_result["dataset"],
        "setting": prior_result["setting"],
        "training_seed": prior_result["training_seed"],
        "generation_seed": prior_result["generation_seed"],
        "checkpoint": prior_result["checkpoint"],
        "sample_count": int(binary.shape[0]),
        "reference_sample_count": len(reference_signatures),
        "selected_chain_count": int(binary.shape[1]),
        "evaluation_definition": {
            "chain_handling": (
                "each rule/value terminal matrix chain is evaluated separately"
            ),
            "combination_across_chains": "none",
            "binarization": "value >= 0.5 -> 1; otherwise -> 0",
            "self_loops_removed": True,
            "isolates_removed_from_generated": True,
            "largest_generated_connected_component_only": True,
            "structural_metrics": list(METRIC_NAMES),
            "kernels": "same Gaussian-TV kernels and sigmas as stat_rnn.mmd_eval",
            "random_gin_metrics_included": False,
        },
        "unique_graph_count": len(cache),
        "chains": chains,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_path, payload)
    write_chain_csv(output_dir / CSV_NAME, payload)
    print(f"[done] {run_dir}: {output_path}")
    return payload


def main() -> int:
    cli = parse_args()
    if cli.workers < 1:
        raise ValueError("--workers must be positive.")
    os.chdir(REPO_ROOT)
    runs = discover_runs(cli)
    if not runs:
        raise RuntimeError("No per-chain archives found.")
    print(f"Discovered {len(runs)} run(s).")

    completed = []
    failures = []
    for index, run_dir in enumerate(runs, start=1):
        print(f"\n=== {index}/{len(runs)} {run_dir} ===")
        try:
            payload = evaluate_run(run_dir, cli)
            completed.append(
                {
                    "run_dir": str(run_dir),
                    "dataset": payload["dataset"],
                    "setting": payload["setting"],
                    "training_seed": payload["training_seed"],
                    "selected_chain_count": payload["selected_chain_count"],
                }
            )
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "error": repr(exc)})
            print(f"[error] {run_dir}: {exc}", file=sys.stderr)

    summary_root = (
        cli.gather_root.expanduser().resolve()
        if cli.gather_root
        else runs[0].parent
    )
    write_json(
        summary_root / "per_chain_terminal_motif_mmd_batch_summary.json",
        {"completed": completed, "failures": failures},
    )
    print(f"Completed={len(completed)} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
