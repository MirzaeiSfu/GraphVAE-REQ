#!/usr/bin/env python3
"""Compare each generated terminal chain with the same real test chain."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from per_chain_terminal_motif_evaluation import (
    compute_feature_cache,
    evaluate_signature_chain,
    prepare_generated_adjacency,
    reference_metric_context,
)
from terminal_motif_evaluation import write_json


OUTPUT_NAME = "matched_per_chain_structural_mmd.json"
CSV_NAME = "matched_per_chain_structural_mmd.csv"
METRICS = (
    "degree",
    "clustering",
    "sparsity",
    "orbit",
    "spectral",
    "triangle",
    "diameter",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--reference-archive", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepared_by_chain(binary: np.ndarray):
    signatures: List[List[bytes]] = []
    prepared_all = []
    for position in range(binary.shape[1]):
        current = []
        for adjacency in binary[:, position]:
            prepared = prepare_generated_adjacency(adjacency)
            if prepared is not None:
                current.append(prepared.signature)
                prepared_all.append(prepared)
        signatures.append(current)
    return signatures, prepared_all


def write_csv(path: Path, payload: Dict[str, Any]) -> None:
    fields = [
        "dataset",
        "setting",
        "training_seed",
        "chain_index",
        "rule_index",
        "value_index",
        "rule",
        "value",
        "status",
        "generated_nonempty_count",
        "reference_nonempty_count",
        *METRICS,
        "generated_edge_count",
        "reference_edge_count",
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
                "value": json.dumps(chain["value"], sort_keys=True),
                "status": chain["status"],
                "generated_nonempty_count": chain[
                    "generated_nonempty_count"
                ],
                "reference_nonempty_count": chain[
                    "reference_nonempty_count"
                ],
                "generated_edge_count": chain.get("generated_edge_count"),
                "reference_edge_count": chain.get("reference_edge_count"),
            }
            row.update(chain["metrics"])
            writer.writerow(row)


def main() -> int:
    cli = parse_args()
    os.chdir(REPO_ROOT)
    run_dir = cli.run_dir.expanduser().resolve()
    output_dir = run_dir / "terminal_motif_mmd"
    output_path = output_dir / OUTPUT_NAME
    if output_path.exists() and not cli.overwrite:
        print(f"[skip] {run_dir}")
        return 0

    started = time.time()
    prior_per_chain_path = output_dir / "per_chain_structural_mmd.json"
    if prior_per_chain_path.exists():
        prior = load_json(prior_per_chain_path)
        metadata = {
            int(chain["chain_index"]): chain for chain in prior["chains"]
        }
    else:
        prior = load_json(output_dir / "terminal_motif_mmd.json")
        metadata = {
            int(chain["chain_index"]): chain
            for chain in load_json(output_dir / "chain_metadata.json")
        }
    generated_path = output_dir / "terminal_chain_binary_matrices.npz"
    with np.load(generated_path, allow_pickle=False) as archive:
        generated = archive["binary_terminal_matrices"]
        generated_indices = archive["selected_chain_indices"].astype(int)
    with np.load(
        cli.reference_archive.expanduser().resolve(), allow_pickle=False
    ) as archive:
        reference = archive["binary_terminal_matrices"]
        reference_indices = archive["selected_chain_indices"].astype(int)
    if not np.array_equal(generated_indices, reference_indices):
        raise ValueError("Generated/reference selected chain indices differ.")
    if generated.shape[1] != reference.shape[1]:
        raise ValueError("Generated/reference chain counts differ.")

    generated_signatures, prepared_generated = prepared_by_chain(generated)
    reference_signatures, prepared_reference = prepared_by_chain(reference)
    cache = compute_feature_cache(
        [*prepared_generated, *prepared_reference],
        workers=cli.workers,
    )
    chains = []
    for position, chain_index in enumerate(generated_indices.tolist()):
        generated_sigs = generated_signatures[position]
        reference_sigs = reference_signatures[position]
        source = metadata[chain_index]
        status = (
            "ok"
            if generated_sigs and reference_sigs
            else (
                "empty_generated_chain"
                if not generated_sigs
                else "empty_reference_chain"
            )
        )
        chain: Dict[str, Any] = {
            "archive_position": position,
            "chain_index": chain_index,
            "rule_index": source["rule_index"],
            "value_index": source["value_index"],
            "rule": source["rule"],
            "states": source["states"],
            "value": source["value"],
            "rule_source": source.get("rule_source"),
            "status": status,
            "generated_input_count": int(generated.shape[0]),
            "reference_input_count": int(reference.shape[0]),
            "generated_nonempty_count": len(generated_sigs),
            "reference_nonempty_count": len(reference_sigs),
            "combined_with_other_chains": False,
            "binary_before_metrics": True,
        }
        if status == "ok":
            context = reference_metric_context(reference_sigs, cache)
            context["edge_count"] = {
                "samples": [
                    float(cache[signature]["edge_count"])
                    for signature in reference_sigs
                ]
            }
            chain.update(
                evaluate_signature_chain(
                    generated_sigs,
                    reference_context=context,
                    cache=cache,
                )
            )
        else:
            chain.update(
                {
                    "metrics": {metric: None for metric in METRICS},
                    "metric_generated_sample_counts": {
                        metric: 0 for metric in METRICS
                    },
                    "generated_edge_count": None,
                    "reference_edge_count": None,
                }
            )
        chains.append(chain)
        print(
            f"[chain] {position + 1}/{len(generated_indices)} "
            f"index={chain_index} gen={len(generated_sigs)} "
            f"ref={len(reference_sigs)}"
        )

    payload = {
        "metric_family": "matched_independent_terminal_chain_structural_mmd",
        "run_dir": str(run_dir),
        "dataset": prior["dataset"],
        "setting": prior["setting"],
        "training_seed": prior["training_seed"],
        "generation_seed": prior["generation_seed"],
        "checkpoint": prior["checkpoint"],
        "reference_archive": str(
            cli.reference_archive.expanduser().resolve()
        ),
        "comparison": "generated chain j versus real test chain j",
        "combination_across_chains": "none",
        "binarization_threshold": 0.5,
        "selected_chain_count": len(chains),
        "chains": chains,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_path, payload)
    write_csv(output_dir / CSV_NAME, payload)
    print(f"[done] {run_dir}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
