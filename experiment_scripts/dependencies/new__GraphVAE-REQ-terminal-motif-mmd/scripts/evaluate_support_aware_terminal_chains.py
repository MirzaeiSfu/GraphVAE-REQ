#!/usr/bin/env python3
"""Count-aligned, support-aware evaluation of terminal motif-matrix chains.

This evaluator fixes the main limitations of the earlier representative-chain
table:

* chain identity is canonicalized from the rule/state/value payload;
* eligibility is defined from training support, not test performance;
* motif occurrence/count use every valid terminal-matrix cell, including the
  diagonal, matching count mode's final all-spatial-cell sum;
* graph-structural metrics use a separate diagonal-cleared projection;
* motif occurrence-rate error is measured on all graphs;
* disconnected nonempty motif graphs are retained for Random-GIN;
* edge/node/component distributions include empty graphs;
* Random-GIN is reported only when both sides have adequate sample support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from scipy.stats import wasserstein_distance


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stat_rnn import ggm_gnn_quality_stats
from terminal_motif_evaluation import write_json


OUTPUT_NAME = "count_aligned_per_chain.json"
CSV_NAME = "count_aligned_per_chain.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--training-support", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gin-runs", type=int, default=10)
    parser.add_argument("--min-training-support", type=int, default=20)
    parser.add_argument("--min-test-support", type=int, default=20)
    parser.add_argument("--min-generated-support", type=int, default=20)
    parser.add_argument("--skip-gin", action="store_true")
    parser.add_argument(
        "--reuse-compatible-gin",
        action="store_true",
        help=(
            "Reuse an existing support_aware_per_chain.json Random-GIN result "
            "when its canonical chain, seed, graph projection, and repetition "
            "count are identical."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_chain_payload(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rule": metadata["rule"],
        "states": metadata["states"],
        "value": metadata["value"],
        "rule_source": metadata.get("rule_source"),
    }


def canonical_chain_key(metadata: Dict[str, Any]) -> str:
    serialized = json.dumps(
        canonical_chain_payload(metadata),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def count_and_structural_graph_summary(
    adjacency: np.ndarray,
) -> Tuple[nx.Graph | None, Dict[str, float]]:
    array = (np.asarray(adjacency) >= 0.5).astype(np.uint8, copy=True)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected square adjacency, got {array.shape}.")
    # The motif counter's scalar count is the sum of every valid cell in this
    # terminal matrix.  Do not remove the diagonal for count/presence.
    motif_count = float(array.sum())
    motif_occurrence = float(motif_count > 0)

    # A graph adjacency has different semantics: diagonal cells are self-loops.
    # Construct a separate diagonal-cleared projection for structural metrics.
    structural_array = array.copy()
    np.fill_diagonal(structural_array, 0)
    graph = nx.from_numpy_array(structural_array, create_using=nx.Graph)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_edges() == 0:
        return None, {
            "motif_occurrence": motif_occurrence,
            "motif_count": motif_count,
            "active_nodes": 0.0,
            "edges": 0.0,
            "components": 0.0,
        }
    graph = nx.convert_node_labels_to_integers(
        graph, ordering="sorted", label_attribute=None
    )
    return graph, {
        "motif_occurrence": motif_occurrence,
        "motif_count": motif_count,
        "active_nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "components": float(nx.number_connected_components(graph)),
    }


def summarize_collection(
    matrices: np.ndarray,
) -> Tuple[List[nx.Graph], Dict[str, List[float]]]:
    graphs: List[nx.Graph] = []
    samples = {
        "motif_occurrence": [],
        "motif_count": [],
        "active_nodes": [],
        "edges": [],
        "components": [],
    }
    for adjacency in matrices:
        graph, summary = count_and_structural_graph_summary(adjacency)
        if graph is not None:
            graphs.append(graph)
        for key, value in summary.items():
            samples[key].append(float(value))
    return graphs, samples


def distribution_summary(
    generated: Dict[str, List[float]],
    reference: Dict[str, List[float]],
) -> Dict[str, Any]:
    generated_rate = float(np.mean(generated["motif_occurrence"]))
    reference_rate = float(np.mean(reference["motif_occurrence"]))
    result: Dict[str, Any] = {
        "generated_occurrence_rate": generated_rate,
        "reference_occurrence_rate": reference_rate,
        "occurrence_rate_absolute_error": abs(
            generated_rate - reference_rate
        ),
    }
    generated_counts = np.asarray(generated["motif_count"], dtype=float)
    reference_counts = np.asarray(reference["motif_count"], dtype=float)
    result.update(
        {
            "generated_binary_motif_count_mean_all_graphs": float(
                generated_counts.mean()
            ),
            "reference_binary_motif_count_mean_all_graphs": float(
                reference_counts.mean()
            ),
            "binary_motif_count_mean_absolute_error": abs(
                float(generated_counts.mean() - reference_counts.mean())
            ),
            "binary_motif_count_wasserstein": float(
                wasserstein_distance(reference_counts, generated_counts)
            ),
        }
    )
    for metric in ("active_nodes", "edges", "components"):
        generated_values = np.asarray(generated[metric], dtype=float)
        reference_values = np.asarray(reference[metric], dtype=float)
        result[f"generated_{metric}_mean_all_graphs"] = float(
            generated_values.mean()
        )
        result[f"reference_{metric}_mean_all_graphs"] = float(
            reference_values.mean()
        )
        result[f"{metric}_mean_absolute_error"] = abs(
            float(generated_values.mean() - reference_values.mean())
        )
        result[f"{metric}_wasserstein"] = float(
            wasserstein_distance(reference_values, generated_values)
        )
    return result


def deterministic_eval_seed(training_seed: int, canonical_key: str) -> int:
    chain_bits = int(canonical_key[:8], 16)
    return int((training_seed * 1_000_003 + chain_bits) % (2**31 - 1))


def run_gin(
    reference_graphs: List[nx.Graph],
    generated_graphs: List[nx.Graph],
    *,
    training_seed: int,
    canonical_key: str,
    device: str,
    repetitions: int,
) -> Dict[str, Any]:
    seed = deterministic_eval_seed(training_seed, canonical_key)
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
    return {
        "evaluation_seed": seed,
        "repetitions": repetitions,
        "uses_full_disconnected_nonempty_graph": True,
        "metrics": metrics,
    }


def write_csv(path: Path, payload: Dict[str, Any]) -> None:
    fields = [
        "dataset",
        "setting",
        "training_seed",
        "chain_index",
        "canonical_chain_key",
        "training_nonempty_count",
        "reference_nonempty_count",
        "generated_nonempty_count",
        "training_eligible",
        "reference_eligible",
        "gin_eligible",
        "generated_occurrence_rate",
        "reference_occurrence_rate",
        "occurrence_rate_absolute_error",
        "generated_binary_motif_count_mean_all_graphs",
        "reference_binary_motif_count_mean_all_graphs",
        "binary_motif_count_mean_absolute_error",
        "binary_motif_count_wasserstein",
        "active_nodes_wasserstein",
        "edges_wasserstein",
        "components_wasserstein",
        "degree",
        "clustering",
        "sparsity",
        "orbit",
        "spectral",
        "triangle",
        "diameter",
        "mmd_rbf",
        "precision",
        "recall",
        "f1_pr",
        "gin_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for chain in payload["chains"]:
            gin_metrics = (
                chain["random_gin"]["metrics"]
                if chain.get("random_gin")
                else {}
            )
            row = {
                "dataset": payload["dataset"],
                "setting": payload["setting"],
                "training_seed": payload["training_seed"],
                "chain_index": chain["chain_index"],
                "canonical_chain_key": chain["canonical_chain_key"],
                "training_nonempty_count": chain["training_nonempty_count"],
                "reference_nonempty_count": chain["reference_nonempty_count"],
                "generated_nonempty_count": chain["generated_nonempty_count"],
                "training_eligible": chain["training_eligible"],
                "reference_eligible": chain["reference_eligible"],
                "gin_eligible": chain["gin_eligible"],
                "gin_error": gin_metrics.get("error"),
            }
            row.update(chain["distribution_metrics"])
            row.update(chain["conditional_largest_component_structural_mmd"])
            for metric in ("mmd_rbf", "precision", "recall", "f1_pr"):
                row[metric] = gin_metrics.get(metric)
            writer.writerow({field: row.get(field) for field in fields})


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

    matched = load_json(
        output_dir / "matched_per_chain_structural_mmd.json"
    )
    reusable_gin_by_key: Dict[str, Dict[str, Any]] = {}
    reusable_gin_path = output_dir / "support_aware_per_chain.json"
    if cli.reuse_compatible_gin and reusable_gin_path.exists():
        reusable_payload = load_json(reusable_gin_path)
        if (
            int(reusable_payload["training_seed"])
            == int(matched["training_seed"])
            and int(reusable_payload["generation_seed"])
            == int(matched["generation_seed"])
        ):
            reusable_gin_by_key = {
                chain["canonical_chain_key"]: chain["random_gin"]
                for chain in reusable_payload["chains"]
                if chain.get("random_gin")
            }
    training_support_path = cli.training_support.expanduser().resolve()
    training_support = load_json(training_support_path)
    training_by_key = {
        chain["canonical_chain_key"]: chain
        for chain in training_support["chains"]
    }
    metadata_by_index = {
        int(chain["chain_index"]): chain
        for chain in load_json(output_dir / "chain_metadata.json")
    }

    generated_path = output_dir / "terminal_chain_binary_matrices.npz"
    with np.load(generated_path, allow_pickle=False) as archive:
        generated = archive["binary_terminal_matrices"]
        generated_indices = archive["selected_chain_indices"].astype(int)
    reference_path = Path(matched["reference_archive"])
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = archive["binary_terminal_matrices"]
        reference_indices = archive["selected_chain_indices"].astype(int)
    if not np.array_equal(generated_indices, reference_indices):
        raise ValueError("Generated/reference selected chain indices differ.")

    prior_by_index = {
        int(chain["chain_index"]): chain for chain in matched["chains"]
    }
    chains = []
    for position, chain_index in enumerate(generated_indices.tolist()):
        metadata = metadata_by_index[chain_index]
        canonical_key = canonical_chain_key(metadata)
        if canonical_key not in training_by_key:
            raise KeyError(
                f"Chain {chain_index} missing from training support map."
            )
        support = training_by_key[canonical_key]
        generated_graphs, generated_samples = summarize_collection(
            generated[:, position]
        )
        reference_graphs, reference_samples = summarize_collection(
            reference[:, position]
        )
        generated_count = len(generated_graphs)
        reference_count = len(reference_graphs)
        training_count = int(support["training_nonempty_count"])
        training_eligible = training_count >= int(cli.min_training_support)
        reference_eligible = reference_count >= int(cli.min_test_support)
        gin_eligible = (
            training_eligible
            and reference_eligible
            and generated_count >= int(cli.min_generated_support)
        )
        prior = prior_by_index[chain_index]
        chain: Dict[str, Any] = {
            "archive_position": position,
            "chain_index": chain_index,
            "canonical_chain_key": canonical_key,
            "canonical_chain": canonical_chain_payload(metadata),
            "rule_index": metadata["rule_index"],
            "value_index": metadata["value_index"],
            "training_graph_count": int(
                support["training_graph_count"]
            ),
            "training_nonempty_count": training_count,
            "training_nonempty_rate": float(
                support["training_nonempty_rate"]
            ),
            "generated_input_count": int(generated.shape[0]),
            "reference_input_count": int(reference.shape[0]),
            "generated_nonempty_count": generated_count,
            "reference_nonempty_count": reference_count,
            "generated_structural_nonempty_count": generated_count,
            "reference_structural_nonempty_count": reference_count,
            "training_eligible": training_eligible,
            "reference_eligible": reference_eligible,
            "gin_eligible": gin_eligible,
            "distribution_metrics": distribution_summary(
                generated_samples, reference_samples
            ),
            "conditional_largest_component_structural_mmd": prior[
                "metrics"
            ],
            "conditional_metric_status": prior["status"],
            "random_gin": None,
        }
        if gin_eligible and not cli.skip_gin:
            reusable = reusable_gin_by_key.get(canonical_key)
            if (
                reusable
                and int(reusable.get("repetitions", -1)) == int(cli.gin_runs)
                and reusable.get("uses_full_disconnected_nonempty_graph")
                and reusable.get("metrics", {}).get("error") is None
            ):
                chain["random_gin"] = dict(reusable)
                chain["random_gin"]["reused_from"] = str(reusable_gin_path)
            else:
                try:
                    chain["random_gin"] = run_gin(
                        reference_graphs,
                        generated_graphs,
                        training_seed=int(matched["training_seed"]),
                        canonical_key=canonical_key,
                        device=str(cli.device),
                        repetitions=int(cli.gin_runs),
                    )
                except Exception as exc:
                    chain["random_gin"] = {
                        "evaluation_seed": deterministic_eval_seed(
                            int(matched["training_seed"]), canonical_key
                        ),
                        "repetitions": int(cli.gin_runs),
                        "uses_full_disconnected_nonempty_graph": True,
                        "metrics": {"error": repr(exc)},
                    }
        chains.append(chain)
        gin_status = (
            "done"
            if chain["random_gin"]
            and chain["random_gin"]["metrics"].get("error") is None
            else ("error" if chain["random_gin"] else "not-eligible")
        )
        print(
            f"[chain] {position + 1}/{len(generated_indices)} "
            f"index={chain_index} train={training_count} "
            f"gen={generated_count} ref={reference_count} gin={gin_status}"
        )

    payload = {
        "metric_family": (
            "count_aligned_support_aware_independent_terminal_chain"
        ),
        "protocol_version": "count_aligned_v2",
        "run_dir": str(run_dir),
        "dataset": matched["dataset"],
        "setting": matched["setting"],
        "training_seed": matched["training_seed"],
        "generation_seed": matched["generation_seed"],
        "checkpoint": matched["checkpoint"],
        "training_support": str(training_support_path),
        "reference_archive": str(reference_path),
        "thresholds": {
            "minimum_training_nonempty_graphs": int(
                cli.min_training_support
            ),
            "minimum_test_reference_nonempty_graphs": int(
                cli.min_test_support
            ),
            "minimum_generated_nonempty_graphs_for_gin": int(
                cli.min_generated_support
            ),
        },
        "chain_identity": "sha256(canonical rule/states/value/source JSON)",
        "chain_selection_uses_test_metric_values": False,
        "motif_count_and_occurrence": {
            "binarization_threshold": 0.5,
            "valid_terminal_matrix_cells": "all cells",
            "diagonal_included": True,
            "definition": (
                "sum of all valid binarized terminal-matrix cells; occurrence "
                "is one iff that sum is positive"
            ),
            "alignment": (
                "matches count mode's all-spatial-cell sum, after the "
                "requested evaluation-time binarization"
            ),
        },
        "empty_graph_handling": {
            "occurrence_and_distribution_metrics": (
                "all graphs included; motif count/occurrence use every valid "
                "cell including diagonal; structural summaries use the "
                "diagonal-cleared projection and contribute zeros when empty"
            ),
            "conditional_structural_mmd": (
                "nonempty largest components, retained for compatibility"
            ),
            "random_gin": (
                "nonempty full graphs with every connected component retained"
            ),
        },
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
