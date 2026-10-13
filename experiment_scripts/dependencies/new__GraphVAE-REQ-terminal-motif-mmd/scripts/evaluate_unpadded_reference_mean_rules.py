#!/usr/bin/env python3
"""Evaluate every thresholded terminal rule against its matching test rule.

No occurrence statistic is computed. Padded cells are cropped using the stored
true node count before graph construction and before every downstream metric.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import pickle
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from per_chain_terminal_motif_evaluation import (
    METRIC_SPECS,
    PreparedGraph,
    compute_feature_cache,
    evaluate_signature_chain,
    reference_metric_context,
)
from stat_rnn import ggm_gnn_quality_stats
from terminal_motif_evaluation import write_json


STRUCTURAL_METRICS = tuple(METRIC_SPECS)
GIN_METRICS = ("mmd_rbf", "precision", "recall", "f1_pr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-archive", required=True, type=Path)
    parser.add_argument("--reference-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gin-runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-gin", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_chain_key(metadata: Dict[str, Any]) -> str:
    payload = {
        "rule": metadata["rule"],
        "states": metadata["states"],
        "value": metadata["value"],
        "rule_source": metadata.get("rule_source"),
    }
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def deterministic_eval_seed(training_seed: int, canonical_key: str) -> int:
    chain_bits = int(canonical_key[:8], 16)
    return int((training_seed * 1_000_003 + chain_bits) % (2**31 - 1))


def cropped_matrices(
    matrices: np.ndarray,
    node_counts: np.ndarray,
    position: int,
) -> List[np.ndarray]:
    return [
        np.asarray(matrices[index, position, :node_count, :node_count])
        for index, node_count in enumerate(node_counts.astype(int).tolist())
    ]


def prepare_all_real_nodes(matrix: np.ndarray) -> PreparedGraph:
    """Create a structural graph while retaining every non-padded real node."""

    array = np.asarray(matrix, dtype=np.uint8).copy()
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected square matrix, got {array.shape}.")
    np.fill_diagonal(array, 0)
    graph = nx.from_numpy_array(array, create_using=nx.Graph)
    graph = nx.convert_node_labels_to_integers(
        graph, ordering="sorted", label_attribute=None
    )
    packed = np.packbits(array.reshape(-1), bitorder="little").tobytes()
    signature = int(graph.number_of_nodes()).to_bytes(4, "little") + packed
    return PreparedGraph(signature, graph)


def structural_prepared(
    matrices: Sequence[np.ndarray],
) -> Tuple[List[bytes], List[Any]]:
    signatures: List[bytes] = []
    prepared_all: List[Any] = []
    for matrix in matrices:
        prepared = prepare_all_real_nodes(matrix)
        signatures.append(prepared.signature)
        prepared_all.append(prepared)
    return signatures, prepared_all


def full_real_node_graphs(
    matrices: Sequence[np.ndarray],
) -> List[nx.Graph]:
    graphs: List[nx.Graph] = []
    for matrix in matrices:
        array = np.asarray(matrix, dtype=np.uint8).copy()
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"Expected square matrix, got {array.shape}.")
        np.fill_diagonal(array, 0)
        graph = nx.from_numpy_array(array, create_using=nx.Graph)
        graphs.append(
            nx.convert_node_labels_to_integers(
                graph, ordering="sorted", label_attribute=None
            )
        )
    return graphs


def cached_reference_features(
    reference_path: Path,
    prepared: Sequence[PreparedGraph],
    *,
    workers: int,
) -> Dict[bytes, Dict[str, Any]]:
    """Reuse exact reference features across settings with the same test archive."""

    cache_path = reference_path.with_suffix(".structural_features.pkl")
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    expected_signatures = {graph.signature for graph in prepared}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                cache = pickle.load(handle)
            if expected_signatures.issubset(cache):
                print(
                    f"[reference-cache] loaded {len(cache)} unique graphs "
                    f"from {cache_path}"
                )
                return cache
        cache = compute_feature_cache(prepared, workers=workers)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(cache_path)
        print(
            f"[reference-cache] saved {len(cache)} unique graphs "
            f"to {cache_path}"
        )
        return cache


def mean_available(chains: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values: List[float] = []
    for chain in chains:
        if key in STRUCTURAL_METRICS:
            value = chain["structural_mmd"].get(key)
        else:
            value = chain["random_gin"]["metrics"].get(key)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return {
        "mean": float(np.mean(values)) if values else None,
        "rule_count": len(values),
        "total_rule_count": len(chains),
    }


def write_csv(path: Path, payload: Dict[str, Any]) -> None:
    fields = [
        "dataset",
        "setting",
        "training_seed",
        "archive_position",
        "chain_index",
        "rule_index",
        "value_index",
        "reference_mean_threshold",
        "reference_graphs_evaluated",
        "generated_graphs_evaluated",
        *STRUCTURAL_METRICS,
        *GIN_METRICS,
        "gin_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for chain in payload["rules"]:
            row = {
                "dataset": payload["dataset"],
                "setting": payload["setting"],
                "training_seed": payload["training_seed"],
                "archive_position": chain["archive_position"],
                "chain_index": chain["chain_index"],
                "rule_index": chain["rule_index"],
                "value_index": chain["value_index"],
                "reference_mean_threshold": chain[
                    "reference_mean_threshold"
                ],
                "reference_graphs_evaluated": chain[
                    "reference_graphs_evaluated"
                ],
                "generated_graphs_evaluated": chain[
                    "generated_graphs_evaluated"
                ],
                "gin_error": chain["random_gin"]["metrics"].get("error"),
            }
            row.update(chain["structural_mmd"])
            for metric in GIN_METRICS:
                row[metric] = chain["random_gin"]["metrics"].get(metric)
            writer.writerow(row)


def main() -> int:
    cli = parse_args()
    os.chdir(REPO_ROOT)
    generated_path = cli.generated_archive.expanduser().resolve()
    reference_path = cli.reference_archive.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    output_path = output_dir / "per_rule_metrics.json"
    partial_path = output_dir / "per_rule_metrics.partial.json"
    csv_path = output_dir / "per_rule_metrics.csv"
    if output_path.exists() and not cli.overwrite:
        print(f"[skip] {output_path}")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    generated_metadata = load_json(generated_path.with_suffix(".json"))
    reference_metadata = load_json(reference_path.with_suffix(".json"))
    with np.load(generated_path, allow_pickle=False) as archive:
        generated = archive["binary_terminal_matrices"]
        generated_indices = archive["selected_chain_indices"].astype(np.int64)
        generated_nodes = archive["node_counts"].astype(np.int32)
        generated_thresholds = archive["rule_thresholds"].astype(np.float64)
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = archive["binary_terminal_matrices"]
        reference_indices = archive["selected_chain_indices"].astype(np.int64)
        reference_nodes = archive["node_counts"].astype(np.int32)
        reference_thresholds = archive["rule_thresholds"].astype(np.float64)
    if not np.array_equal(generated_indices, reference_indices):
        raise ValueError("Generated/reference selected chain indices differ.")
    if not np.array_equal(generated_nodes, reference_nodes):
        raise ValueError("Generated/reference true node counts differ.")
    if not np.array_equal(generated_thresholds, reference_thresholds):
        raise ValueError("Generated/reference rule thresholds differ.")
    if generated.shape[:2] != reference.shape[:2]:
        raise ValueError(
            f"Generated/reference graph-rule shapes differ: "
            f"{generated.shape[:2]} vs {reference.shape[:2]}."
        )

    selected_metadata = reference_metadata["selected_chains"]
    if len(selected_metadata) != len(generated_indices):
        raise ValueError("Selected rule metadata length differs from archive.")

    # Prepare and cache structural statistics once for all rules.
    generated_structural_signatures: List[List[bytes]] = []
    reference_structural_signatures: List[List[bytes]] = []
    generated_full_graphs: List[List[nx.Graph]] = []
    reference_full_graphs: List[List[nx.Graph]] = []
    prepared_generated: List[PreparedGraph] = []
    prepared_reference: List[PreparedGraph] = []
    for position in range(len(generated_indices)):
        generated_current = cropped_matrices(
            generated, generated_nodes, position
        )
        reference_current = cropped_matrices(
            reference, reference_nodes, position
        )
        gen_sigs, gen_prepared = structural_prepared(generated_current)
        ref_sigs, ref_prepared = structural_prepared(reference_current)
        generated_structural_signatures.append(gen_sigs)
        reference_structural_signatures.append(ref_sigs)
        prepared_generated.extend(gen_prepared)
        prepared_reference.extend(ref_prepared)
        generated_full_graphs.append(full_real_node_graphs(generated_current))
        reference_full_graphs.append(full_real_node_graphs(reference_current))
        print(
            f"[prepare] rule={position + 1}/{len(generated_indices)} "
            f"gen={len(gen_sigs)} ref={len(ref_sigs)}"
        )
    workers = max(1, int(cli.workers))
    reference_feature_cache = cached_reference_features(
        reference_path,
        prepared_reference,
        workers=workers,
    )
    generated_feature_cache = compute_feature_cache(
        prepared_generated, workers=workers
    )
    feature_cache = dict(reference_feature_cache)
    feature_cache.update(generated_feature_cache)

    completed_by_position: Dict[int, Dict[str, Any]] = {}
    if partial_path.exists() and not cli.overwrite:
        partial = load_json(partial_path)
        if (
            partial.get("generated_archive") == str(generated_path)
            and partial.get("reference_archive") == str(reference_path)
            and int(partial.get("gin_runs", -1)) == int(cli.gin_runs)
        ):
            completed_by_position = {
                int(rule["archive_position"]): rule
                for rule in partial.get("rules", [])
            }
            print(
                f"[resume] loaded {len(completed_by_position)} completed rules"
            )

    rules: List[Dict[str, Any]] = []
    training_seed = int(generated_metadata["training_seed"])
    for position, chain_index in enumerate(generated_indices.tolist()):
        if position in completed_by_position:
            rules.append(completed_by_position[position])
            continue
        metadata = selected_metadata[position]
        key = canonical_chain_key(metadata)
        gen_sigs = generated_structural_signatures[position]
        ref_sigs = reference_structural_signatures[position]
        structural = {metric: None for metric in STRUCTURAL_METRICS}
        structural_error = None
        if gen_sigs and ref_sigs:
            try:
                context = reference_metric_context(ref_sigs, feature_cache)
                evaluated = evaluate_signature_chain(
                    gen_sigs,
                    reference_context=context,
                    cache=feature_cache,
                )
                structural = evaluated["metrics"]
            except Exception as exc:
                structural_error = repr(exc)

        gin_seed = deterministic_eval_seed(training_seed, key)
        gin_graphs_generated = generated_full_graphs[position]
        gin_graphs_reference = reference_full_graphs[position]
        if cli.skip_gin:
            gin_metrics = {
                **{metric: None for metric in GIN_METRICS},
                "error": "skipped",
            }
        else:
            random.seed(gin_seed)
            np.random.seed(gin_seed)
            torch.manual_seed(gin_seed)
            if str(cli.device).startswith("cuda"):
                torch.cuda.manual_seed_all(gin_seed)
            os.environ["GRAPHVAE_GGM_GNN_DEVICE"] = str(cli.device)
            gin_metrics = ggm_gnn_quality_stats(
                gin_graphs_reference,
                gin_graphs_generated,
                runs=int(cli.gin_runs),
            )

        rule = {
            "archive_position": position,
            "chain_index": int(chain_index),
            "canonical_chain_key": key,
            "rule_index": metadata["rule_index"],
            "value_index": metadata["value_index"],
            "rule": metadata["rule"],
            "states": metadata["states"],
            "value": metadata["value"],
            "rule_source": metadata.get("rule_source"),
            "reference_mean_threshold": float(reference_thresholds[position]),
            "reference_input_graphs": int(reference.shape[0]),
            "generated_input_graphs": int(generated.shape[0]),
            "reference_graphs_evaluated": len(gin_graphs_reference),
            "generated_graphs_evaluated": len(gin_graphs_generated),
            "structural_projection": (
                "crop to true nodes and clear diagonal; retain every real node, "
                "including legitimate isolates and edgeless graphs"
            ),
            "structural_mmd": structural,
            "structural_error": structural_error,
            "random_gin": {
                "evaluation_seed": gin_seed,
                "repetitions": int(cli.gin_runs),
                "graph_projection": (
                    "crop to true nodes and clear diagonal; retain every real "
                    "node and every connected component"
                ),
                "metrics": gin_metrics,
            },
        }
        rules.append(rule)
        partial_payload = {
            "generated_archive": str(generated_path),
            "reference_archive": str(reference_path),
            "gin_runs": int(cli.gin_runs),
            "rules": sorted(
                rules, key=lambda item: int(item["archive_position"])
            ),
        }
        write_json(partial_path, partial_payload)
        print(
            f"[metric] rule={position + 1}/{len(generated_indices)} "
            f"index={chain_index} mmd_rbf={gin_metrics.get('mmd_rbf')} "
            f"f1_pr={gin_metrics.get('f1_pr')}"
        )

    rules.sort(key=lambda item: int(item["archive_position"]))
    macro = {
        metric: mean_available(rules, metric)
        for metric in (*STRUCTURAL_METRICS, *GIN_METRICS)
    }
    payload = {
        "metric_family": "reference_mean_unpadded_per_rule_mmd_and_gin",
        "protocol_version": "reference_mean_per_rule_unpadded_v1",
        "dataset": generated_metadata["dataset"],
        "setting": generated_metadata["setting"],
        "training_seed": training_seed,
        "generation_seed": generated_metadata["generation_seed"],
        "checkpoint": generated_metadata["checkpoint"],
        "generated_archive": str(generated_path),
        "reference_archive": str(reference_path),
        "sample_count": int(generated.shape[0]),
        "selected_rule_count": int(len(generated_indices)),
        "padding_handling": (
            "Motif counting and all metrics use only [:true_n, :true_n]. "
            "Padded nodes never enter the motif algorithm or any metric."
        ),
        "threshold_definition": (
            "One threshold per rule: mean of all valid entries across all "
            "reference test terminal matrices; terminal_value > threshold."
        ),
        "threshold_applied_identically_to_reference_and_generated": True,
        "diagonal_included_in_threshold": True,
        "diagonal_removed_for_graph_metrics": True,
        "occurrence_metric_computed": False,
        "rule_aggregation": (
            "Arithmetic mean of each metric over every rule for which that "
            "metric is defined; coverage is saved as rule_count/total_rule_count."
        ),
        "macro_average_over_rules": macro,
        "rules": rules,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_path, payload)
    write_csv(csv_path, payload)
    partial_path.unlink(missing_ok=True)
    print(f"[done] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
