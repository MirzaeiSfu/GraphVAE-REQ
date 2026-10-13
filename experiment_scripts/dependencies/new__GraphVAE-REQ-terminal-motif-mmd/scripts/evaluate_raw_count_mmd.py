#!/usr/bin/env python3
"""Compare raw training/generated motif-count distributions per rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from scipy.stats import wasserstein_distance


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terminal_motif_evaluation import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-count-archive", required=True, type=Path)
    parser.add_argument("--generated-count-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_chain_key(metadata: Dict[str, Any]) -> str:
    payload = {
        "rule": metadata["rule"],
        "states": metadata["states"],
        "value": metadata["value"],
        "rule_source": metadata.get("rule_source"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def reference_bandwidth(values: np.ndarray, max_samples: int = 512) -> float:
    """Reference-only median positive pairwise distance, with a safe fallback."""

    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if len(ordered) > max_samples:
        indices = np.linspace(0, len(ordered) - 1, max_samples).round().astype(int)
        ordered = ordered[indices]
    distances = np.abs(ordered[:, None] - ordered[None, :])
    positive = distances[np.triu_indices(len(ordered), k=1)]
    positive = positive[positive > 0]
    if positive.size:
        return float(np.median(positive))
    standard_deviation = float(np.std(values))
    return max(standard_deviation, 1.0)


def compressed(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(
        np.asarray(values, dtype=np.float64).reshape(-1), return_counts=True
    )
    return unique, counts.astype(np.float64) / counts.sum()


def rbf_mean(
    first_values: np.ndarray,
    first_weights: np.ndarray,
    second_values: np.ndarray,
    second_weights: np.ndarray,
    sigma: float,
) -> float:
    distance = first_values[:, None] - second_values[None, :]
    kernel = np.exp(-(distance**2) / (2.0 * sigma**2))
    return float(first_weights @ kernel @ second_weights)


def biased_rbf_mmd(
    reference: np.ndarray,
    generated: np.ndarray,
    sigma: float,
) -> float:
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Invalid RBF bandwidth: {sigma}.")
    ref_values, ref_weights = compressed(reference)
    gen_values, gen_weights = compressed(generated)
    value = (
        rbf_mean(ref_values, ref_weights, ref_values, ref_weights, sigma)
        + rbf_mean(gen_values, gen_weights, gen_values, gen_weights, sigma)
        - 2.0
        * rbf_mean(ref_values, ref_weights, gen_values, gen_weights, sigma)
    )
    return float(max(value, 0.0))


def linear_mmd(reference: np.ndarray, generated: np.ndarray) -> float:
    """Squared MMD for the scalar linear kernel k(x, y) = x * y."""

    mean_difference = float(np.mean(reference) - np.mean(generated))
    return mean_difference**2


def mean_metric(rules: Iterable[Dict[str, Any]], key: str) -> float:
    values = [float(rule[key]) for rule in rules]
    return float(np.mean(values))


def write_per_rule_csv(path: Path, payload: Dict[str, Any]) -> None:
    fields = [
        "dataset",
        "setting",
        "training_seed",
        "generation_seed",
        "checkpoint",
        "training_graph_count",
        "generated_graph_count",
        "rule_position",
        "chain_index",
        "canonical_chain_key",
        "rule_index",
        "value_index",
        "train_count_mean",
        "train_count_std",
        "train_count_min",
        "train_count_max",
        "generated_count_mean",
        "generated_count_std",
        "generated_count_min",
        "generated_count_max",
        "mean_count_absolute_error",
        "count_wasserstein",
        "reference_rbf_sigma",
        "raw_count_mmd_rbf",
        "raw_count_mmd_rbf_sigma1",
        "raw_count_mmd_linear",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rule in payload["rules"]:
            row = {
                "dataset": payload["dataset"],
                "setting": payload["setting"],
                "training_seed": payload["training_seed"],
                "generation_seed": payload["generation_seed"],
                "checkpoint": payload["checkpoint"],
                "training_graph_count": payload["training_graph_count"],
                "generated_graph_count": payload["generated_graph_count"],
                **rule,
            }
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    cli = parse_args()
    training_path = cli.training_count_archive.expanduser().resolve()
    generated_path = cli.generated_count_archive.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    output_path = output_dir / "raw_count_mmd.json"
    csv_path = output_dir / "raw_count_mmd_per_rule.csv"
    if output_path.exists() and csv_path.exists() and not cli.overwrite:
        print(f"[skip] {output_path}")
        return 0

    with training_path.with_suffix(".json").open(
        "r", encoding="utf-8"
    ) as handle:
        training_metadata = json.load(handle)
    with generated_path.with_suffix(".json").open(
        "r", encoding="utf-8"
    ) as handle:
        generated_metadata = json.load(handle)
    with np.load(training_path, allow_pickle=False) as archive:
        training_counts = archive["raw_motif_counts"].astype(np.float64)
        training_indices = archive["selected_chain_indices"].astype(np.int64)
        training_nodes = archive["node_counts"].astype(np.int32)
    with np.load(generated_path, allow_pickle=False) as archive:
        generated_counts = archive["raw_motif_counts"].astype(np.float64)
        generated_indices = archive["selected_chain_indices"].astype(np.int64)
        generated_nodes = archive["node_counts"].astype(np.int32)

    if training_counts.shape != generated_counts.shape:
        raise ValueError(
            f"Training/generated count shapes differ: {training_counts.shape} "
            f"vs {generated_counts.shape}."
        )
    if not np.array_equal(training_indices, generated_indices):
        raise ValueError("Training/generated selected chain indices differ.")
    if not np.array_equal(training_nodes, generated_nodes):
        raise ValueError("Training/generated node-count schedules differ.")
    if not np.isfinite(training_counts).all() or not np.isfinite(
        generated_counts
    ).all():
        raise ValueError("Non-finite raw motif count detected.")

    selected_metadata = training_metadata["selected_chains"]
    if len(selected_metadata) != training_counts.shape[1]:
        raise ValueError("Selected chain metadata/count width differs.")
    rules = []
    for position, (chain_index, metadata) in enumerate(
        zip(training_indices.tolist(), selected_metadata)
    ):
        reference = training_counts[:, position]
        generated = generated_counts[:, position]
        sigma = reference_bandwidth(reference)
        rules.append(
            {
                "rule_position": int(position),
                "chain_index": int(chain_index),
                "canonical_chain_key": canonical_chain_key(metadata),
                "rule_index": metadata["rule_index"],
                "value_index": metadata["value_index"],
                "train_count_mean": float(np.mean(reference)),
                "train_count_std": float(np.std(reference)),
                "train_count_min": float(np.min(reference)),
                "train_count_max": float(np.max(reference)),
                "generated_count_mean": float(np.mean(generated)),
                "generated_count_std": float(np.std(generated)),
                "generated_count_min": float(np.min(generated)),
                "generated_count_max": float(np.max(generated)),
                "mean_count_absolute_error": float(
                    abs(np.mean(reference) - np.mean(generated))
                ),
                "count_wasserstein": float(
                    wasserstein_distance(reference, generated)
                ),
                "reference_rbf_sigma": sigma,
                "raw_count_mmd_rbf": biased_rbf_mmd(
                    reference, generated, sigma
                ),
                "raw_count_mmd_rbf_sigma1": biased_rbf_mmd(
                    reference, generated, 1.0
                ),
                "raw_count_mmd_linear": linear_mmd(reference, generated),
            }
        )

    macro_keys = (
        "raw_count_mmd_rbf",
        "raw_count_mmd_rbf_sigma1",
        "raw_count_mmd_linear",
        "count_wasserstein",
        "mean_count_absolute_error",
        "train_count_mean",
        "generated_count_mean",
    )
    payload = {
        "protocol": "raw_training_count_mmd_v1",
        "dataset": generated_metadata["dataset"],
        "setting": generated_metadata["setting"],
        "training_seed": generated_metadata["training_seed"],
        "generation_seed": generated_metadata["generation_seed"],
        "checkpoint": generated_metadata["checkpoint"],
        "training_count_archive": str(training_path),
        "generated_count_archive": str(generated_path),
        "training_graph_count": int(training_counts.shape[0]),
        "generated_graph_count": int(generated_counts.shape[0]),
        "selected_rule_count": int(training_counts.shape[1]),
        "count_definition": (
            "sum of all valid unbinarized terminal-matrix cells, including "
            "the diagonal; padded nodes excluded before motif counting"
        ),
        "mmd_definition": (
            "biased squared MMD computed independently per rule using a "
            "scalar RBF kernel; primary bandwidth is the median positive "
            "pairwise distance from training counts only, with fallback 1.0; "
            "linear MMD uses k(x,y)=x*y and equals the squared difference "
            "between the training and generated mean counts"
        ),
        "rule_aggregation": "arithmetic mean of independently computed rule metrics",
        "occurrence_metric_computed": False,
        "padded_nodes_enter_motif_algorithm": False,
        "macro_average_over_rules": {
            key: mean_metric(rules, key) for key in macro_keys
        },
        "rules": rules,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)
    write_per_rule_csv(csv_path, payload)
    print(
        f"[done] {payload['dataset']} {payload['setting']} "
        f"seed={payload['training_seed']} rules={len(rules)} "
        f"mmd={payload['macro_average_over_rules']['raw_count_mmd_rbf']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
