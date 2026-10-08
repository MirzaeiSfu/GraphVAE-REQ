#!/usr/bin/env python3
"""Evaluate hard DeFoG PTC samples on GraphVAE's exact FactorBase rule rows.

The implementation is specialized to the five PTC rules.  It evaluates the
182 combinations retained by the saved CP-smoothed training selection and the
complete 6,899-row universe used for normalized correlation.  Graphs are
paired by sorted node count only for the Gaussian-aligned diagnostic; all
aggregate and distributional metrics are permutation invariant.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pruned-template", type=Path, required=True)
    parser.add_argument("--full-template", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_collection(path: Path):
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "ggm-eval-pyg-tensors":
        raise ValueError(f"Unexpected safe graph format: {path}")
    return payload["graphs"], payload.get("metadata", {}), payload.get("collection_sha256")


def graph_statistics(item):
    n = int(item["num_nodes"])
    x = torch.as_tensor(item["x"], dtype=torch.float64)
    if tuple(x.shape) != (n, 19):
        raise ValueError(f"Expected PTC node one-hot shape ({n}, 19), got {tuple(x.shape)}")
    labels = x.argmax(dim=1).numpy().astype(np.int64) + 1
    edge_index = torch.as_tensor(item["edge_index"], dtype=torch.long)
    adjacency = np.zeros((n, n), dtype=np.float64)
    adjacency[edge_index[0].numpy(), edge_index[1].numpy()] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    onehot = np.eye(19, dtype=np.float64)[labels - 1]
    neighbour_labels = adjacency @ onehot
    triples = np.zeros((19, 19, 19), dtype=np.float64)
    for middle_label in range(1, 20):
        selected = neighbour_labels[labels == middle_label]
        if selected.size:
            triples[:, middle_label - 1, :] = np.einsum("na,nc->ac", selected, selected)
    return {
        "node_count": n,
        "directed_edge_count": float(adjacency.sum()),
        "node_label_counts": np.bincount(labels, minlength=20)[1:].astype(np.float64),
        "path_label_triples": triples,
    }


def entry_count(stat, entry):
    rule_index = int(entry["rule_index"])
    row = entry["value_row"]
    if rule_index in (0, 1):
        if row[0] != "T":
            raise ValueError(f"Unexpected PTC unit-edge state: {row[0]}")
        return stat["directed_edge_count"]
    if rule_index == 2:
        label0, edge01, edge12, label1, label2 = row[1:6]
        if edge01 != "T" or edge12 != "T":
            raise ValueError(f"Unexpected PTC path-edge states: {edge01}, {edge12}")
        return stat["path_label_triples"][int(label0) - 1, int(label1) - 1, int(label2) - 1]
    if rule_index in (3, 4):
        return stat["node_label_counts"][int(row[0]) - 1]
    raise ValueError(f"Unexpected PTC rule index: {rule_index}")


def count_entries(graphs, entries):
    stats = [graph_statistics(item) for item in graphs]
    counts = np.asarray(
        [[entry_count(stat, entry) for entry in entries] for stat in stats],
        dtype=np.float64,
    )
    node_counts = np.asarray([stat["node_count"] for stat in stats], dtype=np.int64)
    return counts, node_counts


def metric_summary(observed_np, generated_np, min_log_sigma=-6.0, eps=1e-12):
    if observed_np.shape != generated_np.shape:
        raise ValueError(f"Count shape mismatch: {observed_np.shape} versus {generated_np.shape}")
    observed = torch.as_tensor(observed_np, dtype=torch.float64)
    generated = torch.as_tensor(generated_np, dtype=torch.float64)
    observed_total = observed.sum(dim=0)
    generated_total = generated.sum(dim=0)
    difference = generated_total - observed_total
    observed_mean = observed.mean(dim=0)
    generated_mean = generated.mean(dim=0)
    mean_difference = generated_mean - observed_mean
    aggregate_rmse = torch.sqrt(torch.mean(difference.square()))
    mean_vector_rmse = torch.sqrt(torch.mean(mean_difference.square()))
    observed_mean_rms = torch.sqrt(torch.mean(observed_mean.square()))
    relative_rmse = mean_vector_rmse / observed_mean_rms

    per_motif_mse = torch.mean((generated - observed).square(), dim=0)
    per_motif_rmse = per_motif_mse.clamp_min(float(eps) ** 2).sqrt()
    per_motif_log_sigma = float(min_log_sigma) + F.softplus(
        torch.log(per_motif_rmse) - float(min_log_sigma)
    )
    per_motif_nll = (
        0.5 * per_motif_mse / torch.exp(2.0 * per_motif_log_sigma)
        + per_motif_log_sigma
        + 0.5 * math.log(2.0 * math.pi)
    )

    observed_std = observed.std(dim=0, unbiased=False)
    standardized_rmse = torch.sqrt(
        torch.mean((mean_difference / observed_std.clamp_min(1.0)).square())
    )
    log_mean_rmse = torch.sqrt(
        torch.mean(
            (torch.log1p(generated_mean.clamp_min(0.0)) - torch.log1p(observed_mean.clamp_min(0.0))).square()
        )
    )
    observed_sorted = torch.sort(torch.log1p(observed.clamp_min(0.0)), dim=0).values
    generated_sorted = torch.sort(torch.log1p(generated.clamp_min(0.0)), dim=0).values
    per_rule_wasserstein = torch.mean(torch.abs(generated_sorted - observed_sorted), dim=0)
    sorted_wasserstein = torch.sort(per_rule_wasserstein).values
    trim = int(math.floor(0.1 * sorted_wasserstein.numel()))
    robust = sorted_wasserstein[:-trim].mean() if trim else sorted_wasserstein.mean()
    paired_rmse = torch.sqrt(torch.mean((generated - observed).square(), dim=1))
    return {
        "aggregate_count_distance": float(aggregate_rmse),
        "aggregate_mean_absolute_count_difference": float(difference.abs().mean()),
        "mean_vector_count_distance": float(mean_vector_rmse),
        "mean_vector_absolute_count_difference": float(mean_difference.abs().mean()),
        "relative_count_distance": float(relative_rmse),
        "standardized_mean_vector_count_distance": float(standardized_rmse),
        "log1p_mean_vector_count_distance": float(log_mean_rmse),
        "robust_count_distance": float(robust),
        "log1p_wasserstein_mean": float(per_rule_wasserstein.mean()),
        "log1p_wasserstein_median": float(per_rule_wasserstein.median()),
        "gaussian_aligned_score": float(per_motif_nll.mean()),
        "paired_graph_count_distance_mean": float(paired_rmse.mean()),
        "paired_graph_count_distance_median": float(paired_rmse.median()),
        "paired_graph_count_distance_std": float(paired_rmse.std(unbiased=False)),
        "gaussian_min_log_sigma": float(min_log_sigma),
        "observed_aggregate_counts": observed_total.tolist(),
        "generated_aggregate_counts": generated_total.tolist(),
    }


def full_state_tv(observed, generated, entries):
    grouped = {}
    rules = {}
    for entry in entries:
        if len(entry["rule"]) < 2:
            continue
        rule_index = int(entry["rule_index"])
        grouped.setdefault(rule_index, []).append(int(entry["index"]))
        rules[rule_index] = entry["rule"]
    details = []
    for rule_index, indices in sorted(grouped.items()):
        obs = observed[:, indices].sum(axis=0)
        gen = generated[:, indices].sum(axis=0)
        if obs.sum() <= 0 or gen.sum() <= 0:
            continue
        tv = 0.5 * np.abs(obs / obs.sum() - gen / gen.sum()).sum()
        details.append({
            "rule_index": rule_index,
            "rule": rules[rule_index],
            "state_count": len(indices),
            "aggregate_state_distribution_tv": float(tv),
        })
    return {
        "multi_atom_rule_count": len(details),
        "mean_per_rule_aggregate_state_tv": fmean(
            item["aggregate_state_distribution_tv"] for item in details
        ),
        "rules": details,
    }


def main():
    args = parse_args()
    generated_graphs, generated_meta, generated_sha = load_collection(args.generated)
    reference_graphs, reference_meta, reference_sha = load_collection(args.reference)
    if len(generated_graphs) != len(reference_graphs):
        raise ValueError(
            f"Need equal graph counts for literal aggregate comparison: {len(generated_graphs)} vs {len(reference_graphs)}"
        )
    pruned_template = json.loads(args.pruned_template.read_text())
    full_template = json.loads(args.full_template.read_text())
    pruned_entries = pruned_template["motif_entries"]
    full_entries = full_template["motif_entries"]
    if len(pruned_entries) != 182 or len(full_entries) != 6899:
        raise ValueError("Unexpected PTC motif universe sizes")

    observed_pruned, observed_nodes = count_entries(reference_graphs, pruned_entries)
    generated_pruned, generated_nodes = count_entries(generated_graphs, pruned_entries)
    # DeFoG samples node counts unconditionally. Pair sorted sizes only for the
    # graph-wise Gaussian diagnostic; aggregates/distributions ignore ordering.
    observed_order = np.argsort(observed_nodes, kind="stable")
    generated_order = np.argsort(generated_nodes, kind="stable")
    metrics = metric_summary(
        observed_pruned[observed_order], generated_pruned[generated_order]
    )
    observed_full, _ = count_entries(reference_graphs, full_entries)
    generated_full, _ = count_entries(generated_graphs, full_entries)

    payload = {
        "schema_version": "defog-ptc-pruned-rule-evaluation-v1",
        "dataset": "PTC",
        "model": "DeFoG",
        "training_seed": args.seed,
        "graph_count": len(generated_graphs),
        "feature_schema": "gin-node-label-v2|export=decoded_node",
        "node_feature_dim": 19,
        "edge_feature_dim": 0,
        "pruned_rule_combination_count": len(pruned_entries),
        "full_combination_count": len(full_entries),
        "hard_pruned_rule_metrics": metrics,
        "normalized_complete_state_correlation": full_state_tv(
            observed_full, generated_full, full_entries
        ),
        "gaussian_pairing": (
            "reference and generated graphs independently sorted by node count; "
            "DeFoG is unconditional, so this is a size-matched diagnostic rather than reconstruction pairing"
        ),
        "aggregation": "sum exact counts over all graphs for aggregate RMSE",
        "generated_node_count": {
            "mean": float(generated_nodes.mean()), "minimum": int(generated_nodes.min()),
            "maximum": int(generated_nodes.max()),
        },
        "reference_node_count": {
            "mean": float(observed_nodes.mean()), "minimum": int(observed_nodes.min()),
            "maximum": int(observed_nodes.max()),
        },
        "artifacts": {
            "generated": str(args.generated.resolve()), "generated_sha256": generated_sha,
            "reference": str(args.reference.resolve()), "reference_sha256": reference_sha,
            "pruned_template": str(args.pruned_template.resolve()),
            "full_template": str(args.full_template.resolve()),
        },
        "generated_metadata": generated_meta,
        "reference_metadata": reference_meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
