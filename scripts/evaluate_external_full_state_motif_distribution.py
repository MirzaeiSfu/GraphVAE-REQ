#!/usr/bin/env python3
"""Evaluate complete FactorBase state distributions for an external graph set.

This intentionally uses every cached state of every multi-atom rule (no
training-time pruning). Counts are summed independently over the reference and
generated collections, normalized within each rule, and compared by total
variation distance. The two collections need not contain the same number of
graphs because this is an aggregate distribution metric, not a paired metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data import DataWrapper, merge_datasets
from motif_counting.motif_counter import RelationalMotifCounter
from scripts.evaluate_motif_count_distance_correlation import (
    count_exact_graph_records,
    count_training_graphs,
    hard_graph_postprocess,
    load_yaml,
    make_counter_args,
    motif_entry_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--motif-cache-dir", required=True)
    parser.add_argument("--generated-graphs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count-batch-size", type=int, default=128)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def graphs_from_bundle(path: Path) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    payload = load_torch(path)
    if isinstance(payload, Mapping) and "graphs" in payload:
        return list(payload["graphs"]), dict(payload.get("metadata") or {})
    if isinstance(payload, Sequence):
        return list(payload), {}
    raise TypeError(f"Unsupported graph collection in {path}: {type(payload)}")


def external_records(
    graphs: Sequence[Mapping[str, Any]], relation_keys: Sequence[str], node_dim: int
) -> list[dict[str, Any]]:
    if len(relation_keys) != 1:
        raise ValueError(f"Expected one adjacency relation, got {relation_keys}")
    relation = relation_keys[0]
    records = []
    for index, graph in enumerate(graphs):
        n = int(graph["num_nodes"])
        x = graph.get("x")
        if x is None:
            raise ValueError(f"Graph {index} has no categorical node features")
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2 and x.shape == (n, 1) and node_dim > 1:
            # Frozen topology-only exports use one constant channel, whereas
            # the historical synthetic cache may retain unused intrinsic
            # feature metadata. Expand to a deterministic dummy category;
            # edge-only motif rules are unaffected.
            expanded = torch.zeros((n, node_dim), dtype=torch.float32)
            expanded[:, 0] = 1.0
            x = expanded
        if x.ndim != 2 or x.shape != (n, node_dim):
            raise ValueError(
                f"Graph {index} node features {tuple(x.shape)} != {(n, node_dim)}"
            )
        # Enforce the same categorical hardening used by GraphVAE evaluation.
        labels = x.argmax(dim=1)
        onehot = torch.nn.functional.one_hot(labels, num_classes=node_dim).float()
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Graph {index} has invalid edge_index {edge_index.shape}")
        adjacency = torch.zeros((n, n), dtype=torch.float32)
        if edge_index.numel():
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= n:
                raise ValueError(f"Graph {index} edge index is outside 0..{n - 1}")
            adjacency[edge_index[0], edge_index[1]] = 1.0
        record = {
            "features": onehot,
            "feat_onehot": onehot,
            "adj": {relation: adjacency},
            "edge": None,
        }
        records.append(hard_graph_postprocess(record))
    return records


def aggregate_full_state_tv(
    observed: torch.Tensor,
    generated: torch.Tensor,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[int, list[int]] = {}
    rules: dict[int, Any] = {}
    for entry in entries:
        if len(entry.get("rule", [])) < 2:
            continue
        rule_index = int(entry["rule_index"])
        grouped.setdefault(rule_index, []).append(int(entry["index"]))
        rules[rule_index] = entry["rule"]

    summaries = []
    for rule_index, indices in sorted(grouped.items()):
        obs = observed[:, indices].clamp_min(0).sum(dim=0)
        gen = generated[:, indices].clamp_min(0).sum(dim=0)
        obs_total = obs.sum()
        gen_total = gen.sum()
        if obs_total <= 0 or gen_total <= 0:
            summaries.append({
                "rule_index": rule_index,
                "rule": rules[rule_index],
                "state_count": len(indices),
                "status": "undefined_zero_total",
                "reference_total": float(obs_total),
                "generated_total": float(gen_total),
                "total_variation": None,
            })
            continue
        p = obs / obs_total
        q = gen / gen_total
        tv = 0.5 * torch.abs(p - q).sum()
        summaries.append({
            "rule_index": rule_index,
            "rule": rules[rule_index],
            "state_count": len(indices),
            "status": "ok",
            "reference_total": float(obs_total),
            "generated_total": float(gen_total),
            "reference_distribution": p.tolist(),
            "generated_distribution": q.tolist(),
            "total_variation": float(tv),
        })
    values = [row["total_variation"] for row in summaries if row["total_variation"] is not None]
    return {
        "score": sum(values) / len(values) if values else None,
        "eligible_multi_atom_rules": len(grouped),
        "evaluated_multi_atom_rules": len(values),
        "rules": summaries,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    dataset_path = Path(args.dataset_cache).resolve()
    motif_dir = Path(args.motif_cache_dir).resolve()
    generated_path = Path(args.generated_graphs).resolve()
    output_path = Path(args.output).resolve()
    _, flat = load_yaml(config_path)
    device = torch.device(args.device)

    with dataset_path.open("rb") as handle:
        cache = pickle.load(handle)
    train_dataset = cache["list_graphs"]
    node_info = cache.get("node_onehot_info")
    edge_info = cache.get("edge_onehot_info")
    # Topology-only synthetic datasets intentionally have no native node
    # attributes. Their interchange artifacts still carry a single constant
    # channel so the external evaluator can use the same graph contract.  This
    # channel is not a learned/data feature and does not enter topology rules.
    node_dim = len(node_info or {}) or 1

    counter = RelationalMotifCounter(
        database_name=str(flat["database_name"]),
        args=make_counter_args(flat, motif_dir, device),
    )
    node_counts = [int(adjacency.shape[0]) for adjacency in train_dataset.list_adjs]
    _, reference_hard, _ = count_training_graphs(
        counter, train_dataset, node_info, edge_info, node_counts,
        args.count_batch_size, device, selected_rules_values=None,
    )

    graphs, metadata = graphs_from_bundle(generated_path)
    records = external_records(graphs, counter.relation_keys, node_dim)
    mapping = DataWrapper(
        merge_datasets(train_dataset), counter.relation_keys, node_info,
        edge_onehot_info=edge_info,
        edge_feature_info_mapping=counter.feature_info_mapping,
        device=str(device),
    ).feature_onehot_mapping
    generated_hard = count_exact_graph_records(
        counter, records, mapping, args.count_batch_size, device,
        selected_rules_values=None,
    )
    entries = motif_entry_metadata(counter, selected_rules_values=None)
    result = aggregate_full_state_tv(reference_hard, generated_hard, entries)
    payload = {
        "schema_version": "external-full-state-motif-distribution-v1",
        "dataset": args.dataset_label,
        "generator": metadata.get("generator", "external"),
        "training_seed": metadata.get("training_seed"),
        "generation_seed": metadata.get("generation_seed"),
        "database_name": flat["database_name"],
        "metric": (
            "Mean over multi-atom rules of TV(p_r,q_r), where state counts are "
            "summed independently over each entire graph collection and normalized "
            "within the complete, unpruned FactorBase state table of rule r."
        ),
        "reference_collection": "GraphVAE training split used by the historical full-state table",
        "reference_graph_count": int(reference_hard.shape[0]),
        "generated_graph_count": int(generated_hard.shape[0]),
        "complete_state_count": int(reference_hard.shape[1]),
        "result": result,
        "artifacts": {
            "config": str(config_path),
            "dataset_cache": str(dataset_path),
            "motif_cache": str(motif_dir / f"{flat['database_name']}.pkl"),
            "generated_graphs": str(generated_path),
            "generated_graphs_sha256": sha256(generated_path),
        },
        "generated_metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{args.dataset_label}: score={result['score']} -> {output_path}")


if __name__ == "__main__":
    main()
