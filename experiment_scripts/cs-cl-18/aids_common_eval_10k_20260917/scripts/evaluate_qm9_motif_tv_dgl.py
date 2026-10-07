#!/usr/bin/env python3
"""Compute hard aggregate QM9 motif-state TV on the exact training rule universe."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import dgl
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--motif-cache-dir", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def graph_records(graphs, relation):
    records = []
    for graph in graphs:
        node_features = graph.ndata["attr"].float().cpu()
        n = int(graph.num_nodes())
        source, target = graph.edges()
        source, target = source.long().cpu(), target.long().cpu()
        adjacency = torch.zeros((n, n), dtype=torch.float32)
        adjacency[source, target] = 1.0

        edge = None
        if "attr" in graph.edata:
            edge_rows = graph.edata["attr"].float().cpu()
            packed = torch.zeros(
                (edge_rows.shape[1], n, n), dtype=torch.float32
            )
            packed[:, source, target] = edge_rows.transpose(0, 1)
            edge = [packed]

        # The exact categorical information is carried by feat_onehot and the
        # training cache's feature_onehot_mapping.  The scalar feature column
        # is retained only to satisfy the historical graph-wrapper contract.
        records.append(
            {
                "features": node_features.argmax(dim=1, keepdim=True).float() + 1,
                "feat_onehot": node_features,
                "adj": {relation: adjacency},
                "edge": edge,
            }
        )
    return records


def main():
    args = parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))

    from data import DataWrapper, merge_datasets
    from motif_counting.motif_counter import RelationalMotifCounter
    from evaluate_motif_count_distance import (
        count_exact_graph_records,
        hard_graph_postprocess,
        load_yaml,
        make_counter_args,
        motif_entry_metadata,
    )

    _, flat = load_yaml(args.config)
    # These values are taken from the actual command line recorded in the
    # training log.  The copied YAML predates its launch-time overrides.
    flat["motif_cp_table_source"] = "cp_smoothed"
    flat["rule_prune"] = False
    flat["use_syntactic_literal_rules"] = False
    flat["syntactic_literal_rule_mode"] = "original"
    device = torch.device(args.device)
    counter = RelationalMotifCounter(
        str(flat["database_name"]),
        make_counter_args(flat, args.motif_cache_dir.resolve(), device),
    )

    with args.dataset_cache.open("rb") as handle:
        cache = pickle.load(handle)
    train_dataset = cache["list_graphs"]
    wrapper = DataWrapper(
        merge_datasets(train_dataset),
        counter.relation_keys,
        cache.get("node_onehot_info"),
        edge_onehot_info=cache.get("edge_onehot_info"),
        edge_feature_info_mapping=counter.feature_info_mapping,
        device=str(device),
    )
    mapping = wrapper.feature_onehot_mapping

    generated, _ = dgl.load_graphs(str(args.generated))
    reference, _ = dgl.load_graphs(str(args.reference))
    count = min(len(generated), len(reference))
    generated_records = [
        hard_graph_postprocess(record)
        for record in graph_records(generated[:count], counter.relation_keys[0])
    ]
    reference_records = [
        hard_graph_postprocess(record)
        for record in graph_records(reference[:count], counter.relation_keys[0])
    ]
    generated_counts = count_exact_graph_records(
        counter, generated_records, mapping, args.batch_size, device
    )
    reference_counts = count_exact_graph_records(
        counter, reference_records, mapping, args.batch_size, device
    )

    entries = motif_entry_metadata(counter)
    grouped = {}
    rules = {}
    for entry in entries:
        if len(entry.get("rule", [])) < 2:
            continue
        rule_index = int(entry["rule_index"])
        grouped.setdefault(rule_index, []).append(int(entry["index"]))
        rules[rule_index] = entry["rule"]

    rows = []
    for rule_index, indices in sorted(grouped.items()):
        observed = reference_counts[:, indices].clamp_min(0).sum(dim=0)
        sampled = generated_counts[:, indices].clamp_min(0).sum(dim=0)
        if observed.sum() <= 0 or sampled.sum() <= 0:
            tv = None
        else:
            tv = float(
                (0.5 * ((observed / observed.sum()) - (sampled / sampled.sum())).abs().sum()).item()
            )
        rows.append(
            {
                "rule_index": rule_index,
                "rule": rules[rule_index],
                "state_count": len(indices),
                "total_variation": tv,
                "reference_total": float(observed.sum()),
                "generated_total": float(sampled.sum()),
            }
        )
    values = [row["total_variation"] for row in rows if row["total_variation"] is not None]
    payload = {
        "schema_version": "qm9-training-rule-state-tv-v1",
        "label": args.label,
        "training_seed": args.seed,
        "reference": "held-out test collection",
        "graph_count": count,
        "database": flat["database_name"],
        "cp_table_source": "cp_smoothed",
        "rule_prune": False,
        "loaded_rule_count": len(counter.rules),
        "loaded_state_count": sum(len(value) for value in counter.values),
        "eligible_multi_atom_rule_count": len(grouped),
        "mean_rule_total_variation": sum(values) / len(values) if values else None,
        "rules": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
