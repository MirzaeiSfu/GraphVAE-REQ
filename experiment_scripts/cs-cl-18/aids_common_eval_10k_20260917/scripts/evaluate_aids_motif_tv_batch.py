#!/usr/bin/env python3
"""Batch AIDs pruned-training-rule TV evaluation with one reference pass."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import dgl
import torch

from evaluate_qm9_motif_tv_dgl import graph_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--motif-cache-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--item", action="append", nargs=4, metavar=("LABEL", "SEED", "GENERATED", "OUTPUT"), required=True
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

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
    # Preserve the actual AIDs training rule selection. In particular, do not
    # disable rule pruning: the requested metric must use the retained states
    # loaded by the same cp_smoothed/pruned configuration as motif training.
    flat.update(
        motif_cp_table_source="cp_smoothed",
        use_syntactic_literal_rules=False,
        syntactic_literal_rule_mode="original",
    )
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
    reference_graphs, _ = dgl.load_graphs(str(args.reference))
    reference_records = [
        hard_graph_postprocess(record)
        for record in graph_records(reference_graphs, counter.relation_keys[0])
    ]
    print(f"Counting shared reference collection ({len(reference_records)} graphs)")
    reference_counts = count_exact_graph_records(
        counter, reference_records, mapping, args.batch_size, device
    )

    entries = motif_entry_metadata(counter)
    grouped, rules = {}, {}
    for entry in entries:
        if len(entry.get("rule", [])) < 2:
            continue
        rule_index = int(entry["rule_index"])
        grouped.setdefault(rule_index, []).append(int(entry["index"]))
        rules[rule_index] = entry["rule"]

    for label, seed_text, generated_text, output_text in args.item:
        generated_path, output_path = Path(generated_text), Path(output_text)
        generated_graphs, _ = dgl.load_graphs(str(generated_path))
        count = min(len(reference_graphs), len(generated_graphs))
        generated_records = [
            hard_graph_postprocess(record)
            for record in graph_records(generated_graphs[:count], counter.relation_keys[0])
        ]
        print(f"Counting {label} seed {seed_text} ({count} graphs)")
        generated_counts = count_exact_graph_records(
            counter, generated_records, mapping, args.batch_size, device
        )
        rows = []
        for rule_index, indices in sorted(grouped.items()):
            observed = reference_counts[:count, indices].clamp_min(0).sum(dim=0)
            sampled = generated_counts[:, indices].clamp_min(0).sum(dim=0)
            tv = None
            if observed.sum() > 0 and sampled.sum() > 0:
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
            "schema_version": "aids-pruned-training-rule-state-tv-v1",
            "label": label,
            "training_seed": int(seed_text),
            "reference": "same frozen 400 held-out test graphs",
            "graph_count": count,
            "database": flat["database_name"],
            "cp_table_source": "cp_smoothed",
            "rule_prune": bool(flat.get("rule_prune", False)),
            "loaded_rule_count": len(counter.rules),
            "loaded_state_count": sum(len(value) for value in counter.values),
            "eligible_multi_atom_rule_count": len(grouped),
            "mean_rule_total_variation": sum(values) / len(values) if values else None,
            "rules": rows,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
