#!/usr/bin/env python3
"""Compute soft PTC TV on the exact retained training states."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

import run_ptc_pruned_state_tv_train_test as base


OUT = Path("/local-scratch2/mirzaei/ptc_pruned_state_tv_soft_20260909")


def train_soft(method: str, seed: int, entries: list[dict]) -> dict:
    d = json.loads((base.RULE_ROOT / f"results/{method}_seed{seed}.json").read_text())
    return base.tv_from_aggregate(
        np.asarray(d["soft"]["observed_aggregate_counts"]),
        np.asarray(d["soft"]["generated_aggregate_counts"]),
        entries,
    )


def test_soft(method: str, seed: int, entries: list[dict], test_graphs: list[dict], device: torch.device) -> dict:
    config_path, checkpoint = base.graphvae_paths(method, seed)
    _, flat = base.ev.load_yaml(config_path)
    _, selection_flat = base.ev.load_yaml(base.RULE_ROOT / "configs/ptc_total_count_effective.yaml")
    base.ev.configure_rng(seed)
    cache = base.ev.load_pickle(base.DATASET_CACHE)
    train_dataset = cache["list_graphs"]
    counter = base.ev.RelationalMotifCounter(
        database_name=str(selection_flat["database_name"]),
        args=base.ev.make_counter_args(selection_flat, base.MOTIF_CACHE, device),
    )
    selected: dict[int, list[int]] = {}
    for entry in entries:
        selected.setdefault(int(entry["rule_index"]), []).append(int(entry["value_index"]))
    state = base.ev.load_state_dict(checkpoint)
    decoder, node_decoder, edge_decoder, graph_dim = base.ev.build_decoders(state, flat, train_dataset, device)
    node_counts = [int(g["num_nodes"]) for g in test_graphs]
    generated_soft, _, _ = base.ev.count_generated_graphs(
        counter=counter, decoder=decoder, node_decoder=node_decoder,
        edge_decoder=edge_decoder, graph_dim=graph_dim, node_counts=node_counts,
        node_onehot_info=cache.get("node_onehot_info"),
        edge_onehot_info=cache.get("edge_onehot_info"), generation_batch_size=16,
        count_batch_size=16, adj_threshold=0.5, device=device,
        selected_rules_values=selected,
    )
    observed, _ = base.de.count_entries(test_graphs, entries)
    result = base.tv_from_aggregate(
        observed.sum(axis=0), generated_soft.numpy().sum(axis=0), entries
    )
    result["reference_graph_count"] = len(test_graphs)
    result["generated_graph_count"] = int(generated_soft.shape[0])
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    template = json.loads((base.RULE_ROOT / "results/total_count_seed0.json").read_text())
    entries = template["motif_entries"]
    test_graphs, _, _ = base.de.load_collection(base.DEFOG_ROOT / "artifacts/ptc/real_test_graphs.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}
    for method in ("false", "total_count", "full_matrix"):
        results[method] = {}
        for seed in range(3):
            print(f"[start] {method} seed {seed}", flush=True)
            record = {
                "schema_version": "ptc-pruned-state-soft-tv-v1",
                "method": method, "seed": seed,
                "retained_path_state_count": 142,
                "train_reference": train_soft(method, seed, entries),
                "test_reference": test_soft(method, seed, entries, test_graphs, device),
                "definition": "TV after summing differentiable soft counts and normalizing across the 142 retained states of the trained path rule",
            }
            (OUT / f"{method}_seed{seed}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n"
            )
            results[method][str(seed)] = record
            print(f"[done] {method} seed {seed}", flush=True)
    summary = {"schema_version": "ptc-pruned-state-soft-tv-summary-v1", "methods": {}}
    for method, seeds in results.items():
        summary["methods"][method] = {}
        for split in ("train_reference", "test_reference"):
            vals = [seeds[str(s)][split]["mean_per_rule_tv"] for s in range(3)]
            summary["methods"][method][split] = {
                "mean": mean(vals), "sample_sd": stdev(vals), "n": 3, "values": vals
            }
    summary["methods"]["defog"] = {
        "train_reference": None, "test_reference": None,
        "reason": "DeFoG exports discrete graphs and has no GraphVAE-style soft decoder probabilities",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
