#!/usr/bin/env python3
"""Compute PTC TV on the exact 142 retained states of the trained path rule.

The 182-row training selection contains two singleton edge rows, 142 retained
states of one five-literal path rule, and two 19-state unary node-label rows.
Correlation is defined on the retained states of the multi-atom path rule only.
Train-reference GraphVAE counts reuse the already-computed exact-rule outputs;
test-reference GraphVAE counts are regenerated from the saved checkpoints with
test graph node counts. DeFoG uses its saved 240-train and 70-test exports.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch


OUT = Path("/local-scratch2/mirzaei/ptc_pruned_state_tv_train_test_20260909")
REPO = Path("/local-scratch2/mirzaei/count_distance_manifest_fix/GraphVAE-REQ")
RULE_ROOT = Path("/local-scratch2/mirzaei/ptc_rule_metrics_20260906")
TRUE_ROOT = Path("/local-scratch2/mirzaei/motif_true_clean_20260906/ptc")
FALSE_ROOT = Path("/local-scratch2/new/gather/datasets/ptc/setting_01")
DEFOG_ROOT = Path("/local-scratch2/mirzaei/defog_ptc_frozen_20260906")
DEFOG_240 = Path("/local-scratch2/mirzaei/defog_ptc_full_metrics_20260907/generated_240")
DATASET_CACHE = RULE_ROOT / "cache/PTC_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-gin-node-label-v2.pkl"
MOTIF_CACHE = Path("/local-scratch2/mirzaei/fb/GraphVAE-REQ/cache_motifs/multihop_smoothed")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import evaluate_motif_count_distance as ev  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "defog_ptc_eval", "/local-scratch/localhome/mirzaei/evaluate_defog_ptc_rule_metrics_20260907.py"
)
de = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(de)


def tv_from_aggregate(reference: np.ndarray, generated: np.ndarray, entries: list[dict]) -> dict:
    groups: dict[int, list[int]] = {}
    rules = {}
    for position, entry in enumerate(entries):
        if len(entry["rule"]) < 2:
            continue
        ri = int(entry["rule_index"])
        groups.setdefault(ri, []).append(position)
        rules[ri] = entry["rule"]
    details = []
    for ri, positions in sorted(groups.items()):
        ref = reference[positions].astype(np.float64)
        gen = generated[positions].astype(np.float64)
        ref_total = float(ref.sum())
        gen_total = float(gen.sum())
        if ref_total <= 0:
            # The reference does not define a target distribution for this rule.
            continue
        # If generated retained-state mass is zero, its conditional distribution
        # is undefined. Treat it as complete failure on the evaluated support
        # rather than silently dropping the rule or fabricating a distribution.
        value = (
            1.0
            if gen_total <= 0
            else 0.5 * np.abs(ref / ref_total - gen / gen_total).sum()
        )
        details.append({
            "rule_index": ri,
            "rule": rules[ri],
            "retained_state_count": len(positions),
            "total_variation": float(value),
            "reference_total": ref_total,
            "generated_total": gen_total,
            "zero_generated_retained_mass": bool(gen_total <= 0),
        })
    if not details:
        raise RuntimeError("No eligible retained multi-atom rule states")
    return {
        "mean_per_rule_tv": float(mean(x["total_variation"] for x in details)),
        "eligible_rule_count": len(details),
        "rules": details,
    }


def graphvae_train(method: str, seed: int, entries: list[dict]) -> dict:
    d = json.loads((RULE_ROOT / f"results/{method}_seed{seed}.json").read_text())
    return tv_from_aggregate(
        np.asarray(d["hard"]["observed_aggregate_counts"]),
        np.asarray(d["hard"]["generated_aggregate_counts"]),
        entries,
    )


def graphvae_paths(method: str, seed: int) -> tuple[Path, Path]:
    if method == "false":
        root = FALSE_ROOT / f"seed_{seed}"
        return root / "run_config_used.yaml", root / "best_validation_mmd_model"
    return (
        RULE_ROOT / f"configs/ptc_{method}_effective.yaml",
        TRUE_ROOT / method / f"seed_{seed}/best_validation_mmd_model",
    )


def graphvae_test(method: str, seed: int, entries: list[dict], test_graphs: list[dict], device: torch.device) -> dict:
    config_path, checkpoint = graphvae_paths(method, seed)
    _, flat = ev.load_yaml(config_path)
    _, selection_flat = ev.load_yaml(RULE_ROOT / "configs/ptc_total_count_effective.yaml")
    ev.configure_rng(seed)
    cache = ev.load_pickle(DATASET_CACHE)
    train_dataset = cache["list_graphs"]
    counter = ev.RelationalMotifCounter(
        database_name=str(selection_flat["database_name"]),
        args=ev.make_counter_args(selection_flat, MOTIF_CACHE, device),
    )
    selected: dict[int, list[int]] = {}
    for entry in entries:
        selected.setdefault(int(entry["rule_index"]), []).append(int(entry["value_index"]))
    state = ev.load_state_dict(checkpoint)
    decoder, node_decoder, edge_decoder, graph_dim = ev.build_decoders(state, flat, train_dataset, device)
    node_counts = [int(g["num_nodes"]) for g in test_graphs]
    _, generated_hard, _ = ev.count_generated_graphs(
        counter=counter,
        decoder=decoder,
        node_decoder=node_decoder,
        edge_decoder=edge_decoder,
        graph_dim=graph_dim,
        node_counts=node_counts,
        node_onehot_info=cache.get("node_onehot_info"),
        edge_onehot_info=cache.get("edge_onehot_info"),
        generation_batch_size=16,
        count_batch_size=16,
        adj_threshold=0.5,
        device=device,
        selected_rules_values=selected,
    )
    observed, _ = de.count_entries(test_graphs, entries)
    result = tv_from_aggregate(observed.sum(axis=0), generated_hard.numpy().sum(axis=0), entries)
    result["reference_graph_count"] = len(test_graphs)
    result["generated_graph_count"] = int(generated_hard.shape[0])
    return result


def defog_eval(seed: int, entries: list[dict], split: str) -> dict:
    if split == "train":
        generated = DEFOG_240 / f"seed_{seed}/generated_graphs.pt"
        reference = DEFOG_ROOT / "artifacts/ptc/real_train_graphs.pt"
    else:
        generated = DEFOG_ROOT / f"artifacts/ptc/generated/seed_{seed}/generated_graphs.pt"
        reference = DEFOG_ROOT / "artifacts/ptc/real_test_graphs.pt"
    gg, _, _ = de.load_collection(generated)
    rg, _, _ = de.load_collection(reference)
    gc, _ = de.count_entries(gg, entries)
    rc, _ = de.count_entries(rg, entries)
    result = tv_from_aggregate(rc.sum(axis=0), gc.sum(axis=0), entries)
    result["reference_graph_count"] = len(rg)
    result["generated_graph_count"] = len(gg)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    template = json.loads((RULE_ROOT / "results/total_count_seed0.json").read_text())
    entries = template["motif_entries"]
    if len(entries) != 182:
        raise RuntimeError(f"Expected 182 retained entries, found {len(entries)}")
    test_graphs, _, _ = de.load_collection(DEFOG_ROOT / "artifacts/ptc/real_test_graphs.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_results = {}
    for method in ("false", "total_count", "full_matrix", "defog"):
        all_results[method] = {}
        for seed in range(3):
            target = OUT / f"{method}_seed{seed}.json"
            if target.exists():
                record = json.loads(target.read_text())
            else:
                print(f"[start] {method} seed {seed}", flush=True)
                if method == "defog":
                    train = defog_eval(seed, entries, "train")
                    test = defog_eval(seed, entries, "test")
                else:
                    train = graphvae_train(method, seed, entries)
                    test = graphvae_test(method, seed, entries, test_graphs, device)
                record = {
                    "schema_version": "ptc-pruned-state-tv-train-test-v1",
                    "dataset": "PTC",
                    "method": method,
                    "seed": seed,
                    "training_selection_count": 182,
                    "eligible_path_rule_retained_state_count": 142,
                    "train_reference": train,
                    "test_reference": test,
                    "definition": "sum counts across collection, normalize across the 142 retained states of the trained multi-atom path rule, then total variation",
                }
                target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                print(f"[done] {target}", flush=True)
            all_results[method][str(seed)] = record
    summary = {"schema_version": "ptc-pruned-state-tv-train-test-summary-v1", "methods": {}}
    for method, seeds in all_results.items():
        summary["methods"][method] = {}
        for split in ("train_reference", "test_reference"):
            vals = [seeds[str(s)][split]["mean_per_rule_tv"] for s in range(3)]
            summary["methods"][method][split] = {
                "mean": mean(vals), "sample_sd": stdev(vals), "n": 3, "values": vals
            }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
