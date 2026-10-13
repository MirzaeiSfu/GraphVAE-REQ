#!/usr/bin/env python3
"""Build the PTC three-seed and repaired PROTEINS report addendum."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


ROOT = Path("/local-scratch2/mirzaei")
PTC_FINAL = ROOT / "ptc_motif_true_vs_baseline_gather_corrected_20260906/comparison.json"
PTC_RULE = ROOT / "ptc_rule_metrics_20260906/results"
PTC_FULL = ROOT / "ptc_full_state_correlation_20260906/results"
PROTEIN_REPAIRED = ROOT / "proteins_third_party_repaired_20260906/summary.csv"
PROTEIN_FALSE = Path("/local-scratch2/new/gather/datasets/proteins/setting_01")
OUT = ROOT / "PTC_3SEED_AND_PROTEINS_REPAIRED_METRICS_20260906.md"
OUT_JSON = ROOT / "PTC_3SEED_AND_PROTEINS_REPAIRED_METRICS_20260906.json"

SETTINGS = ("false", "total_count", "full_matrix")
FINAL_NAMES = {
    "false": "motif_false_baseline_gather_setting_01",
    "total_count": "motif_true_total_count",
    "full_matrix": "motif_true_full_matrix",
}
LABELS = {"false": "Motif=False", "total_count": "True total", "full_matrix": "True full"}


def summary(values):
    values = [float(v) for v in values]
    return {"mean": fmean(values), "sample_sd": stdev(values), "values": values, "n": len(values)}


def fmt(item):
    mean, sd = item["mean"], item["sample_sd"]
    if abs(mean) >= 10000 or (mean and abs(mean) < 1e-4):
        return f"{mean:.6e} ± {sd:.6e}"
    return f"{mean:.6f} ± {sd:.6f}"


def improvement(base, value, higher):
    gain = value - base if higher else base - value
    pct = 100.0 * gain / abs(base) if base != 0 else math.nan
    return gain, pct


def comparison_table(title, rows, aggregates):
    lines = [f"### {title}", "", "| Metric | Better | Motif=False | True total | Δ better | Improvement | True full | Δ better | Improvement |", "| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, key, higher in rows:
        base = aggregates["false"][key]
        total = aggregates["total_count"][key]
        full = aggregates["full_matrix"][key]
        dt, pt = improvement(base["mean"], total["mean"], higher)
        df, pf = improvement(base["mean"], full["mean"], higher)
        lines.append(
            f"| {label} | {'↑' if higher else '↓'} | {fmt(base)} | {fmt(total)} | {dt:+.6g} | {pt:+.2f}% | {fmt(full)} | {df:+.6g} | {pf:+.2f}% |"
        )
    return lines


def full_state_tv(payload):
    entries = payload["motif_entries"]
    obs = payload["hard"]["observed_aggregate_counts"]
    gen = payload["hard"]["generated_aggregate_counts"]
    grouped = defaultdict(list)
    rule_text = {}
    for entry in entries:
        if len(entry["rule"]) < 2:
            continue
        grouped[int(entry["rule_index"])].append(int(entry["index"]))
        rule_text[int(entry["rule_index"])] = entry["rule"]
    values = []
    details = []
    for rule_idx, indices in sorted(grouped.items()):
        o = [max(0.0, float(obs[i])) for i in indices]
        g = [max(0.0, float(gen[i])) for i in indices]
        osum, gsum = sum(o), sum(g)
        if osum <= 0 or gsum <= 0:
            continue
        tv = 0.5 * sum(abs(x / osum - y / gsum) for x, y in zip(o, g))
        values.append(tv)
        details.append({"rule_index": rule_idx, "state_count": len(indices), "tv": tv, "rule": rule_text[rule_idx]})
    if not values:
        raise RuntimeError("No eligible PTC full-state correlation rules")
    return fmean(values), details


def load_ptc():
    final = json.loads(PTC_FINAL.read_text())
    out = {"final": {}, "pruned": {}, "full_state": {}, "rules": []}
    for setting in SETTINGS:
        src = final["aggregates"][FINAL_NAMES[setting]]
        values = {}
        for key in ("degree", "clustering", "orbit", "spectral", "diameter", "sparsity", "triangle", "generated_edge_count", "reference_edge_count"):
            item = src["table2"][key]
            values[key] = summary(item["values"])
        gen = src["table2"]["generated_edge_count"]["values"]
        ref = src["table2"]["reference_edge_count"]["values"]
        values["edge_count_absolute_error"] = summary(abs(a - b) for a, b in zip(gen, ref))
        for key in ("local_mmd_rbf", "local_precision", "local_recall", "local_f1_pr", "third_party_mmd_rbf", "third_party_mmd_linear", "third_party_precision", "third_party_recall", "third_party_f1_pr"):
            item = src["table3"][key]
            values[key] = summary(item["values"])
        final_root = (
            Path("/local-scratch2/new/gather/datasets/ptc/setting_01")
            if setting == "false"
            else ROOT / f"motif_true_clean_20260906/ptc/{setting}"
        )
        table3 = [
            json.loads((final_root / f"seed_{seed}/final_table3_metrics.json").read_text())
            for seed in range(3)
        ]
        values["third_party_mmd_linear_median"] = summary(
            x["third_party_eval_metrics"]["metrics"]["mmd_linear"]["median"]
            for x in table3
        )
        values["third_party_mmd_linear_trimmed"] = summary(
            x["third_party_eval_metrics"]["metrics"]["mmd_linear"]["trimmed_mean"]
            for x in table3
        )
        out["final"][setting] = values

        per_seed = [json.loads((PTC_RULE / f"{setting}_seed{seed}.json").read_text()) for seed in range(3)]
        metric_map = {
            "soft_gaussian": ("soft", "gaussian_aligned_score"),
            "hard_gaussian": ("hard", "gaussian_aligned_score"),
            "soft_aggregate_rmse": ("soft", "aggregate_count_distance"),
            "hard_aggregate_rmse": ("hard", "aggregate_count_distance"),
            "soft_robust_wasserstein": ("soft", "robust_count_distance"),
            "hard_robust_wasserstein": ("hard", "robust_count_distance"),
            "soft_mean_vector_rmse": ("soft", "mean_vector_count_distance"),
            "hard_mean_vector_rmse": ("hard", "mean_vector_count_distance"),
        }
        out["pruned"][setting] = {
            key: summary(item[side][metric] for item in per_seed)
            for key, (side, metric) in metric_map.items()
        }
        out["pruned"][setting]["selection"] = per_seed[0]["motif_selection"]
        if not out["rules"]:
            seen = set()
            for entry in per_seed[0]["motif_entries"]:
                rule = tuple(entry["rule"])
                if rule not in seen:
                    seen.add(rule)
                    out["rules"].append(list(rule))

        full_items = [json.loads((PTC_FULL / f"{setting}_seed{seed}.json").read_text()) for seed in range(3)]
        tvs, details = zip(*(full_state_tv(item) for item in full_items))
        out["full_state"][setting] = {
            "aggregate_state_distribution_tv": summary(tvs),
            "multi_atom_rule_count": len(details[0]),
            "full_combination_count": full_items[0]["motif_selection"]["full_combinations"],
            "per_seed_rules": details,
        }
    return out


def load_proteins():
    rows = list(csv.DictReader(PROTEIN_REPAIRED.open()))
    result = {}
    false_items = [json.loads((PROTEIN_FALSE / f"seed_{s}/graph_realism_random_gin.json").read_text()) for s in range(3)]
    result["false"] = {
        "f1_pr": summary(x["metrics"]["f1_pr"]["mean"] for x in false_items),
        "precision": summary(x["metrics"]["precision"]["mean"] for x in false_items),
        "recall": summary(x["metrics"]["recall"]["mean"] for x in false_items),
        "mmd_rbf": summary(x["metrics"]["mmd_rbf"]["mean"] for x in false_items),
        "mmd_linear_mean": summary(x["metrics"]["mmd_linear"]["mean"] for x in false_items),
        "mmd_linear_median": summary(x["metrics"]["mmd_linear"]["median"] for x in false_items),
        "mmd_linear_trimmed": summary(x["metrics"]["mmd_linear"]["trimmed_mean"] for x in false_items),
        "graph_counts": [[x["num_generated_graphs"], x["num_reference_graphs"]] for x in false_items],
    }
    for setting, marker in (("total_count", "/total_count/"), ("full_matrix", "/full_matrix/")):
        selected = sorted((r for r in rows if marker in r["run_dir"]), key=lambda r: r["run_dir"])
        if len(selected) != 3:
            raise RuntimeError(f"Expected three repaired PROTEINS rows for {setting}, got {len(selected)}")
        result[setting] = {
            "f1_pr": summary(r["f1_pr_mean"] for r in selected),
            "precision": summary(r["precision_mean"] for r in selected),
            "recall": summary(r["recall_mean"] for r in selected),
            "mmd_rbf": summary(r["mmd_rbf_mean"] for r in selected),
            "mmd_linear_mean": summary(r["mmd_linear_mean"] for r in selected),
            "mmd_linear_median": summary(r["mmd_linear_median"] for r in selected),
            "mmd_linear_trimmed": summary(r["mmd_linear_trimmed_mean"] for r in selected),
            "graph_counts": [[int(r["num_generated_graphs"]), int(r["num_reference_graphs"])] for r in selected],
        }
    return result


def build_markdown(ptc, proteins):
    lines = [
        "# PTC three-seed comparison and repaired PROTEINS third-party evaluation",
        "",
        "Generated 2026-09-06. Every aggregate is the mean ± sample SD across training seeds 0, 1, and 2. Each Random-GIN seed value is itself the mean of ten evaluator initializations. Positive improvement means motif=True is better after metric direction is applied.",
        "",
        "## PTC",
        "",
        "PTC motif=True uses `_CP_smoothed`, five FactorBase rules, `rule_prune=true`, score threshold 0.0, cap 256 values per rule, and 182 retained combinations out of 6,899. Rule-level diagnostics generate one node-count-matched graph per each of 240 training graphs. The complete correlation calculation separately restores all 6,899 states.",
        "",
        "### Comparability warning",
        "",
        "These PTC columns are useful but are **not a motif-only controlled ablation**. Gather motif=False uses latent dimension 1024, batch 240, and edge-feature loss 0; motif=True uses latent dimension 128, batch 256, and edge-feature loss 1. Reference preprocessing also differs slightly, as shown by reference edge counts. Do not attribute every difference solely to motif loss.",
        "",
        "| Setting | Seeds | Epochs | Latent | Batch | Kernel/BCE | KL | Node | Edge | Motif | CP/rules |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        "| Gather motif=False setting_01 | 3 | 20,000 | 1024 | 240 | 1 | 1 | 1 | 0 | 0 | motif disabled |",
        "| Motif=True total/full | 3 | 20,000 | 128 | 256 | 1 | 1 | 1 | 1 | 0.1 | `_CP_smoothed`; threshold 0; cap 256 |",
        "",
    ]
    structural = [
        ("Degree MMD", "degree", False), ("Clustering MMD", "clustering", False),
        ("Orbit MMD", "orbit", False), ("Spectral MMD", "spectral", False),
        ("Diameter MMD", "diameter", False), ("Sparsity error", "sparsity", False),
        ("Triangle metric", "triangle", False), ("Edge-count absolute error", "edge_count_absolute_error", False),
    ]
    local = [("Local MMD-RBF", "local_mmd_rbf", False), ("Local precision", "local_precision", True), ("Local recall", "local_recall", True), ("Local F1-PR", "local_f1_pr", True)]
    third = [("Third-party MMD-RBF", "third_party_mmd_rbf", False), ("Third-party linear MMD, arithmetic mean", "third_party_mmd_linear", False), ("Third-party linear MMD, median", "third_party_mmd_linear_median", False), ("Third-party linear MMD, 10%-trimmed", "third_party_mmd_linear_trimmed", False), ("Third-party precision", "third_party_precision", True), ("Third-party recall", "third_party_recall", True), ("Third-party F1-PR", "third_party_f1_pr", True)]
    lines += comparison_table("Final structural metrics", structural, ptc["final"]) + [""]
    lines += comparison_table("Final local Random-GIN metrics", local, ptc["final"]) + [""]
    lines += comparison_table("Final third-party Random-GIN metrics", third, ptc["final"]) + [""]
    lines += ["Generated and reference edge counts are descriptive:", "", "| Setting | Generated edges | Reference edges |", "| --- | ---: | ---: |"]
    for setting in SETTINGS:
        lines.append(f"| {LABELS[setting]} | {fmt(ptc['final'][setting]['generated_edge_count'])} | {fmt(ptc['final'][setting]['reference_edge_count'])} |")
    lines += [""]
    pruned_rows = [
        ("Soft Gaussian-aligned NLL", "soft_gaussian", False),
        ("Hard Gaussian-aligned NLL", "hard_gaussian", False),
        ("Soft aggregate raw-count RMSE", "soft_aggregate_rmse", False),
        ("Hard aggregate raw-count RMSE", "hard_aggregate_rmse", False),
        ("Soft robust log1p-Wasserstein", "soft_robust_wasserstein", False),
        ("Hard robust log1p-Wasserstein", "hard_robust_wasserstein", False),
        ("Soft mean-vector RMSE diagnostic", "soft_mean_vector_rmse", False),
        ("Hard mean-vector RMSE diagnostic", "hard_mean_vector_rmse", False),
    ]
    lines += comparison_table("Exact pruned-rule diagnostics (182/6,899 combinations)", pruned_rows, ptc["pruned"]) + [""]
    corr_agg = {s: {"corr": ptc["full_state"][s]["aggregate_state_distribution_tv"]} for s in SETTINGS}
    lines += comparison_table("Normalized complete full-state motif correlation", [("Mean per-rule aggregate-state TV", "corr", False)], corr_agg) + [""]
    lines += [
        "Correlation uses all 6,899 states and the three eligible multi-atom rules, not only the 182 training-retained combinations. For each rule, state counts are summed over all graphs, normalized within rule, compared using total-variation distance, and then averaged equally across eligible rules.",
        "",
        "### Per-seed PTC rule metrics",
        "",
        "| Setting | Seed | Soft Gaussian | Hard Gaussian | Soft aggregate RMSE | Hard aggregate RMSE | Soft robust | Hard robust | Full-state TV |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for setting in SETTINGS:
        for seed in range(3):
            p = ptc["pruned"][setting]
            c = ptc["full_state"][setting]["aggregate_state_distribution_tv"]["values"][seed]
            lines.append(f"| {LABELS[setting]} | {seed} | {p['soft_gaussian']['values'][seed]:.6f} | {p['hard_gaussian']['values'][seed]:.6f} | {p['soft_aggregate_rmse']['values'][seed]:.6f} | {p['hard_aggregate_rmse']['values'][seed]:.6f} | {p['soft_robust_wasserstein']['values'][seed]:.6f} | {p['hard_robust_wasserstein']['values'][seed]:.6f} | {c:.6f} |")
    lines += ["", "### PTC training rule structures", "", "Combination values are intentionally omitted:", ""]
    for idx, rule in enumerate(ptc["rules"], 1):
        lines.append(f"{idx}. `{' AND '.join(rule)}`")
    lines += [
        "", "### PTC artifacts and models", "",
        "- Motif=False metrics/models: `/local-scratch2/new/gather/datasets/ptc/setting_01/seed_{0,1,2}`",
        "- Motif=True metrics/models: `/local-scratch2/mirzaei/motif_true_clean_20260906/ptc/{total_count,full_matrix}/seed_{0,1,2}`",
        "- Pruned rule metrics: `/local-scratch2/mirzaei/ptc_rule_metrics_20260906/results`",
        "- Complete correlation metrics: `/local-scratch2/mirzaei/ptc_full_state_correlation_20260906/results`",
        "- Every seed directory above contains `best_validation_mmd_model`.",
        "",
        "## PROTEINS repaired topology-oriented third-party Random-GIN",
        "",
        "The earlier failure was caused by unequal postprocessing counts. The repaired evaluator removes empty graphs and then deterministically truncates both collections to the same length. All motif=True runs now evaluate 209 generated versus 209 reference graphs. This is the structural-feature Random-GIN evaluation; the separate decoded-node-feature evaluation remains unchanged elsewhere in the master report.",
        "",
    ]
    protein_rows = [
        ("F1-PR", "f1_pr", True),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("MMD-RBF", "mmd_rbf", False),
        ("MMD-linear, arithmetic mean", "mmd_linear_mean", False),
        ("MMD-linear, median", "mmd_linear_median", False),
        ("MMD-linear, 10%-trimmed", "mmd_linear_trimmed", False),
    ]
    lines += comparison_table("Repaired three-seed PROTEINS results", protein_rows, proteins) + [""]
    lines += ["### PROTEINS per-seed evaluator means", "", "| Setting | Seed | Gen/ref | F1-PR | Precision | Recall | MMD-RBF | Linear mean | Linear median | Linear trimmed |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for setting in SETTINGS:
        for seed in range(3):
            p = proteins[setting]
            gc = p["graph_counts"][seed]
            lines.append(f"| {LABELS[setting]} | {seed} | {gc[0]}/{gc[1]} | {p['f1_pr']['values'][seed]:.6f} | {p['precision']['values'][seed]:.6f} | {p['recall']['values'][seed]:.6f} | {p['mmd_rbf']['values'][seed]:.6f} | {p['mmd_linear_mean']['values'][seed]:.6f} | {p['mmd_linear_median']['values'][seed]:.6f} | {p['mmd_linear_trimmed']['values'][seed]:.6f} |")
    lines += [
        "", "Repaired artifacts:", "",
        "- `/local-scratch2/mirzaei/proteins_third_party_repaired_20260906/summary.csv`",
        "- Per-run `graph_realism_random_gin_repaired_20260906.json` files under `/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/proteins/{total_count,full_matrix}/seed_{0,1,2}`.",
        "",
        "## Metric definitions", "",
        "- **Legacy aggregate RMSE:** sum each retained rule/value count over all graphs, then take RMSE between generated and FactorBase/reference aggregate vectors.",
        "- **Gaussian-aligned:** calibrated Gaussian NLL over graph-wise residuals with the training loss's smooth `min_log_sigma=-6` floor. Soft is closest to training; hard shows threshold/postprocessing effects.",
        "- **Robust distance:** upper-10%-trimmed mean of per-combination 1-D Wasserstein distances between `log1p` per-graph counts.",
        "- **Full-state correlation:** mean total-variation error between normalized aggregate state distributions of complete multi-atom rules. It measures relative joint-state structure, not absolute count scale.",
        "- **Random-GIN:** ten evaluator initializations within each training seed; this report then uses mean and sample SD over the three independently trained seeds.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ptc = load_ptc()
    proteins = load_proteins()
    payload = {"ptc": ptc, "proteins_repaired_third_party": proteins}
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT.write_text(build_markdown(ptc, proteins))
    print(OUT)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
