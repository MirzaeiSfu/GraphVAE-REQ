#!/usr/bin/env python3
"""Aggregate attributed LOBSTER GraphCL-GIN and semantic-consistency results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODES = (
    "topology_control",
    "decoded_node",
    "decoded_edge",
    "decoded_node_edge",
)
CONDITIONS = (
    "lobster_graphvae_mm_fixed_split_native40_legacy",
    "lobster_kiarash_parity_kia40_2000_feature40_legacy",
    "lobster_semantic_hybrid_r001_legacy",
    "lobster_semantic_hybrid_r001_edgecount01_legacy",
)
LABELS = {
    CONDITIONS[0]: "Matched manual Kiarash",
    CONDITIONS[1]: "Motif bundle only",
    CONDITIONS[2]: "Relational 0.01",
    CONDITIONS[3]: "Relational 0.01 + edge count 0.1",
}
METRICS = (
    "f1_pr",
    "mmd_rbf",
    "fid",
    "precision",
    "recall",
    "f1_dc",
    "density",
    "coverage",
)
NODE_FEATURES = (
    "node_degree",
    "distance_to_spine",
    "subtree_size",
    "eccentricity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--encoders", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summary(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_across_generator_seeds": float(array.std()),
        "values_by_generator_seed": array.tolist(),
    }


def load_graphcl(root: Path) -> tuple[dict, list[dict]]:
    aggregate = {}
    csv_rows = []
    for mode in MODES:
        aggregate[mode] = {}
        for condition in CONDITIONS:
            seed_payloads = []
            for seed in range(3):
                path = root / mode / condition / f"seed_{seed}" / "evaluation.json"
                payload = json.loads(path.read_text())
                if payload["checkpoint_count"] != 3:
                    raise RuntimeError(f"{path} does not contain three encoder seeds.")
                if payload["feature_mode"] != mode:
                    raise RuntimeError(
                        f"{path} has mode {payload['feature_mode']}, expected {mode}."
                    )
                seed_payloads.append(payload)
            metric_summary = {
                metric: summary(
                    [payload["summary"][metric]["mean"] for payload in seed_payloads]
                )
                for metric in METRICS
            }
            aggregate[mode][condition] = {
                "label": LABELS[condition],
                "metrics": metric_summary,
            }
            row = {"mode": mode, "condition": condition, "label": LABELS[condition]}
            for metric in METRICS:
                row[f"{metric}_mean"] = metric_summary[metric]["mean"]
                row[f"{metric}_std_generator"] = metric_summary[metric][
                    "std_across_generator_seeds"
                ]
            csv_rows.append(row)
    return aggregate, csv_rows


def load_audits(path: Path) -> tuple[dict, list[dict], dict]:
    campaign = json.loads(path.read_text())
    by_condition = {}
    csv_rows = []
    for condition in CONDITIONS:
        audits = sorted(
            (
                row
                for row in campaign["audits"]
                if row["condition"] == condition
            ),
            key=lambda row: row["training_seed"],
        )
        if [row["training_seed"] for row in audits] != [0, 1, 2]:
            raise RuntimeError(f"Incomplete attribute audit for {condition}.")
        consistency = {
            feature: summary(
                [
                    row["semantic_consistency"]["node_accuracy"][feature]
                    for row in audits
                ]
            )
            for feature in NODE_FEATURES
        }
        consistency["all_node_features"] = summary(
            [
                row["semantic_consistency"]["node_all_features_accuracy"]
                for row in audits
            ]
        )
        consistency["edge_type"] = summary(
            [row["semantic_consistency"]["edge_accuracy"] for row in audits]
        )
        marginal_tv = {
            feature: summary(
                [
                    row["marginal_distance_to_heldout"]["node"][feature][
                        "total_variation"
                    ]
                    for row in audits
                ]
            )
            for feature in NODE_FEATURES
        }
        marginal_tv["edge_type"] = summary(
            [
                row["marginal_distance_to_heldout"]["edge"]["edge_type"][
                    "total_variation"
                ]
                for row in audits
            ]
        )
        redecoded = {
            "exact_graphs": int(
                sum(
                    row["redecoded_topology_audit"]["exact_lcc_count"]
                    for row in audits
                )
            ),
            "total_graphs": int(
                sum(
                    row["redecoded_topology_audit"]["total_count"]
                    for row in audits
                )
            ),
            "all_exports_match_frozen_topology": all(
                row["topology_exact_match"] for row in audits
            ),
        }
        by_condition[condition] = {
            "label": LABELS[condition],
            "semantic_consistency": consistency,
            "marginal_total_variation": marginal_tv,
            "redecoded_topology": redecoded,
        }
        row = {"condition": condition, "label": LABELS[condition]}
        for feature, values in consistency.items():
            row[f"{feature}_accuracy_mean"] = values["mean"]
            row[f"{feature}_accuracy_std_generator"] = values[
                "std_across_generator_seeds"
            ]
        for feature, values in marginal_tv.items():
            row[f"{feature}_tv_mean"] = values["mean"]
            row[f"{feature}_tv_std_generator"] = values[
                "std_across_generator_seeds"
            ]
        csv_rows.append(row)
    return by_condition, csv_rows, campaign


def paired_deltas(graphcl: dict, audits: dict) -> dict:
    baseline = CONDITIONS[1]
    result = {}
    for condition in CONDITIONS:
        if condition == baseline:
            continue
        mode_deltas = {}
        for mode in MODES:
            mode_deltas[mode] = {}
            for metric in ("f1_pr", "mmd_rbf", "fid"):
                values = np.asarray(
                    graphcl[mode][condition]["metrics"][metric][
                        "values_by_generator_seed"
                    ]
                )
                baseline_values = np.asarray(
                    graphcl[mode][baseline]["metrics"][metric][
                        "values_by_generator_seed"
                    ]
                )
                delta = values - baseline_values
                mode_deltas[mode][metric] = {
                    "mean_delta": float(delta.mean()),
                    "paired_seed_deltas": delta.tolist(),
                    "favorable_seed_count": int(
                        (delta > 0).sum()
                        if metric == "f1_pr"
                        else (delta < 0).sum()
                    ),
                }
        consistency_delta = {}
        for feature in (*NODE_FEATURES, "all_node_features", "edge_type"):
            values = np.asarray(
                audits[condition]["semantic_consistency"][feature][
                    "values_by_generator_seed"
                ]
            )
            baseline_values = np.asarray(
                audits[baseline]["semantic_consistency"][feature][
                    "values_by_generator_seed"
                ]
            )
            delta = values - baseline_values
            consistency_delta[feature] = {
                "mean_delta": float(delta.mean()),
                "paired_seed_deltas": delta.tolist(),
                "favorable_seed_count": int((delta > 0).sum()),
            }
        result[condition] = {
            "label": LABELS[condition],
            "baseline": LABELS[baseline],
            "graphcl": mode_deltas,
            "semantic_consistency": consistency_delta,
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: dict, digits: int = 3) -> str:
    return (
        f"{values['mean']:.{digits}f} ± "
        f"{values['std_across_generator_seeds']:.{digits}f}"
    )


def write_report(path: Path, result: dict) -> None:
    graphcl = result["graphcl"]
    audits = result["attribute_audits"]
    deltas = result["paired_deltas"][CONDITIONS[2]]
    lines = [
        "# Attributed GraphCL-GIN analysis",
        "",
        "## Why the topology-control encoder is necessary",
        "",
        "The topology-control encoder receives the same graph edges but replaces "
        "all decoded node features with a constant and removes edge features. "
        "It measures how much a result can be explained by topology alone. The "
        "node, edge, and node+edge encoders are independently trained on the "
        "same 70-graph real training split, with three encoder seeds each.",
        "",
        "## GraphCL-GIN held-out results",
        "",
        "Each cell is the mean over three generator seeds after first averaging "
        "the three frozen GraphCL encoder seeds; uncertainty is the standard "
        "deviation across generator seeds. F1-PR is higher-is-better; RBF MMD "
        "and FID are lower-is-better.",
        "",
    ]
    for mode in MODES:
        lines += [
            f"### {mode}",
            "",
            "| Method | F1-PR ↑ | RBF MMD ↓ | FID ↓ |",
            "|---|---:|---:|---:|",
        ]
        for condition in CONDITIONS:
            metrics = graphcl[mode][condition]["metrics"]
            lines.append(
                f"| {LABELS[condition]} | {mean_std(metrics['f1_pr'])} | "
                f"{mean_std(metrics['mmd_rbf'])} | {mean_std(metrics['fid'], 1)} |"
            )
        lines.append("")

    lines += [
        "## Direct decoded-attribute checks",
        "",
        "The old_v1 LOBSTER attributes are deterministic functions of topology. "
        "The following accuracy recomputes each label from the frozen generated "
        "topology and compares it with the decoder's categorical argmax. A "
        "perfectly coherent decoder would score 1.0. TV is the total-variation "
        "distance between the generated and held-out categorical marginals.",
        "",
        "| Method | Degree acc. | Spine acc. | Subtree acc. | Ecc. acc. | "
        "All-node acc. | Edge-type acc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = audits[condition]["semantic_consistency"]
        lines.append(
            f"| {LABELS[condition]} | "
            f"{mean_std(values['node_degree'])} | "
            f"{mean_std(values['distance_to_spine'])} | "
            f"{mean_std(values['subtree_size'])} | "
            f"{mean_std(values['eccentricity'])} | "
            f"{mean_std(values['all_node_features'])} | "
            f"{mean_std(values['edge_type'])} |"
        )
    lines += [
        "",
        "| Method | Degree TV ↓ | Spine TV ↓ | Subtree TV ↓ | Ecc. TV ↓ | "
        "Edge-type TV ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = audits[condition]["marginal_total_variation"]
        lines.append(
            f"| {LABELS[condition]} | "
            f"{mean_std(values['node_degree'])} | "
            f"{mean_std(values['distance_to_spine'])} | "
            f"{mean_std(values['subtree_size'])} | "
            f"{mean_std(values['eccentricity'])} | "
            f"{mean_std(values['edge_type'])} |"
        )

    full_delta = deltas["graphcl"]["decoded_node_edge"]
    node_delta = deltas["graphcl"]["decoded_node"]
    semantic_delta = deltas["semantic_consistency"]
    lines += [
        "",
        "## Critical interpretation",
        "",
        "The narrow positive result is that relational 0.01 improves attributed "
        "neighbourhood precision/recall over bundle-only: its mean F1-PR delta "
        f"is {node_delta['f1_pr']['mean_delta']:+.3f} in the node view "
        f"({node_delta['f1_pr']['favorable_seed_count']}/3 generator seeds) and "
        f"{full_delta['f1_pr']['mean_delta']:+.3f} in the node+edge view "
        f"({full_delta['f1_pr']['favorable_seed_count']}/3 seeds). Its "
        "topology-only F1-PR does not improve, so this signal is not explained "
        "by topology alone.",
        "",
        "That is not enough to accept the hypothesis. In the combined view, "
        f"the relational model changes RBF MMD by "
        f"{full_delta['mmd_rbf']['mean_delta']:+.3f} and FID by "
        f"{full_delta['fid']['mean_delta']:+.1f} versus bundle-only; only "
        f"{full_delta['mmd_rbf']['favorable_seed_count']}/3 seeds improve RBF "
        "MMD. The direct semantic check is more negative: relational 0.01 "
        f"changes all-node consistency by "
        f"{semantic_delta['all_node_features']['mean_delta']:+.3f} and "
        f"edge-type consistency by {semantic_delta['edge_type']['mean_delta']:+.3f}; "
        "neither improves in any paired generator seed.",
        "",
        "The bundle-only control itself has the best semantic consistency even "
        "though its structural bundle contains no typed literal information. "
        "That means optimization/run variability is large enough that the "
        "difference cannot be credited to relational motifs without a stronger "
        "ablation. Edge-count 0.1 is consistently harmful and should be dropped.",
        "",
        "Two scope limitations matter. First, this Relational 0.01 condition "
        "adds the multi-atom full-matrix motif loss but keeps the single-atom "
        "literal-marginal weight at zero; it does not test whether a direct "
        "literal loss improves attributes. Second, old_v1 LOBSTER attributes "
        "are deterministic structural descriptors, not independent semantic "
        "labels. They are missed by a scalar such as diameter but are, in "
        "principle, recoverable from topology.",
        "",
        "Conclusion: there is a reproducible F1-PR hint that attributed GraphCL "
        "detects something missed by topology-only criteria, especially in the "
        "node+edge view, but the stronger semantic and distance metrics reject "
        "the claim that the current relational motif objective improves "
        "generated attributes overall.",
        "",
        "## Reproducibility notes",
        "",
        "- All 12 exported attributed collections exactly reuse the frozen "
        "rollout-0 topology.",
        "- Attributes and adjacency were decoded from the same CUDA latent draw.",
        "- Re-decoding on a different GPU model reproduced "
        f"{result['redecoded_topology_exact_graphs']}/"
        f"{result['redecoded_topology_total_graphs']} LCCs byte-for-byte; the "
        "remaining threshold-boundary cases used the already-frozen topology "
        "while preserving the recovered decoder-node alignment.",
        "- Only 20 held-out graphs and three generator seeds are available. "
        "Standard deviations are descriptive, not confidence intervals.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    inputs = args.inputs.expanduser().resolve()
    evaluations = args.evaluations.expanduser().resolve()
    encoders = args.encoders.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    for mode in MODES:
        for seed in range(3):
            checkpoint = encoders / mode / f"seed_{seed}" / "checkpoint.pt"
            training = encoders / mode / f"seed_{seed}" / "training.json"
            if not checkpoint.is_file() or not training.is_file():
                raise FileNotFoundError(
                    f"Missing trained {mode} encoder seed {seed}: {checkpoint}."
                )

    graphcl, graphcl_rows = load_graphcl(evaluations)
    audits, audit_rows, campaign = load_audits(inputs / "campaign.json")
    result = {
        "design": {
            "generator_conditions": list(CONDITIONS),
            "generator_seeds": [0, 1, 2],
            "encoder_modes": list(MODES),
            "encoder_seeds": [0, 1, 2],
            "heldout_graphs": 20,
            "graphcl_training_graphs": 70,
        },
        "graphcl": graphcl,
        "attribute_audits": audits,
    }
    result["paired_deltas"] = paired_deltas(graphcl, audits)
    result["redecoded_topology_exact_graphs"] = sum(
        audits[condition]["redecoded_topology"]["exact_graphs"]
        for condition in CONDITIONS
    )
    result["redecoded_topology_total_graphs"] = sum(
        audits[condition]["redecoded_topology"]["total_graphs"]
        for condition in CONDITIONS
    )
    result["all_exports_match_frozen_topology"] = all(
        audits[condition]["redecoded_topology"][
            "all_exports_match_frozen_topology"
        ]
        for condition in CONDITIONS
    )
    result["feature_schema"] = campaign["feature_schema"]

    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "graphcl_by_method_and_view.csv", graphcl_rows)
    write_csv(output / "attribute_consistency.csv", audit_rows)
    write_report(output / "analysis.md", result)
    print(f"Wrote attributed GraphCL summary to {output}")


if __name__ == "__main__":
    main()
