#!/usr/bin/env python3
"""Build DeFoG versus three-seed GraphVAE motif=True synthetic report."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from statistics import fmean, stdev

import networkx as nx
import numpy as np


ROOT = Path("/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906")
OUT = ROOT / "DEFOG_VS_GRAPHVAE_MOTIF_TRUE_SYNTHETIC_20260906.md"
OUT_JSON = ROOT / "DEFOG_VS_GRAPHVAE_MOTIF_TRUE_SYNTHETIC_20260906.json"
DATASETS = ("grid", "lobster", "triangular_grid")
LABEL = {"grid": "GRID", "lobster": "LOBSTER", "triangular_grid": "TRIANGULAR_GRID"}
MODES = ("total_count", "full_matrix")
MODE_LABEL = {"defog": "DeFoG", "total_count": "GraphVAE total", "full_matrix": "GraphVAE full"}


def summarize(values):
    values = [float(value) for value in values]
    return {
        "mean": fmean(values),
        "sample_sd": stdev(values) if len(values) > 1 else None,
        "values": values,
        "n": len(values),
    }


def fmt(summary):
    mean, sd, n = summary["mean"], summary["sample_sd"], summary["n"]
    if abs(mean) >= 10000 or (mean != 0 and abs(mean) < 1e-4):
        value = f"{mean:.6e}"
        spread = f"{sd:.6e}" if sd is not None else None
    else:
        value = f"{mean:.6f}"
        spread = f"{sd:.6f}" if sd is not None else None
    return f"{value} ± {spread}" if spread is not None else f"{value} (n={n})"


def advantage(defog, graphvae, higher):
    gain = graphvae - defog if higher else defog - graphvae
    if defog == 0:
        return gain, None
    return gain, 100.0 * gain / abs(defog)


def fmt_adv(defog, graphvae, higher):
    gain, percentage = advantage(defog, graphvae, higher)
    if percentage is None or not math.isfinite(percentage):
        return f"Δ={gain:+.6g}"
    return f"{percentage:+.2f}%"


def normalize_npy_graph(item):
    graph = nx.from_numpy_array(np.asarray(item))
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() == 0:
        raise ValueError("Empty graph in GraphVAE evaluation collection")
    component = max(nx.connected_components(graph), key=len)
    graph = nx.Graph(graph.subgraph(component))
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def load_npy_graphs(path):
    return [normalize_npy_graph(item) for item in np.load(path, allow_pickle=True)]


def canonical_graph_digest(graphs):
    digest = hashlib.sha256()
    for graph in graphs:
        digest.update(struct.pack(">Q", graph.number_of_nodes()))
        edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges())
        digest.update(struct.pack(">Q", len(edges)))
        for u, v in edges:
            digest.update(struct.pack(">QQ", u, v))
    return digest.hexdigest()


def edge_path_state_counts(graphs):
    totals = np.zeros(4, dtype=np.float64)  # FF, TT, FT, TF
    for graph in graphs:
        n = graph.number_of_nodes()
        degrees = np.asarray([degree for _, degree in graph.degree()], dtype=np.int64)
        tt = int(np.dot(degrees, degrees))
        mixed = int(np.dot(degrees, n - degrees))
        totals += np.asarray([n ** 3 - tt - 2 * mixed, tt, mixed, mixed])
    return totals


def full_state_tv(reference, generated):
    ref = edge_path_state_counts(reference)
    gen = edge_path_state_counts(generated)
    p, q = ref / ref.sum(), gen / gen.sum()
    return {
        "total_variation": float(0.5 * np.abs(p - q).sum()),
        "state_order": ["FF", "TT", "FT", "TF"],
        "reference_counts": ref.tolist(),
        "generated_counts": gen.tolist(),
        "reference_distribution": p.tolist(),
        "generated_distribution": q.tolist(),
    }


def load_graphvae_seed(dataset, mode, seed):
    run = ROOT / "graphvae_linkcorr" / dataset / mode / f"seed_{seed}"
    final = json.loads((run / "final_metrics_summary.json").read_text())
    gin = json.loads((run / "graph_realism_random_gin.json").read_text())
    generated = load_npy_graphs(run / "Single_comp_generatedGraphs_adj_final_eval.npy")
    reference = load_npy_graphs(run / "testGraphs_adj_.npy")
    table2 = final["table2"]
    result = {
        "structural": dict(table2["metrics"]),
        "random_gin": {
            "f1_pr": gin["metrics"]["f1_pr"]["mean"],
            "precision": gin["metrics"]["precision"]["mean"],
            "recall": gin["metrics"]["recall"]["mean"],
            "mmd_rbf": gin["metrics"]["mmd_rbf"]["mean"],
            "mmd_linear_mean": gin["metrics"]["mmd_linear"]["mean"],
            "mmd_linear_median": gin["metrics"]["mmd_linear"]["median"],
            "mmd_linear_trimmed": gin["metrics"]["mmd_linear"]["trimmed_mean"],
        },
        "motif_correlation": full_state_tv(reference, generated),
        "reference_canonical_sha256": canonical_graph_digest(reference),
        "generated_graph_count": len(generated),
        "reference_graph_count": len(reference),
        "run_dir": str(run),
    }
    extra = table2["extra_metrics"]
    result["structural"].update(
        {
            "sparsity": extra["sparsity"],
            "triangle": extra["triangle"],
            "generated_edge_count": extra["generated_edge_count"],
            "reference_edge_count": extra["reference_edge_count"],
            "edge_count_absolute_error": abs(
                extra["generated_edge_count"] - extra["reference_edge_count"]
            ),
        }
    )
    return result


def load_defog_seed(dataset, seed):
    path = ROOT / "metrics" / f"{dataset}_seed{seed}.json"
    payload = json.loads(path.read_text())
    gin = payload["third_party_random_gin_structural_features"]["metrics"]
    return {
        "structural": payload["structural_mmd"],
        "random_gin": {
            "f1_pr": gin["f1_pr"]["mean"],
            "precision": gin["precision"]["mean"],
            "recall": gin["recall"]["mean"],
            "mmd_rbf": gin["mmd_rbf"]["mean"],
            "mmd_linear_mean": gin["mmd_linear"]["mean"],
            "mmd_linear_median": gin["mmd_linear"]["median"],
            "mmd_linear_trimmed": gin["mmd_linear"]["trimmed_mean"],
        },
        "motif_correlation": payload["linkcorr_full_state_distribution"],
        "reference_canonical_sha256": payload["reference_canonical_graph_sha256"],
        "generated_graph_count": payload["generated_graph_count"],
        "reference_graph_count": payload["reference_graph_count"],
        "run_dir": str(path.parent),
        "artifact_path": payload["generated_path"],
    }


def aggregate(seed_records):
    result = {}
    for family in ("structural", "random_gin"):
        keys = seed_records[0][family]
        result[family] = {
            key: summarize(item[family][key] for item in seed_records) for key in keys
        }
    result["motif_correlation"] = {
        "total_variation": summarize(
            item["motif_correlation"]["total_variation"] for item in seed_records
        )
    }
    return result


def load_all():
    payload = {"datasets": {}, "protocol": {}}
    for dataset in DATASETS:
        defog_seeds = (0, 1) if dataset == "grid" else (0, 1, 2)
        records = {
            "defog": [load_defog_seed(dataset, seed) for seed in defog_seeds]
        }
        for mode in MODES:
            records[mode] = [load_graphvae_seed(dataset, mode, seed) for seed in (0, 1, 2)]
        frozen_digest = records["defog"][0]["reference_canonical_sha256"]
        reference_checks = {
            mode: [item["reference_canonical_sha256"] == frozen_digest for item in records[mode]]
            for mode in MODES
        }
        payload["datasets"][dataset] = {
            "per_seed": records,
            "aggregate": {setting: aggregate(items) for setting, items in records.items()},
            "frozen_reference_canonical_sha256": frozen_digest,
            "graphvae_reference_matches_frozen": reference_checks,
        }
    payload["protocol"] = {
        "defog_training_seeds": {"grid": [0, 1], "lobster": [0, 1, 2], "triangular_grid": [0, 1, 2]},
        "graphvae_training_seeds": [0, 1, 2],
        "test_graphs_per_seed": 20,
        "third_party_evaluator_repeats": 10,
        "third_party_structural_node_features": ["degree", "clustering", "square_clustering"],
        "motif_rule": ["edges(nodes0,nodes1)", "edges(nodes1,nodes2)"],
        "motif_state_order": ["FF", "TT", "FT", "TF"],
        "motif_aggregation": "sum exact state counts over all 20 graphs, normalize once, then TV",
    }
    return payload


def table(lines, dataset, family, family_label, rows, data):
    agg = data["aggregate"]
    lines += [
        f"### {LABEL[dataset]} — {family_label}", "",
        "| Metric | Better | DeFoG | GraphVAE total | GraphVAE advantage | GraphVAE full | GraphVAE advantage |",
        "| --- | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key, higher in rows:
        d = agg["defog"][family][key]
        t = agg["total_count"][family][key]
        f = agg["full_matrix"][family][key]
        lines.append(
            f"| {label} | {'↑' if higher else '↓'} | {fmt(d)} | {fmt(t)} | "
            f"{fmt_adv(d['mean'], t['mean'], higher)} | {fmt(f)} | "
            f"{fmt_adv(d['mean'], f['mean'], higher)} |"
        )
    lines.append("")


def build_markdown(payload):
    lines = [
        "# DeFoG versus GraphVAE-REQ motif=True on synthetic datasets",
        "",
        "Generated 2026-09-06. This report evaluates DeFoG GRID, LOBSTER, and TRIANGULAR_GRID samples and compares them with the earlier controlled three-seed GraphVAE-REQ LinkCorrelation motif=True total-count and full-matrix runs. Positive `GraphVAE advantage` means GraphVAE is better after metric direction is applied; a negative value means DeFoG is better.",
        "",
        "## Completion and fairness status", "",
        "- LOBSTER DeFoG: seeds 0–2 complete and evaluated.",
        "- TRIANGULAR_GRID DeFoG: seeds 0–2 complete and evaluated.",
        "- GRID DeFoG: seeds 0–1 complete and evaluated; seed 2 is still training, so all GRID DeFoG aggregates are provisional (`n=2`).",
        "- Every evaluated collection contains exactly 20 generated and 20 reference graphs.",
        "- LOBSTER uses exactly the same frozen test graph identities for DeFoG and GraphVAE.",
        "- GRID and TRIANGULAR_GRID use the same dataset definitions and 70/10/20 split policy, but the saved LinkCorrelation GraphVAE test identities do not match the DeFoG frozen test identities. Their side-by-side results are therefore dataset-level comparisons, not identical-test-set paired comparisons.",
        "- The Random-GIN implementation files are byte-identical between the DeFoG and GraphVAE repositories. Each training seed is averaged over evaluator seeds 0–9.",
        "",
        "## Compared training protocols", "",
        "| Dataset | DeFoG epochs / batch | DeFoG seeds used | GraphVAE epochs | GraphVAE seeds | GraphVAE motif setting |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        "| GRID | 10,000 / 1 | 0, 1 (seed 2 running) | 20,000 | 0, 1, 2 | LinkCorr; total/full; motif weight 0.1 |",
        "| LOBSTER | 1,000 / 4 | 0, 1, 2 | 20,000 | 0, 1, 2 | LinkCorr; total/full; motif weight 0.1 |",
        "| TRIANGULAR_GRID | 10,000 / 1 | 0, 1, 2 | 20,000 | 0, 1, 2 | LinkCorr; total/full; motif weight 0.1 |",
        "",
    ]
    gin_rows = [
        ("F1-PR", "f1_pr", True), ("Precision", "precision", True),
        ("Recall", "recall", True), ("MMD-RBF", "mmd_rbf", False),
        ("MMD-linear mean", "mmd_linear_mean", False),
        ("MMD-linear median", "mmd_linear_median", False),
        ("MMD-linear 10%-trimmed", "mmd_linear_trimmed", False),
    ]
    structural_rows = [
        ("Degree MMD", "degree", False), ("Clustering MMD", "clustering", False),
        ("Orbit MMD", "orbit", False), ("Spectral MMD", "spectral", False),
        ("Diameter MMD", "diameter", False), ("Triangle MMD", "triangle", False),
        ("Sparsity MMD", "sparsity", False),
        ("Edge-count absolute error", "edge_count_absolute_error", False),
    ]
    lines += ["## Third-party structural-feature Random-GIN", "", "Node inputs are degree, clustering coefficient, and square-clustering coefficient. These are the same settings used for the saved GraphVAE third-party results.", ""]
    for dataset in DATASETS:
        table(lines, dataset, "random_gin", "third-party Random-GIN", gin_rows, payload["datasets"][dataset])
    lines += ["## Direct structural MMD", ""]
    for dataset in DATASETS:
        table(lines, dataset, "structural", "direct structural metrics", structural_rows, payload["datasets"][dataset])
        agg = payload["datasets"][dataset]["aggregate"]
        lines += [
            f"Generated/reference mean edge counts: DeFoG {fmt(agg['defog']['structural']['generated_edge_count'])} / {fmt(agg['defog']['structural']['reference_edge_count'])}; "
            f"GraphVAE total {fmt(agg['total_count']['structural']['generated_edge_count'])} / {fmt(agg['total_count']['structural']['reference_edge_count'])}; "
            f"GraphVAE full {fmt(agg['full_matrix']['structural']['generated_edge_count'])} / {fmt(agg['full_matrix']['structural']['reference_edge_count'])}.", "",
        ]
    lines += [
        "## Normalized full-state LinkCorrelation motif metric", "",
        "The rule is `edges(nodes0,nodes1) AND edges(nodes1,nodes2)`. Its four states are `FF`, `TT`, `FT`, and `TF`. For every graph, assignments include repeated variables exactly as the repository motif counter does. Counts are summed over all 20 graphs, normalized once into a four-state distribution, and compared with total-variation distance. Lower is better.", "",
        "| Dataset | DeFoG | GraphVAE total | GraphVAE advantage | GraphVAE full | GraphVAE advantage | Reference identity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for dataset in DATASETS:
        data = payload["datasets"][dataset]
        agg = data["aggregate"]
        d = agg["defog"]["motif_correlation"]["total_variation"]
        t = agg["total_count"]["motif_correlation"]["total_variation"]
        f = agg["full_matrix"]["motif_correlation"]["total_variation"]
        matched = all(data["graphvae_reference_matches_frozen"]["total_count"] + data["graphvae_reference_matches_frozen"]["full_matrix"])
        lines.append(
            f"| {LABEL[dataset]} | {fmt(d)} | {fmt(t)} | {fmt_adv(d['mean'], t['mean'], False)} | "
            f"{fmt(f)} | {fmt_adv(d['mean'], f['mean'], False)} | {'Exact match' if matched else 'Different held-out identities'} |"
        )
    lines += [
        "", "### Per-seed motif TV", "",
        "| Dataset | Model | Seed | Full-state TV |", "| --- | --- | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for setting in ("defog", "total_count", "full_matrix"):
            for seed, record in enumerate(payload["datasets"][dataset]["per_seed"][setting]):
                lines.append(f"| {LABEL[dataset]} | {MODE_LABEL[setting]} | {seed} | {record['motif_correlation']['total_variation']:.9f} |")
    lines += [
        "", "## Main findings", "",
        "- **LOBSTER (exact matched test set):** DeFoG has lower degree, clustering, orbit, spectral, triangle, and sparsity MMD, lower edge-count error, and lower Random-GIN MMD. GraphVAE full has the best normalized motif-state TV (0.004021 versus 0.006782 for DeFoG), while GraphVAE has slightly higher Random-GIN F1/recall.",
        "- **GRID (provisional):** GraphVAE full is better on most structural, third-party, and motif-distribution metrics. DeFoG seed 2 is still training, and the held-out graph identities do not match, so this is not yet a final paired result. The GraphVAE total-count aggregate also contains a large seed-level outlier in linear MMD and edge count.",
        "- **TRIANGULAR_GRID:** GraphVAE total and especially GraphVAE full are better across the reported structural, third-party, and motif-distribution metrics. The test identities differ, and DeFoG seed 2 is a strong outlier, so both facts should accompany any reported comparison.",
        "- Overall, motif=True GraphVAE is not uniformly better than DeFoG: its clearest advantage is TRIANGULAR_GRID and normalized motif-state agreement for the LOBSTER full-matrix setting, whereas DeFoG is stronger on most LOBSTER structural-distribution metrics.",
        "", "## Artifact locations", "",
        "- Central DeFoG evaluation package: `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906` on system 18.",
        "- Per-seed combined DeFoG metrics: `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/metrics/<dataset>_seed<seed>.json`.",
        "- Pinned topology-control Random-GIN audit (constant node input; retained for reproducibility but not used in the main comparison tables): `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/random_gin/<dataset>/seed_<seed>/evaluation.json`.",
        "- The main third-party tables use structural node inputs (degree, clustering, square clustering); those results are embedded in the per-seed combined metric JSON files above.",
        "- GraphVAE comparison copies: `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/graphvae_linkcorr/<dataset>/{total_count,full_matrix}/seed_<seed>`.",
        "- Original DeFoG models: LOBSTER on system 16 under `/local-scratch/mirzaei/defog_frozen_benchmark_20260903/GraphVAE-REQ-full/runs/defog/frozen_eval/jobs/lobster`; GRID on system 18 under `/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/frozen_eval/jobs/grid`; TRIANGULAR_GRID on system 19 under `/local-scratch2/mirzaei/defog_frozen_benchmark_20260903/GraphVAE-REQ-full/runs/defog/frozen_eval/jobs/triangular_grid`. Within each completed seed directory, checkpoints are under `training/checkpoints/frozen_<dataset>_seed_<seed>/` as `best-epoch=...ckpt` and `last.ckpt`; GRID seed 2 is not final yet.",
        "", "## Interpretation caution", "",
        "A lower motif TV shows better agreement in the relative four-state edge-path distribution. It does not measure absolute motif counts, and it is not the DeFoG training objective. GRID remains provisional until DeFoG seed 2 finishes. For GRID and TRIANGULAR_GRID, the reference-identity mismatch must be disclosed in any paper table unless GraphVAE is regenerated/evaluated on the frozen DeFoG test identities.",
    ]
    return "\n".join(lines) + "\n"


def main():
    payload = load_all()
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT.write_text(build_markdown(payload))
    print(OUT)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
