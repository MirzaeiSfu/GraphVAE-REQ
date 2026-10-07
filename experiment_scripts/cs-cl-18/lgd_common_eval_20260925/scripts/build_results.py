#!/usr/bin/env python3
"""Build <BASE>/<DATASET>/RESULTS.md (+ results_table.json) comparing LGD with the
existing corrected results of GraphVAE motif=False, GraphVAE-REQ motif=True full
matrix and DeFoG.  Existing values are read (never written) from their result files.

Usage: build_results.py DATASET
"""
import json
import statistics
import sys
from pathlib import Path

BASE = Path(__import__("os").environ.get("LGD_EVAL_BASE", "/local-scratch2/mirzaei/lgd_common_eval_20260925"))
CE = Path("/local-scratch2/mirzaei/common_eval_fix_20260923")
LGD_SEEDS = [0, 1, 2]


def load(p):
    return json.loads(Path(p).read_text())


def get(d, path):
    for part in path.split("."):
        if d is None:
            return None
        d = d.get(part)
    return d


STRUCT = [("Degree MMD", "degree"), ("Clustering MMD", "clustering"), ("Orbit MMD", "orbit"),
          ("Spectral MMD", "spectral"), ("Diameter MMD", "diameter"), ("Triangle MMD", "triangle"),
          ("Sparsity error", "sparsity"), ("Mean-edge absolute error", "edge_count_absolute_error"),
          ("Generated mean edges (descriptive)", "generated_edge_count"),
          ("Reference mean edges (descriptive)", "reference_edge_count")]
GIN = [("F1-PR", "f1_pr.mean"), ("Precision", "precision.mean"), ("Recall", "recall.mean"),
       ("MMD-RBF", "mmd_rbf.mean"), ("Linear MMD mean", "mmd_linear.mean"),
       ("Linear MMD median", "mmd_linear.median"), ("Linear MMD trimmed mean", "mmd_linear.trimmed_mean")]
MODE_GIN = [("F1-PR", "f1_pr.mean"), ("Precision", "precision.mean"), ("Recall", "recall.mean"),
            ("MMD-RBF", "mmd_rbf.mean"), ("Linear MMD mean", "mmd_linear.mean")]


def struct_from(d):
    if "structural_mmd" in d:
        return d["structural_mmd"]
    s = dict(d["metrics"])  # GraphVAE final_table2_metrics.json
    ex = d["extra_metrics"]
    s.update(sparsity=ex["sparsity"], triangle=ex["triangle"], generated_edge_count=ex["generated_edge_count"],
             reference_edge_count=ex["reference_edge_count"],
             edge_count_absolute_error=abs(ex["reference_edge_count"] - ex["generated_edge_count"]))
    return s


def synthetic_rows(dataset, slug):
    methods = {"Motif=False": ("motif_false", [0, 1, 2]), "Motif=True full": ("motif_true_full_matrix", [0, 1, 2]),
               "DeFoG": ("defog", [0, 4, 5] if dataset == "GRID" else [0, 1, 3])}
    rows = {}
    sources = {}
    for name, (prefix, seeds) in methods.items():
        files = [CE / dataset / "results" / f"{prefix}_seed_{s}.json" for s in seeds]
        rows[name] = (seeds, [load(f) for f in files])
        sources[name] = [str(f) for f in files]
    files = [BASE / dataset / "results" / f"lgd_seed_{s}.json" for s in LGD_SEEDS]
    rows["LGD"] = (LGD_SEEDS, [load(f) if f.exists() else None for f in files])
    sources["LGD"] = [str(f) for f in files]
    sections = [
        ("Structural MMD (common %s-graph reference)" % "20", [(lab, lambda d, k=k: d["structural_mmd"][k]) for lab, k in STRUCT]),
        ("Random-GIN, structural node features (degree/clustering/square clustering), 10 evaluator seeds",
         [(lab, lambda d, k=k: get(d["third_party_random_gin_structural_features"]["metrics"], k)) for lab, k in GIN]),
        ("Auxiliary: edge-path state TV (edges(n0,n1),edges(n1,n2))",
         [("Total variation", lambda d: d["linkcorr_full_state_distribution"]["total_variation"])]),
    ]
    return rows, sources, sections


def proteins_rows():
    methods = {"Motif=False": ("false", [0, 1, 2]), "Motif=True full": ("true_full_003", [0, 1, 2]),
               "DeFoG": ("defog", [0, 1, 2])}
    rows, sources = {}, {}
    for name, (sub, seeds) in methods.items():
        files = [CE / "PROTEINS" / "results" / sub / f"seed_{s}" / "common_metrics.json" for s in seeds]
        rows[name] = (seeds, [load(f) for f in files])
        sources[name] = [str(f) for f in files]
    files = [BASE / "PROTEINS" / "results" / "lgd" / f"seed_{s}" / "common_metrics.json" for s in LGD_SEEDS]
    rows["LGD"] = (LGD_SEEDS, [load(f) if f.exists() else None for f in files])
    sources["LGD"] = [str(f) for f in files]
    sup = [BASE / "PROTEINS/supplementary_lgd_own_test_split/results/lgd" / f"seed_{s}" / "common_metrics.json" for s in LGD_SEEDS]
    if all(f.exists() for f in sup):
        rows["LGD vs own test split (suppl., not comparable)"] = (LGD_SEEDS, [load(f) for f in sup])
        sources["LGD vs own test split (suppl., not comparable)"] = [str(f) for f in sup]
    sections = [("Structural MMD (common 209-graph reference; suppl. column: LGD's own 209-graph test split)", [(lab, lambda d, k=k: d["structural_mmd"][k]) for lab, k in STRUCT])]
    for mode, title in [("topology_control", "Random-GIN topology-control (constant node input)"),
                        ("decoded_node", "Random-GIN native node labels (3-way one-hot)")]:
        sections.append((title + ", 10 evaluator seeds",
                         [(lab, lambda d, k=k, m=mode: get(d["random_gin"]["modes"][m]["summary"], k)) for lab, k in MODE_GIN]))
    return rows, sources, sections


def qm9_rows():
    Q = Path("/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluations")
    rows, sources = {}, {}
    for name, sub in (("Motif=False", "false"), ("Motif=True full", "true_full")):
        items, src = [], []
        for s in range(3):
            p = Q / "graphvae" / sub / f"seed_{s}"
            d = load(p / "common_structural.json")
            d["random_gin"] = load(p / "attributed_random_gin.json")["evaluation"]
            items.append(d)
            src += [str(p / "common_structural.json"), str(p / "attributed_random_gin.json")]
        rows[name] = ([0, 1, 2], items)
        sources[name] = src
    files = [Q / "defog" / f"seed_{s}" / "common_metrics.json" for s in range(3)]
    rows["DeFoG"] = ([0, 1, 2], [load(f) for f in files])
    sources["DeFoG"] = [str(f) for f in files]
    files = [BASE / "QM9" / "results" / "lgd" / f"seed_{s}" / "common_metrics.json" for s in LGD_SEEDS]
    rows["LGD"] = (LGD_SEEDS, [load(f) if f.exists() else None for f in files])
    sources["LGD"] = [str(f) for f in files]
    sections = [("Structural MMD (common 512-graph reference)", [(lab, lambda d, k=k: d["structural_mmd"][k]) for lab, k in STRUCT])]
    for mode, title in [("topology_control", "Random-GIN without node features (topology only)"),
                        ("decoded_node", "Random-GIN with node features (atom type 5 + H count 4)")]:
        sections.append((title + ", 10 evaluator seeds",
                         [(lab, lambda d, k=k, m=mode: get(d["random_gin"]["modes"][m]["summary"], k)) for lab, k in MODE_GIN]))
    return rows, sources, sections


def ptc_rows():
    D = Path("/local-scratch2/mirzaei")
    rows, sources = {}, {}

    def pack(struct_path, sgin_path, sgin_key, topo, native):
        return {"struct": struct_from(load(struct_path)), "sgin": get(load(sgin_path), sgin_key),
                "topo": topo, "native": native}

    def modes(path):
        return load(path)["evaluation"]["modes"]

    items, src = [], []
    for s in range(3):
        st = Path(f"/local-scratch2/new/gather/datasets/ptc/setting_01/seed_{s}/final_table2_metrics.json")
        sg = D / f"ptc_matched_reference_eval_20260913/topology_feature/seed_{s}/evaluation.json"
        tp = D / f"ptc_matched_reference_eval_20260913/random_gin/motif_false_seed{s}.json"
        items.append(pack(st, sg, "metrics", modes(tp)["topology_control"]["summary"], None))
        src += [str(st), str(sg), str(tp)]
    rows["Motif=False"] = ([0, 1, 2], items)
    sources["Motif=False"] = src
    items, src = [], []
    for s in range(3):
        p = D / f"motif_true_clean_20260906/ptc/full_matrix/seed_{s}"
        t3 = load(p / "final_table3_metrics.json")["local_eval_metrics"]
        topo = {"f1_pr": {"mean": t3["f1_pr"]}, "precision": {"mean": t3["precision"]},
                "recall": {"mean": t3["recall"]}, "mmd_rbf": {"mean": t3["mmd_rbf"]}}
        items.append(pack(p / "final_table2_metrics.json", p / "graph_realism_random_gin.json", "metrics", topo, None))
        src += [str(p / "final_table2_metrics.json"), str(p / "graph_realism_random_gin.json"),
                str(p / "final_table3_metrics.json") + " (local_eval_metrics)"]
    rows["Motif=True full"] = ([0, 1, 2], items)
    sources["Motif=True full"] = src
    items, src = [], []
    for s in range(3):
        st = D / f"defog_ptc_full_metrics_20260907/structural/seed_{s}.json"
        rg = D / f"defog_ptc_frozen_20260906/random_gin/ptc/seed_{s}/evaluation.json"
        m = modes(rg)
        items.append(pack(st, st, "third_party_random_gin_structural_features.metrics",
                          m["topology_control"]["summary"], m["decoded_node"]["summary"]))
        src += [str(st), str(rg)]
    rows["DeFoG"] = ([0, 1, 2], items)
    sources["DeFoG"] = src
    items, src = [], []
    for s in LGD_SEEDS:
        st = BASE / f"PTC/results/lgd/seed_{s}/structural.json"
        rg = BASE / f"PTC/results/lgd/seed_{s}/random_gin.json"
        if st.exists() and rg.exists():
            m = modes(rg)
            items.append(pack(st, st, "third_party_random_gin_structural_features.metrics",
                              m["topology_control"]["summary"], m["decoded_node"]["summary"]))
        else:
            items.append(None)
        src += [str(st), str(rg)]
    rows["LGD"] = (LGD_SEEDS, items)
    sources["LGD"] = src
    sup = BASE / "PTC/supplementary_motif_false_common_reference/results"
    if all((sup / f"motif_false_seed_{s}.json").exists() for s in range(3)):
        rows["Motif=False (common-ref rescore, suppl.)"] = (
            [0, 1, 2], [{"struct": load(sup / f"motif_false_seed_{s}.json")["structural_mmd"],
                         "sgin": load(sup / f"motif_false_seed_{s}.json")["third_party_random_gin_structural_features"]["metrics"],
                         "topo": None, "native": None} for s in range(3)])
        sources["Motif=False (common-ref rescore, suppl.)"] = [str(sup / f"motif_false_seed_{s}.json") for s in range(3)]
    sections = [
        ("Structural MMD (70 generated vs 70 test graphs)", [(lab, lambda d, k=k: d["struct"][k]) for lab, k in STRUCT]),
        ("Random-GIN, structural node features (primary PTC protocol), 10 evaluator seeds",
         [(lab, lambda d, k=k: get(d["sgin"], k)) for lab, k in GIN]),
        ("Random-GIN topology-only (constant node input), 10 evaluator seeds",
         [(lab, lambda d, k=k: get(d["topo"], k) if d["topo"] else None) for lab, k in MODE_GIN]),
        ("Random-GIN native node labels (19-way one-hot, decoded_node), 10 evaluator seeds",
         [(lab, lambda d, k=k: get(d["native"], k) if d["native"] else None) for lab, k in MODE_GIN]),
    ]
    return rows, sources, sections


def fmt(x):
    if x is None:
        return "N/A"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.4e}"
    return f"{x:.6f}" if ax < 10 else f"{x:.4f}"


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], None, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def main():
    dataset = sys.argv[1]
    if dataset in ("GRID", "TRIANGULAR_GRID"):
        rows, sources, sections = synthetic_rows(dataset, dataset.lower())
    elif dataset == "PROTEINS":
        rows, sources, sections = proteins_rows()
    elif dataset == "PTC":
        rows, sources, sections = ptc_rows()
    elif dataset == "QM9":
        rows, sources, sections = qm9_rows()
    else:
        raise SystemExit(f"no table definition for {dataset}")
    methods = list(rows)
    table = {"dataset": dataset, "aggregation": "mean and sample SD over generator-training seeds",
             "sources": sources, "sections": []}
    md = [f"# {dataset}: LGD vs GraphVAE motif=False / motif=True full / DeFoG", "",
          "Values: mean ± sample SD across generator-training seeds (seeds in header). "
          "Existing methods are copied from their stored corrected result files (see Sources); "
          "LGD is new. Lower is better except precision, recall and F1-PR. See PROTOCOL.md for protocol and caveats.", ""]
    header = "| Metric | " + " | ".join(f"{m} (seeds {','.join(map(str, rows[m][0]))})" for m in methods) + " |"
    for title, metrics in sections:
        md += [f"## {title}", "", header, "|---|" + "---:|" * len(methods)]
        sec = {"title": title, "metrics": []}
        for label, fn in metrics:
            cells, entry = [], {"metric": label}
            for m in methods:
                vals = []
                for d in rows[m][1]:
                    try:
                        vals.append(None if d is None else fn(d))
                    except (KeyError, TypeError):
                        vals.append(None)
                mean, sd, n = agg(vals)
                entry[m] = {"per_seed": dict(zip(map(str, rows[m][0]), vals)), "mean": mean, "sd": sd, "n": n}
                cells.append("N/A" if mean is None else (fmt(mean) + (f" ± {fmt(sd)}" if sd is not None else " (n=1)")
                                                        + ("" if n == len(rows[m][0]) else f" [n={n}]")))
            md.append(f"| {label} | " + " | ".join(cells) + " |")
            sec["metrics"].append(entry)
        md.append("")
        table["sections"].append(sec)
    md += ["## LGD per-seed values", ""]
    lgd_seeds = rows["LGD"][0]
    md += ["| Section | Metric | " + " | ".join(f"seed {s}" for s in lgd_seeds) + " |",
           "|---|---|" + "---:|" * len(lgd_seeds)]
    for sec in table["sections"]:
        for e in sec["metrics"]:
            ps = e["LGD"]["per_seed"]
            md.append(f"| {sec['title'].split(' (')[0].split(', 10')[0]} | {e['metric']} | " + " | ".join(fmt(ps[str(s)]) for s in lgd_seeds) + " |")
    md += ["", "## Sources", ""]
    for m, fl in sources.items():
        md.append(f"- **{m}**:")
        md += [f"  - `{f}`" for f in fl]
    out = BASE / dataset
    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    (out / "results_table.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(out / "RESULTS.md")


if __name__ == "__main__":
    main()
