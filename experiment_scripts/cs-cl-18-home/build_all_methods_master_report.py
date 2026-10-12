#!/usr/bin/env python3
"""Build the consolidated GraphVAE/DeFoG comparison requested on 2026-09-07.

The script reads only existing result artifacts. It deliberately does not invent
values for unfinished or unavailable seeds, and deliberately suppresses the PTC
motif-correlation result while that metric is under review.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


OUT = Path("/local-scratch2/mirzaei/ALL_DATASETS_ALL_METHODS_FULL_REPORT_20260908.md")
MASTER = Path("/local-scratch2/mirzaei/FINAL_ALL_MOTIF_RESULTS_PRUNED_COUNT_DISTANCE.md")
SYNTH_JSON = Path("/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/DEFOG_VS_GRAPHVAE_MOTIF_TRUE_SYNTHETIC_20260906.json")
PTC_JSON = Path("/local-scratch2/mirzaei/defog_ptc_full_metrics_20260907/PTC_DEFOG_VS_MOTIF_TRUE_FALSE_ALL_METRICS_20260907.json")

METHODS = ["false", "total_count", "full_matrix", "defog"]
METHOD_LABEL = {
    "false": "Motif=False",
    "total_count": "Motif=True total",
    "full_matrix": "Motif=True full",
    "defog": "DeFoG",
}


def load(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def remote_json(host: int, path: str):
    raw = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"mirzaei@cs-cl-{host}.cmpt.sfu.ca", "cat", path],
        text=True,
    )
    return json.loads(raw)


def remote_text(host: int, path: str):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"mirzaei@cs-cl-{host}.cmpt.sfu.ca", "cat", path],
        text=True,
    )


def fmt(x):
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    x = float(x)
    if math.isnan(x):
        return "N/A"
    if x == 0:
        return "0"
    if abs(x) >= 1e6 or abs(x) < 1e-5:
        return f"{x:.6e}"
    return f"{x:.6f}"


def agg(vals):
    vals = [float(x) for x in vals if x is not None]
    if not vals:
        return None
    return {"mean": statistics.mean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else None, "n": len(vals), "values": vals}


def afmt(a):
    if not a:
        return "N/A"
    if a["n"] == 1:
        return f"{fmt(a['mean'])} (n=1)"
    return f"{fmt(a['mean'])} ± {fmt(a['sd'])} (n={a['n']})"


def improvement(base, candidate, better):
    if not base or not candidate or better not in ("up", "down"):
        return "—"
    b, c = base["mean"], candidate["mean"]
    delta = c - b if better == "up" else b - c
    if b <= 0 or abs(b) < 1e-15:
        return f"Δbetter {delta:+.6f}"
    return f"{100 * delta / abs(b):+.2f}%"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def extract_final(d):
    out = {}
    t2 = d.get("table2", {})
    for k, v in t2.get("metrics", {}).items():
        out[f"structural.{k}"] = v
    for k, v in t2.get("extra_metrics", {}).items():
        out[f"structural.{k}"] = v
    if "structural.generated_edge_count" in out and "structural.reference_edge_count" in out:
        out["structural.edge_count_absolute_error"] = abs(out["structural.generated_edge_count"] - out["structural.reference_edge_count"])
    local = d.get("table3", {}).get("local_eval_metrics", {})
    for k in ("mmd_rbf", "precision", "recall", "f1_pr"):
        if k in local:
            out[f"local_random_gin.{k}"] = local[k]
    tp = d.get("table3", {}).get("third_party_eval_metrics", {}).get("metrics", {})
    for k in ("mmd_rbf", "precision", "recall", "f1_pr"):
        if k in tp:
            out[f"third_party_random_gin.{k}"] = tp[k].get("mean")
    if "mmd_linear" in tp:
        out["third_party_random_gin.mmd_linear_mean"] = tp["mmd_linear"].get("mean")
        out["third_party_random_gin.mmd_linear_median"] = tp["mmd_linear"].get("median")
        out["third_party_random_gin.mmd_linear_trimmed"] = tp["mmd_linear"].get("trimmed_mean")
    return out


LOG_RE = re.compile(
    r"degree: (?P<degree>\S+) clustering: (?P<clustering>\S+) sparsity: (?P<sparsity>\S+) "
    r"orbits: (?P<orbit>\S+) Spec: (?P<spectral>\S+) Tri: (?P<triangle>\S+) "
    r"average edge # in test set: (?P<reference_edge_count>\S+) average edge # in grnrated set: (?P<generated_edge_count>\S+) "
    r"diameter:(?P<diameter>\S+) mmd_rbf: (?P<mmd_rbf>\S+).*?precision: (?P<precision>\S+).*?"
    r"recall: (?P<recall>\S+).*?f1_pr: (?P<f1_pr>\S+)"
)


def extract_last_log(path):
    text = Path(path).read_text(errors="replace")
    matches = list(LOG_RE.finditer(text))
    if not matches:
        return {}
    d = {k: float(v) for k, v in matches[-1].groupdict().items()}
    out = {}
    for k in ("degree", "clustering", "sparsity", "orbit", "spectral", "triangle", "diameter", "reference_edge_count", "generated_edge_count"):
        out[f"structural.{k}"] = d[k]
    out["structural.edge_count_absolute_error"] = abs(d["generated_edge_count"] - d["reference_edge_count"])
    for k in ("mmd_rbf", "precision", "recall", "f1_pr"):
        out[f"local_random_gin.{k}"] = d[k]
    return out


def repaired_tp_proteins():
    result = {m: {} for m in ("false", "total_count", "full_matrix")}
    mapping = {"Motif-False": "false", "Motif=False": "false", "True total": "total_count", "True full": "full_matrix"}
    with open("/local-scratch2/mirzaei/proteins_third_party_repaired_20260906/summary.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = row["run_dir"]
            if "/total_count/" in p:
                method = "total_count"
            elif "/full_matrix/" in p:
                method = "full_matrix"
            else:
                method = "false"
            seed = int(re.search(r"seed_(\d+)", p).group(1))
            x = {
                "third_party_random_gin.f1_pr": float(row["f1_pr_mean"]),
                "third_party_random_gin.precision": float(row["precision_mean"]),
                "third_party_random_gin.recall": float(row["recall_mean"]),
                "third_party_random_gin.mmd_rbf": float(row["mmd_rbf_mean"]),
                "third_party_random_gin.mmd_linear_mean": float(row["mmd_linear_mean"]),
                "third_party_random_gin.mmd_linear_median": float(row["mmd_linear_median"]),
                "third_party_random_gin.mmd_linear_trimmed": float(row["mmd_linear_trimmed_mean"]),
            }
            result[method][seed] = x
    return result


def graphvae_standard():
    data = {ds: {m: {} for m in METHODS} for ds in ("mutag", "proteins", "grid", "lobster", "triangular_grid", "ptc")}
    # MUTAG baseline and full are local; total is on system 19.
    for s in range(3):
        data["mutag"]["false"][s] = extract_final(load(f"/local-scratch2/new/gather/datasets/mutag/setting_01/seed_{s}/final_metrics_summary.json"))
        data["mutag"]["full_matrix"][s] = extract_final(load(f"/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed/mutag/full_matrix/seed_{s}/final_metrics_summary.json"))
        p = f"/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed/mutag/total_count/seed_{s}/final_metrics_summary.json"
        data["mutag"]["total_count"][s] = extract_final(remote_json(19, p))

    # PROTEINS baseline summaries are local; motif=True final lines are in train logs.
    tp = repaired_tp_proteins()
    for s in range(3):
        data["proteins"]["false"][s] = extract_final(load(f"/local-scratch2/new/gather/datasets/proteins/setting_01/seed_{s}/final_metrics_summary.json"))
        for m, folder in (("total_count", "total_count"), ("full_matrix", "full_matrix")):
            p = f"/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/proteins/{folder}/seed_{s}/train.log"
            data["proteins"][m][s] = extract_last_log(p)
        for m in ("false", "total_count", "full_matrix"):
            data["proteins"][m][s].update(tp[m].get(s, {}))

    # Decoded categorical-node evaluation for MUTAG/PROTEINS, all GraphVAE seeds.
    short = {"false": "false", "total_count": "total", "full_matrix": "full"}
    for ds in ("mutag", "proteins"):
        for m in ("false", "total_count", "full_matrix"):
            for s in range(3):
                p = f"/local-scratch2/mirzaei/node_feature_evaluation_20260901/{ds}/{short[m]}_s{s}/attributed_random_gin.json"
                summary = load(p)["evaluation"]["modes"]["decoded_node"]["summary"]
                for metric, a in summary.items():
                    data[ds][m][s][f"decoded_node_random_gin.{metric}"] = a["mean"]

    # DeFoG MUTAG/PROTEINS: one supplied model seed. Add topology-control and
    # decoded-node Random-GIN values, but do not reinterpret evaluator-repeat SD
    # as generator-training-seed SD.
    with open("/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/structural_preview/summary.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = "proteins" if "proteins" in row["run_dir"].lower() else "mutag"
            rec = data[ds]["defog"].setdefault(0, {})
            for metric, col in (("f1_pr", "f1_pr_mean"), ("precision", "precision_mean"), ("recall", "recall_mean"),
                                ("mmd_rbf", "mmd_rbf_mean"), ("mmd_linear_mean", "mmd_linear_mean"),
                                ("mmd_linear_median", "mmd_linear_median"), ("mmd_linear_trimmed", "mmd_linear_trimmed_mean")):
                rec[f"third_party_random_gin.{metric}"] = float(row[col])
    for ds in ("mutag", "proteins"):
        p = f"/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/preview_random_gin/{ds}/evaluation.json"
        modes = load(p)["evaluation"]["modes"]
        for metric, a in modes["decoded_node"]["summary"].items():
            data[ds]["defog"][0][f"decoded_node_random_gin.{metric}"] = a["mean"]
        for metric, a in modes["topology_control"]["summary"].items():
            data[ds]["defog"][0][f"local_random_gin.{metric}"] = a["mean"]

    # Synthetic link-correlation motif=True and DeFoG results are centralized.
    syn = load(SYNTH_JSON)["datasets"]
    false_hosts = {"grid": 19, "lobster": 16, "triangular_grid": 19}
    false_roots = {
        "grid": "/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/grid",
        "lobster": "/local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/lobster",
        "triangular_grid": "/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/triangular_grid",
    }
    true_hosts = {"grid": 19, "lobster": 16, "triangular_grid": 18}
    true_roots = {
        "grid": "/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/grid",
        "lobster": "/local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/lobster",
        "triangular_grid": "/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/triangular_grid",
    }
    for ds in false_hosts:
        for s in range(3):
            data[ds]["false"][s] = extract_final(remote_json(false_hosts[ds], f"{false_roots[ds]}/seed_{s}/final_metrics_summary.json"))
        for m in ("total_count", "full_matrix", "defog"):
            for s, rec in enumerate(syn[ds]["per_seed"][m]):
                x = {f"structural.{k}": v for k, v in rec["structural"].items()}
                x.update({f"third_party_random_gin.{k}": v for k, v in rec["random_gin"].items()})
                x["motif_correlation.full_state_tv_20graph"] = rec["motif_correlation"]["total_variation"]
                data[ds][m][s] = x
        # The centralized synthetic comparison contains the structural-feature
        # evaluator but omitted the built-in topology-only evaluator. Recover
        # that already-computed result from each original motif=True run.
        for m in ("total_count", "full_matrix"):
            for s in range(3):
                p = f"{true_roots[ds]}/{m}/seed_{s}/final_metrics_summary.json"
                original = extract_final(remote_json(true_hosts[ds], p))
                data[ds][m][s].update({k: v for k, v in original.items() if k.startswith("local_random_gin.")})

    # PTC already has exact per-seed arrays for all four methods.
    ptc = load(PTC_JSON)
    for m in METHODS:
        pm = m
        for metric, a in ptc["held_out_final_metrics"][pm].items():
            for s, v in enumerate(a["values"]):
                data["ptc"][m].setdefault(s, {})[f"final.{metric}"] = v
        for family, prefix in (("topology_control_random_gin", "topology_random_gin"), ("decoded_node_random_gin", "decoded_node_random_gin")):
            for metric, a in ptc[family][pm].items():
                vals = a.get("training_seed_values", [])
                for s, v in enumerate(vals):
                    data["ptc"][m].setdefault(s, {})[f"{prefix}.{metric}"] = v
    return data


def count_data():
    out = {ds: {m: {} for m in METHODS} for ds in ("mutag", "proteins", "grid", "lobster", "triangular_grid", "ptc")}
    universe = {"mutag": "no_link", "proteins": "no_link", "grid": "linkcorr", "lobster": "linkcorr", "triangular_grid": "linkcorr"}
    for ds, u in universe.items():
        for m, file_m in (("false", "false"), ("total_count", "total"), ("full_matrix", "full")):
            files = sorted(glob.glob(f"/local-scratch2/mirzaei/gaussian_aligned_v7_collected/**/{u}/{ds}/{file_m}_seed*.json", recursive=True))
            for p in files:
                d = load(p); s = int(d["seed"]); x = {}
                for state in ("soft", "hard"):
                    for k, v in d[state].items():
                        if isinstance(v, (int, float)) and k not in ("gaussian_min_log_sigma",):
                            x[f"{state}.{k}"] = v
                out[ds][m][s] = x
    ptc = load(PTC_JSON)["exact_pruned_rule_metrics"]
    for m in METHODS:
        for metric, a in ptc[m].items():
            for s, v in enumerate(a["values"]):
                out["ptc"][m].setdefault(s, {})[metric] = v
    return out


def metric_aggregates(per_seed):
    metrics = sorted({k for seed in per_seed.values() for k in seed})
    return {k: agg([per_seed[s].get(k) for s in sorted(per_seed)]) for k in metrics}


def direction(metric):
    if any(x in metric for x in ("precision", "recall", "f1_pr")):
        return "up"
    if any(x in metric for x in ("generated_edge_count", "reference_edge_count")):
        return None
    return "down"


def comparison_table(data_by_method, metrics=None):
    aggs = {m: metric_aggregates(data_by_method[m]) for m in METHODS}
    if metrics is None:
        metrics = sorted({k for m in METHODS for k in aggs[m]})
    rows = []
    for metric in metrics:
        a = {m: aggs[m].get(metric) for m in METHODS}
        better = direction(metric)
        rows.append([
            metric, "↑" if better == "up" else ("↓" if better == "down" else "—"), afmt(a["false"]), afmt(a["total_count"]),
            improvement(a["false"], a["total_count"], better), afmt(a["full_matrix"]), improvement(a["false"], a["full_matrix"], better), afmt(a["defog"]),
        ])
    return md_table(["Metric", "Better", "Motif=False", "True total", "Total improvement", "True full", "Full improvement", "DeFoG"], rows)


def per_seed_tables(data_by_method):
    structural = sorted({k for m in METHODS for r in data_by_method[m].values() for k in r if k.startswith(("structural.", "final.")) and "third_party" not in k})
    evaluator = sorted({k for m in METHODS for r in data_by_method[m].values() for k in r if k not in structural and not k.startswith("motif_correlation.")})
    sections = []
    for title, metrics in (("Structural/final metrics", structural), ("Evaluator metrics", evaluator)):
        if not metrics:
            continue
        rows = []
        for m in METHODS:
            seeds = data_by_method[m]
            if not seeds:
                rows.append([METHOD_LABEL[m], "—", "N/A"])
            for s in sorted(seeds):
                vals = "; ".join(f"{k}={fmt(seeds[s].get(k))}" for k in metrics if k in seeds[s]) or "N/A"
                rows.append([METHOD_LABEL[m], s, vals])
        sections.append(f"#### {title}\n\n" + md_table(["Method", "Seed", "Metrics"], rows))
    return "\n\n".join(sections)


CORR_AGG = {
    "mutag": {"false": (0.146266, 0.004566, 3), "total_count": (0.107568, 0.013488, 3), "full_matrix": (0.119875, 0.009594, 3), "defog": (0.025682, None, 1)},
    "proteins": {"false": (0.124721, 0.009596, 3), "total_count": (0.066547, 0.001737, 3), "full_matrix": (0.067793, 0.017094, 3), "defog": (0.131917, None, 1)},
    "grid": {"false": (0.012910, 0.003124, 3), "total_count": (0.059774, 0.078219, 3), "full_matrix": (0.005437, 0.000625, 3)},
    "lobster": {"false": (0.032247, 0.002789, 3), "total_count": (0.022316, 0.004500, 3), "full_matrix": (0.018693, 0.001786, 3)},
    "triangular_grid": {"false": (0.018245, 0.003490, 3), "total_count": (0.013144, 0.000966, 3), "full_matrix": (0.010917, 0.002605, 3)},
}


def corr_table(ds, standard):
    rows = []
    for m in METHODS:
        if m in CORR_AGG.get(ds, {}):
            mean, sd, n = CORR_AGG[ds][m]
            val = f"{fmt(mean)} (n=1)" if n == 1 else f"{fmt(mean)} ± {fmt(sd)} (n={n})"
            imp = "—"
            if m in ("total_count", "full_matrix"):
                base = {"mean": CORR_AGG[ds]["false"][0]}; cand = {"mean": mean}
                imp = improvement(base, cand, "down")
            rows.append([METHOD_LABEL[m], "historical training-collection complete-state TV", val, imp])
        # The held-out 20-graph protocol is separately available for synthetic true/DeFoG.
        held = [r.get("motif_correlation.full_state_tv_20graph") for r in standard[m].values() if r.get("motif_correlation.full_state_tv_20graph") is not None]
        if held:
            rows.append([METHOD_LABEL[m], "held-out 20-graph full-state TV", afmt(agg(held)), "— (different protocol)"])
    return md_table(["Method", "Protocol", "Full-state distribution TV ↓", "Improvement vs false"], rows)


PATHS = {
    "mutag": {
        "false": "system 18: /local-scratch2/new/gather/datasets/mutag/setting_01/seed_<s>",
        "total_count": "system 19: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed/mutag/total_count/seed_<s>",
        "full_matrix": "system 18: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed/mutag/full_matrix/seed_<s>",
        "defog": "system 18: /local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/mutag (training seed 0 only)",
    },
    "proteins": {
        "false": "system 18: /local-scratch2/new/gather/datasets/proteins/setting_01/seed_<s>",
        "total_count": "system 18: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/proteins/total_count/seed_<s>",
        "full_matrix": "system 18: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/proteins/full_matrix/seed_<s>",
        "defog": "system 18: /local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/proteins (training seed 0 only)",
    },
    "grid": {
        "false": "system 19: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/grid/seed_<s>",
        "total_count": "system 19: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/grid/total_count/seed_<s>",
        "full_matrix": "system 19: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/grid/full_matrix/seed_<s>",
        "defog": "system 18: /local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/frozen_eval/jobs/grid/seed_<s> (seed 2 still training)",
    },
    "lobster": {
        "false": "system 16: /local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/lobster/seed_<s>",
        "total_count": "system 16: /local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/lobster/total_count/seed_<s>",
        "full_matrix": "system 16: /local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/lobster/full_matrix/seed_<s>",
        "defog": "system 16: /local-scratch/mirzaei/defog_frozen_benchmark_20260903/GraphVAE-REQ-full/runs/defog/frozen_eval/jobs/lobster/seed_<s>",
    },
    "triangular_grid": {
        "false": "system 19: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/triangular_grid/seed_<s>",
        "total_count": "system 18: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/triangular_grid/total_count/seed_<s>",
        "full_matrix": "system 18: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/triangular_grid/full_matrix/seed_<s>",
        "defog": "system 19: /local-scratch2/mirzaei/defog_frozen_benchmark_20260903/GraphVAE-REQ-full/runs/defog/frozen_eval/jobs/triangular_grid/seed_<s>",
    },
    "ptc": {
        "false": "system 18: /local-scratch2/new/gather/datasets/ptc/setting_01/seed_<s>",
        "total_count": "system 18: /local-scratch2/mirzaei/motif_true_clean_20260906/ptc/total_count/seed_<s>",
        "full_matrix": "system 18: /local-scratch2/mirzaei/motif_true_clean_20260906/ptc/full_matrix/seed_<s>",
        "defog": "system 18: /local-scratch2/mirzaei/defog_ptc_frozen_20260906/jobs/ptc/seed_<s>",
    },
}


def main():
    standard = graphvae_standard()
    counts = count_data()
    title = "# All datasets and all available methods — full consolidated report\n\n"
    intro = """Generated: 2026-09-08 (America/Vancouver).

This report consolidates **MUTAG, PROTEINS, GRID, LOBSTER, TRIANGULAR_GRID, and PTC** for GraphVAE motif=False, GraphVAE motif=True total-count, GraphVAE motif=True full-matrix, and DeFoG. Values with `n>1` are mean ± **sample** SD across independent training seeds. Improvement is direction-aware and compares each motif=True mean with motif=False; positive is better. Per-seed evaluator values are themselves means over the evaluator's repeated GNN initializations where applicable.

## Critical completeness and fairness notes

| Dataset | False | True total | True full | DeFoG | Important qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| MUTAG | 3 | 3 | 3 | 1 | DeFoG has one training checkpoint; GraphVAE baseline/true reference preprocessing is not perfectly identical. |
| PROTEINS | 3 | 3 | 3 | 1 | DeFoG has one training checkpoint; repaired GraphVAE third-party evaluator equalizes each run internally. |
| GRID LinkCorr=1 | 3 | 3 | 3 | **2 complete** | DeFoG seed 2 is still training; DeFoG aggregate is provisional. Frozen DeFoG reference identity differs from GraphVAE's saved reference. |
| LOBSTER LinkCorr=1 | 3 | 3 | 3 | 3 | Frozen reference identity matches exactly. |
| TRIANGULAR_GRID LinkCorr=1 | 3 | 3 | 3 | 3 | Frozen DeFoG reference identity differs from GraphVAE's saved reference. |
| PTC | 3 | 3 | 3 | 3 | GraphVAE false/true architecture and preprocessing differ; this is not a motif-only ablation. **PTC motif-correlation results are intentionally omitted pending validation.** |

`N/A` means the metric was not computed for that method, not zero. AUC/AP are absent because these runs did not perform link-prediction evaluation. DeFoG MUTAG/PROTEINS evaluator SDs in their source files describe evaluator repeats inside one model seed and are not presented as training-seed uncertainty.

## Metric conventions

- Structural MMD, Random-GIN MMD, count errors, Gaussian NLL, Wasserstein distances, and full-state TV: lower is better.
- Precision, recall, and F1-PR: higher is better.
- Generated/reference edge counts are descriptive.
- Legacy aggregate raw-count RMSE sums every retained rule-state count over all graphs before comparing the generated/reference vectors.
- Soft/hard Gaussian-aligned scores reproduce the calibrated Gaussian count-loss family on decoder probabilities / thresholded graphs.
- Robust count distance is the upper-10%-trimmed mean of per-rule 1-D Wasserstein distances on `log1p` graph-wise counts.
- Full-state distribution TV sums counts over each whole graph collection, normalizes within each complete multi-atom rule state table, computes TV, then averages rules. This is the custom FactorBase-aligned correlation diagnostic, not the unfinished directed PTC paper metric.
"""
    parts = [title, intro]
    names = {"mutag": "MUTAG", "proteins": "PROTEINS", "grid": "GRID", "lobster": "LOBSTER", "triangular_grid": "TRIANGULAR_GRID", "ptc": "PTC"}
    core_count = [
        "soft.gaussian_aligned_score", "hard.gaussian_aligned_score", "soft.aggregate_count_distance", "hard.aggregate_count_distance",
        "soft.robust_count_distance", "hard.robust_count_distance", "hard.relative_count_distance", "hard.standardized_mean_vector_count_distance",
        "hard.log1p_wasserstein_mean", "hard.log1p_wasserstein_median", "hard.paired_graph_count_distance_mean", "hard.paired_graph_count_distance_median",
        "soft_gaussian", "hard_gaussian", "soft_aggregate_rmse", "hard_aggregate_rmse", "soft_robust_wasserstein", "hard_robust_wasserstein",
        "hard_relative_rmse", "hard_standardized_mean_rmse", "hard_wasserstein_mean", "hard_wasserstein_median", "hard_paired_graph_rmse_mean", "hard_paired_graph_rmse_median",
    ]
    for ds in names:
        suffix = " — LinkCorrelation ON" if ds in ("grid", "lobster", "triangular_grid") else ""
        parts.append(f"\n## {names[ds]}{suffix}\n")
        if suffix:
            parts.append("`LinkCorrelations=1` applies only to motif=True total-count and full-matrix. Motif=False and DeFoG are unchanged baselines reused for both LinkCorrelation ON and OFF comparisons.\n")
        parts.append("### Aggregated predictive and structural results\n\n" + comparison_table(standard[ds]))
        existing_count_metrics = [m for m in core_count if any(m in metric_aggregates(counts[ds][x]) for x in METHODS)]
        parts.append("\n### Aggregated pruned-rule count results\n\n" + comparison_table(counts[ds], existing_count_metrics))
        if ds == "ptc":
            parts.append("\n### Motif correlation\n\n**Intentionally omitted.** The PTC directed motif-correlation calculation remains under validation and must not be cited from this report.")
        else:
            parts.append("\n### Normalized full-state motif distribution/correlation\n\n" + corr_table(ds, standard[ds]))
        parts.append("\n### Per-seed predictive and structural results\n\n" + per_seed_tables(standard[ds]))
        count_rows = []
        for m in METHODS:
            if not counts[ds][m]:
                count_rows.append([METHOD_LABEL[m], "—", "N/A"])
            for s in sorted(counts[ds][m]):
                vals = "; ".join(f"{k}={fmt(counts[ds][m][s].get(k))}" for k in existing_count_metrics if k in counts[ds][m][s]) or "N/A"
                count_rows.append([METHOD_LABEL[m], s, vals])
        parts.append("\n#### Per-seed pruned-rule count results\n\n" + md_table(["Method", "Seed", "Metrics"], count_rows))
        if ds != "ptc":
            corr_rows = []
            for m in METHODS:
                vals = [(s, r.get("motif_correlation.full_state_tv_20graph")) for s, r in standard[ds][m].items() if r.get("motif_correlation.full_state_tv_20graph") is not None]
                if vals:
                    corr_rows.extend([[METHOD_LABEL[m], s, fmt(v), "held-out 20-graph protocol"] for s, v in vals])
                elif m in CORR_AGG.get(ds, {}):
                    corr_rows.append([METHOD_LABEL[m], "not retained centrally", "see aggregate", "historical training-collection protocol"])
                else:
                    corr_rows.append([METHOD_LABEL[m], "—", "N/A", "not computed"])
            parts.append("\n#### Per-seed correlation availability\n\n" + md_table(["Method", "Seed", "Full-state TV", "Protocol/status"], corr_rows))
        path_rows = [[METHOD_LABEL[m], PATHS[ds][m], PATHS[ds][m].replace("seed_<s>", "seed_<s>/best_validation_mmd_model") if m != "defog" else PATHS[ds][m]] for m in METHODS]
        parts.append("\n### Result and model addresses\n\n" + md_table(["Method", "Result directory", "Best model/checkpoint"], path_rows))

    sources = """
## DeFoG synthetic outlier sensitivity

The primary tables above retain every completed seed. A separate sensitivity analysis found two broad, multi-metric anomalous DeFoG seeds:

| Dataset | Primary DeFoG seeds | Sensitivity exclusion | F1-PR before | F1-PR after | Status |
| --- | --- | --- | ---: | ---: | --- |
| GRID | 0, 1 | seed 1 | 0.466860 ± 0.506842 | 0.825252 (seed 0 only) | Exploratory only; `n=1`, while seed 2 is still training |
| LOBSTER | 0, 1, 2 | none | 0.970886 ± 0.024528 | unchanged | No broad failed seed |
| TRIANGULAR_GRID | 0, 1, 2 | seed 2 | 0.392299 ± 0.339927 | 0.588440 ± 0.016584 | Exploratory two-seed sensitivity result |

The complete filtered Random-GIN, structural-MMD, and motif-TV tables are in `/local-scratch/localhome/mirzaei/DEFOG_VS_GRAPHVAE_MOTIF_TRUE_SYNTHETIC_20260906.md`. These filtered values must not replace the all-seed primary results without a predefined failure rule.

## Source-of-truth artifacts

- GraphVAE master (all legacy/new count definitions, configurations, rules, features, and model paths): `/local-scratch2/mirzaei/FINAL_ALL_MOTIF_RESULTS_PRUNED_COUNT_DISTANCE.md`
- Motif=False gather archive: `/local-scratch2/new/gather/`
- GraphVAE Gaussian/count per-seed JSON: `/local-scratch2/mirzaei/gaussian_aligned_v7_collected/`
- Repaired PROTEINS third-party metrics: `/local-scratch2/mirzaei/proteins_third_party_repaired_20260906/summary.csv`
- MUTAG/PROTEINS decoded-node GraphVAE evaluation: `/local-scratch2/mirzaei/node_feature_evaluation_20260901/`
- MUTAG/PROTEINS DeFoG topology evaluator: `/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/structural_preview/summary.csv`
- MUTAG/PROTEINS DeFoG decoded-node evaluator: `/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/preview_random_gin/`
- MUTAG/PROTEINS DeFoG full-state TV: `/local-scratch2/mirzaei/defog_full_state_motif_20260902/`
- Synthetic machine-readable comparison and per-seed files: `/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/`
- Synthetic all-seed and outlier-sensitivity report: `/local-scratch/localhome/mirzaei/DEFOG_VS_GRAPHVAE_MOTIF_TRUE_SYNTHETIC_20260906.md`
- PTC machine-readable comparison and per-seed files: `/local-scratch2/mirzaei/defog_ptc_full_metrics_20260907/`

## Interpretation guardrails

1. Do not treat DeFoG MUTAG/PROTEINS `n=1` as a three-seed estimate.
2. Do not treat GRID DeFoG `n=2` as final while seed 2 is training.
3. Do not compare historical training-collection TV directly with held-out 20-graph TV; both are shown with protocol labels.
4. Do not claim PTC motif correlation from this file; it is omitted by request.
5. PTC motif=False versus motif=True differs in latent size, batch size, edge-feature loss, and preprocessing, so improvements are descriptive rather than a pure motif-loss causal effect.
"""
    parts.append(sources)
    OUT.write_text("\n\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
