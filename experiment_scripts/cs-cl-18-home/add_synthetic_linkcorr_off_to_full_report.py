#!/usr/bin/env python3
"""Add original (LinkCorrelation OFF) synthetic results to the 2026-09-08 master report."""

import glob
import json
from pathlib import Path

import build_all_methods_master_report as b


REPORT = Path("/local-scratch2/mirzaei/ALL_DATASETS_ALL_METHODS_FULL_REPORT_20260908.md")
MARKER = "## Synthetic datasets: LinkCorrelation OFF results"
METHODS = b.METHODS


def original_true_standard():
    # Start with the existing false and DeFoG results; replace motif=True ON
    # records with the original/no-link run records.
    current = b.graphvae_standard()
    result = {ds: {m: dict(current[ds][m]) for m in METHODS} for ds in ("grid", "lobster", "triangular_grid")}
    hosts = {"grid": 19, "lobster": 16, "triangular_grid": 19}
    roots = {
        "grid": "/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/grid",
        "lobster": "/local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/lobster",
        "triangular_grid": "/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/triangular_grid",
    }
    for ds in result:
        for method in METHODS:
            for record in result[ds][method].values():
                record.pop("motif_correlation.full_state_tv_20graph", None)
        for method in ("total_count", "full_matrix"):
            result[ds][method] = {}
            for seed in range(3):
                p = f"{roots[ds]}/{method}/seed_{seed}/final_metrics_summary.json"
                result[ds][method][seed] = b.extract_final(b.remote_json(hosts[ds], p))
    return result


def original_counts():
    out = {ds: {m: {} for m in METHODS} for ds in ("grid", "lobster", "triangular_grid")}
    for ds in out:
        for method, stem in (("false", "false"), ("total_count", "total"), ("full_matrix", "full")):
            paths = sorted(glob.glob(
                f"/local-scratch2/mirzaei/gaussian_aligned_v7_collected/**/no_link/{ds}/{stem}_seed*.json",
                recursive=True,
            ))
            for p in paths:
                d = b.load(p); seed = int(d["seed"]); rec = {}
                for state in ("soft", "hard"):
                    for key, value in d[state].items():
                        if isinstance(value, (int, float)) and key != "gaussian_min_log_sigma":
                            rec[f"{state}.{key}"] = value
                out[ds][method][seed] = rec
    return out


text = REPORT.read_text(encoding="utf-8")
if MARKER in text:
    raise SystemExit("LinkCorrelation OFF section already present")

standard = original_true_standard()
counts = original_counts()
names = {"grid": "GRID", "lobster": "LOBSTER", "triangular_grid": "TRIANGULAR_GRID"}
paths = {
    "grid": "system 19: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/grid/{total_count,full_matrix}/seed_<s>",
    "lobster": "system 16: /local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/lobster/{total_count,full_matrix}/seed_<s>",
    "triangular_grid": "system 19: /local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/triangular_grid/{total_count,full_matrix}/seed_<s>",
}

core_count = [
    "soft.gaussian_aligned_score", "hard.gaussian_aligned_score",
    "soft.aggregate_count_distance", "hard.aggregate_count_distance",
    "soft.robust_count_distance", "hard.robust_count_distance",
    "hard.relative_count_distance", "hard.standardized_mean_vector_count_distance",
    "hard.log1p_wasserstein_mean", "hard.log1p_wasserstein_median",
    "hard.paired_graph_count_distance_mean", "hard.paired_graph_count_distance_median",
]

parts = [MARKER, """
The main GRID, LOBSTER, and TRIANGULAR_GRID sections above are the **LinkCorrelation ON** (`LinkCorrelations=1`) motif=True runs. This section adds the earlier **LinkCorrelation OFF** motif=True runs so both rule universes are preserved in one report.

| Rule universe | Full combinations | Retained combinations | Unique rules | Correlation metric |
| --- | ---: | ---: | ---: | --- |
| LinkCorrelation OFF | 2 | 2 | 2 unary edge rules | N/A: no multi-atom rule |
| LinkCorrelation ON | 6 | 4 | 2, including `edges(nodes0,nodes1) AND edges(nodes1,nodes2)` | Full-state TV is defined |

The motif=False and DeFoG columns are reused as topology baselines: neither is trained with the FactorBase LinkCorrelation switch. DeFoG motif-correlation is not reported for OFF because the OFF rule universe has no multi-atom correlation rule. GRID DeFoG remains provisional over seeds 0–1; seed 2 is still training. GraphVAE values use seeds 0–2.
""".strip()]

for ds, label in names.items():
    parts.append(f"\n### {label} — LinkCorrelation OFF")
    parts.append("\n#### Aggregated predictive and structural metrics\n\n" + b.comparison_table(standard[ds]))
    present = [m for m in core_count if any(m in b.metric_aggregates(counts[ds][method]) for method in METHODS)]
    parts.append("\n#### Aggregated exact-rule count metrics\n\n" + b.comparison_table(counts[ds], present))
    parts.append("\n#### Normalized full-state motif correlation\n\n`N/A` for every method under LinkCorrelation OFF: both rules are unary, so there is no joint multi-literal state distribution to compare.")
    parts.append("\n#### Per-seed predictive and structural metrics\n\n" + b.per_seed_tables(standard[ds]))
    rows = []
    for method in METHODS:
        if not counts[ds][method]:
            rows.append([b.METHOD_LABEL[method], "—", "N/A: no compatible count artifact"])
        for seed in sorted(counts[ds][method]):
            values = "; ".join(f"{key}={b.fmt(counts[ds][method][seed].get(key))}" for key in present if key in counts[ds][method][seed])
            rows.append([b.METHOD_LABEL[method], str(seed), values or "N/A"])
    parts.append("\n#### Per-seed exact-rule count metrics\n\n" + b.md_table(["Method", "Seed", "Metrics"], rows))
    parts.append(f"\n#### Original motif=True result/model address\n\n- `{paths[ds]}`\n- Append `/best_validation_mmd_model` for the selected GraphVAE checkpoint.")

parts.append("""
### ON versus OFF interpretation

- Compare ON and OFF motif=True runs as a rule-universe ablation; do not pool their seeds into one aggregate.
- Structural and Random-GIN metrics are directly interpretable within each dataset, subject to the already documented DeFoG frozen-reference mismatch for GRID and TRIANGULAR_GRID.
- Raw count distances cannot be compared numerically across ON and OFF as though they used the same target dimension: OFF has 2 retained combinations, whereas ON has 4 retained combinations.
- Full-state correlation is defined only for ON. An OFF value of `N/A` is mathematically different from zero.
""".strip())

section = "\n\n".join(parts) + "\n\n"
text = text.replace("## DeFoG synthetic outlier sensitivity", section + "## DeFoG synthetic outlier sensitivity")
REPORT.write_text(text, encoding="utf-8")
print(REPORT)
