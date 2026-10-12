#!/usr/bin/env python3
"""Normalize metric names in the consolidated report without changing values."""

from pathlib import Path


REPORT = Path("/local-scratch2/mirzaei/ALL_DATASETS_ALL_METHODS_FULL_REPORT_20260908.md")


REPLACEMENTS = [
    # Random-GIN families: the name explicitly records the node input.
    ("decoded_node_random_gin.", "random_gin.decoded_node_features."),
    ("third_party_random_gin.", "random_gin.structural_node_features."),
    ("topology_random_gin.", "random_gin.no_node_features."),
    ("local_random_gin.", "random_gin.no_node_features."),
    ("final.third_party_", "random_gin.structural_node_features."),
    ("final.local_", "random_gin.no_node_features."),
    # Structural metric aliases.
    ("final.clustering", "structural_mmd.clustering"),
    ("final.degree", "structural_mmd.degree"),
    ("final.diameter", "structural_mmd.diameter"),
    ("final.orbit", "structural_mmd.orbit"),
    ("final.sparsity", "structural_mmd.sparsity"),
    ("final.spectral", "structural_mmd.spectral"),
    ("final.triangle", "structural_mmd.triangle"),
    ("structural.clustering", "structural_mmd.clustering"),
    ("structural.degree", "structural_mmd.degree"),
    ("structural.diameter", "structural_mmd.diameter"),
    ("structural.orbit", "structural_mmd.orbit"),
    ("structural.sparsity", "structural_mmd.sparsity"),
    ("structural.spectral", "structural_mmd.spectral"),
    ("structural.triangle", "structural_mmd.triangle"),
    ("final.edge_count_absolute_error", "structural_summary.edge_count_absolute_error"),
    ("final.generated_edge_count", "structural_summary.generated_edge_count"),
    ("final.reference_edge_count", "structural_summary.reference_edge_count"),
    ("structural.edge_count_absolute_error", "structural_summary.edge_count_absolute_error"),
    ("structural.generated_edge_count", "structural_summary.generated_edge_count"),
    ("structural.reference_edge_count", "structural_summary.reference_edge_count"),
    # Correlation protocols.
    ("motif_correlation.full_state_tv_20graph", "motif_full_state_tv.heldout_reference.n20"),
    # PTC count aliases to the same names used elsewhere.
    ("soft_gaussian", "rule_metric.soft.gaussian_aligned"),
    ("hard_gaussian", "rule_metric.hard.gaussian_aligned"),
    ("soft_aggregate_rmse", "rule_metric.soft.aggregate_raw_count_rmse"),
    ("hard_aggregate_rmse", "rule_metric.hard.aggregate_raw_count_rmse"),
    ("soft_robust_wasserstein", "rule_metric.soft.robust_log1p_wasserstein"),
    ("hard_robust_wasserstein", "rule_metric.hard.robust_log1p_wasserstein"),
    ("hard_relative_rmse", "rule_metric.hard.relative_count_rmse"),
    ("hard_standardized_mean_rmse", "rule_metric.hard.standardized_mean_count_rmse"),
    ("hard_wasserstein_mean", "rule_metric.hard.log1p_wasserstein_mean"),
    ("hard_wasserstein_median", "rule_metric.hard.log1p_wasserstein_median"),
    ("hard_paired_graph_rmse_mean", "rule_metric.hard.paired_graph_count_rmse_mean"),
    ("hard_paired_graph_rmse_median", "rule_metric.hard.paired_graph_count_rmse_median"),
    ("soft.gaussian_aligned_score", "rule_metric.soft.gaussian_aligned"),
    ("hard.gaussian_aligned_score", "rule_metric.hard.gaussian_aligned"),
    ("soft.aggregate_count_distance", "rule_metric.soft.aggregate_raw_count_rmse"),
    ("hard.aggregate_count_distance", "rule_metric.hard.aggregate_raw_count_rmse"),
    ("soft.robust_count_distance", "rule_metric.soft.robust_log1p_wasserstein"),
    ("hard.robust_count_distance", "rule_metric.hard.robust_log1p_wasserstein"),
    ("hard.relative_count_distance", "rule_metric.hard.relative_count_rmse"),
    ("hard.standardized_mean_vector_count_distance", "rule_metric.hard.standardized_mean_count_rmse"),
    ("hard.log1p_wasserstein_mean", "rule_metric.hard.log1p_wasserstein_mean"),
    ("hard.log1p_wasserstein_median", "rule_metric.hard.log1p_wasserstein_median"),
    ("hard.paired_graph_count_distance_mean", "rule_metric.hard.paired_graph_count_rmse_mean"),
    ("hard.paired_graph_count_distance_median", "rule_metric.hard.paired_graph_count_rmse_median"),
]


def merge_duplicate_table_rows(lines: list[str]) -> list[str]:
    """Merge complementary N/A rows created when two legacy aliases become one."""
    out: list[str] = []
    block: list[str] = []

    def flush() -> None:
        nonlocal block
        if not block:
            return
        seen: dict[str, int] = {}
        merged: list[list[str]] = []
        raw: list[str] = []
        for line in block:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 8 or cells[0] in ("Metric", "---"):
                raw.append(line)
                continue
            key = cells[0]
            if key not in seen:
                seen[key] = len(merged)
                merged.append(cells)
            else:
                dst = merged[seen[key]]
                for i in range(1, 8):
                    if dst[i] in ("N/A", "—") and cells[i] not in ("N/A", "—"):
                        dst[i] = cells[i]
        # Headers always precede data in these report tables.
        out.extend(raw)
        out.extend("| " + " | ".join(c) + " |" for c in merged)
        block = []

    for line in lines:
        if line.startswith("|"):
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return out


text = REPORT.read_text(encoding="utf-8")
for old, new in REPLACEMENTS:
    text = text.replace(old, new)
# Keep the postprocessor idempotent when a standardized report is checked again.
while "rule_metric.rule_metric." in text:
    text = text.replace("rule_metric.rule_metric.", "rule_metric.")

marker = "## Metric conventions\n"
legend = """## Stable metric-name and Random-GIN input conventions

The same metric names are used for every dataset and method. Random-GIN names explicitly identify their node input:

| Stable prefix | Node input | Semantic dataset node features used? |
| --- | --- | --- |
| `random_gin.no_node_features.*` | One constant placeholder per node; adjacency/message passing supplies topology | No |
| `random_gin.structural_node_features.*` | Degree, clustering coefficient, square-clustering coefficient | No; these inputs are derived from topology |
| `random_gin.decoded_node_features.*` | Model-decoded categorical node label | Yes |

Legacy source files may call these families `local_random_gin`/`final.local`, `third_party_random_gin`/`final.third_party`, and `decoded_node_random_gin`, respectively. Those aliases are normalized only in this report; underlying result values are unchanged.

Structural distribution metrics use `structural_mmd.*`; descriptive edge-count summaries use `structural_summary.*`; pruned-rule metrics use `rule_metric.*`. Controlled correlation uses `motif_full_state_tv.heldout_reference.n20`; historical correlation remains separately labeled by protocol in its table.

"""
if "## Stable metric-name and Random-GIN input conventions" not in text:
    text = text.replace(marker, legend + marker)

lines = merge_duplicate_table_rows(text.splitlines())
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(REPORT)
