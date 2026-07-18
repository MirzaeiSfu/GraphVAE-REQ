#!/usr/bin/env python3
"""Summarize exact LOBSTER setting-6 reproduction runs.

The report intentionally separates:
  - validation generated samples from mmd.log,
  - the best-validation metadata,
  - final test evaluation after loading the best-validation model.

That distinction matters because the strong historical setting-6 number was
recorded as a post-hoc evaluation of a saved validation generated sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ranking_score import compute_validation_mmd_score  # noqa: E402
DEFAULT_ROOT = ROOT / "collected_runs/20260708/lobster_setting6_exact_no_guard"
DEFAULT_OUT_CSV = ROOT / "reports/lobster_setting6_exact_no_guard_reproduction_20260708.csv"
DEFAULT_OUT_MD = ROOT / "reports/lobster_setting6_exact_no_guard_reproduction_20260708.md"
GUARD_COMPARISON_CSV = (
    ROOT / "reports/lobster_graphvae_original_temp_guard_sweep_comparison_20260708.csv"
)

MMD_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MMD_RESULT_LABELS = {
    "degree": "degree",
    "clustering": "clustering",
    "orbit": "orbits",
    "spectral": "Spec",
    "triangle": "Tri",
    "sparsity": "sparsity",
    "diameter": "diameter",
    "mmd_rbf": "mmd_rbf",
    "mmd_rbf_std": "mmd_rbf_std",
    "precision": "precision",
    "precision_std": "precision_std",
    "recall": "recall",
    "recall_std": "recall_std",
    "f1_pr": "f1_pr",
    "f1_pr_std": "f1_pr_std",
}
MMD_EDGE_COUNT_LABELS = {
    "reference_edge_count": "average edge # in test set",
    "generated_edge_count": "average edge # in grnrated set",
}

FIELDNAMES = [
    "Source",
    "Run",
    "Row Type",
    "Setting",
    "Replicate",
    "Epoch",
    "Split",
    "Model Source",
    "Normalized Score",
    "Score Mode",
    "Degree",
    "Clustering",
    "Orbit",
    "Spectral",
    "Diameter",
    "MMD RBF",
    "MMD RBF Std",
    "MMD Linear",
    "MMD Linear Std",
    "MMD Linear Median",
    "MMD Linear Trimmed Mean",
    "Precision",
    "Precision Std",
    "Recall",
    "Recall Std",
    "F1-PR",
    "F1-PR Std",
    "Avg Edges Test",
    "Avg Edges Generated",
    "Generated Graphs",
    "Reference Graphs",
    "Run Dir",
    "Notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--dataset", default="LOBSTER")
    parser.add_argument("--score-mode", default="normalized_table2_table3")
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=GUARD_COMPARISON_CSV,
        help="Existing consolidated CSV used to import the historical setting-6 row.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return value


def base_row() -> dict[str, object]:
    return {name: "" for name in FIELDNAMES}


def parse_graph_quality_result(mmd_result: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for metric_name, result_label in MMD_RESULT_LABELS.items():
        match = re.search(
            rf"{re.escape(result_label)}\s*:\s*({MMD_FLOAT_PATTERN})",
            str(mmd_result),
        )
        metrics[metric_name] = float(match.group(1)) if match else None
    for metric_name, result_label in MMD_EDGE_COUNT_LABELS.items():
        match = re.search(
            rf"{re.escape(result_label)}\s*:\s*({MMD_FLOAT_PATTERN})",
            str(mmd_result),
        )
        metrics[metric_name] = float(match.group(1)) if match else None
    return metrics


def score_metrics(metrics: dict, score_mode: str, dataset: str):
    score_input = {
        "degree": maybe_float(metrics.get("degree")),
        "clustering": maybe_float(metrics.get("clustering")),
        "orbit": maybe_float(metrics.get("orbit")),
        "spectral": maybe_float(metrics.get("spectral")),
        "diameter": maybe_float(metrics.get("diameter")),
        "mmd_rbf": maybe_float(metrics.get("mmd_rbf")),
        "f1_pr": maybe_float(metrics.get("f1_pr")),
    }
    if any(value is None for value in score_input.values()):
        return None
    return compute_validation_mmd_score(score_input, score_mode, dataset)


def fill_metric_row(row: dict[str, object], metrics: dict, score_mode: str, dataset: str) -> None:
    row["Degree"] = metrics.get("degree")
    row["Clustering"] = metrics.get("clustering")
    row["Orbit"] = metrics.get("orbit")
    row["Spectral"] = metrics.get("spectral")
    row["Diameter"] = metrics.get("diameter")
    row["MMD RBF"] = metrics.get("mmd_rbf")
    row["MMD RBF Std"] = metrics.get("mmd_rbf_std")
    row["Precision"] = metrics.get("precision")
    row["Precision Std"] = metrics.get("precision_std")
    row["Recall"] = metrics.get("recall")
    row["Recall Std"] = metrics.get("recall_std")
    row["F1-PR"] = metrics.get("f1_pr")
    row["F1-PR Std"] = metrics.get("f1_pr_std")
    row["Avg Edges Test"] = metrics.get("reference_edge_count")
    row["Avg Edges Generated"] = metrics.get("generated_edge_count")
    row["Normalized Score"] = score_metrics(metrics, score_mode, dataset)
    row["Score Mode"] = score_mode


def third_party_metric(third_party_metrics: dict | None, name: str, stat: str = "mean"):
    if not third_party_metrics:
        return None
    metric = third_party_metrics.get("metrics", {}).get(name)
    if not isinstance(metric, dict):
        return None
    return metric.get(stat)


def discover_run_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    candidates = set()
    for marker in ("mmd.log", "best_validation_mmd.json", "final_metrics_summary.json"):
        for path in root_dir.rglob(marker):
            candidates.add(path.parent)
    return sorted(candidates)


def replicate_from_name(name: str) -> str:
    match = re.search(r"rep([0-9]+)", name)
    return match.group(1) if match else ""


def rows_from_mmd_log(run_dir: Path, score_mode: str, dataset: str) -> list[dict[str, object]]:
    path = run_dir / "mmd.log"
    if not path.is_file():
        return []
    rows = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if " @ Val @ , " not in raw_line:
            continue
        step_text, val_text = raw_line.split(" @ Val @ , ", 1)
        step_match = re.match(r"\s*([0-9]+)\s+@", step_text)
        epoch = step_match.group(1) if step_match else ""
        metrics = parse_graph_quality_result(val_text)
        row = base_row()
        row.update(
            {
                "Source": "setting6_exact_no_guard",
                "Run": run_dir.name,
                "Row Type": "validation_generated_sample",
                "Setting": "06",
                "Replicate": replicate_from_name(run_dir.name),
                "Epoch": epoch,
                "Split": "validation",
                "Model Source": "validation_epoch_model",
                "Generated Graphs": str(
                    run_dir / f"Single_comp_generatedGraphs_adj_{int(epoch) - 1}.npy"
                )
                if str(epoch).isdigit()
                else "",
                "Reference Graphs": str(run_dir / "testGraphs_adj_.npy"),
                "Run Dir": str(run_dir),
            }
        )
        fill_metric_row(row, metrics, score_mode, dataset)
        rows.append(row)
    return rows


def row_from_best_validation(run_dir: Path, score_mode: str, dataset: str) -> dict[str, object] | None:
    path = run_dir / "best_validation_mmd.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    metrics = payload.get("metrics", {})
    row = base_row()
    row.update(
        {
            "Source": "setting6_exact_no_guard",
            "Run": run_dir.name,
            "Row Type": "best_validation_metadata",
            "Setting": "06",
            "Replicate": replicate_from_name(run_dir.name),
            "Epoch": payload.get("epoch_1_based", ""),
            "Split": "validation",
            "Model Source": "best_validation_mmd_model",
            "Generated Graphs": payload.get("validation_generated_graphs", ""),
            "Reference Graphs": str(run_dir / "testGraphs_adj_.npy"),
            "Run Dir": str(run_dir),
            "Notes": "Same validation sample as the best-validation checkpoint metadata.",
        }
    )
    fill_metric_row(row, metrics, score_mode, dataset)
    if payload.get("score") is not None:
        row["Normalized Score"] = payload.get("score")
    if payload.get("score_mode"):
        row["Score Mode"] = payload.get("score_mode")
    return row


def row_from_final_eval(run_dir: Path, score_mode: str, dataset: str) -> dict[str, object] | None:
    table2_path = run_dir / "final_table2_metrics.json"
    table3_path = run_dir / "final_table3_metrics.json"
    if not table2_path.is_file() or not table3_path.is_file():
        return None
    table2 = load_json(table2_path)
    table3 = load_json(table3_path)
    table2_metrics = table2.get("metrics", {})
    table2_extra = table2.get("extra_metrics", {})
    local_table3 = table3.get("local_eval_metrics", {})
    third_party = table3.get("third_party_eval_metrics")
    metrics = {
        **table2_metrics,
        "mmd_rbf": third_party_metric(third_party, "mmd_rbf") or local_table3.get("mmd_rbf"),
        "mmd_rbf_std": third_party_metric(third_party, "mmd_rbf", "std")
        or local_table3.get("mmd_rbf_std"),
        "precision": third_party_metric(third_party, "precision") or local_table3.get("precision"),
        "precision_std": third_party_metric(third_party, "precision", "std")
        or local_table3.get("precision_std"),
        "recall": third_party_metric(third_party, "recall") or local_table3.get("recall"),
        "recall_std": third_party_metric(third_party, "recall", "std")
        or local_table3.get("recall_std"),
        "f1_pr": third_party_metric(third_party, "f1_pr") or local_table3.get("f1_pr"),
        "f1_pr_std": third_party_metric(third_party, "f1_pr", "std")
        or local_table3.get("f1_pr_std"),
        "reference_edge_count": table2_extra.get("reference_edge_count"),
        "generated_edge_count": table2_extra.get("generated_edge_count"),
    }
    row = base_row()
    row.update(
        {
            "Source": "setting6_exact_no_guard",
            "Run": run_dir.name,
            "Row Type": "final_test_eval",
            "Setting": "06",
            "Replicate": replicate_from_name(run_dir.name),
            "Split": "test",
            "Model Source": table2.get("model_source", ""),
            "Generated Graphs": table2.get("generated_graphs", ""),
            "Reference Graphs": table2.get("reference_graphs", ""),
            "Run Dir": str(run_dir),
        }
    )
    fill_metric_row(row, metrics, score_mode, dataset)
    row["MMD Linear"] = third_party_metric(third_party, "mmd_linear")
    row["MMD Linear Std"] = third_party_metric(third_party, "mmd_linear", "std")
    row["MMD Linear Median"] = third_party_metric(third_party, "mmd_linear", "median")
    row["MMD Linear Trimmed Mean"] = third_party_metric(
        third_party, "mmd_linear", "trimmed_mean"
    )
    return row


def historical_setting6_row(comparison_csv: Path) -> dict[str, object] | None:
    if not comparison_csv.is_file():
        return None
    with comparison_csv.open(newline="", encoding="utf-8") as handle:
        for source_row in csv.DictReader(handle):
            if (
                source_row.get("Source") == "lobster_docx_20260708"
                and source_row.get("Setting") == "06"
            ):
                row = base_row()
                row.update(
                    {
                        "Source": source_row.get("Source", ""),
                        "Run": source_row.get("Model", ""),
                        "Row Type": "historical_docx_posthoc",
                        "Setting": "06",
                        "Split": "saved validation sample",
                        "Model Source": "post-hoc saved generated sample",
                        "Normalized Score": source_row.get("Normalized Score", ""),
                        "Score Mode": source_row.get("Normalized Score Source", ""),
                        "Degree": source_row.get("Degree", ""),
                        "Clustering": source_row.get("Clustering", ""),
                        "Orbit": source_row.get("Orbit", ""),
                        "Spectral": source_row.get("Spectral", ""),
                        "Diameter": source_row.get("Diameter", ""),
                        "MMD RBF": source_row.get("MMD RBF", ""),
                        "MMD Linear": source_row.get("MMD Linear", ""),
                        "Precision": source_row.get("Precision", ""),
                        "Recall": source_row.get("Recall", ""),
                        "F1-PR": source_row.get("F1-PR", ""),
                        "Avg Edges Test": source_row.get("Avg Edges Test", ""),
                        "Avg Edges Generated": source_row.get("Avg Edges Generated", ""),
                        "Notes": source_row.get("Notes", ""),
                    }
                )
                return row
    return None


def sort_key(row: dict[str, object]):
    source_order = {
        "historical_docx_posthoc": 0,
        "best_validation_metadata": 1,
        "validation_generated_sample": 2,
        "final_test_eval": 3,
    }
    score = maybe_float(row.get("Normalized Score"))
    return (
        source_order.get(str(row.get("Row Type")), 99),
        score if score is not None else float("inf"),
        str(row.get("Run")),
        str(row.get("Epoch")),
    )


def write_outputs(rows: list[dict[str, object]], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: fmt(row.get(name, "")) for name in FIELDNAMES})

    best_rows = sorted(
        [row for row in rows if row.get("Row Type") != "validation_generated_sample"],
        key=sort_key,
    )
    val_rows = sorted(
        [row for row in rows if row.get("Row Type") == "validation_generated_sample"],
        key=sort_key,
    )[:10]
    lines = [
        "# LOBSTER Setting 6 Exact No-Guard Reproduction",
        "",
        f"Rows written: {len(rows)}",
        f"CSV: `{out_csv}`",
        "",
        "## Key Rows",
        "",
        "| Source | Run | Type | Epoch | Score | Edges Gen | MMD RBF | F1-PR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows[:20]:
        lines.append(
            "| {Source} | {Run} | {Row Type} | {Epoch} | {Normalized Score} | "
            "{Avg Edges Generated} | {MMD RBF} | {F1-PR} |".format(
                **{name: fmt(row.get(name, "")) for name in FIELDNAMES}
            )
        )
    lines.extend(
        [
            "",
            "## Best Validation Samples By Score",
            "",
            "| Run | Epoch | Score | Edges Gen | Degree | Clustering | Orbit | Spectral | Diameter | MMD RBF | F1-PR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in val_rows:
        values = {name: fmt(row.get(name, "")) for name in FIELDNAMES}
        lines.append(
            "| {Run} | {Epoch} | {Normalized Score} | {Avg Edges Generated} | "
            "{Degree} | {Clustering} | {Orbit} | {Spectral} | {Diameter} | "
            "{MMD RBF} | {F1-PR} |".format(**values)
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root_dir = args.root_dir.expanduser()
    rows: list[dict[str, object]] = []
    historical = historical_setting6_row(args.comparison_csv.expanduser())
    if historical is not None:
        rows.append(historical)

    for run_dir in discover_run_dirs(root_dir):
        final_row = row_from_final_eval(run_dir, args.score_mode, args.dataset)
        if final_row is not None:
            rows.append(final_row)
        best_row = row_from_best_validation(run_dir, args.score_mode, args.dataset)
        if best_row is not None:
            rows.append(best_row)
        rows.extend(rows_from_mmd_log(run_dir, args.score_mode, args.dataset))

    rows = sorted(rows, key=sort_key)
    write_outputs(rows, args.out_csv.expanduser(), args.out_md.expanduser())
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print(f"Wrote summary to {args.out_md}")
    if not discover_run_dirs(root_dir):
        print(f"No reproduction run directories found yet under {root_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
