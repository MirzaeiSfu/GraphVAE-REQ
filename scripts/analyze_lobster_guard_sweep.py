#!/usr/bin/env python3
"""Build a consolidated LOBSTER guarded loss-weight sweep comparison report."""

from __future__ import annotations

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "collected_runs/20260708/lobster_graphvae_original_temp_guard_sweep"
MANIFEST = ROOT / "configs/loss_weight_sweeps/lobster_graphvae_motif_original_temp_guard/manifest.csv"
LOBSTER_DOCX = ROOT / "reports/lobster.docx"
LOSS_SWEEP_DOCX = ROOT / "reports/lobster_loss_weight_sweep_results_20260708.docx"

OUT_CSV = ROOT / "reports/lobster_graphvae_original_temp_guard_sweep_comparison_20260708.csv"
OUT_MD = ROOT / "reports/lobster_graphvae_original_temp_guard_sweep_analysis_20260708.md"

DENOMS = {
    "degree": 0.081,
    "clustering": 0.739,
    "orbit": 0.372,
    "spectral": 0.056,
    "diameter": 0.129,
    "mmd_rbf": 0.1,
    "f1_pr_error": 0.05,
}
SCORE_METRICS = ("degree", "clustering", "orbit", "spectral", "diameter", "mmd_rbf", "f1_pr")
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
LOBSTER_DOCX_CAVEATS = {
    "06": (
        "Source metrics CSV says this was a post-hoc eval on a saved best-validation "
        "generated graph sample; the source run crashed before normal finalization."
    ),
}
LOBSTER_DOCX_EDGE_COUNTS = {
    "06": {"Avg Edges Generated": 28.7, "Avg Edges Test": 45.4},
}


FIELDNAMES = [
    "Source",
    "Setting",
    "Model",
    "Dataset",
    "Status",
    "Rank",
    "Normalized Score",
    "Normalized Score Source",
    "Best Val Score",
    "Best Val Epoch",
    "Best Val Degree",
    "Best Val Clustering",
    "Best Val Orbit",
    "Best Val Spectral",
    "Best Val Diameter",
    "Best Val MMD RBF",
    "Best Val Precision",
    "Best Val Recall",
    "Best Val F1-PR",
    "alpha_node_feat",
    "alpha_edge_feat",
    "alpha_motif_loss",
    "alpha_syntactic_literal_motif_loss",
    "alpha_kernel_cost",
    "alpha_adj_recon",
    "rule_mode",
    "motif_temperature_start",
    "motif_temperature_end",
    "motif_temperature_guard_ratio",
    "motif_temperature_guard_relax_factor",
    "motif_temperature_guard_sharpen_factor",
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
    "MMD Linear Min",
    "MMD Linear Max",
    "MMD Linear Mean/Median",
    "Precision",
    "Precision Std",
    "Recall",
    "Recall Std",
    "F1-PR",
    "F1-PR Std",
    "Local MMD RBF",
    "Local MMD RBF Std",
    "Local Precision",
    "Local Precision Std",
    "Local Recall",
    "Local Recall Std",
    "Local F1-PR",
    "Local F1-PR Std",
    "Triangles",
    "Sparsity",
    "Avg Edges Test",
    "Avg Edges Generated",
    "Num Generated Graphs",
    "Num Reference Graphs",
    "Repeats",
    "Run Dir",
    "Config",
    "Notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def docx_cell_text(cell) -> str:
    return "".join(t.text or "" for t in cell.findall(".//w:t", WORD_NS)).strip()


def extract_docx_tables(path: Path) -> list[list[list[str]]]:
    with ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    tables = []
    for tbl in root.findall(".//w:tbl", WORD_NS):
        rows = []
        for tr in tbl.findall("./w:tr", WORD_NS):
            rows.append([docx_cell_text(tc) for tc in tr.findall("./w:tc", WORD_NS)])
        tables.append(rows)
    return tables


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def as_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def metric(metrics: dict, name: str, stat: str = "mean"):
    value = metrics.get(name)
    if isinstance(value, dict):
        return value.get(stat)
    return value


def coalesce(*values):
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def normalized_score(values: dict[str, object]) -> float | None:
    components = []
    for name in SCORE_METRICS:
        value = as_float(values.get(name))
        if value is None:
            return None
        if name == "f1_pr":
            component = max(0.0, (1.0 - value) / DENOMS["f1_pr_error"])
        else:
            component = value / DENOMS[name]
        components.append(min(component, 10.0))
    return sum(components) / len(components)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.12g}"
    return value


def base_row() -> dict[str, object]:
    return {name: "" for name in FIELDNAMES}


def parse_doc_number(value: str, *, field: str = "", notes: list[str] | None = None):
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if field == "f1_pr" and number > 1.0 and text.isdigit():
        fixed = float("0." + text)
        if notes is not None:
            notes.append(f"DOCX F1 PR cell `{text}` interpreted as {fixed:.6g}")
        return fixed
    return number


def first_note(*notes: str) -> str:
    return "; ".join(note for note in notes if note)


def run_prefix(run_dir: Path) -> str:
    return run_dir.name.split("__", 1)[0]


def read_tail(path: Path, max_bytes: int = 40000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-max_bytes:].decode(errors="replace")


def infer_run_note(run_dir: Path, has_final_eval: bool) -> str:
    notes = []
    stdout_tail = read_tail(run_dir / "stdout.log")
    train_tail = read_tail(run_dir / "train.log")
    combined = stdout_tail + "\n" + train_tail
    if not has_final_eval:
        notes.append("missing final test eval")
    if re.search(r"\bloss?:\s*0*nan\b|Binary_Cross_Entropy:nan|KL-D:nan", combined):
        notes.append("training log contains NaN loss")
    if "Traceback" in stdout_tail:
        if "StopIteration" in stdout_tail:
            notes.append("crashed during plotting with StopIteration")
        else:
            notes.append("stdout contains traceback")
    return first_note(*notes)


def load_manifest() -> dict[str, dict[str, str]]:
    return {row["name"]: row for row in read_csv(MANIFEST)}


def add_best_validation(row: dict[str, object], run_dir: Path) -> None:
    path = run_dir / "best_validation_mmd.json"
    if not path.exists():
        return
    best = load_json(path)
    row["Best Val Score"] = best.get("score")
    row["Best Val Epoch"] = best.get("epoch_1_based") or best.get("step")
    metrics = best.get("metrics", {})
    row["Best Val Degree"] = metrics.get("degree")
    row["Best Val Clustering"] = metrics.get("clustering")
    row["Best Val Orbit"] = metrics.get("orbit")
    row["Best Val Spectral"] = metrics.get("spectral")
    row["Best Val Diameter"] = metrics.get("diameter")
    row["Best Val MMD RBF"] = metrics.get("mmd_rbf")
    row["Best Val Precision"] = metrics.get("precision")
    row["Best Val Recall"] = metrics.get("recall")
    row["Best Val F1-PR"] = metrics.get("f1_pr")


def build_guard_rows() -> list[dict[str, object]]:
    manifest = load_manifest()
    rows = []
    for run_dir in sorted(p for p in SWEEP_DIR.iterdir() if p.is_dir()):
        prefix = run_prefix(run_dir)
        config = manifest.get(prefix, {})
        final_path = run_dir / "final_metrics_summary.json"
        has_final = final_path.exists()

        row = base_row()
        row["Source"] = "guard_sweep_20260708"
        row["Setting"] = prefix.replace("lobster_graphvae_original_temp_guard_", "")
        row["Model"] = "GraphVAE+Motif original temp guard"
        row["Dataset"] = config.get("dataset", "LOBSTER")
        row["Status"] = "evaluated" if has_final else "partial_no_final_eval"
        row["Run Dir"] = str(run_dir.relative_to(ROOT))
        row["Config"] = config.get("config", "")
        row["Notes"] = infer_run_note(run_dir, has_final)

        for key in (
            "alpha_node_feat",
            "alpha_edge_feat",
            "alpha_motif_loss",
            "alpha_syntactic_literal_motif_loss",
            "alpha_kernel_cost",
            "alpha_adj_recon",
            "rule_mode",
            "motif_temperature_start",
            "motif_temperature_end",
            "motif_temperature_guard_ratio",
            "motif_temperature_guard_relax_factor",
            "motif_temperature_guard_sharpen_factor",
        ):
            row[key] = config.get(key, "")

        add_best_validation(row, run_dir)

        if has_final:
            summary = load_json(final_path)
            table2 = summary.get("table2", {})
            table2_metrics = table2.get("metrics", {})
            extra = table2.get("extra_metrics", {})
            table3 = summary.get("table3", {})
            local = table3.get("local_eval_metrics", {})
            third = table3.get("third_party_eval_metrics", {})
            third_metrics = third.get("metrics", {})

            row["Degree"] = table2_metrics.get("degree")
            row["Clustering"] = table2_metrics.get("clustering")
            row["Orbit"] = table2_metrics.get("orbit")
            row["Spectral"] = table2_metrics.get("spectral")
            row["Diameter"] = table2_metrics.get("diameter")
            row["Triangles"] = extra.get("triangle")
            row["Sparsity"] = extra.get("sparsity")
            row["Avg Edges Test"] = extra.get("reference_edge_count")
            row["Avg Edges Generated"] = extra.get("generated_edge_count")

            row["Local MMD RBF"] = local.get("mmd_rbf")
            row["Local MMD RBF Std"] = local.get("mmd_rbf_std")
            row["Local Precision"] = local.get("precision")
            row["Local Precision Std"] = local.get("precision_std")
            row["Local Recall"] = local.get("recall")
            row["Local Recall Std"] = local.get("recall_std")
            row["Local F1-PR"] = local.get("f1_pr")
            row["Local F1-PR Std"] = local.get("f1_pr_std")

            row["MMD RBF"] = coalesce(metric(third_metrics, "mmd_rbf"), local.get("mmd_rbf"))
            row["MMD RBF Std"] = coalesce(metric(third_metrics, "mmd_rbf", "std"), local.get("mmd_rbf_std"))
            row["Precision"] = coalesce(metric(third_metrics, "precision"), local.get("precision"))
            row["Precision Std"] = coalesce(metric(third_metrics, "precision", "std"), local.get("precision_std"))
            row["Recall"] = coalesce(metric(third_metrics, "recall"), local.get("recall"))
            row["Recall Std"] = coalesce(metric(third_metrics, "recall", "std"), local.get("recall_std"))
            row["F1-PR"] = coalesce(metric(third_metrics, "f1_pr"), local.get("f1_pr"))
            row["F1-PR Std"] = coalesce(metric(third_metrics, "f1_pr", "std"), local.get("f1_pr_std"))

            linear = third_metrics.get("mmd_linear", {})
            if isinstance(linear, dict):
                row["MMD Linear"] = linear.get("mean")
                row["MMD Linear Std"] = linear.get("std")
                row["MMD Linear Median"] = linear.get("median")
                row["MMD Linear Trimmed Mean"] = linear.get("trimmed_mean")
                row["MMD Linear Min"] = linear.get("min")
                row["MMD Linear Max"] = linear.get("max")
                row["MMD Linear Mean/Median"] = linear.get("mean_to_median_ratio")

            row["Num Generated Graphs"] = third.get("num_generated_graphs", "")
            row["Num Reference Graphs"] = third.get("num_reference_graphs", "")
            row["Repeats"] = third.get("repeats", "")

            score_values = {
                "degree": row["Degree"],
                "clustering": row["Clustering"],
                "orbit": row["Orbit"],
                "spectral": row["Spectral"],
                "diameter": row["Diameter"],
                "mmd_rbf": row["MMD RBF"],
                "f1_pr": row["F1-PR"],
            }
            row["Normalized Score"] = normalized_score(score_values)
            row["Normalized Score Source"] = "test_table2_plus_third_party_table3"

        rows.append(row)
    return rows


def build_lobster_docx_rows() -> list[dict[str, object]]:
    table2_structural, table3_gnn = extract_docx_tables(LOBSTER_DOCX)[:2]
    structural = {row[0]: row for row in table2_structural[1:]}
    gnn = {row[0]: row for row in table3_gnn[1:]}

    rows = []
    for setting in sorted(structural):
        t2 = structural[setting]
        t3 = gnn.get(setting, [])
        notes = [LOBSTER_DOCX_CAVEATS[setting]] if setting in LOBSTER_DOCX_CAVEATS else []
        row = base_row()
        row["Source"] = "lobster_docx_20260708"
        row["Setting"] = setting
        row["Model"] = t2[1]
        row["Dataset"] = "LOBSTER"
        row["Status"] = "evaluated"
        row["Degree"] = parse_doc_number(t2[2])
        row["Clustering"] = parse_doc_number(t2[3])
        row["Orbit"] = parse_doc_number(t2[4])
        row["Spectral"] = parse_doc_number(t2[5])
        row["Diameter"] = parse_doc_number(t2[6])
        if t3:
            row["MMD RBF"] = parse_doc_number(t3[2])
            row["MMD Linear"] = parse_doc_number(t3[3])
            row["Precision"] = parse_doc_number(t3[4])
            row["Recall"] = parse_doc_number(t3[5])
            row["F1-PR"] = parse_doc_number(t3[6], field="f1_pr", notes=notes)
        row["Config"] = str(LOBSTER_DOCX.relative_to(ROOT))
        for key, value in LOBSTER_DOCX_EDGE_COUNTS.get(setting, {}).items():
            row[key] = value
        row["Notes"] = first_note(*notes)

        score_values = {
            "degree": row["Degree"],
            "clustering": row["Clustering"],
            "orbit": row["Orbit"],
            "spectral": row["Spectral"],
            "diameter": row["Diameter"],
            "mmd_rbf": row["MMD RBF"],
            "f1_pr": row["F1-PR"],
        }
        row["Normalized Score"] = normalized_score(score_values)
        row["Normalized Score Source"] = "docx_table2_plus_docx_table3"
        rows.append(row)
    return rows


def build_loss_sweep_docx_rows() -> list[dict[str, object]]:
    table2_structural, table3_gnn = extract_docx_tables(LOSS_SWEEP_DOCX)[:2]
    structural = {row[0]: row for row in table2_structural[1:]}
    gnn = {row[0]: row for row in table3_gnn[1:]}

    rows = []
    for setting in sorted(structural):
        t2 = structural[setting]
        t3 = gnn.get(setting, [])
        row = base_row()
        row["Source"] = "loss_weight_sweep_docx_20260708"
        row["Setting"] = f"coarse_{setting}"
        row["Model"] = "GraphVAE+Motif both no-temp"
        row["Dataset"] = "LOBSTER"
        row["Status"] = "evaluated"
        row["alpha_node_feat"] = parse_doc_number(t2[1])
        row["alpha_edge_feat"] = parse_doc_number(t2[2])
        row["alpha_motif_loss"] = parse_doc_number(t2[3])
        row["alpha_syntactic_literal_motif_loss"] = parse_doc_number(t2[3])
        row["rule_mode"] = "both"
        row["Degree"] = parse_doc_number(t2[4])
        row["Clustering"] = parse_doc_number(t2[5])
        row["Orbit"] = parse_doc_number(t2[6])
        row["Spectral"] = parse_doc_number(t2[7])
        row["Diameter"] = parse_doc_number(t2[8])
        row["Avg Edges Generated"] = parse_doc_number(t2[9])
        row["Avg Edges Test"] = parse_doc_number(t2[10])
        if t3:
            row["MMD RBF"] = parse_doc_number(t3[4])
            row["MMD Linear"] = parse_doc_number(t3[5])
            row["Precision"] = parse_doc_number(t3[6])
            row["Recall"] = parse_doc_number(t3[7])
            row["F1-PR"] = parse_doc_number(t3[8])
        row["Config"] = str(LOSS_SWEEP_DOCX.relative_to(ROOT))

        score_values = {
            "degree": row["Degree"],
            "clustering": row["Clustering"],
            "orbit": row["Orbit"],
            "spectral": row["Spectral"],
            "diameter": row["Diameter"],
            "mmd_rbf": row["MMD RBF"],
            "f1_pr": row["F1-PR"],
        }
        row["Normalized Score"] = normalized_score(score_values)
        row["Normalized Score Source"] = "docx_table2_plus_docx_table3"
        rows.append(row)
    return rows


def assign_ranks(rows: list[dict[str, object]]) -> None:
    rankable = [
        row
        for row in rows
        if row.get("Status") == "evaluated" and as_float(row.get("Normalized Score")) is not None
    ]
    rankable.sort(key=lambda row: as_float(row["Normalized Score"]))
    for idx, row in enumerate(rankable, start=1):
        row["Rank"] = idx


def write_csv(rows: list[dict[str, object]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in FIELDNAMES})


def md_table(rows: list[dict[str, object]], limit: int = 12) -> str:
    rankable = [
        row
        for row in rows
        if row.get("Status") == "evaluated" and as_float(row.get("Normalized Score")) is not None
    ]
    rankable.sort(key=lambda row: as_float(row["Normalized Score"]))
    headers = [
        "Rank",
        "Source",
        "Setting",
        "Model",
        "Score",
        "Degree",
        "Clustering",
        "Orbit",
        "Spectral",
        "Diameter",
        "MMD RBF",
        "MMD Linear",
        "Precision",
        "Recall",
        "F1-PR",
        "Notes",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rankable[:limit]:
        cells = [
            row.get("Rank", ""),
            row.get("Source", ""),
            row.get("Setting", ""),
            row.get("Model", ""),
            fmt(row.get("Normalized Score")),
            fmt(row.get("Degree")),
            fmt(row.get("Clustering")),
            fmt(row.get("Orbit")),
            fmt(row.get("Spectral")),
            fmt(row.get("Diameter")),
            fmt(row.get("MMD RBF")),
            fmt(row.get("MMD Linear")),
            fmt(row.get("Precision")),
            fmt(row.get("Recall")),
            fmt(row.get("F1-PR")),
            row.get("Notes", ""),
        ]
        lines.append("| " + " | ".join(str(cell).replace("|", "/") for cell in cells) + " |")
    return "\n".join(lines)


def write_markdown(rows: list[dict[str, object]]) -> None:
    guard_rows = [row for row in rows if row["Source"] == "guard_sweep_20260708"]
    evaluated_guard = [row for row in guard_rows if row["Status"] == "evaluated"]
    partial_guard = [row for row in guard_rows if row["Status"] != "evaluated"]
    best_guard = sorted(
        evaluated_guard,
        key=lambda row: as_float(row.get("Normalized Score")) if as_float(row.get("Normalized Score")) is not None else 999,
    )[:3]

    lines = [
        "# LOBSTER GraphVAE Original Temp Guard Sweep Analysis",
        "",
        f"Generated CSV: `{OUT_CSV.relative_to(ROOT)}`",
        "",
        "Lower normalized score is better. The score averages degree, clustering, orbit, spectral, diameter, "
        "MMD RBF, and F1-PR error after normalization by the same denominator family used for validation "
        "selection. Table 3 values use third-party Random GIN metrics when available.",
        "",
        "## Ranking",
        "",
        md_table(rows),
        "",
        "## Guarded Sweep Takeaways",
        "",
    ]

    if best_guard:
        top = best_guard[0]
        lines.append(
            f"- Best evaluated guarded run: `{top['Setting']}` "
            f"(alpha_node/edge={top['alpha_node_feat']}, alpha_motif={top['alpha_motif_loss']}), "
            f"score {fmt(top['Normalized Score'])}, F1-PR {fmt(top['F1-PR'])}, "
            f"MMD RBF {fmt(top['MMD RBF'])}."
        )
    for row in best_guard[1:]:
        lines.append(
            f"- Next guarded candidate: `{row['Setting']}` with score {fmt(row['Normalized Score'])}, "
            f"F1-PR {fmt(row['F1-PR'])}, MMD RBF {fmt(row['MMD RBF'])}."
        )
    if partial_guard:
        for row in partial_guard:
            lines.append(
                f"- `{row['Setting']}` is not final-test comparable: {row['Notes']}. "
                f"Its best validation score was {fmt(row['Best Val Score'])} at epoch {fmt(row['Best Val Epoch'])}."
            )

    lines.extend(
        [
            "",
            "## Source Notes",
            "",
            "- Requested DOCX files `lobster.docx` and `lobster_loss_weight_sweep_results_20260708.docx` "
            "are now parsed directly from `reports/`. The older 20260630 CSV exports are intentionally not "
            "used because they do not match these later DOCX reports.",
            "- Guarded sweep rows come from `final_metrics_summary.json`, `best_validation_mmd.json`, "
            "and `manifest.csv` under the 20260708 sweep paths.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    rows = build_lobster_docx_rows() + build_loss_sweep_docx_rows() + build_guard_rows()
    assign_ranks(rows)
    write_csv(rows)
    write_markdown(rows)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
