#!/usr/bin/env python3
"""Export full validation and held-out comparison metrics for the rule sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_SELECTION_DIR = Path(
    "collected_runs/20260719/"
    "lobster_matrix_motif_rule_database_normalized_table2_table3_selection"
)
DEFAULT_PRIOR_COMPARISON = Path(
    "collected_runs/20260718/lobster_matrix_motif_posthoc_selection/"
    "winner_matrix_vs_graphvae_vs_graphvae_mm.csv"
)
METRIC_COLUMNS = [
    "Precision",
    "Recall",
    "Degree MMD",
    "Clustering MMD",
    "Orbit MMD",
    "Spectral MMD",
    "Diameter MMD",
    "3rd-party F1-PR",
    "3rd-party MMD RBF",
]
METRIC_KEYS = {
    "Precision": "precision",
    "Recall": "recall",
    "Degree MMD": "degree",
    "Clustering MMD": "clustering",
    "Orbit MMD": "orbit",
    "Spectral MMD": "spectral",
    "Diameter MMD": "diameter",
    "3rd-party F1-PR": "f1_pr",
    "3rd-party MMD RBF": "mmd_rbf",
}
PROBABILITY_COLUMNS = ("Precision", "Recall", "3rd-party F1-PR")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: float) -> str:
    return f"{float(value):.12f}"


def load_candidates(paths: list[Path]) -> tuple[list[dict], str]:
    candidates = []
    reference_hash = None
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("heldout_loaded") is not False:
            raise RuntimeError(f"Candidate file is not validation-only: {path}")
        current_hash = payload["validation_reference_sha256"]
        if reference_hash is None:
            reference_hash = current_hash
        elif current_hash != reference_hash:
            raise RuntimeError(f"Validation reference mismatch in {path}")
        candidates.extend(payload["candidates"])
    unique = {candidate["checkpoint_path"]: candidate for candidate in candidates}
    if len(unique) != 20:
        raise RuntimeError(f"Expected 20 unique validation candidates, found {len(unique)}")
    if len({candidate["artifact_dir"] for candidate in unique.values()}) != 4:
        raise RuntimeError("Expected candidates from exactly four runs")
    return list(unique.values()), reference_hash


def validation_row(candidate: dict) -> dict:
    metrics = candidate["summary"]["metrics"]
    row = {
        "Model": candidate["run"],
        "Checkpoint": candidate["checkpoint"],
        "Normalized Table2+Table3 Score": format_metric(
            candidate["selection_score"]
        ),
    }
    row.update(
        {
            column: format_metric(metrics[key]["mean"])
            for column, key in METRIC_KEYS.items()
        }
    )
    return row


def comparison_row(label: str, metrics: dict) -> dict:
    row = {"Model": label}
    row.update(
        {
            column: format_metric(metrics[key])
            for column, key in METRIC_KEYS.items()
        }
    )
    return row


def clamp_prior_probabilities(row: dict) -> dict:
    clamped = dict(row)
    adjustments = {}
    for column in PROBABILITY_COLUMNS:
        raw = float(clamped[column])
        projected = min(max(raw, 0.0), 1.0)
        clamped[column] = format_metric(projected)
        if projected != raw:
            adjustments[column] = {"unclamped": raw, "clamped": projected}
    return clamped, adjustments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument(
        "--prior-comparison", type=Path, default=DEFAULT_PRIOR_COMPARISON
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_dir = args.selection_dir.resolve()
    candidate_paths = sorted(selection_dir.glob("validation_clamped_*.json"))
    if len(candidate_paths) != 4:
        raise RuntimeError(
            f"Expected four clamped validation shards, found {len(candidate_paths)}"
        )
    candidates, validation_hash = load_candidates(candidate_paths)
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["summary"]["score"]["std"],
            candidate["checkpoint_path"],
        ),
    )
    all_validation_rows = [validation_row(candidate) for candidate in ranked]
    validation_fields = [
        "Model",
        "Checkpoint",
        "Normalized Table2+Table3 Score",
        *METRIC_COLUMNS,
    ]
    all_validation_path = selection_dir / "validation_all_candidates_metrics.csv"
    write_csv(all_validation_path, validation_fields, all_validation_rows)

    per_run = {}
    for candidate in ranked:
        per_run.setdefault(candidate["artifact_dir"], candidate)
    per_run_rows = sorted(
        (validation_row(candidate) for candidate in per_run.values()),
        key=lambda row: float(row["Normalized Table2+Table3 Score"]),
    )
    per_run_path = selection_dir / "validation_best_per_run_metrics.csv"
    write_csv(per_run_path, validation_fields, per_run_rows)

    test_path = selection_dir / "test_evaluation.json"
    test_payload = json.loads(test_path.read_text(encoding="utf-8"))
    frozen_path = selection_dir / "validation_selection.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["candidate_count"] != 20 or frozen["run_count"] != 4:
        raise RuntimeError("Frozen selection does not contain the complete 20/4 inventory")
    if frozen["winner"]["checkpoint_path"] != test_payload["selected_checkpoint_path"]:
        raise RuntimeError("Held-out result does not match the frozen validation winner")

    with args.prior_comparison.open(newline="", encoding="utf-8") as handle:
        prior_rows = list(csv.DictReader(handle))
    baselines = {
        row["Model"]: row
        for row in prior_rows
        if row["Model"] in {"GraphVAE (standard)", "GraphVAE-MM"}
    }
    if set(baselines) != {"GraphVAE (standard)", "GraphVAE-MM"}:
        raise RuntimeError("Prior comparison is missing GraphVAE baseline rows")

    comparison_rows = [
        comparison_row(
            "Winner matrix method (rule-database sweep)", test_payload["metrics"]
        )
    ]
    baseline_adjustments = {}
    for label in ("GraphVAE (standard)", "GraphVAE-MM"):
        row, adjustments = clamp_prior_probabilities(baselines[label])
        comparison_rows.append(row)
        if adjustments:
            baseline_adjustments[label] = adjustments
    comparison_path = (
        selection_dir / "winner_rule_database_matrix_vs_graphvae_vs_graphvae_mm.csv"
    )
    write_csv(comparison_path, ["Model", *METRIC_COLUMNS], comparison_rows)

    manifest = {
        "schema_version": 1,
        "selection_split": "validation",
        "winner_test_split": "heldout_test",
        "selection_frozen_before_heldout_load": True,
        "candidate_count": 20,
        "run_count": 4,
        "selected_run": test_payload["selected_run"],
        "selected_checkpoint": test_payload["selected_checkpoint"],
        "validation_reference_sha256": validation_hash,
        "test_reference_sha256": test_payload["test_reference_sha256"],
        "test_generation_seed": test_payload["test_generation_seed"],
        "test_gin_runs": test_payload["test_gin_runs"],
        "test_gin_seed": test_payload["test_gin_seed"],
        "random_gin_structural_features": [
            "degree",
            "clustering",
            "square_clustering",
        ],
        "metric_domain_policy": {
            "MMD": "max(value, 0)",
            "precision_recall_f1": "clamp(value, 0, 1)",
        },
        "baseline_source_csv": str(args.prior_comparison.resolve()),
        "baseline_source_sha256": sha256_file(args.prior_comparison.resolve()),
        "baseline_probability_adjustments": baseline_adjustments,
        "outputs": {
            "comparison": str(comparison_path),
            "validation_all_candidates": str(all_validation_path),
            "validation_best_per_run": str(per_run_path),
        },
    }
    (selection_dir / "metrics_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {comparison_path}")
    print(f"Wrote {per_run_path}")
    print(f"Wrote {all_validation_path}")


if __name__ == "__main__":
    main()
