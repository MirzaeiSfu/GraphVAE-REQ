#!/usr/bin/env python3
"""Aggregate gathered GraphCL and existing graph metrics into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path(
    "/local-scratch2/new/gather/pretrained_graphcl_evaluation"
)
GRAPHCL_METRICS = (
    "f1_pr",
    "mmd_rbf",
    "fid",
    "precision",
    "recall",
    "density",
    "coverage",
    "f1_dc",
    "mmd_linear",
)
STRUCTURAL_METRICS = (
    "degree",
    "clustering",
    "orbit",
    "spectral",
    "diameter",
)
RANDOM_GIN_METRICS = ("f1_pr", "mmd_rbf", "precision", "recall")


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def artifact_manifest(path_text: str) -> dict | None:
    if not path_text:
        return None
    path = Path(path_text)
    return load_json(path.with_suffix(path.suffix + ".json"))


def base_fieldnames() -> list[str]:
    fields = [
        "dataset",
        "setting",
        "generator_seed",
        "evaluation_status",
        "evaluation_error",
        "evaluator_source",
        "evaluator_encoder",
        "evaluator_feature_mode",
        "evaluator_seed_count",
        "evaluator_seeds",
        "evaluator_epochs",
        "feature_schema",
        "node_feature_dim",
        "edge_feature_dim",
        "generation_seed",
        "generation_attempts",
        "reference_graph_count",
        "generated_graph_count",
        "reference_mean_nodes",
        "generated_mean_nodes",
        "reference_mean_undirected_edges",
        "generated_mean_undirected_edges",
        "checkpoint_sha256",
        "generated_collection_sha256",
        "reference_collection_sha256",
        "run_dir",
        "checkpoint",
        "generated_artifact",
        "reference_artifact",
        "evaluation_json",
    ]
    for metric in GRAPHCL_METRICS:
        fields.extend(
            f"graphcl_{metric}_{stat}"
            for stat in ("mean", "std", "min", "max")
        )
    for evaluator_seed in (0, 1, 2):
        for metric in GRAPHCL_METRICS:
            fields.append(f"graphcl_{metric}_evaluator_seed_{evaluator_seed}")
    for metric in STRUCTURAL_METRICS:
        fields.append(f"structural_{metric}_mmd")
    fields.extend(
        (
            "structural_reference_mean_edges",
            "structural_generated_mean_edges",
            "legacy_local_f1_pr",
            "legacy_local_f1_pr_std",
            "legacy_local_mmd_rbf",
            "legacy_local_mmd_rbf_std",
            "legacy_local_precision",
            "legacy_local_precision_std",
            "legacy_local_recall",
            "legacy_local_recall_std",
        )
    )
    for metric in RANDOM_GIN_METRICS:
        fields.extend(
            (f"random_gin_{metric}_mean", f"random_gin_{metric}_std")
        )
    fields.extend(
        (
            "random_gin_repeats",
            "random_gin_structural_features",
            "existing_metrics_source",
        )
    )
    return fields


def fill_artifact_summary(
    output: dict,
    manifest: dict | None,
    *,
    prefix: str,
):
    if not manifest:
        return
    summary = manifest.get("summary", {})
    output[f"{prefix}_graph_count"] = summary.get("graph_count", "")
    output[f"{prefix}_collection_sha256"] = manifest.get(
        "collection_sha256", ""
    )
    graph_count = int(summary.get("graph_count", 0) or 0)
    if graph_count:
        output[f"{prefix}_mean_nodes"] = (
            float(summary.get("total_nodes", 0)) / graph_count
        )
        output[f"{prefix}_mean_undirected_edges"] = (
            float(summary.get("directed_edge_count", 0)) / (2 * graph_count)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else output_root / "campaign_manifest.csv"
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else output_root
        / "all_datasets_all_settings_all_seeds_graphcl_metrics.csv"
    )
    rows = load_rows(manifest_path)
    outputs = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        evaluation_json_path = Path(row["evaluation_dir"]) / "evaluation.json"
        evaluation = load_json(evaluation_json_path)
        failure = load_json(Path(row["evaluation_dir"]) / "failure.json")
        output = {
            "dataset": row["dataset"],
            "setting": row["setting"],
            "generator_seed": row["generator_seed"],
            "evaluation_status": "complete" if evaluation else (
                "failed" if failure else "missing"
            ),
            "evaluation_error": (
                failure.get("error", "") if failure else row.get("error", "")
            ),
            "evaluator_source": row["evaluator_source"],
            "evaluator_encoder": "",
            "evaluator_feature_mode": row["feature_mode"],
            "evaluator_seed_count": "",
            "evaluator_seeds": "",
            "evaluator_epochs": "",
            "feature_schema": row["feature_schema"],
            "node_feature_dim": row["node_feature_dim"],
            "edge_feature_dim": row["edge_feature_dim"],
            "generation_seed": row["generation_seed"],
            "generation_attempts": row["generation_attempts"],
            "reference_graph_count": row["reference_graph_count"],
            "generated_graph_count": "",
            "reference_mean_nodes": "",
            "generated_mean_nodes": "",
            "reference_mean_undirected_edges": "",
            "generated_mean_undirected_edges": "",
            "checkpoint_sha256": row["checkpoint_sha256"],
            "generated_collection_sha256": "",
            "reference_collection_sha256": "",
            "run_dir": row["run_dir"],
            "checkpoint": row["checkpoint"],
            "generated_artifact": row["generated"],
            "reference_artifact": row["reference"],
            "evaluation_json": str(evaluation_json_path.resolve()),
            "existing_metrics_source": str(
                (run_dir / "final_metrics_summary.json").resolve()
            ),
        }
        fill_artifact_summary(
            output, artifact_manifest(row["generated"]), prefix="generated"
        )
        fill_artifact_summary(
            output, artifact_manifest(row["reference"]), prefix="reference"
        )

        if evaluation:
            per_checkpoint = evaluation.get("per_checkpoint", [])
            output["evaluator_encoder"] = evaluation.get("encoder", "")
            output["evaluator_feature_mode"] = evaluation.get(
                "feature_mode", row["feature_mode"]
            )
            output["evaluator_seed_count"] = len(per_checkpoint)
            output["evaluator_seeds"] = ";".join(
                str(item.get("checkpoint_seed", ""))
                for item in per_checkpoint
            )
            epochs = {
                item.get("training", {}).get("epochs")
                for item in per_checkpoint
            }
            output["evaluator_epochs"] = (
                next(iter(epochs)) if len(epochs) == 1 else ";".join(
                    str(value) for value in sorted(epochs, key=str)
                )
            )
            for metric in GRAPHCL_METRICS:
                summary = evaluation.get("summary", {}).get(metric, {})
                for stat in ("mean", "std", "min", "max"):
                    output[f"graphcl_{metric}_{stat}"] = summary.get(stat, "")
            for item in per_checkpoint:
                evaluator_seed = item.get("checkpoint_seed")
                if evaluator_seed not in (0, 1, 2):
                    continue
                for metric in GRAPHCL_METRICS:
                    output[
                        f"graphcl_{metric}_evaluator_seed_{evaluator_seed}"
                    ] = item.get("metrics", {}).get(metric, "")

        existing = load_json(run_dir / "final_metrics_summary.json") or {}
        table2 = existing.get("table2", {})
        for metric in STRUCTURAL_METRICS:
            output[f"structural_{metric}_mmd"] = (
                table2.get("metrics", {}).get(metric, "")
            )
        extra = table2.get("extra_metrics", {})
        output["structural_reference_mean_edges"] = extra.get(
            "reference_edge_count", ""
        )
        output["structural_generated_mean_edges"] = extra.get(
            "generated_edge_count", ""
        )
        table3 = existing.get("table3", {})
        local = table3.get("local_eval_metrics", {})
        for metric in ("f1_pr", "mmd_rbf", "precision", "recall"):
            output[f"legacy_local_{metric}"] = local.get(metric, "")
            output[f"legacy_local_{metric}_std"] = local.get(
                f"{metric}_std", ""
            )
        random_gin = table3.get("third_party_eval_metrics", {})
        random_metrics = random_gin.get("metrics", {})
        for metric in RANDOM_GIN_METRICS:
            summary = random_metrics.get(metric, {})
            output[f"random_gin_{metric}_mean"] = summary.get("mean", "")
            output[f"random_gin_{metric}_std"] = summary.get("std", "")
        output["random_gin_repeats"] = random_gin.get("repeats", "")
        output["random_gin_structural_features"] = random_gin.get(
            "structural_features", ""
        )
        outputs.append(output)

    fieldnames = base_fieldnames()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(outputs)

    completed = sum(row["evaluation_status"] == "complete" for row in outputs)
    summary = {
        "schema_version": "gathered-graphcl-results-v1",
        "output_csv": str(output_csv),
        "row_count": len(outputs),
        "complete_count": completed,
        "incomplete_count": len(outputs) - completed,
        "datasets": sorted({row["dataset"] for row in outputs}),
        "settings": sorted(
            {f"{row['dataset']}/{row['setting']}" for row in outputs}
        ),
    }
    (output_csv.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if completed != len(outputs):
        raise SystemExit(
            f"Only {completed}/{len(outputs)} evaluations are complete."
        )


if __name__ == "__main__":
    main()
