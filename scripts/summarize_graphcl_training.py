#!/usr/bin/env python3
"""Validate and summarize collected GraphCL-GIN training runs.

The cluster launcher writes one ``training_summary.json`` per dataset. This
command discovers those summaries after collection, checks that every
scheduled dataset and seed is present, verifies the copied checkpoints, and
writes machine-readable JSON/CSV plus a compact Markdown report.

Exit status is nonzero when a scheduled run is missing, incomplete, malformed,
or has a missing checkpoint. This makes the command suitable as the final
gate in a reusable distributed-training workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def _schedule_datasets(path: Path) -> dict[str, dict]:
    """Parse ``HOST GPU DATASET FEATURE_MODE`` rows into dataset metadata."""

    datasets = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"{path}:{line_number}: expected four fields, got {fields!r}."
            )
        host, gpu, dataset, feature_mode = fields
        if dataset in datasets:
            raise ValueError(f"Dataset {dataset!r} appears twice in {path}.")
        datasets[dataset] = {
            "host": host,
            "gpu": int(gpu),
            "feature_mode": feature_mode,
        }
    if not datasets:
        raise ValueError(f"No jobs found in schedule {path}.")
    return datasets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_run_for_dataset(root: Path, dataset: str) -> tuple[Path, dict]:
    """Find exactly one collected training summary declaring ``dataset``."""

    matches = []
    for summary_path in root.rglob("training_summary.json"):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs = payload.get("training_runs") or []
        metadata = runs[0].get("training_metadata", {}) if runs else {}
        if metadata.get("dataset") == dataset:
            matches.append((summary_path, payload))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one training_summary.json for {dataset}, found "
            f"{len(matches)} under {root}."
        )
    return matches[0]


def collect_summary(
    *,
    collected_root: Path,
    schedule_path: Path,
    expected_seeds: list[int],
) -> dict:
    """Return a validated aggregate of all scheduled training jobs."""

    scheduled = _schedule_datasets(schedule_path)
    rows = []
    datasets = {}
    errors = []

    for dataset, schedule in scheduled.items():
        try:
            summary_path, payload = _find_run_for_dataset(
                collected_root, dataset
            )
            run_dir = summary_path.parent
            if not (run_dir / "COMPLETED").is_file():
                raise ValueError(f"Missing COMPLETED marker in {run_dir}.")
            if payload.get("engine") != "contrastive-pyg-upstream":
                raise ValueError(
                    f"Unexpected engine in {summary_path}: "
                    f"{payload.get('engine')!r}."
                )
            if payload.get("encoder") != "graphcl":
                raise ValueError(
                    f"Unexpected encoder in {summary_path}: "
                    f"{payload.get('encoder')!r}."
                )
            if payload.get("feature_mode") != schedule["feature_mode"]:
                raise ValueError(
                    f"Feature mode mismatch for {dataset}: "
                    f"{payload.get('feature_mode')!r} versus "
                    f"{schedule['feature_mode']!r}."
                )

            declared_seeds = [int(seed) for seed in payload.get("seeds", [])]
            if declared_seeds != expected_seeds:
                raise ValueError(
                    f"Seed mismatch for {dataset}: {declared_seeds} versus "
                    f"{expected_seeds}."
                )
            training_runs = payload.get("training_runs") or []
            by_seed = {int(run["seed"]): run for run in training_runs}
            if sorted(by_seed) != sorted(expected_seeds):
                raise ValueError(
                    f"Training runs for {dataset} have seeds "
                    f"{sorted(by_seed)}, expected {sorted(expected_seeds)}."
                )

            dataset_rows = []
            for seed in expected_seeds:
                run = by_seed[seed]
                checkpoint = run_dir / f"seed_{seed}" / "checkpoint.pt"
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise ValueError(f"Missing checkpoint: {checkpoint}.")
                metadata = run.get("training_metadata") or {}
                if metadata.get("dataset") != dataset:
                    raise ValueError(
                        f"Checkpoint metadata dataset mismatch for "
                        f"{dataset}/seed_{seed}: {metadata.get('dataset')!r}."
                    )
                row = {
                    "dataset": dataset,
                    "host": schedule["host"],
                    "gpu": schedule["gpu"],
                    "feature_mode": schedule["feature_mode"],
                    "feature_schema": metadata.get("feature_schema"),
                    "seed": seed,
                    "epochs": int(run["training"]["epochs"]),
                    "training_loss": float(run["training_loss"]),
                    "elapsed_seconds": float(run["elapsed_seconds"]),
                    "graph_count": int(run["training_graphs"]["graph_count"]),
                    "node_feature_dim": int(
                        run["training_graphs"]["node_feature_dim"]
                    ),
                    "edge_feature_dim": int(
                        run["training_graphs"]["edge_feature_dim"]
                    ),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": _sha256(checkpoint),
                    "training_collection_sha256": run[
                        "training_collection_sha256"
                    ],
                    "upstream_revision": run["upstream"]["revision"],
                }
                dataset_rows.append(row)
                rows.append(row)

            losses = [row["training_loss"] for row in dataset_rows]
            elapsed = [row["elapsed_seconds"] for row in dataset_rows]
            first = dataset_rows[0]
            datasets[dataset] = {
                **schedule,
                "feature_schema": first["feature_schema"],
                "seeds": list(expected_seeds),
                "epochs": first["epochs"],
                "graph_count": first["graph_count"],
                "node_feature_dim": first["node_feature_dim"],
                "edge_feature_dim": first["edge_feature_dim"],
                "mean_training_loss": statistics.fmean(losses),
                "population_std_training_loss": statistics.pstdev(losses),
                "total_elapsed_seconds": sum(elapsed),
                "summary_path": str(summary_path.resolve()),
                "checkpoints": [
                    row["checkpoint"] for row in dataset_rows
                ],
            }
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{dataset}: {exc}")

    return {
        "status": "complete" if not errors else "invalid",
        "collected_root": str(collected_root.resolve()),
        "schedule": str(schedule_path.resolve()),
        "expected_datasets": list(scheduled),
        "expected_seeds": expected_seeds,
        "dataset_count": len(datasets),
        "checkpoint_count": len(rows),
        "datasets": datasets,
        "runs": rows,
        "errors": errors,
    }


def _write_csv(path: Path, rows: list[dict]):
    fields = [
        "dataset",
        "host",
        "gpu",
        "feature_mode",
        "feature_schema",
        "seed",
        "epochs",
        "training_loss",
        "elapsed_seconds",
        "graph_count",
        "node_feature_dim",
        "edge_feature_dim",
        "checkpoint",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "training_collection_sha256",
        "upstream_revision",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict):
    lines = [
        "# GraphCL-GIN training summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Datasets complete: {payload['dataset_count']}/"
        f"{len(payload['expected_datasets'])}",
        f"- Checkpoints verified: {payload['checkpoint_count']}",
        f"- Seeds per dataset: {payload['expected_seeds']}",
        "",
        "| Dataset | Graphs | Node dim | Edge dim | "
        "Mean loss ± population SD | Total time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in payload["expected_datasets"]:
        item = payload["datasets"].get(dataset)
        if item is None:
            lines.append(f"| {dataset} | missing |  |  |  |  |")
            continue
        lines.append(
            f"| {dataset} | {item['graph_count']} | "
            f"{item['node_feature_dim']} | {item['edge_feature_dim']} | "
            f"{item['mean_training_loss']:.6g} ± "
            f"{item['population_std_training_loss']:.6g} | "
            f"{item['total_elapsed_seconds']:.1f} |"
        )
    if payload["errors"]:
        lines.extend(["", "## Validation errors", ""])
        lines.extend(f"- {error}" for error in payload["errors"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-root", type=Path, required=True)
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("CLUSTER_GRAPHCL_GIN_20260725.txt"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    collected_root = args.collected_root.expanduser().resolve()
    if not collected_root.is_dir():
        raise FileNotFoundError(f"Collected run root not found: {collected_root}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = collect_summary(
        collected_root=collected_root,
        schedule_path=args.schedule.expanduser().resolve(),
        expected_seeds=[int(seed) for seed in args.seeds],
    )
    (output_dir / "graphcl_training_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "graphcl_training_runs.csv", payload["runs"])
    _write_markdown(
        output_dir / "graphcl_training_summary.md",
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
