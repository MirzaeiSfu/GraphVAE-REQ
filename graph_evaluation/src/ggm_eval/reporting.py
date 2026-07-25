"""Common JSON and Markdown reporting across isolated evaluator engines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def summarize_values(values: Iterable[float]) -> dict:
    """Return population summary statistics used by existing reports."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric sequence.")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_checkpoint_results(results: list) -> dict:
    """Aggregate numeric metrics across independently trained checkpoints."""

    if not results:
        raise ValueError("At least one checkpoint result is required.")
    checkpoint_seeds = [result.get("checkpoint_seed") for result in results]
    if len(set(checkpoint_seeds)) != len(checkpoint_seeds):
        raise ValueError(
            "Checkpoint seeds must be unique to represent independent runs."
        )
    comparable_fields = (
        "engine",
        "encoder",
        "feature_mode",
        "model",
        "training",
        "training_metadata",
        "schema_identity",
        "upstream_revision",
        "generated_sha256",
        "reference_sha256",
    )
    expected = {
        field: results[0].get(field) for field in comparable_fields
    }
    for index, result in enumerate(results[1:], start=1):
        actual = {field: result.get(field) for field in comparable_fields}
        if actual != expected:
            raise ValueError(
                f"Checkpoint result {index} is not comparable to result 0: "
                f"expected {expected}, got {actual}."
            )
    metric_names = set(results[0]["metrics"])
    for index, result in enumerate(results[1:], start=1):
        if set(result["metrics"]) != metric_names:
            raise ValueError(
                f"Checkpoint result {index} has different metrics."
            )
    summary = {
        metric: summarize_values(
            float(result["metrics"][metric]) for result in results
        )
        for metric in sorted(metric_names)
    }
    return {
        "engine": "contrastive-pyg-upstream",
        "encoder": results[0]["encoder"],
        "feature_mode": results[0]["feature_mode"],
        "checkpoint_count": len(results),
        "summary": summary,
        "per_checkpoint": results,
    }


def write_json(path, payload: dict):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_contrastive_markdown(path, payload: dict):
    """Write a compact report for matched PyG encoders."""

    lines = [
        "# PyG graph-generative-model evaluation",
        "",
        f"- Encoder: `{payload['encoder']}`",
        f"- Feature mode: `{payload['feature_mode']}`",
        f"- Independent checkpoints: `{payload['checkpoint_count']}`",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, summary in payload["summary"].items():
        lines.append(
            f"| {metric} | {summary['mean']:.6g} | {summary['std']:.6g} | "
            f"{summary['min']:.6g} | {summary['max']:.6g} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_legacy_markdown(path, payload: dict):
    """Write a report without reinterpreting the legacy metric values."""

    evaluation = payload["evaluation"]
    lines = [
        "# Legacy DGL Random-GIN evaluation",
        "",
        "The input PyG collection was converted directly to DGL before the "
        "existing evaluator was invoked.",
        "",
        "| Mode | Metric | Mean | Std | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, mode_result in evaluation["modes"].items():
        for metric, summary in mode_result["summary"].items():
            lines.append(
                f"| {mode} | {metric} | {summary['mean']:.6g} | "
                f"{summary['std']:.6g} | {summary['min']:.6g} | "
                f"{summary['max']:.6g} |"
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
