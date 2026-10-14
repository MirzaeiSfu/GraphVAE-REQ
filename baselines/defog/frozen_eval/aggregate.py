#!/usr/bin/env python3
"""Aggregate 10 Random-GIN runs within seed, then sample SD across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev


METRICS = ("f1_pr", "precision", "recall", "mmd_rbf", "mmd_linear")


def load_seed_result(path: Path, dataset: str, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    campaign = payload.get("campaign") or {}
    if campaign.get("dataset") != dataset or int(campaign.get("training_seed", -1)) != seed:
        raise ValueError(f"Campaign identity mismatch in {path}")
    if campaign.get("evaluator_seeds") != list(range(10)):
        raise ValueError(f"Evaluator seeds are not 0..9 in {path}")
    evaluation = payload["evaluation"]
    if int(evaluation["repeats"]) != 10 or int(evaluation["base_seed"]) != 0:
        raise ValueError(f"Random-GIN repetition contract changed in {path}")
    return payload


def aggregate_dataset(input_root: Path, dataset: str) -> dict:
    per_seed = {}
    reference_digest = None
    for seed in (0, 1, 2):
        path = input_root / dataset.lower() / f"seed_{seed}" / "evaluation.json"
        payload = load_seed_result(path, dataset, seed)
        if reference_digest is None:
            reference_digest = payload["reference_sha256"]
        elif payload["reference_sha256"] != reference_digest:
            raise ValueError(f"Reference collection differs for {dataset} seed {seed}")
        modes = payload["evaluation"]["modes"]
        per_seed[str(seed)] = {
            mode: {
                metric: float(mode_result["summary"][metric]["mean"])
                for metric in METRICS
            }
            for mode, mode_result in modes.items()
        }
    expected_modes = set(per_seed["0"])
    if any(set(result) != expected_modes for result in per_seed.values()):
        raise ValueError(f"Feature modes differ across {dataset} training seeds")
    aggregate = {}
    for mode in sorted(expected_modes):
        aggregate[mode] = {}
        for metric in METRICS:
            values = [per_seed[str(seed)][mode][metric] for seed in (0, 1, 2)]
            aggregate[mode][metric] = {
                "mean": fmean(values),
                "sample_sd": stdev(values),
                "training_seed_values": values,
            }
    return {
        "dataset": dataset,
        "training_seeds": [0, 1, 2],
        "evaluator_seeds_per_training_seed": list(range(10)),
        "reference_sha256": reference_digest,
        "per_training_seed_evaluator_means": per_seed,
        "aggregate": aggregate,
    }


def write_reports(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for mode, metrics in result["aggregate"].items():
        for metric, summary in metrics.items():
            rows.append(
                {
                    "dataset": result["dataset"],
                    "mode": mode,
                    "metric": metric,
                    "mean": summary["mean"],
                    "sample_sd": summary["sample_sd"],
                }
            )
    with (output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# DeFoG {result['dataset']} frozen Random-GIN evaluation",
        "",
        "Values are means across training seeds 0, 1, and 2; each training-seed "
        "value is first averaged across evaluator seeds 0 through 9. SD is the "
        "sample standard deviation across training seeds.",
        "",
        "| Mode | Metric | Mean | Sample SD |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['metric']} | {row['mean']:.6g} | "
            f"{row['sample_sd']:.6g} |"
        )
    (output_dir / "aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_dataset(args.input_root.expanduser().resolve(), args.dataset.upper())
    write_reports(args.output_dir.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

