#!/usr/bin/env python3
"""Aggregate all matched terminal-chain structural metrics across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terminal_motif_evaluation import write_json


METRICS = (
    "degree",
    "clustering",
    "sparsity",
    "orbit",
    "spectral",
    "triangle",
    "diameter",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stat(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def fmt(value: Dict[str, Any]) -> str:
    if value["mean"] is None:
        return "—"
    return f"{value['mean']:.5f} ± {value['std']:.5f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather-root", required=True, type=Path)
    cli = parser.parse_args()
    root = cli.gather_root.expanduser().resolve()
    paths = sorted(
        root.rglob(
            "terminal_motif_mmd/matched_per_chain_structural_mmd.json"
        )
    )
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for path in paths:
        payload = load_json(path)
        grouped[(payload["dataset"], payload["setting"])].append(payload)

    settings = []
    for (dataset, setting), runs in sorted(grouped.items()):
        runs.sort(key=lambda run: int(run["training_seed"]))
        seeds = [int(run["training_seed"]) for run in runs]
        by_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for run in runs:
            for chain in run["chains"]:
                by_index[int(chain["chain_index"])].append(chain)
        chains = []
        for chain_index, rows in sorted(by_index.items()):
            representative = rows[0]
            okay = [row for row in rows if row["status"] == "ok"]
            chains.append(
                {
                    "chain_index": chain_index,
                    "rule_index": representative["rule_index"],
                    "value_index": representative["value_index"],
                    "rule": representative["rule"],
                    "states": representative["states"],
                    "value": representative["value"],
                    "seed_count_total": len(runs),
                    "seed_count_evaluable": len(okay),
                    "metrics": {
                        metric: stat(
                            [
                                float(row["metrics"][metric])
                                for row in okay
                                if row["metrics"][metric] is not None
                            ]
                        )
                        for metric in METRICS
                    },
                    "generated_nonempty_count": stat(
                        [
                            float(row["generated_nonempty_count"])
                            for row in rows
                        ]
                    ),
                    "reference_nonempty_count": stat(
                        [
                            float(row["reference_nonempty_count"])
                            for row in rows
                        ]
                    ),
                    "generated_edge_count": stat(
                        [
                            float(row["generated_edge_count"])
                            for row in okay
                            if row["generated_edge_count"] is not None
                        ]
                    ),
                    "reference_edge_count": stat(
                        [
                            float(row["reference_edge_count"])
                            for row in okay
                            if row["reference_edge_count"] is not None
                        ]
                    ),
                }
            )
        settings.append(
            {
                "dataset": dataset,
                "setting": setting,
                "seeds": seeds,
                "chain_count": len(chains),
                "chains_evaluable_all_seeds": sum(
                    chain["seed_count_evaluable"] == len(seeds)
                    for chain in chains
                ),
                "chains": chains,
            }
        )

    payload = {
        "metric_family": "matched_independent_terminal_chain_structural_mmd",
        "comparison": "generated chain j versus real test chain j",
        "combination_across_chains": "none",
        "aggregation": (
            "mean and sample standard deviation across training seeds"
        ),
        "run_count": len(paths),
        "setting_count": len(settings),
        "settings": settings,
    }
    json_path = root / "matched_per_chain_terminal_motif_mmd_aggregated.json"
    csv_path = root / "matched_per_chain_terminal_motif_mmd_aggregated.csv"
    report_path = root / "MATCHED_ALL_TERMINAL_CHAIN_RESULTS.md"
    write_json(json_path, payload)

    fields = [
        "dataset",
        "setting",
        "seeds",
        "chain_index",
        "rule_index",
        "value_index",
        "rule",
        "states",
        "value",
        "seed_count_evaluable",
        "seed_count_total",
        "generated_nonempty_mean",
        "reference_nonempty_mean",
        "generated_edges_mean",
        "reference_edges_mean",
        *[
            f"{metric}_{suffix}"
            for metric in METRICS
            for suffix in ("mean", "std", "seed_count")
        ],
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for setting in settings:
            for chain in setting["chains"]:
                row = {
                    "dataset": setting["dataset"],
                    "setting": setting["setting"],
                    "seeds": json.dumps(setting["seeds"]),
                    "chain_index": chain["chain_index"],
                    "rule_index": chain["rule_index"],
                    "value_index": chain["value_index"],
                    "rule": json.dumps(chain["rule"], sort_keys=True),
                    "states": json.dumps(chain["states"], sort_keys=True),
                    "value": json.dumps(chain["value"], sort_keys=True),
                    "seed_count_evaluable": chain["seed_count_evaluable"],
                    "seed_count_total": chain["seed_count_total"],
                    "generated_nonempty_mean": chain[
                        "generated_nonempty_count"
                    ]["mean"],
                    "reference_nonempty_mean": chain[
                        "reference_nonempty_count"
                    ]["mean"],
                    "generated_edges_mean": chain["generated_edge_count"][
                        "mean"
                    ],
                    "reference_edges_mean": chain["reference_edge_count"][
                        "mean"
                    ],
                }
                for metric in METRICS:
                    row[f"{metric}_mean"] = chain["metrics"][metric]["mean"]
                    row[f"{metric}_std"] = chain["metrics"][metric]["std"]
                    row[f"{metric}_seed_count"] = chain["metrics"][metric][
                        "count"
                    ]
                writer.writerow(row)

    lines = [
        "# All matched terminal-chain structural results",
        "",
        "Every generated rule/value chain is compared with the same rule/value "
        "chain computed on attributed real test graphs. Both sides are "
        "binarized and cleaned identically; chains are never combined. Values "
        "are mean ± sample standard deviation across training seeds.",
    ]
    for setting in settings:
        lines.extend(
            [
                "",
                f"## {setting['dataset']} — {setting['setting']}",
                "",
                f"Seeds: {','.join(map(str, setting['seeds']))}; "
                f"chains: {setting['chain_count']}; evaluable in every seed: "
                f"{setting['chains_evaluable_all_seeds']}.",
                "",
                "| Chain | Rule/value | Seeds | Gen/ref nonempty | Gen/ref "
                "edges | Degree | Clustering | Orbit | Spectral | Diameter |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for chain in setting["chains"]:
            rule = json.dumps(
                [chain["rule"], chain["value"]],
                separators=(",", ":"),
            )
            if len(rule) > 180:
                rule = rule[:179] + "…"
            gen_nonempty = chain["generated_nonempty_count"]["mean"]
            ref_nonempty = chain["reference_nonempty_count"]["mean"]
            gen_edges = chain["generated_edge_count"]["mean"]
            ref_edges = chain["reference_edge_count"]["mean"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(chain["chain_index"]),
                        rule.replace("|", "\\|"),
                        (
                            f"{chain['seed_count_evaluable']}/"
                            f"{chain['seed_count_total']}"
                        ),
                        f"{gen_nonempty:.1f}/{ref_nonempty:.1f}",
                        (
                            "—"
                            if gen_edges is None or ref_edges is None
                            else f"{gen_edges:.2f}/{ref_edges:.2f}"
                        ),
                        fmt(chain["metrics"]["degree"]),
                        fmt(chain["metrics"]["clustering"]),
                        fmt(chain["metrics"]["orbit"]),
                        fmt(chain["metrics"]["spectral"]),
                        fmt(chain["metrics"]["diameter"]),
                    ]
                )
                + " |"
            )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Aggregated {len(paths)} runs / {len(settings)} settings.\n"
        f"Report: {report_path}\nCSV: {csv_path}\nJSON: {json_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
