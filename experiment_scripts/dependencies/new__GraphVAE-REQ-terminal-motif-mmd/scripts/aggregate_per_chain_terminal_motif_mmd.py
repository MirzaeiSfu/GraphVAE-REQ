#!/usr/bin/env python3
"""Aggregate independent terminal-chain structural MMD results across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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
DISPLAY_NAMES = {
    "degree": "Degree",
    "clustering": "Clustering",
    "sparsity": "Sparsity",
    "orbit": "Orbit",
    "spectral": "Spectral",
    "triangle": "Triangle",
    "diameter": "Diameter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather-root", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Default: <gather-root>/PER_CHAIN_TERMINAL_MOTIF_MMD_RESULTS.md",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_std(values: Iterable[float]) -> Dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "count": len(values),
    }


def chain_key(chain: Dict[str, Any]) -> str:
    return json.dumps(
        [chain["rule"], chain["states"], chain["value"]],
        sort_keys=True,
        separators=(",", ":"),
    )


def short_json(value: Any, limit: int = 180) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fmt_stat(stat: Dict[str, Any], digits: int = 5) -> str:
    if stat["mean"] is None:
        return "—"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def aggregate(files: List[Path]) -> Dict[str, Any]:
    grouped_runs: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for path in files:
        payload = load_json(path)
        grouped_runs[(payload["dataset"], payload["setting"])].append(payload)

    settings: List[Dict[str, Any]] = []
    individual_rows: List[Dict[str, Any]] = []
    for (dataset, setting), runs in sorted(grouped_runs.items()):
        runs.sort(key=lambda run: int(run["training_seed"]))
        total_seed_count = len(runs)
        chains_by_key: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = (
            defaultdict(list)
        )
        for run in runs:
            for chain in run["chains"]:
                chains_by_key[chain_key(chain)].append((run, chain))
                individual = {
                    "dataset": dataset,
                    "setting": setting,
                    "seed": int(run["training_seed"]),
                    "chain_index": chain["chain_index"],
                    "rule_index": chain["rule_index"],
                    "value_index": chain["value_index"],
                    "rule": chain["rule"],
                    "states": chain["states"],
                    "value": chain["value"],
                    "status": chain["status"],
                    "nonempty_sample_count": chain["nonempty_sample_count"],
                    "nonempty_fraction": chain["nonempty_fraction"],
                    "metrics": chain["metrics"],
                    "generated_edge_count": chain.get("generated_edge_count"),
                    "reference_edge_count": chain.get("reference_edge_count"),
                }
                individual_rows.append(individual)

        aggregated_chains: List[Dict[str, Any]] = []
        for entries in chains_by_key.values():
            representative = entries[0][1]
            successful = [
                (run, chain)
                for run, chain in entries
                if chain["status"] == "ok"
            ]
            metric_stats = {
                metric: mean_std(
                    chain["metrics"][metric]
                    for _, chain in successful
                    if chain["metrics"][metric] is not None
                )
                for metric in METRICS
            }
            complete_coverage = len(successful) == total_seed_count
            score = None
            if complete_coverage and all(
                metric_stats[metric]["mean"] is not None for metric in METRICS
            ):
                score = statistics.fmean(
                    metric_stats[metric]["mean"] for metric in METRICS
                )
            aggregated_chains.append(
                {
                    "chain_index": representative["chain_index"],
                    "rule_index": representative["rule_index"],
                    "value_index": representative["value_index"],
                    "rule": representative["rule"],
                    "states": representative["states"],
                    "value": representative["value"],
                    "rule_source": representative.get("rule_source"),
                    "terminal_shape": representative["terminal_shape"],
                    "seed_count_total": total_seed_count,
                    "seed_count_evaluable": len(successful),
                    "evaluable_seeds": [
                        int(run["training_seed"]) for run, _ in successful
                    ],
                    "complete_seed_coverage": complete_coverage,
                    "nonempty_sample_count": mean_std(
                        chain["nonempty_sample_count"] for _, chain in entries
                    ),
                    "nonempty_fraction": mean_std(
                        chain["nonempty_fraction"] for _, chain in entries
                    ),
                    "generated_edge_count": mean_std(
                        chain["generated_edge_count"]
                        for _, chain in successful
                        if chain.get("generated_edge_count") is not None
                    ),
                    "metrics": metric_stats,
                    "seven_metric_mean_score": score,
                }
            )

        aggregated_chains.sort(key=lambda chain: int(chain["chain_index"]))
        ranked = sorted(
            (
                chain
                for chain in aggregated_chains
                if chain["seven_metric_mean_score"] is not None
            ),
            key=lambda chain: float(chain["seven_metric_mean_score"]),
        )
        settings.append(
            {
                "dataset": dataset,
                "setting": setting,
                "seeds": [int(run["training_seed"]) for run in runs],
                "seed_count": total_seed_count,
                "selected_chain_count": len(aggregated_chains),
                "chains_evaluable_all_seeds": sum(
                    chain["complete_seed_coverage"]
                    for chain in aggregated_chains
                ),
                "chains_evaluable_any_seed": sum(
                    chain["seed_count_evaluable"] > 0
                    for chain in aggregated_chains
                ),
                "best_complete_coverage_chain": ranked[0] if ranked else None,
                "chains": aggregated_chains,
            }
        )
    return {
        "metric_family": "independent_binarized_terminal_chain_structural_mmd",
        "combination_across_chains": "none",
        "aggregation": (
            "arithmetic mean and population standard deviation across evaluable "
            "training seeds; empty chains are excluded and coverage is reported"
        ),
        "best_chain_summary_score": (
            "unweighted arithmetic mean of the seven structural MMD means, "
            "used only to summarize complete-coverage chains; lower is better"
        ),
        "run_count": len(files),
        "setting_count": len(settings),
        "settings": settings,
        "individual_rows": individual_rows,
    }


def write_csv_files(root: Path, payload: Dict[str, Any]) -> None:
    individual_path = root / "per_chain_terminal_motif_mmd_individual.csv"
    with individual_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "dataset",
            "setting",
            "seed",
            "chain_index",
            "rule_index",
            "value_index",
            "rule",
            "states",
            "value",
            "status",
            "nonempty_sample_count",
            "nonempty_fraction",
            *METRICS,
            "generated_edge_count",
            "reference_edge_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["individual_rows"]:
            flat = {
                key: row[key]
                for key in fields
                if key not in METRICS and key not in ("rule", "states", "value")
            }
            flat["rule"] = json.dumps(row["rule"], sort_keys=True)
            flat["states"] = json.dumps(row["states"], sort_keys=True)
            flat["value"] = json.dumps(row["value"], sort_keys=True)
            flat.update(row["metrics"])
            writer.writerow(flat)

    aggregate_path = root / "per_chain_terminal_motif_mmd_aggregated.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
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
            "evaluable_seeds",
            "seed_count_evaluable",
            "seed_count_total",
            "nonempty_sample_count_mean",
            "nonempty_sample_count_std",
            *[
                f"{metric}_{suffix}"
                for metric in METRICS
                for suffix in ("mean", "std", "seed_count")
            ],
            "seven_metric_mean_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for setting in payload["settings"]:
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
                    "evaluable_seeds": json.dumps(chain["evaluable_seeds"]),
                    "seed_count_evaluable": chain["seed_count_evaluable"],
                    "seed_count_total": chain["seed_count_total"],
                    "nonempty_sample_count_mean": chain[
                        "nonempty_sample_count"
                    ]["mean"],
                    "nonempty_sample_count_std": chain[
                        "nonempty_sample_count"
                    ]["std"],
                    "seven_metric_mean_score": chain[
                        "seven_metric_mean_score"
                    ],
                }
                for metric in METRICS:
                    row[f"{metric}_mean"] = chain["metrics"][metric]["mean"]
                    row[f"{metric}_std"] = chain["metrics"][metric]["std"]
                    row[f"{metric}_seed_count"] = chain["metrics"][metric][
                        "count"
                    ]
                writer.writerow(row)


def markdown_report(payload: Dict[str, Any]) -> str:
    lines = [
        "# Independent terminal motif-chain structural MMD",
        "",
        "Each row is one rule/value terminal matrix chain. The terminal matrix "
        "was binarized before graph construction, and matrices from different "
        "chains were **not combined**. Values are mean ± population standard "
        "deviation across the listed evaluable seeds; lower MMD is better.",
        "",
        "The metrics use the same graph cleanup, statistics, Gaussian-TV "
        "kernels, and sigmas as `stat_rnn.mmd_eval`. Random-GIN MMD/F1-PR is "
        "not included because it is not a structural statistic.",
        "",
        "## Coverage and best complete-coverage chain",
        "",
        "| Dataset | Setting | Seeds | Chains any/all seeds | Best chain | "
        "7-metric mean | Degree | Clustering | Orbit | Spectral | Diameter |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in payload["settings"]:
        best = setting["best_complete_coverage_chain"]
        if best is None:
            best_cells = ["—"] * 7
        else:
            best_cells = [
                str(best["chain_index"]),
                f"{best['seven_metric_mean_score']:.5f}",
                fmt_stat(best["metrics"]["degree"]),
                fmt_stat(best["metrics"]["clustering"]),
                fmt_stat(best["metrics"]["orbit"]),
                fmt_stat(best["metrics"]["spectral"]),
                fmt_stat(best["metrics"]["diameter"]),
            ]
        lines.append(
            "| "
            + " | ".join(
                [
                    setting["dataset"],
                    setting["setting"],
                    ",".join(map(str, setting["seeds"])),
                    (
                        f"{setting['chains_evaluable_any_seed']}/"
                        f"{setting['chains_evaluable_all_seeds']}"
                    ),
                    *best_cells,
                ]
            )
            + " |"
        )

    for setting in payload["settings"]:
        lines.extend(
            [
                "",
                f"## {setting['dataset']} — {setting['setting']}",
                "",
                f"Seeds: {', '.join(map(str, setting['seeds']))}. "
                f"Selected chains: {setting['selected_chain_count']}. "
                f"Evaluable in every seed: "
                f"{setting['chains_evaluable_all_seeds']}.",
                "",
                "| Chain | Rule/value chain | Seeds | Nonempty graphs | Degree | "
                "Clustering | Sparsity | Orbit | Spectral | Triangle | Diameter |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for chain in setting["chains"]:
            rule_value = (
                short_json(chain["rule"], 115)
                + "<br>value="
                + short_json(chain["value"], 115)
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(chain["chain_index"]),
                        rule_value.replace("|", "\\|"),
                        (
                            f"{chain['seed_count_evaluable']}/"
                            f"{chain['seed_count_total']}"
                        ),
                        fmt_stat(chain["nonempty_sample_count"], digits=1),
                        *[
                            fmt_stat(chain["metrics"][metric])
                            for metric in METRICS
                        ],
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    cli = parse_args()
    root = cli.gather_root.expanduser().resolve()
    files = sorted(
        root.rglob("terminal_motif_mmd/per_chain_structural_mmd.json")
    )
    if not files:
        raise RuntimeError("No independent per-chain result JSON files found.")
    payload = aggregate(files)
    json_path = root / "per_chain_terminal_motif_mmd_aggregated.json"
    write_json(json_path, payload)
    write_csv_files(root, payload)
    report_path = (
        cli.report.expanduser().resolve()
        if cli.report
        else root / "PER_CHAIN_TERMINAL_MOTIF_MMD_RESULTS.md"
    )
    report_path.write_text(markdown_report(payload), encoding="utf-8")
    print(
        f"Aggregated {payload['run_count']} runs and "
        f"{payload['setting_count']} settings.\n"
        f"Report: {report_path}\nJSON: {json_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
