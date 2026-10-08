#!/usr/bin/env python3
"""Collect the 2026-08-31 GraphVAE runs and render a verified Markdown report."""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SOLAR_ROOT = "/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/runs/solar_kia_bce_motif01_2seed"
RESTART_ROOT = "/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/runs/solar_kia_bce_motif01_2seed_restarts_20260902"
LAB_ROOT = "/local-scratch2/mirzaei/fb/GraphVAE-REQ-kia-false-20260831/runs/kia_motif_false_2seed"
LAB16_ROOT = "/localhome/mirzaei/fb/GraphVAE-REQ-kia-false-20260831/runs/kia_motif_false_2seed"


@dataclass(frozen=True)
class Run:
    dataset: str
    setting: str
    seed: int
    host: str
    path: str
    state: str
    evaluation: str


def ssh_args(host: str) -> list[str]:
    if host == "solar":
        return ["ssh", "-p", "24", "-o", "BatchMode=yes", "mirzaei@solar.cs.sfu.ca"]
    return ["ssh", "-o", "BatchMode=yes", f"mirzaei@cs-cl-{host}.cmpt.sfu.ca"]


def remote_text(host: str, command: str, required: bool = True) -> str | None:
    result = subprocess.run(
        ssh_args(host) + [command], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=45,
    )
    if result.returncode:
        if required:
            raise RuntimeError(f"{host}: {command}\n{result.stderr}")
        return None
    return result.stdout


def remote_json(host: str, path: str, required: bool = True) -> dict | None:
    text = remote_text(host, f"cat {path}", required=required)
    return json.loads(text) if text else None


def build_manifest() -> list[Run]:
    runs: list[Run] = []

    # Solar motif=True. Three GRID runs were preempted and are not final.
    recovered = {
        ("LOBSTER", "Full matrix", 0),
        ("TRIANGULAR_GRID", "Total count", 0),
        ("TRIANGULAR_GRID", "Full matrix", 0),
        ("PROTEINS", "Total count", 0),
        ("PROTEINS", "Total count", 1),
        ("PROTEINS", "Full matrix", 0),
        ("PROTEINS", "Full matrix", 1),
    }
    unfinished = {
        ("GRID", "Total count", 0),
        ("GRID", "Total count", 1),
        ("GRID", "Full matrix", 0),
    }
    slugs = {
        "GRID": "grid", "LOBSTER": "lobster", "TRIANGULAR_GRID": "triangular_grid",
        "MUTAG": "mutag", "PROTEINS": "proteins",
    }
    modes = {"Total count": "total_count", "Full matrix": "full_matrix"}
    for dataset, slug in slugs.items():
        for setting, mode in modes.items():
            for seed in (0, 1):
                key = (dataset, setting, seed)
                path = f"{SOLAR_ROOT}/{slug}/{mode}/seed_{seed}"
                if key in unfinished:
                    runs.append(Run(dataset, setting, seed, "solar", path,
                                    "preempted; restart running", "pending"))
                elif key in recovered:
                    runs.append(Run(dataset, setting, seed, "solar", path,
                                    "training complete", "recovered"))
                else:
                    runs.append(Run(dataset, setting, seed, "solar", path,
                                    "complete", "original"))

    # Matching motif=False baselines distributed over three lab machines.
    placement = {
        ("GRID", 0): ("19", LAB_ROOT), ("GRID", 1): ("19", LAB_ROOT),
        ("MUTAG", 0): ("19", LAB_ROOT), ("MUTAG", 1): ("19", LAB_ROOT),
        ("PROTEINS", 0): ("19", LAB_ROOT), ("PROTEINS", 1): ("19", LAB_ROOT),
        ("LOBSTER", 0): ("16", LAB16_ROOT), ("LOBSTER", 1): ("18", LAB_ROOT),
        ("TRIANGULAR_GRID", 0): ("16", LAB16_ROOT),
        ("TRIANGULAR_GRID", 1): ("18", LAB_ROOT),
    }
    recovered_false = {("GRID", 0), ("TRIANGULAR_GRID", 1), ("PROTEINS", 0), ("PROTEINS", 1)}
    for (dataset, seed), (host, root) in placement.items():
        slug = slugs[dataset]
        path = f"{root}/{slug}/seed_{seed}"
        evaluation = "recovered" if (dataset, seed) in recovered_false else "original"
        state = "training complete" if evaluation == "recovered" else "complete"
        runs.append(Run(dataset, "Motif=False", seed, host, path, state, evaluation))
    return runs


RAW_RE = re.compile(
    r"degree\s*:?\s*(?P<degree>[-+0-9.eE]+).*?clustering\s*:?\s*(?P<clustering>[-+0-9.eE]+)"
    r".*?sparsity\s*:?\s*(?P<sparsity>[-+0-9.eE]+).*?orbits\s*:?\s*(?P<orbit>[-+0-9.eE]+)"
    r".*?Spec\s*:?\s*(?P<spectral>[-+0-9.eE]+).*?Tri\s*:?\s*(?P<triangle>[-+0-9.eE]+)"
    r".*?diameter\s*:?\s*(?P<diameter>[-+0-9.eE]+)"
)


def load_run_metrics(run: Run) -> dict | None:
    if run.evaluation == "pending":
        return None
    eval_name = "graph_realism_random_gin_recovered.json" if run.evaluation == "recovered" else "graph_realism_random_gin.json"
    third = remote_json(run.host, f"{run.path}/{eval_name}")
    summary = remote_json(run.host, f"{run.path}/final_metrics_summary.json", required=False)
    structural: dict[str, float] = {}
    if summary:
        structural.update(summary["table2"]["metrics"])
        extra = summary["table2"].get("extra_metrics", {})
        structural.update({
            "sparsity": extra.get("sparsity"),
            "triangle": extra.get("triangle"),
            "generated_edges": extra.get("generated_edge_count"),
            "reference_edges": extra.get("reference_edge_count"),
        })
    else:
        line = remote_text(
            run.host,
            f"grep -E 'degree.*clustering' {run.path}/console.log | tail -n 1",
            required=False,
        )
        match = RAW_RE.search(line or "")
        if match:
            structural = {key: float(value) for key, value in match.groupdict().items()}
    metrics = third["metrics"]
    return {
        "third": {
            "f1_pr": metrics["f1_pr"]["mean"],
            "precision": metrics["precision"]["mean"],
            "recall": metrics["recall"]["mean"],
            "mmd_rbf": metrics["mmd_rbf"]["mean"],
            "mmd_linear": metrics["mmd_linear"]["mean"],
            "mmd_linear_median": metrics["mmd_linear"]["median"],
            "mmd_linear_trimmed": metrics["mmd_linear"]["trimmed_mean"],
        },
        "structural": structural,
        "num_generated": third["num_generated_graphs"],
        "num_reference": third["num_reference_graphs"],
        "eval_file": f"{run.path}/{eval_name}",
    }


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    if abs(value) >= 10000:
        return f"{value:.4e}"
    return f"{value:.6f}"


def mean_sd(values: list[float]) -> str:
    if not values:
        return "—"
    mean = statistics.mean(values)
    if len(values) == 1:
        return fmt(mean) + " (n=1)"
    return f"{fmt(mean)} ± {fmt(statistics.stdev(values))}"


def improvement(baseline: float | None, candidate: float | None, higher: bool) -> str:
    if baseline is None or candidate is None or baseline == 0:
        return "—"
    gain = (candidate - baseline) / abs(baseline) * 100 if higher else (baseline - candidate) / abs(baseline) * 100
    return f"{gain:+.1f}%"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def main() -> None:
    runs = build_manifest()
    loaded: dict[Run, dict] = {}
    for run in runs:
        metrics = load_run_metrics(run)
        if metrics:
            loaded[run] = metrics

    grouped: dict[tuple[str, str], list[tuple[Run, dict]]] = defaultdict(list)
    for run, metrics in loaded.items():
        grouped[(run.dataset, run.setting)].append((run, metrics))

    out: list[str] = []
    out += [
        "# New GraphVAE hyperparameter experiments: verified results and recovery report",
        "",
        "Generated: 2026-09-02 (America/Vancouver). This report covers the new two-seed runs that use the code-defined GraphVAE loss weights. Values are read directly from the run artifacts; unfinished GRID motif=True runs are excluded from aggregates.",
        "",
        "## Executive status",
        "",
        "- Motif=True: 17/20 original training runs completed. Seven evaluation failures were repaired and rerun successfully. Three preempted GRID runs were restarted from scratch.",
        "- Motif=False: all 10 training runs completed. Four evaluation failures were repaired and rerun successfully.",
        "- Every finished run has a selected `best_validation_mmd_model` and a usable 10-repeat structural Random-GIN evaluation.",
        "- Restart jobs: GRID total-count seeds 0 and 1 are job `264183` on `cs-venus-09` and `cs-venus-13`; GRID full-matrix seed 0 is job `264186` on the normal shared pool (started on `cs-venus-17`).",
        "- The unusable submission `264175` to `schulte-lab-long` was cancelled because that partition currently has zero assigned nodes (`PartitionConfig`).",
        "",
        "## Experiment settings",
        "",
        "All runs use GraphVAE with AvePool encoder, FC decoder, latent graph embedding 128, 20,000 epochs, learning rate `3e-4`, deterministic seed 0 or 1, and the paper split (`70%/10%/20%`, split seed 123). Best-validation-MMD selection is enabled. Evaluation uses the test split and 10 independently initialized Random-GIN evaluators with structural inputs (degree, clustering, square clustering).",
        "",
    ]
    out += table(
        ["Dataset", "Database", "Directed", "Features", "Train batch", "Motif batch", "BCE", "KL", "Node", "Edge", "Motif=True", "Motif=False"],
        [
            ["GRID", "grid_undir_feat_snap_7a58e6_multi_linkcorr", "No", "None", "16", "32", "50", "2000", "0", "0", "5", "0"],
            ["LOBSTER", "lobster_undir_feat_snap_85093d_multi_linkcorr", "No", "None", "64", "64", "40", "2000", "0", "0", "4", "0"],
            ["TRIANGULAR_GRID", "triangular_grid_V2_undir_multi_linkcorr", "No", "None", "8", "16", "50", "2000", "0", "0", "5", "0"],
            ["MUTAG", "mutag_multi", "Yes", "Node + edge", "256", "512", "4", "60", "1", "1", "0.4", "0"],
            ["PROTEINS", "proteins_dir_feat_snap_0094da_multi", "No", "Node", "120", "128", "50", "2000", "1", "0", "5", "0"],
        ],
    )
    out += [
        "",
        "For motif=True, the command-line motif coefficient is exactly `0.1 × BCE`. Both total-count and full-matrix variants use `_CP_smoothed`, calibrated Gaussian motif loss, and pruned rules. Motif=False keeps the same architecture, data split, BCE/KL/node/edge weights, batch size, and schedule, but sets motif loss to zero; total-count versus full-matrix is therefore not applicable to the baseline.",
        "",
        "## Finished-run status and artifact locations",
        "",
    ]
    status_rows: list[list[str]] = []
    for run in sorted(runs, key=lambda x: (x.dataset, x.setting, x.seed)):
        if run.evaluation == "pending":
            mode = "total_count" if run.setting == "Total count" else "full_matrix"
            model = f"{RESTART_ROOT}/grid/{mode}/seed_{run.seed}/best_validation_mmd_model"
            eval_desc = "pending new run"
        else:
            model = f"{run.path}/best_validation_mmd_model"
            tag = "recovered" if run.evaluation == "recovered" else "original"
            eval_desc = f"{tag}: `{loaded[run]['eval_file']}`"
        status_rows.append([run.dataset, run.setting, str(run.seed), f"cs-cl-{run.host}" if run.host != "solar" else "Solar", run.state, f"`{model}`", eval_desc])
    out += table(["Dataset", "Setting", "Seed", "Host", "Status", "Selected model", "Evaluation"], status_rows)

    metric_defs = [
        ("F1-PR ↑", "f1_pr", True), ("Precision ↑", "precision", True),
        ("Recall ↑", "recall", True), ("MMD-RBF ↓", "mmd_rbf", False),
        ("MMD-linear ↓", "mmd_linear", False),
        ("MMD-linear median ↓", "mmd_linear_median", False),
        ("MMD-linear trimmed ↓", "mmd_linear_trimmed", False),
    ]
    out += [
        "",
        "## Aggregated third-party Random-GIN results",
        "",
        "Each cell is the mean across available training seeds ± sample SD across those seeds. Each seed value is itself the mean of 10 evaluator initializations. Positive improvement means motif=True is better: increases for F1/precision/recall and decreases for MMD. GRID motif=True is not aggregated until the three restarts finish.",
        "",
    ]
    for dataset in ("GRID", "LOBSTER", "TRIANGULAR_GRID", "MUTAG", "PROTEINS"):
        out += [f"### {dataset}", ""]
        rows: list[list[str]] = []
        base_items = grouped.get((dataset, "Motif=False"), [])
        for label, key, higher in metric_defs:
            base_values = [m["third"][key] for _, m in base_items]
            base_mean = statistics.mean(base_values) if base_values else None
            row = [label, mean_sd(base_values)]
            for setting in ("Total count", "Full matrix"):
                values = [m["third"][key] for _, m in grouped.get((dataset, setting), [])]
                cand_mean = statistics.mean(values) if values else None
                row += [mean_sd(values), improvement(base_mean, cand_mean, higher)]
            rows.append(row)
        out += table(["Metric", "Motif=False", "Total count", "Improvement", "Full matrix", "Improvement"], rows)
        out.append("")

    out += [
        "## Per-seed third-party results",
        "",
        "These are the evaluator means for each trained seed. MMD-linear is sensitive to extreme evaluator runs, so the median and 10%-trimmed mean are also retained.",
        "",
    ]
    seed_rows: list[list[str]] = []
    for run in sorted(loaded, key=lambda x: (x.dataset, x.setting, x.seed)):
        m = loaded[run]
        t = m["third"]
        seed_rows.append([
            run.dataset, run.setting, str(run.seed), run.evaluation,
            f"{m['num_generated']}/{m['num_reference']}", fmt(t["f1_pr"]),
            fmt(t["precision"]), fmt(t["recall"]), fmt(t["mmd_rbf"]),
            fmt(t["mmd_linear"]), fmt(t["mmd_linear_median"]),
            fmt(t["mmd_linear_trimmed"]),
        ])
    out += table(["Dataset", "Setting", "Seed", "Eval", "Gen/ref", "F1-PR", "Precision", "Recall", "MMD-RBF", "MMD-linear", "Linear median", "Linear trimmed"], seed_rows)

    struct_defs = [
        ("Degree MMD ↓", "degree"), ("Clustering MMD ↓", "clustering"),
        ("Orbit MMD ↓", "orbit"), ("Spectral MMD ↓", "spectral"),
        ("Diameter MMD ↓", "diameter"), ("Triangle MMD ↓", "triangle"),
        ("Sparsity error ↓", "sparsity"),
    ]
    out += [
        "",
        "## Aggregated direct structural metrics",
        "",
        "Lower is better. Metrics were read from `final_metrics_summary.json`; for recovered evaluations, the already-computed final structural line in `console.log` was used. A dash means that metric was not recoverable from the artifact.",
        "",
    ]
    for dataset in ("GRID", "LOBSTER", "TRIANGULAR_GRID", "MUTAG", "PROTEINS"):
        rows = []
        for label, key in struct_defs:
            row = [label]
            for setting in ("Motif=False", "Total count", "Full matrix"):
                values = [m["structural"].get(key) for _, m in grouped.get((dataset, setting), [])]
                values = [v for v in values if v is not None]
                row.append(mean_sd(values))
            rows.append(row)
        out += [f"### {dataset}", ""]
        out += table(["Metric", "Motif=False", "Total count", "Full matrix"], rows)
        out.append("")

    out += [
        "## Recovery details",
        "",
        "The seven Solar failures were evaluation/export failures after training, not failed model training. Three synthetic runs stored an object-dtype adjacency array; the repaired evaluator converts object arrays to numeric `float32` and validates square adjacency matrices. Four PROTEINS runs exported 210 generated graphs versus 209 references; evaluation now applies the evaluator's existing equal-size truncation before comparison. Recovery job `264176` completed all seven array tasks.",
        "",
        "The four lab failures were recovered with the same evaluator fix: GRID seed 0, TRIANGULAR_GRID seed 1, and PROTEINS seeds 0 and 1. Their recovered JSON and CSV files are inside the original run directories, so no model retraining occurred.",
        "",
        "## Roots, configs, and monitoring",
        "",
        f"- Solar motif=True run root: `{SOLAR_ROOT}`",
        f"- Solar restart root: `{RESTART_ROOT}`",
        f"- Lab motif=False root on cs-cl-18/19: `{LAB_ROOT}`",
        f"- Lab motif=False root on cs-cl-16: `{LAB16_ROOT}`",
        "- Motif=True configs: `/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/configs/linkcorr_motif_true_3seed/*.yaml` plus command-line weight/seed/output overrides in the Slurm script.",
        "- Motif=False launcher/config record: `/local-scratch2/mirzaei/fb/GraphVAE-REQ-kia-false-20260831/scripts/run_kia_false_queue.sh` and each run's `run_config_used.yaml`.",
        "- GRID restart script: `/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/slurm/solar_restart_grid_kia_motif_true_20260902.sbatch`.",
        "- Evaluation recovery script: `/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/slurm/solar_recover_kia_evaluations_20260902.sbatch`.",
        "",
        "Monitor GRID restarts with:",
        "",
        "```bash",
        "ssh -p 24 mirzaei@solar.cs.sfu.ca",
        "squeue -j 264183,264186 -o \"%.14i %.18P %.24j %.10T %.10M %.24R\"",
        "tail -f /project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/slurm_logs/grid_restart_264183_0.out",
        "```",
        "",
        "## Interpretation cautions",
        "",
        "- These are two-seed results; uncertainty is therefore only descriptive and should not be treated as a well-powered significance test.",
        "- Recovered runs reuse their already-generated final graph sets and selected best models. They do not rerun training or resample generated graphs.",
        "- MMD-linear can be dominated by one evaluator initialization. Prefer reading it together with its median and trimmed mean, MMD-RBF, and the direct structural MMD metrics.",
        "- A recovered evaluation is comparable to an original evaluation because it uses the same evaluator, 10 repeats, seed, structural features, and graph artifacts; only input normalization/equal-size handling changed to prevent crashes.",
    ]

    target = Path("/local-scratch/localhome/mirzaei/NEW_HYPERPARAMETER_EXPERIMENTS_RESULTS_20260902.md")
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
