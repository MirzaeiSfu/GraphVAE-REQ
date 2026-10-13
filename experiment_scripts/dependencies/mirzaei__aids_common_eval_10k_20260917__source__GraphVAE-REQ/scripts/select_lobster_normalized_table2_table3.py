#!/usr/bin/env python3
"""Select a Lobster checkpoint with the full normalized Table-2/Table-3 score.

The ``evaluate`` subcommand touches validation data only and is shardable.  The
``finalize`` subcommand first writes an immutable validation-selection manifest,
then loads the held-out test split and evaluates the single frozen winner.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import random
import shutil
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import select_lobster_checkpoints as legacy_selector  # noqa: E402
from evaluate_graph_realism_batch import (  # noqa: E402
    evaluate_graph_collections,
    preprocess_graphs,
)
from ranking_score import (  # noqa: E402
    compute_validation_mmd_score,
    score_components_for_mode,
    score_denominators_for_mode,
)
from reproduce_table2_grid import (  # noqa: E402
    compute_table2_metrics,
    locked_orca_tmp,
    to_graphs,
)


SCORE_MODE = "normalized_table2_table3"
DATASET = "LOBSTER"
TABLE2_KEYS = ("degree", "clustering", "orbit", "spectral", "diameter")
TABLE3_KEYS = ("mmd_rbf", "precision", "recall", "f1_pr")
SCORE_KEYS = (*TABLE2_KEYS, "mmd_rbf", "f1_pr")
NONNEGATIVE_METRIC_KEYS = (*TABLE2_KEYS, "mmd_rbf")
PROBABILITY_METRIC_KEYS = ("precision", "recall", "f1_pr")
FORMULA = (
    "mean(min(degree/0.081,10), min(clustering/0.739,10), "
    "min(orbit/0.372,10), min(spectral/0.056,10), "
    "min(diameter/0.129,10), min(mmd_rbf/0.10,10), "
    "min((1-clamp(f1_pr,0,1))/0.05,10))"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_graphs(path: Path, graphs: list[nx.Graph]) -> None:
    matrices = [nx.to_numpy_array(graph, dtype=np.int8) for graph in graphs]
    payload = np.empty(len(matrices), dtype=object)
    payload[:] = matrices
    with path.open("wb") as handle:
        np.save(handle, payload, allow_pickle=True)


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def clamp_metric_domains(metrics: dict[str, float]) -> tuple[dict[str, float], dict]:
    """Project finite evaluator estimates onto their theoretical domains.

    The repository's unbiased MMD estimators can return tiny negative finite-
    sample estimates, while vendored ggmeval adds ``1e-5`` when computing F1
    and can consequently return ``1.00001``.  MMDs are theoretically
    nonnegative and precision/recall/F1 are probabilities, so clamp those
    numerical excursions before both ranking and reporting.  Keep every
    adjustment in the result payload for auditability.
    """

    clamped = dict(metrics)
    adjustments = {}
    for metric_name in NONNEGATIVE_METRIC_KEYS:
        value = clamped.get(metric_name)
        if value is None or not np.isfinite(value):
            continue
        projected = max(float(value), 0.0)
        clamped[metric_name] = projected
        if projected != float(value):
            adjustments[metric_name] = {
                "unclamped": float(value),
                "clamped": projected,
                "domain": "[0,+inf)",
            }
    for metric_name in PROBABILITY_METRIC_KEYS:
        value = clamped.get(metric_name)
        if value is None or not np.isfinite(value):
            continue
        projected = min(max(float(value), 0.0), 1.0)
        clamped[metric_name] = projected
        if projected != float(value):
            adjustments[metric_name] = {
                "unclamped": float(value),
                "clamped": projected,
                "domain": "[0,1]",
            }
    return clamped, adjustments


def summarize_rollouts(rollouts: list[dict]) -> dict:
    metric_names = sorted(
        {
            name
            for rollout in rollouts
            for name, value in rollout["metrics"].items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        }
    )
    component_names = sorted(
        {name for rollout in rollouts for name in rollout["score_components"]}
    )
    return {
        "score": summary([row["normalized_table2_table3"] for row in rollouts]),
        "metrics": {
            name: summary([float(row["metrics"][name]) for row in rollouts])
            for name in metric_names
            if all(name in row["metrics"] for row in rollouts)
        },
        "score_components": {
            name: summary(
                [float(row["score_components"][name]) for row in rollouts]
            )
            for name in component_names
        },
    }


def canonical_gin_graphs(graphs: list[nx.Graph], *, seed: int, shuffle: bool) -> list[nx.Graph]:
    # The saved-artifact evaluator sees adjacency matrices and therefore
    # consecutive node IDs.  Round-trip in memory to reproduce that behavior.
    items = [nx.to_numpy_array(graph, dtype=np.int8) for graph in graphs]
    return preprocess_graphs(items, max_graphs=1000, seed=seed, shuffle=shuffle)


def evaluate_graph_set(
    reference_graphs: list[nx.Graph],
    generated_graphs: list[nx.Graph],
    *,
    gin_runs: int,
    gin_seed: int,
    device: torch.device,
) -> dict:
    captured = io.StringIO()
    with redirect_stdout(captured):
        # Retain the repository lock even though current ORCA inputs are
        # process-unique; it also protects compatibility with older checkouts.
        with locked_orca_tmp():
            table2 = compute_table2_metrics(reference_graphs, generated_graphs)

        gin_reference = canonical_gin_graphs(
            reference_graphs, seed=gin_seed, shuffle=False
        )
        gin_generated = canonical_gin_graphs(
            generated_graphs, seed=gin_seed, shuffle=True
        )
        table3_payload = evaluate_graph_collections(
            generated_graphs=gin_generated,
            reference_graphs=gin_reference,
            repeats=gin_runs,
            seed=gin_seed,
            device=device,
            use_structural_features=True,
        )

    table3 = table3_payload["metrics"]
    raw_metrics = {
        **table2,
        **{name: float(table3[name]["mean"]) for name in TABLE3_KEYS},
        **{f"{name}_std": float(table3[name]["std"]) for name in TABLE3_KEYS},
    }
    metrics, metric_domain_adjustments = clamp_metric_domains(raw_metrics)
    score = compute_validation_mmd_score(metrics, SCORE_MODE, DATASET)
    components = score_components_for_mode(metrics, SCORE_MODE, DATASET)
    if score is None or len(components) != 7:
        raise RuntimeError(
            "Could not compute all seven normalized_table2_table3 components: "
            f"metrics={metrics}, components={components}"
        )
    return {
        "metrics": metrics,
        "score_components": components,
        "normalized_table2_table3": float(score),
        "num_generated_graphs": len(gin_generated),
        "num_reference_graphs": len(gin_reference),
        "metric_domain_adjustments": metric_domain_adjustments,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    reference_graphs: list[nx.Graph],
    *,
    validation_rollouts: int,
    validation_seed: int,
    gin_runs: int,
    gin_seed: int,
    latent_dim: int,
    device: torch.device,
) -> dict:
    decoder = legacy_selector.load_decoder(checkpoint_path, device, latent_dim)
    rollouts = []
    try:
        for rollout_index in range(validation_rollouts):
            generation_seed = validation_seed + rollout_index
            rollout_gin_seed = gin_seed + rollout_index * gin_runs
            raw_graphs, generated_graphs = legacy_selector.generate(
                decoder,
                len(reference_graphs),
                latent_dim,
                device,
                generation_seed,
            )
            result = evaluate_graph_set(
                reference_graphs,
                generated_graphs,
                gin_runs=gin_runs,
                gin_seed=rollout_gin_seed,
                device=device,
            )
            result.update(
                {
                    "rollout": rollout_index,
                    "generation_seed": generation_seed,
                    "gin_seed": rollout_gin_seed,
                    "mean_raw_edges": float(
                        np.mean([graph.number_of_edges() for graph in raw_graphs])
                    ),
                    "mean_lcc_edges": float(
                        np.mean([graph.number_of_edges() for graph in generated_graphs])
                    ),
                }
            )
            rollouts.append(result)
            print(
                f"  rollout={rollout_index + 1}/{validation_rollouts} "
                f"score={result['normalized_table2_table3']:.6f}",
                flush=True,
            )
    finally:
        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rollout_summary = summarize_rollouts(rollouts)
    return {
        "rollouts": rollouts,
        "summary": rollout_summary,
        "selection_score": rollout_summary["score"]["mean"],
    }


def checkpoint_paths(run_dir: Path, expected: int) -> list[Path]:
    paths = sorted(run_dir.glob("periodic_epoch_*.pt"))
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} periodic checkpoints in {run_dir}, found {len(paths)}"
        )
    return paths


def evaluation_protocol(args: argparse.Namespace) -> dict:
    return {
        "dataset": DATASET,
        "score_mode": SCORE_MODE,
        "score_direction": "lower_is_better",
        "score_formula": FORMULA,
        "score_denominators": score_denominators_for_mode(
            SCORE_MODE, DATASET
        ),
        "score_component_cap": 10.0,
        "selection_aggregate": "mean_of_per_rollout_scores",
        "validation_rollouts": args.validation_rollouts,
        "validation_generation_seeds": [
            args.validation_seed + index
            for index in range(args.validation_rollouts)
        ],
        "random_gin_runs_per_rollout": args.gin_runs,
        "random_gin_base_seeds": [
            args.gin_seed + index * args.gin_runs
            for index in range(args.validation_rollouts)
        ],
        "random_gin_structural_features": [
            "degree", "clustering", "square_clustering"
        ],
        "metric_domain_clamps": {
            "nonnegative": list(NONNEGATIVE_METRIC_KEYS),
            "probability_[0,1]": list(PROBABILITY_METRIC_KEYS),
            "adjustments_recorded_per_rollout": True,
        },
        "latent_dim": args.latent_dim,
        "device": str(args.device),
        "validation_reference_filename": args.validation_filename,
    }


def run_evaluate(args: argparse.Namespace) -> None:
    if args.validation_rollouts < 1 or args.gin_runs < 1:
        raise ValueError("validation-rollouts and gin-runs must be positive")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    protocol = evaluation_protocol(args)
    output_path = args.output_json.expanduser().resolve()

    if output_path.exists() and not args.no_resume:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("protocol") != protocol:
            raise RuntimeError(
                f"Existing shard protocol differs from requested protocol: {output_path}"
            )
    else:
        payload = {
            "schema_version": 1,
            "created_at": utc_now(),
            "split": "validation",
            "heldout_loaded": False,
            "protocol": protocol,
            "runs": [],
            "candidates": [],
        }

    existing_paths = {
        candidate["checkpoint_path"] for candidate in payload["candidates"]
    }
    cached_by_sha = {
        candidate["checkpoint_sha256"]: candidate
        for candidate in payload["candidates"]
    }
    reference_hash = payload.get("validation_reference_sha256")

    for raw_run_dir in args.run_dir:
        run_dir = raw_run_dir.expanduser().resolve()
        run_label = (
            run_dir.parent.name if run_dir.name.startswith("seed_") else run_dir.name
        )
        validation_path = run_dir / args.validation_filename
        if not validation_path.is_file():
            raise FileNotFoundError(validation_path)
        current_reference_hash = sha256_file(validation_path)
        if reference_hash is None:
            reference_hash = current_reference_hash
            payload["validation_reference_sha256"] = reference_hash
        elif current_reference_hash != reference_hash:
            raise RuntimeError(
                f"Validation references differ: {validation_path} has "
                f"{current_reference_hash}, expected {reference_hash}"
            )

        reference_items = np.load(validation_path, allow_pickle=True)
        reference_graphs = to_graphs(
            reference_items, keep_largest_component=False
        )
        if not reference_graphs:
            raise RuntimeError(f"No validation graphs in {validation_path}")
        if str(run_dir) not in payload["runs"]:
            payload["runs"].append(str(run_dir))

        for checkpoint_path in checkpoint_paths(
            run_dir, args.expected_checkpoints_per_run
        ):
            resolved_checkpoint = str(checkpoint_path.resolve())
            if resolved_checkpoint in existing_paths:
                print(f"[resume] {run_dir.name}/{checkpoint_path.name}", flush=True)
                continue
            checkpoint_hash = sha256_file(checkpoint_path)
            print(f"[candidate] {run_label}/{checkpoint_path.name}", flush=True)
            if checkpoint_hash in cached_by_sha:
                source = cached_by_sha[checkpoint_hash]
                evaluated = {
                    key: copy.deepcopy(source[key])
                    for key in ("rollouts", "summary", "selection_score")
                }
                reused_from = source["checkpoint_path"]
                print(f"  reused identical checkpoint {reused_from}", flush=True)
            else:
                evaluated = evaluate_checkpoint(
                    checkpoint_path,
                    reference_graphs,
                    validation_rollouts=args.validation_rollouts,
                    validation_seed=args.validation_seed,
                    gin_runs=args.gin_runs,
                    gin_seed=args.gin_seed,
                    latent_dim=args.latent_dim,
                    device=device,
                )
                reused_from = None

            candidate = {
                "run": run_label,
                "artifact_dir": str(run_dir),
                "validation_graphs": str(validation_path),
                "checkpoint": checkpoint_path.name,
                "checkpoint_path": resolved_checkpoint,
                "checkpoint_sha256": checkpoint_hash,
                "reused_from_identical_checkpoint": reused_from,
                **evaluated,
            }
            payload["candidates"].append(candidate)
            existing_paths.add(resolved_checkpoint)
            cached_by_sha.setdefault(checkpoint_hash, candidate)
            payload["updated_at"] = utc_now()
            write_json_atomic(output_path, payload)
            print(
                f"  mean_score={candidate['selection_score']:.6f}", flush=True
            )

    payload["completed_at"] = utc_now()
    write_json_atomic(output_path, payload)
    print(f"Wrote {len(payload['candidates'])} candidates to {output_path}")


def load_candidate_payloads(paths: list[Path]) -> tuple[list[dict], dict, str]:
    candidates_by_path = {}
    protocol = None
    reference_hash = None
    for path in paths:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        if payload.get("heldout_loaded") is not False:
            raise RuntimeError(f"Candidate shard is not validation-only: {path}")
        if protocol is None:
            protocol = payload["protocol"]
            reference_hash = payload["validation_reference_sha256"]
        elif payload["protocol"] != protocol:
            raise RuntimeError(f"Protocol mismatch in {path}")
        elif payload["validation_reference_sha256"] != reference_hash:
            raise RuntimeError(f"Validation reference mismatch in {path}")
        for candidate in payload["candidates"]:
            candidate = copy.deepcopy(candidate)
            artifact_dir = Path(candidate["artifact_dir"])
            if candidate.get("run", "").startswith("seed_"):
                candidate["run"] = artifact_dir.parent.name
            candidates_by_path[candidate["checkpoint_path"]] = candidate
    if protocol is None or reference_hash is None:
        raise RuntimeError("No candidate payloads were loaded")
    return list(candidates_by_path.values()), protocol, reference_hash


def write_summary_csv(path: Path, winner: dict, test: dict) -> None:
    fields = [
        "Split",
        "Model",
        "Checkpoint",
        "Normalized Table2+Table3 Score",
        *[f"Raw {name}" for name in SCORE_KEYS],
        "Precision",
        "Recall",
        *[f"Normalized {name}" for name in (
            "degree", "clustering", "orbit", "spectral", "diameter",
            "mmd_rbf", "f1_pr_error",
        )],
    ]

    def validation_row() -> dict:
        metric_summary = winner["summary"]["metrics"]
        component_summary = winner["summary"]["score_components"]
        row = {
            "Split": "validation_mean_over_rollouts",
            "Model": winner["run"],
            "Checkpoint": winner["checkpoint"],
            "Normalized Table2+Table3 Score": winner["selection_score"],
            "Precision": metric_summary["precision"]["mean"],
            "Recall": metric_summary["recall"]["mean"],
        }
        row.update(
            {f"Raw {name}": metric_summary[name]["mean"] for name in SCORE_KEYS}
        )
        row.update(
            {
                f"Normalized {name}": values["mean"]
                for name, values in component_summary.items()
            }
        )
        return row

    def test_row() -> dict:
        row = {
            "Split": "heldout_test",
            "Model": winner["run"],
            "Checkpoint": winner["checkpoint"],
            "Normalized Table2+Table3 Score": test[
                "normalized_table2_table3"
            ],
            "Precision": test["metrics"]["precision"],
            "Recall": test["metrics"]["recall"],
        }
        row.update(
            {f"Raw {name}": test["metrics"][name] for name in SCORE_KEYS}
        )
        row.update(
            {
                f"Normalized {name}": value
                for name, value in test["score_components"].items()
            }
        )
        return row

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(validation_row())
        writer.writerow(test_row())


def report_markdown(winner: dict, test: dict, protocol: dict) -> str:
    validation = winner["summary"]
    lines = [
        "# Lobster matrix selection: normalized_table2_table3",
        "",
        "Lower is better. Selection used validation only; the held-out test was loaded after the winner was frozen.",
        "",
        "## Selected checkpoint",
        "",
        f"- Run: `{winner['run']}`",
        f"- Checkpoint: `{winner['checkpoint']}`",
        f"- Validation score: `{winner['selection_score']:.12f}` "
        f"(std `{validation['score']['std']:.12f}` across "
        f"{protocol['validation_rollouts']} generation rollouts)",
        f"- Held-out test score: `{test['normalized_table2_table3']:.12f}`",
        "",
        "## Held-out test metrics",
        "",
        "| Metric | Raw | Normalized score component |",
        "| --- | ---: | ---: |",
    ]
    component_for_metric = {
        "degree": "degree",
        "clustering": "clustering",
        "orbit": "orbit",
        "spectral": "spectral",
        "diameter": "diameter",
        "mmd_rbf": "mmd_rbf",
        "f1_pr": "f1_pr_error",
    }
    for metric in SCORE_KEYS:
        component = component_for_metric[metric]
        lines.append(
            f"| {metric} | {test['metrics'][metric]:.12f} | "
            f"{test['score_components'][component]:.12f} |"
        )
    lines += [
        "",
        f"Precision: `{test['metrics']['precision']:.12f}`; "
        f"Recall: `{test['metrics']['recall']:.12f}`. They are reported but are not separate score components.",
        "",
        f"Table-3 test metrics use {test['test_gin_runs']} fresh Random-GIN "
        "initializations with degree, clustering, and square-clustering node features.",
    ]
    return "\n".join(lines) + "\n"


def run_finalize(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_threads)
    candidates, protocol, validation_hash = load_candidate_payloads(
        args.candidate_json
    )
    if len(candidates) != args.expected_candidates:
        raise RuntimeError(
            f"Expected {args.expected_candidates} unique candidates, found {len(candidates)}"
        )
    run_count = len({candidate["artifact_dir"] for candidate in candidates})
    if run_count != args.expected_runs:
        raise RuntimeError(
            f"Expected {args.expected_runs} runs, found {run_count}"
        )
    winner = min(
        candidates,
        key=lambda row: (
            row["selection_score"],
            row["summary"]["score"]["std"],
            row["checkpoint_path"],
        ),
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "validation_selection.json"
    if selection_path.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to replace frozen selection without --force: {selection_path}"
        )
    frozen_selection = {
        "schema_version": 1,
        "frozen_at": utc_now(),
        "selection_split": "validation",
        "selection_frozen_before_heldout_load": True,
        "heldout_loaded": False,
        "validation_reference_sha256": validation_hash,
        "protocol": protocol,
        "candidate_count": len(candidates),
        "run_count": run_count,
        "winner": winner,
        "candidates": sorted(
            candidates, key=lambda row: (row["selection_score"], row["checkpoint_path"])
        ),
    }
    # This write is intentionally completed before constructing or opening the
    # winner's held-out reference path.
    write_json_atomic(selection_path, frozen_selection)

    winner_dir = Path(winner["artifact_dir"])
    heldout_path = winner_dir / args.test_reference_filename
    if not heldout_path.is_file():
        raise FileNotFoundError(heldout_path)
    heldout_hash = sha256_file(heldout_path)
    heldout_items = np.load(heldout_path, allow_pickle=True)
    heldout_graphs = to_graphs(heldout_items, keep_largest_component=False)
    device = torch.device(args.device)
    decoder = legacy_selector.load_decoder(
        Path(winner["checkpoint_path"]), device, protocol["latent_dim"]
    )
    try:
        raw_graphs, generated_graphs = legacy_selector.generate(
            decoder,
            len(heldout_graphs),
            protocol["latent_dim"],
            device,
            args.test_generation_seed,
        )
    finally:
        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_result = evaluate_graph_set(
        heldout_graphs,
        generated_graphs,
        gin_runs=args.test_gin_runs,
        gin_seed=args.test_gin_seed,
        device=device,
    )
    test_result.update(
        {
            "schema_version": 1,
            "evaluated_at": utc_now(),
            "split": "heldout_test",
            "selected_from": str(selection_path),
            "selected_run": winner["run"],
            "selected_checkpoint": winner["checkpoint"],
            "selected_checkpoint_path": winner["checkpoint_path"],
            "test_reference_path": str(heldout_path),
            "test_reference_sha256": heldout_hash,
            "test_generation_seed": args.test_generation_seed,
            "test_gin_runs": args.test_gin_runs,
            "test_gin_seed": args.test_gin_seed,
            "mean_raw_edges": float(
                np.mean([graph.number_of_edges() for graph in raw_graphs])
            ),
            "mean_lcc_edges": float(
                np.mean([graph.number_of_edges() for graph in generated_graphs])
            ),
        }
    )

    selected_model_path = output_dir / "selected_model.pt"
    shutil.copy2(winner["checkpoint_path"], selected_model_path)
    generated_path = output_dir / "selected_test_generated.npy"
    save_graphs(generated_path, generated_graphs)
    test_result["materialized_model"] = str(selected_model_path)
    test_result["generated_test_graphs"] = str(generated_path)
    write_json_atomic(output_dir / "test_evaluation.json", test_result)
    write_summary_csv(output_dir / "summary.csv", winner, test_result)
    (output_dir / "report.md").write_text(
        report_markdown(winner, test_result, protocol), encoding="utf-8"
    )
    print(f"Selected {winner['run']}/{winner['checkpoint']}")
    print(f"Validation score: {winner['selection_score']:.12f}")
    print(
        "Held-out test normalized_table2_table3: "
        f"{test_result['normalized_table2_table3']:.12f}"
    )
    print(f"Wrote results to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate one shard of validation-only candidates."
    )
    evaluate_parser.add_argument("--run-dir", action="append", type=Path, required=True)
    evaluate_parser.add_argument("--output-json", type=Path, required=True)
    evaluate_parser.add_argument("--validation-filename", default="validationGraphs_adj_.npy")
    evaluate_parser.add_argument("--validation-rollouts", type=int, default=10)
    evaluate_parser.add_argument("--validation-seed", type=int, default=20260714)
    evaluate_parser.add_argument("--gin-runs", type=int, default=10)
    evaluate_parser.add_argument("--gin-seed", type=int, default=0)
    evaluate_parser.add_argument("--latent-dim", type=int, default=1024)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.add_argument("--torch-threads", type=int, default=2)
    evaluate_parser.add_argument("--expected-checkpoints-per-run", type=int, default=5)
    evaluate_parser.add_argument("--no-resume", action="store_true")

    finalize_parser = subparsers.add_parser(
        "finalize", help="Freeze the validation winner, then evaluate held-out test."
    )
    finalize_parser.add_argument(
        "--candidate-json", action="append", type=Path, required=True
    )
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser.add_argument("--expected-candidates", type=int, default=40)
    finalize_parser.add_argument("--expected-runs", type=int, default=8)
    finalize_parser.add_argument(
        "--test-reference-filename", default="heldoutTestGraphs_adj_.npy"
    )
    finalize_parser.add_argument("--test-generation-seed", type=int, default=21260714)
    finalize_parser.add_argument("--test-gin-runs", type=int, default=10)
    finalize_parser.add_argument("--test-gin-seed", type=int, default=0)
    finalize_parser.add_argument("--device", default="cpu")
    finalize_parser.add_argument("--torch-threads", type=int, default=2)
    finalize_parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "evaluate":
        run_evaluate(args)
    else:
        run_finalize(args)


if __name__ == "__main__":
    main()
