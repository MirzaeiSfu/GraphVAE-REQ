#!/usr/bin/env python3
"""Select and materialize one validation-best checkpoint per Lobster run.

This is a two-phase post-training workflow:

1. Evaluate every saved checkpoint against only ``validationGraphs_adj_.npy``
   and freeze one winner per run on disk.
2. Load ``heldoutTestGraphs_adj_.npy`` only after every winner is frozen, copy
   the selected models, and generate fresh held-out graph sets.

The generated artifacts use distinct posthoc filenames, so the final-epoch
graphs written during training are preserved.  A ready-to-run command for
``evaluate_graph_realism_batch.py`` is written beside the selection report.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import select_lobster_checkpoints as legacy_selector  # noqa: E402


DEFAULT_MODEL_FILENAME = "posthoc_best_validation_model.pt"
DEFAULT_METADATA_FILENAME = "posthoc_best_validation.json"
DEFAULT_GENERATED_FILENAME = (
    "Single_comp_generatedGraphs_adj_posthoc_best_validation.npy"
)
DEFAULT_REFERENCE_FILENAME = "heldoutTestGraphs_adj_.npy"
DEFAULT_THIRD_PARTY_JSON_FILENAME = (
    "graph_realism_random_gin_posthoc_best_validation.json"
)
DEFAULT_THIRD_PARTY_SUMMARY_FILENAME = (
    "graph_realism_posthoc_best_validation_summary.csv"
)
VALIDATION_REFERENCE_FILENAME = "validationGraphs_adj_.npy"


def validate_artifact_filenames(
    *,
    model_filename: str,
    metadata_filename: str,
    generated_filename: str,
    reference_filename: str,
    third_party_json_filename: str,
) -> None:
    """Reject paths or collisions that could overwrite run inputs/outputs."""
    filenames = {
        "model filename": model_filename,
        "metadata filename": metadata_filename,
        "generated filename": generated_filename,
        "held-out reference filename": reference_filename,
        "third-party JSON filename": third_party_json_filename,
    }
    for label, filename in filenames.items():
        path = Path(filename)
        if (
            not filename
            or path.is_absolute()
            or len(path.parts) != 1
            or path.name in {"", ".", ".."}
        ):
            raise ValueError(f"{label} must be a plain filename: {filename!r}")

    by_filename: dict[str, list[str]] = defaultdict(list)
    for label, filename in filenames.items():
        by_filename[filename].append(label)
    collisions = {
        filename: labels
        for filename, labels in by_filename.items()
        if len(labels) > 1
    }
    if collisions:
        details = "; ".join(
            f"{filename!r}: {', '.join(labels)}"
            for filename, labels in sorted(collisions.items())
        )
        raise ValueError(f"Artifact filenames collide: {details}")

    output_labels = (
        "model filename",
        "metadata filename",
        "generated filename",
        "third-party JSON filename",
    )
    validation_collisions = [
        label
        for label in output_labels
        if filenames[label] == VALIDATION_REFERENCE_FILENAME
    ]
    if validation_collisions:
        raise ValueError(
            f"Artifact filenames collide with validation input "
            f"{VALIDATION_REFERENCE_FILENAME!r}: {', '.join(validation_collisions)}"
        )
    if reference_filename == VALIDATION_REFERENCE_FILENAME:
        raise ValueError(
            "Held-out reference filename must not name the validation input "
            f"{VALIDATION_REFERENCE_FILENAME!r}"
        )


def validate_posthoc_options(args: argparse.Namespace) -> None:
    if args.third_party_repeats <= 0:
        raise ValueError("--third-party-repeats must be positive")
    if args.third_party_max_graphs <= 0:
        raise ValueError("--third-party-max-graphs must be positive")
    validate_artifact_filenames(
        model_filename=args.model_filename,
        metadata_filename=args.metadata_filename,
        generated_filename=args.generated_filename,
        reference_filename=args.reference_filename,
        third_party_json_filename=args.third_party_json_filename,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        action="append",
        required=True,
        help="Collected run root; repeat for multiple collection roots.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=None,
        help="Fail before held-out materialization unless this many runs are found.",
    )
    parser.add_argument(
        "--expected-checkpoints-per-run",
        type=int,
        default=None,
        help="Fail validation selection unless every run has this many checkpoints.",
    )
    parser.add_argument("--validation-rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="Latent seed for each selected model; defaults to --seed + 1000000.",
    )
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--stability-weight", type=float, default=0.25)
    parser.add_argument("--dense-penalty-weight", type=float, default=1.0)
    parser.add_argument(
        "--edge-mean-penalty-weight",
        type=float,
        default=0.25,
        help="Penalty for abs(log(generated validation edges/reference edges)).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--skip-materialization",
        action="store_true",
        help="Freeze validation selections without loading held-out references.",
    )
    parser.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    parser.add_argument("--metadata-filename", default=DEFAULT_METADATA_FILENAME)
    parser.add_argument("--generated-filename", default=DEFAULT_GENERATED_FILENAME)
    parser.add_argument("--reference-filename", default=DEFAULT_REFERENCE_FILENAME)
    parser.add_argument(
        "--run-third-party-eval",
        action="store_true",
        help="Run Random-GIN after all per-run selections and graph sets are frozen.",
    )
    parser.add_argument("--third-party-repeats", type=int, default=10)
    parser.add_argument("--third-party-max-graphs", type=int, default=1000)
    parser.add_argument("--third-party-seed", type=int, default=0)
    parser.add_argument(
        "--third-party-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--third-party-json-filename",
        default=DEFAULT_THIRD_PARTY_JSON_FILENAME,
    )
    parser.add_argument(
        "--no-third-party-structural-features",
        action="store_true",
    )
    args = parser.parse_args()
    if args.validation_rollouts <= 0:
        parser.error("--validation-rollouts must be positive")
    if args.expected_runs is not None and args.expected_runs <= 0:
        parser.error("--expected-runs must be positive")
    if (
        args.expected_checkpoints_per_run is not None
        and args.expected_checkpoints_per_run <= 0
    ):
        parser.error("--expected-checkpoints-per-run must be positive")
    if args.run_third_party_eval and args.skip_materialization:
        parser.error("--run-third-party-eval requires materialization")
    try:
        validate_posthoc_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def discover_validation_paths(runs_roots: list[Path]) -> list[tuple[Path, Path]]:
    """Return unique ``(root, validation_path)`` pairs in stable order."""
    discovered: dict[Path, tuple[Path, Path]] = {}
    missing_roots = []
    for raw_root in runs_roots:
        root = raw_root.expanduser().resolve()
        if not root.exists():
            missing_roots.append(root)
            continue
        for validation_path in sorted(root.rglob("validationGraphs_adj_.npy")):
            resolved = validation_path.resolve()
            discovered.setdefault(resolved, (root, resolved))
    if missing_roots:
        missing = ", ".join(str(path) for path in missing_roots)
        raise FileNotFoundError(f"Collected run roots do not exist: {missing}")
    return list(discovered.values())


def display_run_name(run_dir: Path, runs_root: Path) -> str:
    try:
        relative = run_dir.relative_to(runs_root)
    except ValueError:
        return str(run_dir)
    return str(relative)


def checkpoint_selection_score(
    validation: dict,
    reference_mean_edges: float,
    stability_weight: float,
    dense_penalty_weight: float,
    edge_mean_penalty_weight: float,
) -> tuple[float, float]:
    generated_mean_edges = validation["mean_raw_edges"]["mean"]
    edge_mean_log_error = abs(
        np.log(
            max(generated_mean_edges, 1e-12)
            / max(reference_mean_edges, 1e-12)
        )
    )
    score = (
        validation["score"]["median"]
        + stability_weight * validation["score"]["std"]
        + dense_penalty_weight * validation["dense_rate"]
        + edge_mean_penalty_weight * edge_mean_log_error
    )
    return float(score), float(edge_mean_log_error)


def select_best_per_run(candidates: list[dict]) -> list[dict]:
    """Choose the lowest validation score independently for each run."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["artifact_dir"]].append(candidate)
    winners = [
        min(run_candidates, key=lambda row: row["selection_score"])
        for run_candidates in grouped.values()
    ]
    return sorted(winners, key=lambda row: (row["run"], row["checkpoint"]))


def select_candidates(args: argparse.Namespace, device: torch.device) -> list[dict]:
    """Evaluate all checkpoints without opening any held-out reference file."""
    validation_paths = discover_validation_paths(args.runs_root)
    if not validation_paths:
        roots = ", ".join(str(path) for path in args.runs_root)
        raise FileNotFoundError(f"No validation graph files found under: {roots}")

    candidates = []
    with legacy_selector.locked_orca_tmp():
        for runs_root, validation_path in validation_paths:
            run_dir = validation_path.parent
            refs = legacy_selector.to_graphs(
                np.load(validation_path, allow_pickle=True),
                keep_largest_component=False,
            )
            if not refs:
                raise ValueError(f"Validation reference set is empty: {validation_path}")
            reference_edges = np.asarray(
                [graph.number_of_edges() for graph in refs], dtype=float
            )
            reference_mean_edges = float(reference_edges.mean())
            dense_threshold = float(reference_edges.mean() + 3 * reference_edges.std())
            checkpoint_paths = legacy_selector.checkpoints(run_dir)
            if not checkpoint_paths:
                raise FileNotFoundError(f"No checkpoints found in completed run: {run_dir}")
            expected_checkpoints = getattr(
                args, "expected_checkpoints_per_run", None
            )
            if (
                expected_checkpoints is not None
                and len(checkpoint_paths) != expected_checkpoints
            ):
                raise RuntimeError(
                    f"Expected {expected_checkpoints} checkpoints in {run_dir}, "
                    f"found {len(checkpoint_paths)}"
                )

            for checkpoint_path in checkpoint_paths:
                run_name = display_run_name(run_dir, runs_root)
                print(
                    f"[validation] {run_name}/{checkpoint_path.name}",
                    flush=True,
                )
                decoder = legacy_selector.load_decoder(
                    checkpoint_path, device, args.latent_dim
                )
                validation = legacy_selector.evaluate(
                    decoder,
                    refs,
                    args.validation_rollouts,
                    args.seed,
                    args.latent_dim,
                    device,
                    dense_threshold,
                )
                selection_score, edge_mean_log_error = checkpoint_selection_score(
                    validation,
                    reference_mean_edges,
                    args.stability_weight,
                    args.dense_penalty_weight,
                    args.edge_mean_penalty_weight,
                )
                candidates.append(
                    {
                        "run": run_name,
                        "artifact_dir": str(run_dir),
                        "validation_graphs": str(validation_path),
                        "checkpoint": checkpoint_path.name,
                        "checkpoint_path": str(checkpoint_path),
                        "dense_threshold": dense_threshold,
                        "reference_mean_edges": reference_mean_edges,
                        "edge_mean_log_error": edge_mean_log_error,
                        "selection_score": selection_score,
                        "validation": validation,
                    }
                )
                del decoder
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return candidates


def generated_graph_arrays(graphs: list[nx.Graph]) -> np.ndarray:
    matrices = [nx.to_numpy_array(graph, dtype=np.int8) for graph in graphs]
    result = np.empty(len(matrices), dtype=object)
    result[:] = matrices
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preflight_materialization(winners: list[dict], reference_filename: str) -> None:
    missing = []
    for winner in winners:
        checkpoint_path = Path(winner["checkpoint_path"])
        reference_path = Path(winner["artifact_dir"]) / reference_filename
        if not checkpoint_path.is_file():
            missing.append(str(checkpoint_path))
        if not reference_path.is_file():
            missing.append(str(reference_path))
    if missing:
        raise FileNotFoundError(
            "Cannot materialize frozen selections; missing files: " + ", ".join(missing)
        )


def materialize_selected_runs(
    winners: list[dict],
    *,
    device: torch.device,
    latent_dim: int,
    generation_seed: int,
    model_filename: str,
    metadata_filename: str,
    generated_filename: str,
    reference_filename: str,
) -> list[dict]:
    """Copy winners and generate held-out outputs after selection is frozen."""
    preflight_materialization(winners, reference_filename)
    materialized = []
    for winner in winners:
        run_dir = Path(winner["artifact_dir"])
        checkpoint_path = Path(winner["checkpoint_path"])
        reference_path = run_dir / reference_filename
        model_copy_path = run_dir / model_filename
        generated_path = run_dir / generated_filename
        metadata_path = run_dir / metadata_filename

        # This is the first point at which held-out data are loaded.  The
        # complete per-run selection manifest has already been written.
        heldout_items = np.load(reference_path, allow_pickle=True)
        decoder = legacy_selector.load_decoder(checkpoint_path, device, latent_dim)
        _, largest_components = legacy_selector.generate(
            decoder,
            len(heldout_items),
            latent_dim,
            device,
            generation_seed,
        )

        if checkpoint_path.resolve() != model_copy_path.resolve():
            shutil.copy2(checkpoint_path, model_copy_path)
        # Passing a file handle keeps custom names exact; np.save(path) would
        # silently append .npy and could evade the collision checks above.
        with generated_path.open("wb") as handle:
            np.save(
                handle,
                generated_graph_arrays(largest_components),
                allow_pickle=True,
            )
        per_run_metadata = {
            "selection_split": "validation",
            "selection_frozen_before_heldout_load": True,
            "selected_checkpoint": winner,
            "materialized_model": str(model_copy_path),
            "generated_graphs": str(generated_path),
            "reference_graphs": str(reference_path),
            "generation_seed": generation_seed,
            "num_generated_graphs": len(largest_components),
            "num_reference_graphs": len(heldout_items),
        }
        write_json(metadata_path, per_run_metadata)
        result = {
            "run": winner["run"],
            "artifact_dir": str(run_dir),
            "selected_checkpoint": str(checkpoint_path),
            "materialized_model": str(model_copy_path),
            "metadata": str(metadata_path),
            "generated_graphs": str(generated_path),
            "reference_graphs": str(reference_path),
            "generation_seed": generation_seed,
            "num_generated_graphs": len(largest_components),
            "num_reference_graphs": len(heldout_items),
        }
        materialized.append(result)
        print(
            f"[materialized] {winner['run']}: {checkpoint_path.name} -> "
            f"{generated_path.name}",
            flush=True,
        )
        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return materialized


def build_random_gin_command(
    run_dirs: list[Path],
    *,
    generated_filename: str,
    reference_filename: str,
    json_filename: str,
    summary_csv: Path,
    repeats: int,
    max_graphs: int,
    seed: int,
    device: str,
    structural_features: bool,
    python_executable: str = sys.executable,
) -> list[str]:
    command = [
        python_executable,
        str(ROOT / "scripts" / "evaluate_graph_realism_batch.py"),
    ]
    for run_dir in run_dirs:
        command += ["--run-dir", str(run_dir)]
    command += [
        "--generated-filename",
        generated_filename,
        "--reference-filename",
        reference_filename,
        "--json-filename",
        json_filename,
        "--summary-csv",
        str(summary_csv),
        "--repeats",
        str(repeats),
        "--max-graphs",
        str(max_graphs),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if not structural_features:
        command.append("--no-structural-features")
    return command


def report_markdown(payload: dict) -> str:
    lines = [
        "# Per-run Lobster posthoc checkpoint selection",
        "",
        (
            "Every checkpoint was ranked using validation graphs only. All per-run "
            "winners were frozen before any held-out reference was loaded."
        ),
        "",
        "| Run | Selected checkpoint | Validation median | Std | Dense rate | Selection score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for winner in payload["winners"]:
        validation = winner["validation"]
        lines.append(
            f"| {winner['run']} | {winner['checkpoint']} | "
            f"{validation['score']['median']:.6f} | "
            f"{validation['score']['std']:.6f} | "
            f"{validation['dense_rate']:.2%} | "
            f"{winner['selection_score']:.6f} |"
        )
    global_winner = payload.get("global_validation_winner")
    if global_winner:
        lines += [
            "",
            "## Global validation winner",
            "",
            f"`{global_winner['run']}/{global_winner['checkpoint']}`",
        ]
    if payload.get("materialized"):
        lines += [
            "",
            "## Held-out materialization",
            "",
            (
                f"Fresh graph sets were generated for {len(payload['materialized'])} "
                "frozen per-run winners. Random-GIN must be treated as final reporting, "
                "not as another checkpoint or weight-selection signal."
            ),
            "",
            f"Evaluator command: `{payload['third_party_command']}`",
        ]
    return "\n".join(lines) + "\n"


def write_selection_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "selection.json", payload)
    (output_dir / "report.md").write_text(
        report_markdown(payload), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> dict:
    validate_posthoc_options(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.expanduser().resolve()
    candidates = select_candidates(args, device)
    if not candidates:
        raise RuntimeError("No checkpoint candidates were evaluated")
    winners = select_best_per_run(candidates)
    expected_runs = getattr(args, "expected_runs", None)
    if expected_runs is not None and len(winners) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} completed runs, found {len(winners)}"
        )
    global_winner = min(winners, key=lambda row: row["selection_score"])
    generation_seed = (
        args.seed + 1_000_000
        if args.generation_seed is None
        else args.generation_seed
    )
    payload = {
        "runs_root": [str(path) for path in args.runs_root],
        "device": str(device),
        "validation_rollouts": args.validation_rollouts,
        "validation_seed": args.seed,
        "selection_scope": "one_checkpoint_per_run",
        "selection_split": "validation",
        "selection_frozen_before_heldout_load": True,
        "heldout_materialized": False,
        "selection_formula": (
            "median_normalized_mmd + stability_weight*std + "
            "dense_penalty_weight*dense_rate + "
            "edge_mean_penalty_weight*abs(log(generated_mean_edges/reference_mean_edges))"
        ),
        "stability_weight": args.stability_weight,
        "dense_penalty_weight": args.dense_penalty_weight,
        "edge_mean_penalty_weight": args.edge_mean_penalty_weight,
        "candidates": candidates,
        "winners": winners,
        "global_validation_winner": global_winner,
        "materialized": [],
        "third_party_command": None,
        "third_party_completed": False,
    }

    # Persist all decisions before the code is allowed to load held-out data.
    write_selection_outputs(output_dir, payload)
    if args.skip_materialization:
        print(f"Frozen {len(winners)} per-run validation selections")
        return payload

    materialized = materialize_selected_runs(
        winners,
        device=device,
        latent_dim=args.latent_dim,
        generation_seed=generation_seed,
        model_filename=args.model_filename,
        metadata_filename=args.metadata_filename,
        generated_filename=args.generated_filename,
        reference_filename=args.reference_filename,
    )
    summary_csv = output_dir / DEFAULT_THIRD_PARTY_SUMMARY_FILENAME
    third_party_command = build_random_gin_command(
        [Path(item["artifact_dir"]) for item in materialized],
        generated_filename=args.generated_filename,
        reference_filename=args.reference_filename,
        json_filename=args.third_party_json_filename,
        summary_csv=summary_csv,
        repeats=args.third_party_repeats,
        max_graphs=args.third_party_max_graphs,
        seed=args.third_party_seed,
        device=args.third_party_device,
        structural_features=not args.no_third_party_structural_features,
    )
    payload["heldout_materialized"] = True
    payload["generation_seed"] = generation_seed
    payload["materialized"] = materialized
    payload["third_party_command"] = shlex.join(third_party_command)
    command_path = output_dir / "run_random_gin.sh"
    command_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shlex.join(third_party_command)
        + "\n",
        encoding="utf-8",
    )
    command_path.chmod(0o755)
    write_selection_outputs(output_dir, payload)

    if args.run_third_party_eval:
        subprocess.run(third_party_command, cwd=ROOT, check=True)
        payload["third_party_completed"] = True
        write_selection_outputs(output_dir, payload)
    else:
        print(f"Random-GIN command written to {command_path}")
    print(f"Selected and materialized {len(winners)} runs")
    return payload


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
