#!/usr/bin/env python3
"""Bayesian optimization of GraphVAE attribute reconstruction-loss weights.

The optimization objective is validation Attr-F1PR: the mean ``f1_pr`` from
the attributed Random-GIN evaluator's ``decoded_node_edge`` mode. Training and
evaluation run in subprocesses so every trial starts with fresh process state.
The held-out test split is available only through the explicit
``--evaluate-best-on-test`` mode after a study has selected its best trial.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyYAML is required. Install the repository requirements.") from exc

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:  # pragma: no cover - handled by require_optuna
    optuna = None
    TrialState = None


REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_NAME = "Attr-F1PR"
OBJECTIVE_JSON_PATH = (
    "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
)
PRIMARY_MODE = "decoded_node_edge"
DEFAULT_SPLIT_SEED = 123
DEFAULT_TRAINING_SEED = 0
DEFAULT_GENERATION_SEED = 123
DEFAULT_EVALUATOR_SEED = 0
DEFAULT_EVALUATOR_REPEATS = 5
DEFAULT_CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)_(\d+)$")


class TrialExecutionError(RuntimeError):
    """A recoverable trial failure that Optuna should record and continue past."""


@dataclass(frozen=True)
class SearchRanges:
    alpha_node_feat: tuple[float, float]
    alpha_edge_feat: tuple[float, float]
    alpha_motif_loss: tuple[float, float] | None = None


@dataclass(frozen=True)
class AttrF1PRMetrics:
    f1_pr: float
    precision: float
    recall: float
    graph_count: int


def require_optuna():
    if optuna is None:
        raise RuntimeError(
            "Optuna is required for Bayesian optimization. Install the updated "
            "repository requirements (optuna>=3.6,<5). This workflow never "
            "falls back to grid or random search."
        )


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return payload


def flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the one-level sections accepted by ``main.py``."""

    flat: dict[str, Any] = {}
    for section, value in config.items():
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                if key in flat:
                    raise ValueError(
                        f"Duplicate configuration key {key!r} in section {section!r}."
                    )
                flat[str(key)] = nested_value
        else:
            if section in flat:
                raise ValueError(f"Duplicate configuration key {section!r}.")
            flat[str(section)] = value
    return flat


def set_config_value(
    config: dict[str, Any],
    key: str,
    value: Any,
    *,
    preferred_section: str,
) -> None:
    """Update one existing flattened key or add it to its conventional section."""

    locations: list[dict[str, Any]] = []
    if key in config and not isinstance(config[key], Mapping):
        locations.append(config)
    for section_value in config.values():
        if isinstance(section_value, dict) and key in section_value:
            locations.append(section_value)
    if len(locations) > 1:
        raise ValueError(f"Configuration key {key!r} occurs more than once.")
    if locations:
        locations[0][key] = value
        return
    section = config.setdefault(preferred_section, {})
    if not isinstance(section, dict):
        raise ValueError(
            f"Cannot add {key!r}: preferred section {preferred_section!r} is not a mapping."
        )
    section[key] = value


def validate_log_range(name: str, bounds: tuple[float, float]) -> None:
    low, high = bounds
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"{name} bounds must be finite, got {bounds}.")
    if low <= 0.0 or high <= low:
        raise ValueError(
            f"{name} log-scale bounds require 0 < low < high, got {bounds}."
        )


def build_search_ranges(args: argparse.Namespace) -> SearchRanges:
    ranges = SearchRanges(
        alpha_node_feat=(args.alpha_node_feat_min, args.alpha_node_feat_max),
        alpha_edge_feat=(args.alpha_edge_feat_min, args.alpha_edge_feat_max),
        alpha_motif_loss=(args.alpha_motif_min, args.alpha_motif_max)
        if args.tune_alpha_motif
        else None,
    )
    validate_log_range("alpha_node_feat", ranges.alpha_node_feat)
    validate_log_range("alpha_edge_feat", ranges.alpha_edge_feat)
    if ranges.alpha_motif_loss is not None:
        validate_log_range("alpha_motif_loss", ranges.alpha_motif_loss)
    return ranges


def sample_search_space(trial, ranges: SearchRanges) -> dict[str, float]:
    """Sample only the requested loss weights, always on a log scale."""

    parameters = {
        "alpha_node_feat": trial.suggest_float(
            "alpha_node_feat", *ranges.alpha_node_feat, log=True
        ),
        "alpha_edge_feat": trial.suggest_float(
            "alpha_edge_feat", *ranges.alpha_edge_feat, log=True
        ),
    }
    if ranges.alpha_motif_loss is not None:
        parameters["alpha_motif_loss"] = trial.suggest_float(
            "alpha_motif_loss", *ranges.alpha_motif_loss, log=True
        )
    return parameters


def inject_sampled_parameters(
    base_config: Mapping[str, Any],
    sampled_parameters: Mapping[str, float],
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(base_config))
    allowed = {"alpha_node_feat", "alpha_edge_feat", "alpha_motif_loss"}
    unexpected = sorted(set(sampled_parameters) - allowed)
    if unexpected:
        raise ValueError(f"Unsupported sampled parameters: {', '.join(unexpected)}")
    for key, value in sampled_parameters.items():
        set_config_value(
            resolved,
            key,
            float(value),
            preferred_section="loss",
        )
    return resolved


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def validate_base_config(config: Mapping[str, Any], tune_alpha_motif: bool) -> None:
    """Reject configurations that cannot provide an isolated validation objective."""

    flat = flatten_config(config)
    if flat.get("split_mode", "legacy_80_20") != "paper_70_10_20":
        raise ValueError(
            "Attr-F1PR optimization requires split_mode=paper_70_10_20 so the "
            "validation split is distinct from both training and held-out test data."
        )
    val_fraction = flat.get("val_fraction", 0.1)
    if val_fraction is None:
        val_fraction = 0.1
    if float(val_fraction) <= 0.0:
        raise ValueError("Attr-F1PR optimization requires a positive validation split.")
    if _as_bool(flat.get("ideal_Evalaution", False)):
        raise ValueError(
            "ideal_Evalaution accesses held-out data and must be false for optimization."
        )
    if _as_bool(flat.get("disable_dataset_cache", False)):
        raise ValueError(
            "disable_dataset_cache must be false so all trials use the same cached split."
        )
    if _as_bool(flat.get("sanity_check_only", False)):
        raise ValueError("sanity_check_only must be false so each trial trains a model.")
    if _as_bool(flat.get("tiny_overfit", False)):
        raise ValueError("tiny_overfit is a debug mode and does not save a trial checkpoint.")
    if flat.get("task", "graphGeneration") != "graphGeneration":
        raise ValueError("The base configuration task must be graphGeneration.")
    if int(flat.get("epoch_number", 0)) < 1:
        raise ValueError("The base configuration must define a positive epoch_number.")
    if tune_alpha_motif and not _as_bool(flat.get("motif_loss", False)):
        raise ValueError(
            "--tune-alpha-motif requires motif_loss=true in the base configuration."
        )


def resolve_trial_config(
    base_config: Mapping[str, Any],
    sampled_parameters: Mapping[str, float],
    *,
    trial_number: int,
    trial_directory: Path,
    training_seed: int,
    split_seed: int,
    device: str,
) -> dict[str, Any]:
    """Create the exact training configuration for one isolated trial."""

    resolved = inject_sampled_parameters(base_config, sampled_parameters)
    fixed_values = (
        ("seed", int(training_seed), "data"),
        ("split_seed", int(split_seed), "data"),
        ("device", str(device), "runtime"),
        ("graph_save_path", str((trial_directory / "training").resolve()), "runtime"),
        ("run_label", f"attr-f1pr-bo-trial-{trial_number:05d}", "runtime"),
        ("skip_final_evaluation", True, "runtime"),
        ("third_party_eval", False, "runtime"),
        ("plot_testGraphs", False, "runtime"),
    )
    for key, value, section in fixed_values:
        set_config_value(resolved, key, value, preferred_section=section)
    return resolved


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_study_definition(
    path: Path,
    definition: Mapping[str, Any],
    *,
    existing_trial_count: int,
) -> None:
    """Prevent a resumed SQLite study from mixing incompatible trial settings."""

    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            recorded = json.load(handle)
        if recorded != definition:
            differing = sorted(
                key
                for key in set(recorded) | set(definition)
                if recorded.get(key) != definition.get(key)
            )
            raise ValueError(
                "Study settings differ from the persisted study definition: "
                + ", ".join(differing)
                + ". Use the original settings or a new output directory/study."
            )
        return
    if existing_trial_count:
        raise ValueError(
            f"Cannot safely resume {existing_trial_count} existing trial(s): "
            f"study definition is missing at {path}."
        )
    write_json(path, definition)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_trial_directory(output_dir: Path, trial_number: int) -> Path:
    trial_dir = output_dir / "trials" / f"trial_{trial_number:05d}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    return trial_dir


def run_logged_command(
    command: Sequence[str],
    *,
    log_path: Path,
    environment: Mapping[str, str],
    timeout_seconds: float | None,
) -> float:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND: " + " ".join(command) + "\n\n")
        log_handle.flush()
        try:
            result = subprocess.run(
                list(command),
                cwd=REPO_ROOT,
                env=dict(environment),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TrialExecutionError(
                f"Command exceeded timeout {timeout_seconds}s; see {log_path}."
            ) from exc
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise TrialExecutionError(
            f"Command failed with exit code {result.returncode}; see {log_path}."
        )
    return elapsed


def discover_final_checkpoint(run_dir: Path, epoch_number: int) -> Path:
    expected_epoch = epoch_number - 1
    candidates: list[tuple[int, int, Path]] = []
    for path in run_dir.glob("model_*_*"):
        match = DEFAULT_CHECKPOINT_PATTERN.match(path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), int(match.group(2)), path))
    expected = [item for item in candidates if item[0] == expected_epoch]
    if not expected:
        found = ", ".join(path.name for _, _, path in sorted(candidates)) or "none"
        raise TrialExecutionError(
            f"Final epoch checkpoint model_{expected_epoch}_* was not found in "
            f"{run_dir}; candidates: {found}."
        )
    return max(expected, key=lambda item: (item[1], item[2].stat().st_mtime_ns))[2].resolve()


def validate_feature_head_keys(state_dict_keys: Sequence[str]) -> None:
    normalized = [key[7:] if key.startswith("module.") else key for key in state_dict_keys]
    if not any(key.startswith("node_feature_decoder.") for key in normalized):
        raise TrialExecutionError(
            "Checkpoint has no node_feature_decoder parameters; topology-only "
            "or missing-feature checkpoints cannot define Attr-F1PR."
        )
    if not any(key.startswith("edge_feature_decoder.") for key in normalized):
        raise TrialExecutionError(
            "Checkpoint has no edge_feature_decoder parameters; decoded_node_edge "
            "Attr-F1PR requires generated edge attributes."
        )


def validate_checkpoint_feature_heads(checkpoint_path: Path) -> None:
    """Preflight both decoder heads without importing Torch in unit-only workflows."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training environment guard
        raise RuntimeError("PyTorch is required to inspect GraphVAE checkpoints.") from exc
    try:
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:  # Torch versions before weights_only
        payload = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TrialExecutionError(f"Unsupported checkpoint payload: {checkpoint_path}")
    validate_feature_head_keys(list(payload))


def build_evaluator_command(
    *,
    python_bin: str,
    run_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    split: str,
    generation_seed: int,
    evaluator_seed: int,
    evaluator_repeats: int,
    max_graphs: int,
    generation_batch_size: int,
    nearest_k: int,
    adjacency_threshold: float,
    device: str,
) -> list[str]:
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported evaluator split: {split}")
    return [
        python_bin,
        str(REPO_ROOT / "scripts" / "evaluate_attributed_graph_realism_checkpoints.py"),
        "--run-dir",
        str(run_dir),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        split,
        "--modes",
        PRIMARY_MODE,
        "--max-graphs",
        str(max_graphs),
        "--generation-batch-size",
        str(generation_batch_size),
        "--generation-seed",
        str(generation_seed),
        "--evaluator-seed",
        str(evaluator_seed),
        "--repeats",
        str(evaluator_repeats),
        "--nearest-k",
        str(nearest_k),
        "--adjacency-threshold",
        str(adjacency_threshold),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]


def _finite_metric(summary: Mapping[str, Any], metric_name: str) -> float:
    try:
        value = float(summary[metric_name]["mean"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrialExecutionError(
            f"Attributed evaluator output is missing {metric_name}.mean."
        ) from exc
    if not math.isfinite(value):
        raise TrialExecutionError(
            f"Attributed evaluator returned non-finite {metric_name}.mean={value}."
        )
    if not 0.0 <= value <= 1.0:
        raise TrialExecutionError(
            f"Attributed evaluator returned invalid {metric_name}.mean={value}."
        )
    return value


def parse_attr_f1pr_payload(
    payload: Mapping[str, Any],
    *,
    expected_split: str,
) -> AttrF1PRMetrics:
    """Parse Attr-F1PR structurally and reject every topology/feature fallback."""

    if payload.get("split") != expected_split:
        raise TrialExecutionError(
            f"Expected {expected_split!r} evaluator split, got {payload.get('split')!r}."
        )
    if payload.get("primary_mode") != PRIMARY_MODE:
        raise TrialExecutionError(
            "Attr-F1PR requires primary_mode=decoded_node_edge; topology-only and "
            "node-only evaluator outputs are rejected."
        )
    try:
        evaluation = payload["evaluation"]
        dimensions = evaluation["feature_dimensions"]
        summary = evaluation["modes"][PRIMARY_MODE]["summary"]
        graph_count = int(payload["graph_counts"]["accepted_per_collection"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrialExecutionError(
            f"Evaluator output does not contain {OBJECTIVE_JSON_PATH}."
        ) from exc
    if int(dimensions.get("node", 0)) <= 0 or int(dimensions.get("edge", 0)) <= 0:
        raise TrialExecutionError(
            "decoded_node_edge requires positive matching node and edge feature dimensions."
        )
    feature_source = payload.get("feature_source", {})
    if feature_source.get("hand_made_topology_features") is not False:
        raise TrialExecutionError(
            "Evaluator output did not explicitly disable hand-made topology features."
        )
    generated_source = str(feature_source.get("generated", ""))
    if "node_feature_decoder" not in generated_source or "edge_feature_decoder" not in generated_source:
        raise TrialExecutionError(
            "Evaluator output does not attest that both GraphVAE attribute decoders were used."
        )
    if graph_count < 3:
        raise TrialExecutionError(
            f"Attr-F1PR requires at least three accepted graphs, got {graph_count}."
        )
    return AttrF1PRMetrics(
        f1_pr=_finite_metric(summary, "f1_pr"),
        precision=_finite_metric(summary, "precision"),
        recall=_finite_metric(summary, "recall"),
        graph_count=graph_count,
    )


def parse_attr_f1pr_file(path: Path, *, expected_split: str) -> AttrF1PRMetrics:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_attr_f1pr_payload(payload, expected_split=expected_split)


def _mock_evaluator_payload(
    parameters: Mapping[str, float],
    *,
    split: str,
    graph_count: int = 8,
) -> dict[str, Any]:
    node_log = math.log10(parameters["alpha_node_feat"])
    edge_log = math.log10(parameters["alpha_edge_feat"])
    motif_log = math.log10(parameters.get("alpha_motif_loss", 1.0))
    score = max(
        0.0,
        min(1.0, 0.92 - 0.035 * (node_log - 0.4) ** 2 - 0.045 * (edge_log + 0.1) ** 2 - 0.005 * motif_log**2),
    )
    precision = max(0.0, min(1.0, score + 0.01))
    recall = max(0.0, min(1.0, score - 0.01))
    return {
        "schema_version": "attributed-random-gin-v1",
        "split": split,
        "primary_mode": PRIMARY_MODE,
        "graph_counts": {"accepted_per_collection": graph_count},
        "feature_source": {
            "generated": "GraphVAE node_feature_decoder and edge_feature_decoder",
            "reference": "cached dataset node and edge one-hot attributes",
            "hand_made_topology_features": False,
        },
        "evaluation": {
            "feature_dimensions": {"node": 4, "edge": 3},
            "modes": {
                PRIMARY_MODE: {
                    "summary": {
                        "f1_pr": {"mean": score},
                        "precision": {"mean": precision},
                        "recall": {"mean": recall},
                    }
                }
            },
        },
    }


def _trial_environment(training_seed: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(training_seed)
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return environment


def execute_trial(
    trial,
    *,
    args: argparse.Namespace,
    base_config: Mapping[str, Any],
    ranges: SearchRanges,
    output_dir: Path,
    split_seed: int,
) -> float:
    sampled_parameters = sample_search_space(trial, ranges)
    trial_dir = create_trial_directory(output_dir, trial.number)
    resolved_config = resolve_trial_config(
        base_config,
        sampled_parameters,
        trial_number=trial.number,
        trial_directory=trial_dir,
        training_seed=args.training_seed,
        split_seed=split_seed,
        device=args.device,
    )
    config_path = trial_dir / "resolved_config.yaml"
    write_yaml(config_path, resolved_config)
    flat_config = flatten_config(resolved_config)
    epoch_number = int(flat_config["epoch_number"])
    run_dir = (trial_dir / "training" / f"seed_{args.training_seed}").resolve()
    evaluator_output_dir = (trial_dir / "validation_evaluation").resolve()
    result_path = trial_dir / "trial_result.json"
    started = time.monotonic()
    record: dict[str, Any] = {
        "schema_version": "graphvae-attr-f1pr-bo-trial-v1",
        "objective": OBJECTIVE_NAME,
        "objective_json_path": OBJECTIVE_JSON_PATH,
        "trial_number": trial.number,
        "status": "RUNNING",
        "sampled_weights": sampled_parameters,
        "resolved_config": str(config_path.resolve()),
        "training_seed": args.training_seed,
        "split_seed": split_seed,
        "generation_seed": args.generation_seed,
        "evaluator_seed": args.evaluator_seed,
        "evaluator_seeds": [
            args.evaluator_seed + repeat for repeat in range(args.evaluator_repeats)
        ],
        "evaluator_repeats": args.evaluator_repeats,
        "fixed_generated_graph_limit": args.max_graphs,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "validation_attr_f1pr": None,
        "validation_precision": None,
        "validation_recall": None,
        "accepted_validation_graphs": None,
        "training_elapsed_seconds": None,
        "evaluation_elapsed_seconds": None,
        "total_elapsed_seconds": None,
        "failure_reason": None,
    }
    write_json(result_path, record)

    phase = "training"
    phase_started = time.monotonic()
    try:
        if args.mock:
            if trial.number in set(args.mock_fail_trial):
                raise TrialExecutionError(f"Mock failure requested for trial {trial.number}.")
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = run_dir / f"model_{epoch_number - 1}_0"
            checkpoint_path.write_bytes(
                json.dumps({"mock": True, "parameters": sampled_parameters}).encode("utf-8")
            )
            (trial_dir / "training_subprocess.log").write_text(
                "Mock training completed.\n", encoding="utf-8"
            )
            training_elapsed = time.monotonic() - phase_started
            record.update(
                {
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "training_elapsed_seconds": training_elapsed,
                }
            )
            write_json(result_path, record)
            phase = "evaluation"
            phase_started = time.monotonic()
            evaluator_output_dir.mkdir(parents=True, exist_ok=True)
            evaluator_json_path = evaluator_output_dir / "attributed_random_gin.json"
            write_json(
                evaluator_json_path,
                _mock_evaluator_payload(sampled_parameters, split="validation"),
            )
            (trial_dir / "evaluation_subprocess.log").write_text(
                "Mock decoded_node_edge evaluation completed.\n", encoding="utf-8"
            )
            evaluation_elapsed = time.monotonic() - phase_started
        else:
            environment = _trial_environment(args.training_seed)
            training_command = [
                args.python_bin,
                str(REPO_ROOT / "main.py"),
                "--config",
                str(config_path.resolve()),
            ]
            training_elapsed = run_logged_command(
                training_command,
                log_path=trial_dir / "training_subprocess.log",
                environment=environment,
                timeout_seconds=args.training_timeout,
            )
            checkpoint_path = discover_final_checkpoint(run_dir, epoch_number)
            validate_checkpoint_feature_heads(checkpoint_path)
            record.update(
                {
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "training_elapsed_seconds": training_elapsed,
                }
            )
            write_json(result_path, record)
            phase = "evaluation"
            phase_started = time.monotonic()
            evaluator_command = build_evaluator_command(
                python_bin=args.python_bin,
                run_dir=run_dir,
                config_path=config_path.resolve(),
                checkpoint_path=checkpoint_path,
                output_dir=evaluator_output_dir,
                split="validation",
                generation_seed=args.generation_seed,
                evaluator_seed=args.evaluator_seed,
                evaluator_repeats=args.evaluator_repeats,
                max_graphs=args.max_graphs,
                generation_batch_size=args.generation_batch_size,
                nearest_k=args.nearest_k,
                adjacency_threshold=args.adjacency_threshold,
                device=args.device,
            )
            evaluation_elapsed = run_logged_command(
                evaluator_command,
                log_path=trial_dir / "evaluation_subprocess.log",
                environment=environment,
                timeout_seconds=args.evaluation_timeout,
            )
            evaluator_json_path = evaluator_output_dir / "attributed_random_gin.json"
        record["evaluation_elapsed_seconds"] = evaluation_elapsed
        record["evaluator_output"] = str(evaluator_json_path.resolve())
        write_json(result_path, record)

        metrics = parse_attr_f1pr_file(
            evaluator_json_path,
            expected_split="validation",
        )
        checkpoint_hash = sha256_file(checkpoint_path)
        record.update(
            {
                "status": "COMPLETE",
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "evaluator_output": str(evaluator_json_path.resolve()),
                "validation_attr_f1pr": metrics.f1_pr,
                "validation_precision": metrics.precision,
                "validation_recall": metrics.recall,
                "accepted_validation_graphs": metrics.graph_count,
                "training_elapsed_seconds": training_elapsed,
                "evaluation_elapsed_seconds": evaluation_elapsed,
                "total_elapsed_seconds": time.monotonic() - started,
            }
        )
        for key in (
            "resolved_config",
            "checkpoint",
            "checkpoint_sha256",
            "evaluator_output",
            "training_elapsed_seconds",
            "evaluation_elapsed_seconds",
            "validation_precision",
            "validation_recall",
            "accepted_validation_graphs",
        ):
            trial.set_user_attr(key, record[key])
        trial.set_user_attr("objective_name", OBJECTIVE_NAME)
        trial.set_user_attr("training_seed", args.training_seed)
        trial.set_user_attr("split_seed", split_seed)
        trial.set_user_attr("generation_seed", args.generation_seed)
        trial.set_user_attr("evaluator_seed", args.evaluator_seed)
        trial.set_user_attr("evaluator_repeats", args.evaluator_repeats)
        write_json(result_path, record)
        return metrics.f1_pr
    except Exception as exc:
        elapsed_in_phase = time.monotonic() - phase_started
        if phase == "training" and record["training_elapsed_seconds"] is None:
            record["training_elapsed_seconds"] = elapsed_in_phase
        if phase == "evaluation" and record["evaluation_elapsed_seconds"] is None:
            record["evaluation_elapsed_seconds"] = elapsed_in_phase
        record.update(
            {
                "status": "FAIL",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "failure_traceback": traceback.format_exc(),
                "total_elapsed_seconds": time.monotonic() - started,
            }
        )
        trial.set_user_attr("failure_reason", record["failure_reason"])
        trial.set_user_attr("resolved_config", str(config_path.resolve()))
        trial.set_user_attr("trial_result", str(result_path.resolve()))
        trial.set_user_attr(
            "training_elapsed_seconds", record["training_elapsed_seconds"]
        )
        trial.set_user_attr(
            "evaluation_elapsed_seconds", record["evaluation_elapsed_seconds"]
        )
        if record["checkpoint"] is not None:
            trial.set_user_attr("checkpoint", record["checkpoint"])
            trial.set_user_attr("checkpoint_sha256", record["checkpoint_sha256"])
        if record.get("evaluator_output") is not None:
            trial.set_user_attr("evaluator_output", record["evaluator_output"])
        write_json(result_path, record)
        raise


def sqlite_storage_url(database_path: Path) -> str:
    return "sqlite:///" + database_path.resolve().as_posix()


def create_or_load_study(
    *,
    database_path: Path,
    study_name: str,
    sampler_seed: int,
    startup_trials: int,
):
    require_optuna()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url=sqlite_storage_url(database_path),
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    existing_trial_count = 0
    for summary in optuna.study.get_all_study_summaries(storage=storage):
        if summary.study_name == study_name:
            existing_trial_count = int(summary.n_trials)
            break
    # Optuna sampler RNG state is process-local. Offset the deterministic seed
    # on resume so startup/random proposals do not restart at trial zero and
    # repeat already completed expensive configurations.
    effective_sampler_seed = (
        int(sampler_seed) + existing_trial_count * 1_000_003
    ) % (2**32 - 1)
    sampler = optuna.samplers.TPESampler(
        seed=effective_sampler_seed,
        n_startup_trials=startup_trials,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    return study


def remaining_trial_count(study, target_finished_trials: int) -> int:
    finished = sum(trial.state.is_finished() for trial in study.get_trials(deepcopy=False))
    return max(0, target_finished_trials - finished)


TRIAL_CSV_FIELDS = (
    "trial_number",
    "state",
    "validation_attr_f1pr",
    "alpha_node_feat",
    "alpha_edge_feat",
    "alpha_motif_loss",
    "validation_precision",
    "validation_recall",
    "accepted_validation_graphs",
    "training_elapsed_seconds",
    "evaluation_elapsed_seconds",
    "resolved_config",
    "checkpoint",
    "checkpoint_sha256",
    "failure_reason",
)


def completed_finite_trials(study) -> list[Any]:
    return [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]


def write_study_outputs(
    study,
    *,
    output_dir: Path,
    database_path: Path,
) -> Any | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = study.get_trials(deepcopy=False)
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_CSV_FIELDS)
        writer.writeheader()
        for trial in trials:
            attrs = trial.user_attrs
            writer.writerow(
                {
                    "trial_number": trial.number,
                    "state": trial.state.name,
                    "validation_attr_f1pr": trial.value,
                    "alpha_node_feat": trial.params.get("alpha_node_feat"),
                    "alpha_edge_feat": trial.params.get("alpha_edge_feat"),
                    "alpha_motif_loss": trial.params.get("alpha_motif_loss"),
                    "validation_precision": attrs.get("validation_precision"),
                    "validation_recall": attrs.get("validation_recall"),
                    "accepted_validation_graphs": attrs.get("accepted_validation_graphs"),
                    "training_elapsed_seconds": attrs.get("training_elapsed_seconds"),
                    "evaluation_elapsed_seconds": attrs.get("evaluation_elapsed_seconds"),
                    "resolved_config": attrs.get("resolved_config"),
                    "checkpoint": attrs.get("checkpoint"),
                    "checkpoint_sha256": attrs.get("checkpoint_sha256"),
                    "failure_reason": attrs.get("failure_reason"),
                }
            )

    complete = completed_finite_trials(study)
    best_trial = max(complete, key=lambda item: float(item.value)) if complete else None
    if best_trial is None:
        (output_dir / "SUMMARY.md").write_text(
            "# GraphVAE Attr-F1PR Bayesian Optimization\n\n"
            "No trial has completed with a finite validation Attr-F1PR.\n",
            encoding="utf-8",
        )
        return None

    resolved_config_path = Path(best_trial.user_attrs["resolved_config"])
    best_config = load_yaml_mapping(resolved_config_path)
    write_yaml(output_dir / "best_config.yaml", best_config)
    best_payload = {
        "schema_version": "graphvae-attr-f1pr-bo-best-v1",
        "objective": OBJECTIVE_NAME,
        "objective_json_path": OBJECTIVE_JSON_PATH,
        "study_name": study.study_name,
        "study_database": str(database_path.resolve()),
        "trial_number": best_trial.number,
        "sampled_weights": best_trial.params,
        "validation_attr_f1pr": float(best_trial.value),
        "validation_precision": best_trial.user_attrs.get("validation_precision"),
        "validation_recall": best_trial.user_attrs.get("validation_recall"),
        "accepted_validation_graphs": best_trial.user_attrs.get("accepted_validation_graphs"),
        "training_seed": best_trial.user_attrs.get("training_seed"),
        "split_seed": best_trial.user_attrs.get("split_seed"),
        "generation_seed": best_trial.user_attrs.get("generation_seed"),
        "evaluator_seed": best_trial.user_attrs.get("evaluator_seed"),
        "evaluator_repeats": best_trial.user_attrs.get("evaluator_repeats"),
        "resolved_config": str(resolved_config_path.resolve()),
        "best_config": str((output_dir / "best_config.yaml").resolve()),
        "checkpoint": best_trial.user_attrs.get("checkpoint"),
        "checkpoint_sha256": best_trial.user_attrs.get("checkpoint_sha256"),
        "training_elapsed_seconds": best_trial.user_attrs.get("training_elapsed_seconds"),
        "evaluation_elapsed_seconds": best_trial.user_attrs.get("evaluation_elapsed_seconds"),
    }
    write_json(output_dir / "best_trial.json", best_payload)
    weights = ", ".join(
        f"{key}={value:.8g}" for key, value in sorted(best_trial.params.items())
    )
    (output_dir / "SUMMARY.md").write_text(
        "# GraphVAE Attr-F1PR Bayesian Optimization\n\n"
        f"- Study: `{study.study_name}`\n"
        f"- Best trial: `{best_trial.number}`\n"
        f"- Best weights: `{weights}`\n"
        f"- Validation Attr-F1PR: `{float(best_trial.value):.6f}`\n"
        f"- Objective path: `{OBJECTIVE_JSON_PATH}`\n"
        "- Test split evaluated during optimization: `no`\n",
        encoding="utf-8",
    )
    return best_trial


def _resolved_seed(args: argparse.Namespace, name: str, default: int) -> int:
    value = getattr(args, name)
    return default if value is None else int(value)


def prepare_args(args: argparse.Namespace) -> None:
    args.training_seed = _resolved_seed(args, "training_seed", DEFAULT_TRAINING_SEED)
    args.generation_seed = _resolved_seed(args, "generation_seed", DEFAULT_GENERATION_SEED)
    args.evaluator_seed = _resolved_seed(args, "evaluator_seed", DEFAULT_EVALUATOR_SEED)
    args.evaluator_repeats = _resolved_seed(
        args, "evaluator_repeats", DEFAULT_EVALUATOR_REPEATS
    )
    args.training_timeout = None if args.training_timeout <= 0 else args.training_timeout
    args.evaluation_timeout = None if args.evaluation_timeout <= 0 else args.evaluation_timeout


def optimize(args: argparse.Namespace) -> int:
    require_optuna()
    if args.base_config is None:
        raise ValueError("--base-config is required for optimization.")
    if args.trials < 1:
        raise ValueError("--trials must be positive.")
    if args.evaluator_repeats < 1:
        raise ValueError("--evaluator-repeats must be positive.")
    if args.max_graphs in {1, 2} or args.max_graphs < 0:
        raise ValueError("--max-graphs must be 0 (all) or at least 3.")
    if args.generation_batch_size < 1 or args.nearest_k < 1:
        raise ValueError("Generation batch size and nearest-k must be positive.")
    if not 0.0 <= args.adjacency_threshold <= 1.0:
        raise ValueError("--adjacency-threshold must be in [0, 1].")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_yaml_mapping(args.base_config)
    if not args.mock:
        validate_base_config(base_config, args.tune_alpha_motif)
    else:
        # Mock mode still checks duplicate keys and motif intent, while allowing
        # intentionally tiny fixture configs that do not describe real data.
        flatten_config(base_config)
    ranges = build_search_ranges(args)
    flat = flatten_config(base_config)
    split_seed = (
        int(flat.get("split_seed", DEFAULT_SPLIT_SEED))
        if args.split_seed is None
        else int(args.split_seed)
    )
    database_path = output_dir / "study.sqlite3"
    study = create_or_load_study(
        database_path=database_path,
        study_name=args.study_name,
        sampler_seed=args.sampler_seed,
        startup_trials=args.tpe_startup_trials,
    )
    study_definition = {
        "schema_version": "graphvae-attr-f1pr-bo-study-v1",
        "objective": OBJECTIVE_NAME,
        "objective_json_path": OBJECTIVE_JSON_PATH,
        "primary_mode": PRIMARY_MODE,
        "optimization_split": "validation",
        "test_evaluation_during_optimization": False,
        "study_name": args.study_name,
        "base_config_sha256": sha256_file(args.base_config.expanduser().resolve()),
        "search_ranges": {
            "alpha_node_feat": list(ranges.alpha_node_feat),
            "alpha_edge_feat": list(ranges.alpha_edge_feat),
            "alpha_motif_loss": (
                None
                if ranges.alpha_motif_loss is None
                else list(ranges.alpha_motif_loss)
            ),
        },
        "split_seed": split_seed,
        "training_seed": args.training_seed,
        "generation_seed": args.generation_seed,
        "evaluator_seed": args.evaluator_seed,
        "evaluator_repeats": args.evaluator_repeats,
        "evaluator_seeds": [
            args.evaluator_seed + repeat for repeat in range(args.evaluator_repeats)
        ],
        "max_graphs": args.max_graphs,
        "generation_batch_size": args.generation_batch_size,
        "nearest_k": args.nearest_k,
        "adjacency_threshold": args.adjacency_threshold,
        "device": args.device,
        "sampler": "TPESampler",
        "sampler_seed": args.sampler_seed,
        "tpe_startup_trials": args.tpe_startup_trials,
        "mock": args.mock,
        "epoch_number": int(flat["epoch_number"]),
    }
    ensure_study_definition(
        output_dir / "study_definition.json",
        study_definition,
        existing_trial_count=len(study.trials),
    )
    study.set_user_attr("sampler", "TPESampler")
    study.set_user_attr("sampler_seed", int(args.sampler_seed))
    study.set_user_attr("tpe_startup_trials", int(args.tpe_startup_trials))
    remaining = remaining_trial_count(study, args.trials)

    def objective(trial):
        return execute_trial(
            trial,
            args=args,
            base_config=base_config,
            ranges=ranges,
            output_dir=output_dir,
            split_seed=split_seed,
        )

    interrupted = False
    try:
        if remaining:
            study.optimize(objective, n_trials=remaining, catch=(Exception,))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        write_study_outputs(study, output_dir=output_dir, database_path=database_path)

    complete_count = len(completed_finite_trials(study))
    print(
        f"{OBJECTIVE_NAME} study {study.study_name!r}: "
        f"{complete_count} finite completed trial(s); outputs in {output_dir}"
    )
    if interrupted:
        return 130
    return 0 if complete_count else 2


def evaluate_best_on_test(args: argparse.Namespace) -> int:
    """Explicitly evaluate the already-selected best checkpoint on held-out test."""

    output_dir = args.output_dir.expanduser().resolve()
    best_path = output_dir / "best_trial.json"
    if not best_path.is_file():
        raise FileNotFoundError(
            f"Best-trial metadata not found: {best_path}. Complete optimization first."
        )
    with best_path.open("r", encoding="utf-8") as handle:
        best = json.load(handle)
    for name in (
        "training_seed",
        "generation_seed",
        "evaluator_seed",
        "evaluator_repeats",
    ):
        if not args.seed_arguments_provided[name] and best.get(name) is not None:
            setattr(args, name, int(best[name]))
    checkpoint_path = Path(best["checkpoint"]).resolve()
    config_path = Path(best.get("best_config", best["resolved_config"])).resolve()
    if sha256_file(checkpoint_path) != best["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint hash no longer matches best_trial.json.")
    if args.mock:
        evaluator_output_dir = output_dir / "final_test"
        evaluator_output_dir.mkdir(parents=True, exist_ok=True)
        evaluator_json = evaluator_output_dir / "attributed_random_gin.json"
        write_json(
            evaluator_json,
            _mock_evaluator_payload(best["sampled_weights"], split="test"),
        )
        elapsed = 0.0
    else:
        validate_checkpoint_feature_heads(checkpoint_path)
        training_seed = args.training_seed
        run_dir = checkpoint_path.parent
        evaluator_output_dir = output_dir / "final_test"
        evaluator_output_dir.mkdir(parents=True, exist_ok=True)
        command = build_evaluator_command(
            python_bin=args.python_bin,
            run_dir=run_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=evaluator_output_dir,
            split="test",
            generation_seed=args.generation_seed,
            evaluator_seed=args.evaluator_seed,
            evaluator_repeats=args.evaluator_repeats,
            max_graphs=args.max_graphs,
            generation_batch_size=args.generation_batch_size,
            nearest_k=args.nearest_k,
            adjacency_threshold=args.adjacency_threshold,
            device=args.device,
        )
        elapsed = run_logged_command(
            command,
            log_path=evaluator_output_dir / "evaluation_subprocess.log",
            environment=_trial_environment(training_seed),
            timeout_seconds=args.evaluation_timeout,
        )
        evaluator_json = evaluator_output_dir / "attributed_random_gin.json"
    metrics = parse_attr_f1pr_file(evaluator_json, expected_split="test")
    write_json(
        evaluator_output_dir / "final_test_selection.json",
        {
            "schema_version": "graphvae-attr-f1pr-final-test-v1",
            "selection_objective": OBJECTIVE_NAME,
            "selected_trial_number": best["trial_number"],
            "selected_validation_attr_f1pr": best["validation_attr_f1pr"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": best["checkpoint_sha256"],
            "split": "test",
            "generation_seed": args.generation_seed,
            "evaluator_seed": args.evaluator_seed,
            "evaluator_seeds": [
                args.evaluator_seed + repeat for repeat in range(args.evaluator_repeats)
            ],
            "evaluator_repeats": args.evaluator_repeats,
            "test_attr_f1pr": metrics.f1_pr,
            "test_precision": metrics.precision,
            "test_recall": metrics.recall,
            "accepted_test_graphs": metrics.graph_count,
            "evaluation_elapsed_seconds": elapsed,
            "evaluator_output": str(evaluator_json.resolve()),
        },
    )
    print(
        f"Selected trial {best['trial_number']} held-out test Attr-F1PR: "
        f"{metrics.f1_pr:.6f}"
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--study-name", default="graphvae_attr_f1pr")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--alpha-node-feat-min", type=float, default=1e-3)
    parser.add_argument("--alpha-node-feat-max", type=float, default=1e2)
    parser.add_argument("--alpha-edge-feat-min", type=float, default=1e-3)
    parser.add_argument("--alpha-edge-feat-max", type=float, default=1e2)
    parser.add_argument(
        "--tune-alpha-motif",
        action="store_true",
        help="Also tune the repository's alpha_motif_loss parameter.",
    )
    parser.add_argument("--alpha-motif-min", type=float, default=1e-3)
    parser.add_argument("--alpha-motif-max", type=float, default=1e2)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--generation-seed", type=int, default=None)
    parser.add_argument("--evaluator-seed", type=int, default=None)
    parser.add_argument("--evaluator-repeats", type=int, default=None)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--tpe-startup-trials", type=int, default=5)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--adjacency-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--training-timeout", type=float, default=0.0)
    parser.add_argument("--evaluation-timeout", type=float, default=0.0)
    parser.add_argument(
        "--evaluate-best-on-test",
        action="store_true",
        help=(
            "Do not optimize. Explicitly evaluate the selected best checkpoint "
            "against the held-out test split."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock training/evaluation for tests and smoke checks.",
    )
    parser.add_argument(
        "--mock-fail-trial",
        action="append",
        type=int,
        default=[],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    args.seed_arguments_provided = {
        name: getattr(args, name) is not None
        for name in (
            "training_seed",
            "generation_seed",
            "evaluator_seed",
            "evaluator_repeats",
        )
    }
    prepare_args(args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.evaluate_best_on_test:
        return evaluate_best_on_test(args)
    return optimize(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from exc
