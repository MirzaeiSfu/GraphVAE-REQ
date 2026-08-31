#!/usr/bin/env python3
"""Run fail-closed validation-only generation-seed evaluations for one worker slot."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from graphvae_attr_bo_distributed import atomic_write_json, sha256_file  # noqa: E402
from tune_graphvae_attribute_weights import (  # noqa: E402
    OBJECTIVE_JSON_PATH,
    flatten_config,
    parse_attr_f1pr_payload,
)


EXPECTED_STUDY = "aids_kia_bce_kl_comparison3_20260830a"
EXPECTED_CONTRACT = "262ac59a6d3b5ea96b12b4c8e2130ca98f2cca4c3ed14d06cd13de384006da0c"
EXPECTED_CACHE_SHA = "6edcc3309fb1c3d366b0f87065aa1b2e2c7d23cbff92bc729053f44e874909bb"
EXPECTED_SPLIT_FINGERPRINT = (
    "ea6e38e034feb2c523263172d27f07af3ae1aaa99ea0ba875b739780706d6e66"
)
EXPECTED_NODE_SCHEMA = "630400b38c74e0a51e505e57b7e41ee986deea1eb06e27099ea792cf6876c2c9"
EXPECTED_EDGE_SCHEMA = "1fee16532276d512d7143669f2d3c80a0140ee3c2ded035fa206c323320bf772"
EXPECTED_EVALUATOR_SEEDS = tuple(range(1000, 1010))
EXPECTED_CANDIDATES = {
    (1.0, 1.0),
    (1.0, 0.1),
    (0.1, 1.0),
}


class StabilityError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise StabilityError(f"Required regular JSON file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StabilityError(f"Expected a JSON object: {path}")
    return value


def _within(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise StabilityError(f"Artifact path escapes the study root: {value}") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise StabilityError(f"Artifact is missing or not a regular file: {candidate}")
    return candidate


def _validate_trial(
    study_root: Path,
    *,
    worker_id: str,
    physical_gpu: int,
) -> tuple[dict[str, Any], Path, Path, Path]:
    matches = []
    for path in sorted((study_root / "trials").glob("trial_*/trial_result.json")):
        payload = _load_json(path)
        if payload.get("worker_id") == worker_id:
            matches.append((payload, path.parent))
    if len(matches) != 1:
        raise StabilityError(
            f"Expected exactly one completed trial for {worker_id}, found {len(matches)}."
        )
    result, trial_root = matches[0]
    exact = {
        "status": "COMPLETE",
        "study_contract_sha256": EXPECTED_CONTRACT,
        "training_seed": 0,
        "generation_seed": 123,
        "physical_gpu": physical_gpu,
        "objective_json_path": OBJECTIVE_JSON_PATH,
        "evaluator_backend": "random_gin",
        "evaluator_repeats": 10,
        "evaluator_seeds": list(EXPECTED_EVALUATOR_SEEDS),
        "accepted_validation_graphs": 184,
    }
    for name, expected in exact.items():
        if result.get(name) != expected:
            raise StabilityError(f"Trial result differs for {name}.")
    weights = result.get("sampled_weights") or {}
    pair = (float(weights.get("alpha_node_feat")), float(weights.get("alpha_edge_feat")))
    if pair not in EXPECTED_CANDIDATES or set(weights) != {
        "alpha_node_feat",
        "alpha_edge_feat",
    }:
        raise StabilityError("Trial result contains an unexpected candidate weight vector.")
    checkpoint = _within(study_root, str(result.get("checkpoint")))
    if sha256_file(checkpoint) != result.get("checkpoint_sha256"):
        raise StabilityError("Checkpoint hash differs from the completed trial result.")
    config_path = _within(study_root, str(result.get("resolved_config")))
    if sha256_file(config_path) != result.get("resolved_config_sha256"):
        raise StabilityError("Resolved configuration hash differs from the trial result.")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    flat = flatten_config(config)
    required = {
        "dataset": "AIDS",
        "model": "GraphVAE",
        "motif_loss": False,
        "use_graphvae_mm_bce_kl_weights": True,
        "beta": None,
        "epoch_number": 250,
        "split_mode": "paper_70_10_20",
        "split_seed": 123,
        "seed": 0,
        "skip_final_evaluation": True,
        "ideal_Evalaution": False,
    }
    for name, expected in required.items():
        if flat.get(name) != expected:
            raise StabilityError(f"Resolved configuration differs for {name}.")
    if (
        float(flat.get("alpha_node_feat")) != pair[0]
        or float(flat.get("alpha_edge_feat")) != pair[1]
    ):
        raise StabilityError("Resolved feature weights differ from the trial result.")
    training_root = trial_root / "training" / "seed_0"
    if not training_root.is_dir() or training_root.is_symlink():
        raise StabilityError("Training root is missing or invalid.")
    return result, trial_root, training_root, checkpoint


def build_evaluation_command(
    *,
    python_bin: Path,
    source_root: Path,
    training_root: Path,
    config_path: Path,
    checkpoint: Path,
    generation_seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        str(python_bin),
        str(source_root / "scripts" / "evaluate_attributed_graph_realism_checkpoints.py"),
        "--run-dir",
        str(training_root),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--dataset-cache-dir",
        str(source_root / "cache_datasets"),
        "--split",
        "validation",
        "--modes",
        "decoded_node_edge",
        "--max-graphs",
        "0",
        "--generation-batch-size",
        "20",
        "--generation-seed",
        str(generation_seed),
        "--evaluator-seed",
        "1000",
        "--repeats",
        "10",
        "--nearest-k",
        "5",
        "--adjacency-threshold",
        "0.5",
        "--device",
        "cuda:0",
        "--output-dir",
        str(output_dir),
        "--save-pyg",
    ]


def _validate_evaluation(path: Path, *, generation_seed: int) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("test_access") is not False or payload.get("skip_final_evaluation") is not True:
        raise StabilityError("Evaluation does not explicitly prohibit test/final evaluation.")
    metrics = parse_attr_f1pr_payload(
        payload,
        expected_split="validation",
        expected_graph_count=184,
        expected_cache_sha256=EXPECTED_CACHE_SHA,
        expected_split_fingerprint=EXPECTED_SPLIT_FINGERPRINT,
        expected_node_schema_fingerprint=EXPECTED_NODE_SCHEMA,
        expected_edge_schema_fingerprint=EXPECTED_EDGE_SCHEMA,
        expected_node_feature_dimension=56,
        expected_edge_feature_dimension=3,
        expected_generation_seed=generation_seed,
        expected_evaluator_seed=1000,
        expected_repeats=10,
    )
    if not all(math.isfinite(value) for value in (metrics.f1_pr, metrics.precision, metrics.recall)):
        raise StabilityError("Evaluation returned non-finite metrics.")
    return {
        "generation_seed": generation_seed,
        "f1_pr": metrics.f1_pr,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "evaluation_sha256": sha256_file(path),
        "test_access": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    study_root = args.study_root.resolve()
    python_bin = args.python_bin.resolve()
    if args.study_name != EXPECTED_STUDY or args.study_contract_sha256 != EXPECTED_CONTRACT:
        raise StabilityError("Study identity differs from the frozen comparison contract.")
    if not source_root.is_dir() or not study_root.is_dir() or not python_bin.is_file():
        raise StabilityError("Source, study, or Python path is missing.")
    result, trial_root, training_root, checkpoint = _validate_trial(
        study_root,
        worker_id=args.worker_id,
        physical_gpu=args.physical_gpu,
    )
    config_path = _within(study_root, str(result["resolved_config"]))
    records = []
    for generation_seed in args.generation_seeds:
        if generation_seed not in {124, 125}:
            raise StabilityError("Only the frozen additional generation seeds 124 and 125 are allowed.")
        output_dir = trial_root / "generation_stability" / f"generation_seed_{generation_seed}"
        output_path = output_dir / "attributed_random_gin.json"
        if output_path.is_file():
            records.append(_validate_evaluation(output_path, generation_seed=generation_seed))
            continue
        if output_dir.exists():
            raise StabilityError(f"Ambiguous partial evaluation directory exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        command = build_evaluation_command(
            python_bin=python_bin,
            source_root=source_root,
            training_root=training_root,
            config_path=config_path,
            checkpoint=checkpoint,
            generation_seed=generation_seed,
            output_dir=output_dir,
        )
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", str(trial_root / ".matplotlib"))
        completed = subprocess.run(
            command,
            cwd=str(source_root),
            env=environment,
            check=False,
            timeout=args.timeout_seconds,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise StabilityError(
                f"Generation-seed {generation_seed} evaluation failed with exit {completed.returncode}."
            )
        records.append(_validate_evaluation(output_path, generation_seed=generation_seed))
    manifest = {
        "schema_version": "aids-kia-generation-stability-worker-v1",
        "study_name": EXPECTED_STUDY,
        "study_contract_sha256": EXPECTED_CONTRACT,
        "trial_number": int(result["trial_number"]),
        "budget_index": int(result["budget_index"]),
        "worker_id": args.worker_id,
        "physical_gpu": args.physical_gpu,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "sampled_weights": result["sampled_weights"],
        "evaluations": records,
        "test_access": False,
        "held_out_access": False,
    }
    atomic_write_json(trial_root / "generation_stability" / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--study-contract-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=[124, 125])
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = run(args)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
