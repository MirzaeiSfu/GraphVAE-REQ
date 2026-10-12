#!/usr/bin/env python3
"""Evaluate one frozen confirmation checkpoint on held-out LOBSTER data."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if __package__:
    from .graphvae_attr_bo_distributed import (
        DistributedContractError,
        atomic_write_json,
        canonical_contract_hash,
        runtime_dependency_fingerprint,
        sha256_file,
        verify_deployment_manifest,
    )
    from .tune_graphvae_attribute_weights import (
        build_evaluator_command,
        parse_attr_f1pr_file,
        run_logged_command,
        validate_checkpoint_feature_heads,
    )
else:
    sys.path.insert(0, str(SCRIPT_DIR))
    from graphvae_attr_bo_distributed import (  # noqa: E402
        DistributedContractError,
        atomic_write_json,
        canonical_contract_hash,
        runtime_dependency_fingerprint,
        sha256_file,
        verify_deployment_manifest,
    )
    from tune_graphvae_attribute_weights import (  # noqa: E402
        build_evaluator_command,
        parse_attr_f1pr_file,
        run_logged_command,
        validate_checkpoint_feature_heads,
    )

EXPECTED_OBJECTIVE = "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
EXPECTED_RUNTIME = "e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1"
EXPECTED_PLAN_SHA256 = "831b1c35169cb48bb8c1b06d03b800bf0ed07119b21ebfd7f07689d1d272d881"
EXPECTED_SELECTED = {
    "alpha_node_feat": 5.229045672015893,
    "alpha_edge_feat": 0.05386414830134693,
}
EXPECTED_UNIFORM = {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0}


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise DistributedContractError(f"Expected a JSON object: {path}")
    return value


def validate_heldout_inputs(
    study_dir: Path,
    plan_path: Path,
    trial_number: int,
    *,
    hostname: str,
    physical_gpu: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    study = study_dir.expanduser().resolve()
    plan_file = plan_path.expanduser().resolve()
    plan = _read_json(plan_file)
    definition = _read_json(study / "study_definition.json")
    frozen = _read_json(study / "FROZEN.json")
    report_path = study / "confirmation_report.json"
    report = _read_json(report_path)
    contract_hash = canonical_contract_hash(definition)

    if (
        sha256_file(plan_file) != EXPECTED_PLAN_SHA256
        or plan.get("schema_version")
        != "graphvae-attr-f1pr-confirmation-heldout-plan-v1"
        or plan.get("study_name") != definition.get("study_name")
        or plan.get("study_contract_sha256") != contract_hash
        or plan.get("frozen_confirmation_report_sha256")
        != sha256_file(report_path)
        or plan.get("frozen_confirmation_conclusion") != "no_improvement"
        or plan.get("no_reranking") is not True
        or plan.get("allow_training") is not False
        or plan.get("allow_new_trials") is not False
        or plan.get("split") != "test"
        or plan.get("primary_mode") != "decoded_node_edge"
        or plan.get("objective_json_path") != EXPECTED_OBJECTIVE
        or plan.get("max_graphs") != 0
        or plan.get("expected_test_graphs") != 20
        or plan.get("generation_seed") != 123
        or plan.get("evaluator_seed") != 0
        or plan.get("evaluator_repeats") != 10
        or plan.get("max_parallel") != 3
    ):
        raise DistributedContractError("Held-out plan differs from the frozen contract.")
    if (
        frozen.get("lifecycle") != "FROZEN"
        or frozen.get("study_contract_sha256") != contract_hash
        or report.get("study_contract_sha256") != contract_hash
        or report.get("conclusion") != "no_improvement"
        or report.get("no_reranking") is not True
        or report.get("held_out_access") is not False
        or report.get("test_access") is not False
        or report.get("objective_json_path") != EXPECTED_OBJECTIVE
        or report.get("selection_split") != "validation"
    ):
        raise DistributedContractError("Confirmation was not frozen before held-out access.")
    objective = definition.get("objective") or {}
    if (
        definition.get("reserved_trials") != 6
        or objective.get("json_path") != EXPECTED_OBJECTIVE
        or objective.get("split") != "validation"
        or objective.get("test_access") is not False
        or definition.get("evaluator", {}).get("test_access") is not False
    ):
        raise DistributedContractError("Source study optimization contract is not isolated.")

    entries = plan.get("trials")
    if not isinstance(entries, list) or [row.get("trial_number") for row in entries] != list(range(6)):
        raise DistributedContractError("Held-out plan must contain trials 0 through 5 exactly.")
    entry = entries[trial_number] if 0 <= trial_number < len(entries) else None
    if (
        not isinstance(entry, Mapping)
        or entry.get("trial_number") != trial_number
        or entry.get("hostname") != hostname
        or entry.get("physical_gpu") != physical_gpu
    ):
        raise DistributedContractError("Held-out launch identity differs from the plan.")

    trial_dir = study / "trials" / f"trial_{trial_number:05d}"
    trial = _read_json(trial_dir / "trial_result.json")
    expected_weights = EXPECTED_SELECTED if entry.get("candidate") == "selected" else EXPECTED_UNIFORM
    if (
        trial.get("status") != "COMPLETE"
        or trial.get("trial_number") != trial_number
        or trial.get("study_contract_sha256") != contract_hash
        or trial.get("training_seed") != entry.get("training_seed")
        or trial.get("hostname") != hostname
        or trial.get("physical_gpu") != physical_gpu
        or trial.get("sampled_weights") != expected_weights
        or trial.get("checkpoint_sha256") != entry.get("checkpoint_sha256")
        or trial.get("resolved_config_sha256") != entry.get("resolved_config_sha256")
        or trial.get("objective_json_path") != EXPECTED_OBJECTIVE
    ):
        raise DistributedContractError("Collected confirmation trial differs from the held-out plan.")
    return plan, entry, trial


def evaluate_one(args: argparse.Namespace) -> dict[str, Any]:
    study_dir = args.study_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.relative_to(REPO_ROOT)
    if output_dir.exists():
        raise FileExistsError(f"Held-out output already exists: {output_dir}")
    plan, entry, trial = validate_heldout_inputs(
        study_dir,
        args.plan,
        args.trial_number,
        hostname=socket.gethostname().split(".", 1)[0],
        physical_gpu=args.physical_gpu,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise DistributedContractError("CUDA_VISIBLE_DEVICES differs from the physical GPU plan.")
    if runtime_dependency_fingerprint().get("sha256") != EXPECTED_RUNTIME:
        raise DistributedContractError("Held-out runtime fingerprint mismatch.")
    verify_deployment_manifest(REPO_ROOT, _read_json(REPO_ROOT / "deployment_manifest.json"))
    evaluator_source = REPO_ROOT / "scripts" / "evaluate_attributed_graph_realism_checkpoints.py"
    if sha256_file(evaluator_source) != plan.get("evaluator_source_sha256"):
        raise DistributedContractError("Held-out evaluator source hash mismatch.")

    trial_dir = study_dir / "trials" / f"trial_{args.trial_number:05d}"
    checkpoint = study_dir / str(trial["checkpoint"])
    resolved_config = study_dir / str(trial["resolved_config"])
    if (
        sha256_file(checkpoint) != entry["checkpoint_sha256"]
        or sha256_file(resolved_config) != entry["resolved_config_sha256"]
    ):
        raise DistributedContractError("Held-out checkpoint or configuration hash mismatch.")
    validate_checkpoint_feature_heads(checkpoint)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(f"Held-out staging output already exists: {staging}")
    staging.mkdir()
    started = time.time()
    try:
        command = build_evaluator_command(
            python_bin=args.python_bin,
            run_dir=checkpoint.parent,
            config_path=resolved_config,
            checkpoint_path=checkpoint,
            output_dir=staging,
            split="test",
            generation_seed=int(plan["generation_seed"]),
            evaluator_seed=int(plan["evaluator_seed"]),
            evaluator_repeats=int(plan["evaluator_repeats"]),
            max_graphs=int(plan["max_graphs"]),
            generation_batch_size=int(plan["generation_batch_size"]),
            nearest_k=int(plan["nearest_k"]),
            adjacency_threshold=float(plan["adjacency_threshold"]),
            device="cuda:0",
        )
        elapsed = run_logged_command(
            command,
            log_path=staging / "evaluation_subprocess.log",
            environment=os.environ.copy(),
            timeout_seconds=float(plan["evaluation_timeout_seconds"]),
            process_identity={
                "phase": "heldout_confirmation",
                "trial_number": args.trial_number,
                "study_contract_sha256": plan["study_contract_sha256"],
            },
        )
        evaluator_path = staging / "attributed_random_gin.json"
        metrics = parse_attr_f1pr_file(
            evaluator_path,
            expected_split="test",
            expected_graph_count=int(plan["expected_test_graphs"]),
            expected_node_feature_dimension=int(plan["node_feature_dimension"]),
            expected_edge_feature_dimension=int(plan["edge_feature_dimension"]),
            expected_cache_sha256=plan["cache_sha256"],
            expected_split_fingerprint=plan["test_split_fingerprint"],
            expected_node_schema_fingerprint=plan["node_schema_fingerprint"],
            expected_edge_schema_fingerprint=plan["edge_schema_fingerprint"],
            expected_generation_seed=int(plan["generation_seed"]),
            expected_evaluator_seed=int(plan["evaluator_seed"]),
            expected_repeats=int(plan["evaluator_repeats"]),
        )
        result = {
            "schema_version": "graphvae-attr-f1pr-confirmation-heldout-result-v1",
            "study_name": plan["study_name"],
            "study_contract_sha256": plan["study_contract_sha256"],
            "trial_number": args.trial_number,
            "candidate": entry["candidate"],
            "training_seed": entry["training_seed"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "split": "test",
            "primary_mode": "decoded_node_edge",
            "objective_json_path": EXPECTED_OBJECTIVE,
            "test_attr_f1pr": metrics.f1_pr,
            "test_precision": metrics.precision,
            "test_recall": metrics.recall,
            "accepted_test_graphs": metrics.graph_count,
            "generation_seed": plan["generation_seed"],
            "evaluator_seed": plan["evaluator_seed"],
            "evaluator_repeats": plan["evaluator_repeats"],
            "hostname": entry["hostname"],
            "physical_gpu": entry["physical_gpu"],
            "logical_device": "cuda:0",
            "evaluation_elapsed_seconds": elapsed,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "evaluator_output_sha256": sha256_file(evaluator_path),
            "no_reranking": True,
            "selection_changed": False,
            "new_trial_created": False,
            "training_ran": False,
        }
        atomic_write_json(staging / "heldout_result.json", result)
    except BaseException as exc:
        atomic_write_json(
            staging / "FAILED",
            {
                "schema_version": "graphvae-attr-f1pr-confirmation-heldout-failure-v1",
                "trial_number": args.trial_number,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "selection_changed": False,
                "new_trial_created": False,
                "training_ran": False,
            },
        )
        os.replace(staging, output_dir)
        raise
    os.replace(staging, output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", required=True)
    args = parser.parse_args()
    result = evaluate_one(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from exc
