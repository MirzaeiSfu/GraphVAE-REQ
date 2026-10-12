#!/usr/bin/env python3
"""Probe or terminate one recorded distributed Attr-F1PR child process group."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from graphvae_attr_bo_distributed import (
    atomic_write_json,
    canonical_contract_hash,
    validate_identifier,
)
from tune_graphvae_attribute_weights import (
    TrialExecutionError,
    inspect_recorded_process_group,
    recover_recorded_process_group,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--worker-run-id", required=True)
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Required for a grouped GraphCL replicate; forbidden for legacy trials.",
    )
    parser.add_argument("--phase", choices=("training", "evaluation"), required=True)
    parser.add_argument("--study-contract-sha256", required=True)
    parser.add_argument("--grace-seconds", type=float, default=10.0)
    parser.add_argument("--terminate", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement before signaling the recorded group.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _validated_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    validate_identifier(args.worker_run_id, "worker run ID")
    if args.trial_number < 0:
        raise ValueError("Trial number must be non-negative.")
    if len(args.study_contract_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.study_contract_sha256
    ):
        raise ValueError("Study contract SHA-256 must be 64 lowercase hex characters.")
    if args.grace_seconds < 0:
        raise ValueError("Termination grace must be non-negative.")
    repo_root = args.repo_root.expanduser().resolve()
    study_root = args.study_root.expanduser().resolve()
    study_root.relative_to(repo_root)
    definition_path = study_root / "study_definition.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if canonical_contract_hash(definition) != args.study_contract_sha256:
        raise TrialExecutionError("Local study definition differs from the expected contract.")
    if definition.get("study_name") != study_root.name:
        raise TrialExecutionError("Study root name differs from its immutable definition.")
    objective = definition.get("objective") or {}
    if (
        objective.get("json_path")
        != "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
        or objective.get("split") != "validation"
        or objective.get("test_access") is not False
    ):
        raise TrialExecutionError("Study objective is not the frozen validation contract.")
    evaluator = definition.get("evaluator") or {}
    grouped_graphcl = evaluator.get("backend") == "graphcl_f1pr"
    training_seeds = [int(seed) for seed in definition.get("seeds", {}).get("training_seeds", [])]
    if grouped_graphcl:
        if args.training_seed is None or args.training_seed not in training_seeds:
            raise TrialExecutionError(
                "Grouped GraphCL recovery requires one contracted training seed."
            )
    elif args.training_seed is not None:
        raise TrialExecutionError("Legacy trial recovery forbids --training-seed.")
    worker_root = study_root / "workers" / args.worker_run_id
    trial_root = study_root / "trials" / f"trial_{args.trial_number:05d}"
    if grouped_graphcl:
        trial_root = trial_root / "replicates" / f"seed_{args.training_seed}"
    identity_path = trial_root / f"{args.phase}_subprocess.log.process.json"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.relative_to(worker_root.resolve())
        if output == worker_root.resolve():
            raise ValueError("Recovery output must be a file inside the worker root.")
    else:
        output = worker_root / "PROCESS_RECOVERY.json"
    return repo_root, study_root, identity_path, output


def run(args: argparse.Namespace) -> int:
    if args.execute and not args.terminate:
        raise ValueError("--execute is valid only with --terminate.")
    if args.terminate and not args.execute:
        raise ValueError("Termination requires the explicit --execute acknowledgement.")
    repo_root, study_root, identity_path, output = _validated_paths(args)
    before = inspect_recorded_process_group(
        identity_path,
        expected_cwd=repo_root,
        expected_study_contract_sha256=args.study_contract_sha256,
        expected_worker_run_id=args.worker_run_id,
        expected_trial_number=args.trial_number,
        expected_phase=args.phase,
    )
    terminated = False
    if args.terminate:
        terminated = recover_recorded_process_group(
            identity_path,
            expected_cwd=repo_root,
            expected_study_contract_sha256=args.study_contract_sha256,
            expected_worker_run_id=args.worker_run_id,
            expected_trial_number=args.trial_number,
            expected_phase=args.phase,
            grace_seconds=args.grace_seconds,
        )
    after = inspect_recorded_process_group(
        identity_path,
        expected_cwd=repo_root,
        expected_study_contract_sha256=args.study_contract_sha256,
        expected_worker_run_id=args.worker_run_id,
        expected_trial_number=args.trial_number,
        expected_phase=args.phase,
    )
    if args.terminate and after["status"] != "ABSENT":
        raise TrialExecutionError("Process recovery did not prove the group absent.")
    payload = {
        "schema_version": "graphvae-attr-f1pr-process-recovery-v1",
        "study_name": study_root.name,
        "study_contract_sha256": args.study_contract_sha256,
        "worker_run_id": args.worker_run_id,
        "trial_number": args.trial_number,
        "training_seed": args.training_seed,
        "phase": args.phase,
        "action": "terminate" if args.terminate else "probe",
        "status_before": before["status"],
        "status_after": after["status"],
        "signal_sent": terminated,
        "group_absent_verified": after["status"] == "ABSENT",
        "process_identity": {
            key: before[key]
            for key in (
                "pid",
                "process_group_id",
                "pid_start_ticks",
                "command_sha256",
                "cwd",
            )
        },
        "test_access": False,
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from None
