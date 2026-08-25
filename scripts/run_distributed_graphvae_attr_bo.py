#!/usr/bin/env python3
"""Controller for bounded PostgreSQL GraphVAE Attr-F1PR optimization."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from graphvae_attr_bo_distributed import (  # noqa: E402
    BUDGET_INDEX_ATTR,
    CONTRACT_ATTR,
    CONTROLLER_ATTR,
    DEFINITION_ATTR,
    DEFAULT_GRACE_PERIOD,
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_STARTUP_TRIALS,
    LIFECYCLE_ATTR,
    LIFECYCLE_FROZEN,
    LIFECYCLE_READY,
    LIFECYCLE_RETIRED_PRECLAIM,
    RESERVED_ATTR,
    UNRESERVED_GUARD_ATTR,
    ControllerLocks,
    DistributedContractError,
    assert_quiescent_reserved_study,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    audit_trial_result,
    build_study_definition,
    canonical_contract_hash,
    create_or_load_distributed_study,
    create_portable_snapshot,
    create_postgresql_storage,
    enforce_pinned_versions,
    initialize_reserved_study,
    parse_slots,
    output_root_fingerprint,
    relative_artifact_path,
    redact_secret,
    reservation_audit,
    reserved_trial_states,
    resolve_artifact_path,
    runtime_dependency_fingerprint,
    sampler_seed,
    sha256_file,
    storage_url_from_env,
    trial_semantic_fingerprint,
    validate_guard_rows,
    validate_identifier,
    validate_study_contract,
    worker_run_id,
    write_failure_tombstone,
)
from graphvae_attr_bo_fingerprints import deployment_manifest  # noqa: E402
from tune_graphvae_attribute_weights import (  # noqa: E402
    OBJECTIVE_JSON_PATH,
    OBJECTIVE_NAME,
    build_search_ranges,
    flatten_config,
    load_yaml_mapping,
    validate_base_config,
)


REMOTE_STUDY_INPUTS = (
    "study_definition.json",
    "deployment_manifest.json",
    "dataset_cache_manifest.json",
)
RETRY_SAFE_LAUNCH_PROBE_STATES = {
    "DEFINITE_PRELAUNCH",
    "RECONCILED_PRETRIAL",
    "RECONCILED_TERMINAL",
}
TEST_FAULT_ENV = "GRAPHVAE_BO_ENABLE_TEST_FAULTS"


def _add_storage_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-env", default="GRAPHVAE_BO_STORAGE_URL")
    parser.add_argument("--heartbeat-interval", type=int, default=DEFAULT_HEARTBEAT_INTERVAL)
    parser.add_argument("--grace-period", type=int, default=DEFAULT_GRACE_PERIOD)
    parser.add_argument("--connect-timeout", type=int, default=15)
    parser.add_argument(
        "--allow-insecure-local-postgres",
        action="store_true",
        help="Test-only: allow localhost PostgreSQL without verify-full TLS.",
    )
    parser.add_argument(
        "--allow-sslmode-require-infrastructure-exception",
        action="store_true",
        help=(
            "Documented infrastructure exception: encrypted PostgreSQL without "
            "hostname authentication. Never the default."
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create/validate the immutable study budget.")
    _add_storage_options(init)
    init.add_argument("--base-config", type=Path, required=True)
    init.add_argument("--trials", type=int, required=True)
    init.add_argument("--max-parallel", type=int, default=3)
    init.add_argument("--sampler-seed", type=int, default=0)
    init.add_argument("--tpe-startup-trials", type=int, default=DEFAULT_STARTUP_TRIALS)
    init.add_argument("--alpha-node-feat-min", type=float, default=1e-3)
    init.add_argument("--alpha-node-feat-max", type=float, default=1e2)
    init.add_argument("--alpha-edge-feat-min", type=float, default=1e-3)
    init.add_argument("--alpha-edge-feat-max", type=float, default=1e2)
    init.add_argument("--fixed-alpha-node-feat", type=float, default=None)
    init.add_argument("--fixed-alpha-edge-feat", type=float, default=None)
    init.add_argument(
        "--reservation-plan",
        type=Path,
        default=None,
        help=(
            "Immutable JSON plan with exactly one parameter/seed entry per reservation. "
            "Cannot be combined with study-wide fixed parameters."
        ),
    )
    init.add_argument("--tune-alpha-motif", action="store_true")
    init.add_argument("--alpha-motif-min", type=float, default=1e-3)
    init.add_argument("--alpha-motif-max", type=float, default=1e2)
    init.add_argument("--split-seed", type=int, default=None)
    init.add_argument("--training-seed", type=int, default=0)
    init.add_argument("--generation-seed", type=int, default=123)
    init.add_argument("--evaluator-seed", type=int, default=0)
    init.add_argument("--evaluator-repeats", type=int, default=5)
    init.add_argument(
        "--evaluator-backend",
        choices=("random_gin", "graphcl_f1pr"),
        default="random_gin",
    )
    init.add_argument(
        "--graphcl-contract",
        type=Path,
        default=None,
        help="Immutable LOBSTER GraphCL evaluator backend contract JSON.",
    )
    init.add_argument("--max-graphs", type=int, default=0)
    init.add_argument("--generation-batch-size", type=int, default=16)
    init.add_argument("--nearest-k", type=int, default=5)
    init.add_argument("--adjacency-threshold", type=float, default=0.5)
    init.add_argument("--training-timeout", type=float, default=0.0)
    init.add_argument("--evaluation-timeout", type=float, default=0.0)
    init.add_argument("--termination-grace", type=float, default=10.0)
    init.add_argument("--dataset-cache-manifest", type=Path, default=None)
    init.add_argument("--deployment-manifest", type=Path, default=None)
    init.add_argument("--hardware-policy", type=Path, default=None)
    init.add_argument("--mock", action="store_true")
    init.add_argument("--interrupt-after-reservations", type=int, default=None, help=argparse.SUPPRESS)

    preflight = subparsers.add_parser("preflight", help="Validate local and worker inputs.")
    _add_storage_options(preflight)
    preflight.add_argument("--repo-paths", type=Path, required=True)
    preflight.add_argument("--python-paths", type=Path, required=True)
    preflight.add_argument("--slots", type=Path, required=True)
    preflight.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser("run", help="Dispatch bounded synchronous one-trial waves.")
    _add_storage_options(run)
    run.add_argument("--base-config", type=Path, required=True)
    run.add_argument("--repo-paths", type=Path, required=True)
    run.add_argument("--python-paths", type=Path, required=True)
    run.add_argument("--slots", type=Path, required=True)
    run.add_argument("--max-parallel", type=int, default=3)
    run.add_argument(
        "--credential-env-file",
        type=str,
        default=None,
        help=(
            "Absolute protected file sourced on the remote host before worker "
            "launch. Its contents are never copied into commands or manifests."
        ),
    )
    run.add_argument(
        "--credential-env-paths",
        type=Path,
        default=None,
        help=(
            "Host-to-absolute-protected-environment mapping. Mutually exclusive "
            "with --credential-env-file."
        ),
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--execute-remote",
        action="store_true",
        help="Required safety acknowledgement before SSH/tmux dispatch.",
    )
    run.add_argument(
        "--test-inject-definite-prelaunch-host",
        default=None,
        help=argparse.SUPPRESS,
    )
    run.add_argument(
        "--test-inject-ambiguous-after-ack-host",
        default=None,
        help=argparse.SUPPRESS,
    )

    probe = subparsers.add_parser(
        "probe", help="Reconcile recorded launch attempts without dispatching work."
    )
    _add_storage_options(probe)
    probe.add_argument("--repo-paths", type=Path, required=True)
    probe.add_argument("--python-paths", type=Path, required=True)
    probe.add_argument("--json", type=Path, default=None)

    retire = subparsers.add_parser(
        "retire-preclaim",
        help="Permanently retire a study only when no reservation was claimed.",
    )
    _add_storage_options(retire)
    retire.add_argument(
        "--reason-code",
        required=True,
        choices=("source-contract-superseded", "operator-cancelled-before-claim"),
    )

    for name in ("status", "collect", "finalize"):
        command = subparsers.add_parser(name)
        _add_storage_options(command)
        if name == "status":
            command.add_argument("--json", type=Path, default=None)
        if name == "collect":
            command.add_argument("--source-root", type=Path, required=True)

    hardware = subparsers.add_parser(
        "hardware-audit",
        help="Compare a frozen fixed-parameter study across recorded GPU slots.",
    )
    hardware.add_argument("--study-name", required=True)
    hardware.add_argument("--output-dir", type=Path, required=True)

    restore = subparsers.add_parser(
        "restore",
        help="Regenerate aggregate outputs from a frozen portable SQLite snapshot.",
    )
    restore.add_argument("--study-name", required=True)
    restore.add_argument("--source-output-dir", type=Path, required=True)
    restore.add_argument("--restore-output-dir", type=Path, required=True)

    return parser.parse_args(argv)


def _read_json(path: Path | None, default: Any = None) -> Any:
    if path is None:
        return default
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _controller_uuid(output_dir: Path) -> str:
    path = output_dir / "controller_identity.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload["controller_uuid"])
    identifier = str(uuid.uuid4())
    atomic_write_json(
        path,
        {
            "schema_version": "graphvae-attr-f1pr-controller-v1",
            "controller_uuid": identifier,
        },
    )
    return identifier


def _storage(args: argparse.Namespace, url: str):
    return create_postgresql_storage(
        url,
        heartbeat_interval=args.heartbeat_interval,
        grace_period=args.grace_period,
        connect_timeout=args.connect_timeout,
        allow_insecure_local_test=args.allow_insecure_local_postgres,
        allow_sslmode_require_exception=(
            args.allow_sslmode_require_infrastructure_exception
        ),
    )


def _search_space(args: argparse.Namespace) -> dict[str, Any]:
    ranges = build_search_ranges(args)

    def entry(value):
        return None if value is None else {"low": value[0], "high": value[1], "log": True}

    supplied_fixed = (
        args.fixed_alpha_node_feat is not None,
        args.fixed_alpha_edge_feat is not None,
    )
    if any(supplied_fixed) and not all(supplied_fixed):
        raise ValueError(
            "Fixed-parameter qualification requires both attribute-loss weights."
        )
    fixed_parameters = None
    if all(supplied_fixed):
        fixed_parameters = {
            "alpha_node_feat": float(args.fixed_alpha_node_feat),
            "alpha_edge_feat": float(args.fixed_alpha_edge_feat),
        }
        for name, value in fixed_parameters.items():
            low, high = getattr(ranges, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(
                    f"Fixed {name}={value!r} must be finite and within [{low}, {high}]."
                )
    return {
        "alpha_node_feat": entry(ranges.alpha_node_feat),
        "alpha_edge_feat": entry(ranges.alpha_edge_feat),
        "alpha_motif_loss": entry(ranges.alpha_motif_loss),
        "motif_opt_in": bool(args.tune_alpha_motif),
        "fixed_parameters": fixed_parameters,
    }


def _default_hardware_policy() -> dict[str, Any]:
    return {
        "attr_f1pr_abs_tolerance": 0.02,
        "training_loss_tolerance": {
            "absolute_floor": 0.001,
            "relative": 0.05,
            "formula": "abs(a-b) <= max(1e-3, 0.05*max(abs(a),abs(b)))",
        },
        "checkpoint_byte_equality_expected": False,
        "homogeneous_production_pool": [
            "cs-cl-13:cuda:0",
            "cs-cl-17:cuda:0",
            "cs-cl-17:cuda:1",
        ],
    }


def _definition_for_init(
    args: argparse.Namespace,
    *,
    existing_definition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config_path = args.base_config.expanduser().resolve()
    base_config = load_yaml_mapping(config_path)
    if not args.mock:
        validate_base_config(base_config, args.tune_alpha_motif)
    else:
        flatten_config(base_config)
    flat = flatten_config(base_config)
    qualification = base_config.get("bayesian_optimization_qualification") or {}
    max_graphs = int(qualification.get("max_graphs", args.max_graphs))
    generation_batch_size = int(
        qualification.get("generation_batch_size", args.generation_batch_size)
    )
    training_timeout = float(
        qualification.get("training_timeout_seconds", args.training_timeout)
    )
    evaluation_timeout = float(
        qualification.get("evaluation_timeout_seconds", args.evaluation_timeout)
    )
    termination_grace = float(
        qualification.get("termination_grace_seconds", args.termination_grace)
    )
    split_seed_value = int(flat.get("split_seed", 123) if args.split_seed is None else args.split_seed)

    cache_manifest = _read_json(args.dataset_cache_manifest)
    if cache_manifest is None:
        if not args.mock:
            raise ValueError("--dataset-cache-manifest is required outside mock mode.")
        expected = max_graphs if max_graphs > 0 else 8
        cache_manifest = {
            "schema_version": "graphvae-attr-f1pr-cache-manifest-v1",
            "mock": True,
            "sha256": "mock-cache-sha256",
            "split_fingerprint": "mock-validation-split",
            "expected_validation_graphs": expected,
        }
    source_manifest = _read_json(args.deployment_manifest)
    if source_manifest is None:
        source_manifest = deployment_manifest(REPO_ROOT, require_clean=not args.mock)
    environment = runtime_dependency_fingerprint()
    schemas = {
        "node_sha256": cache_manifest.get("node_schema_fingerprint", "mock-node-schema" if args.mock else None),
        "edge_sha256": cache_manifest.get("edge_schema_fingerprint", "mock-edge-schema" if args.mock else None),
        "node": cache_manifest.get("node_schema"),
        "edge": cache_manifest.get("edge_schema"),
    }
    if not args.mock and (not schemas["node_sha256"] or not schemas["edge_sha256"]):
        raise ValueError("Cache manifest must contain both feature-schema fingerprints.")
    hardware = _read_json(args.hardware_policy, _default_hardware_policy())
    evaluator_seeds = [args.evaluator_seed + index for index in range(args.evaluator_repeats)]
    graphcl_contract = _read_json(args.graphcl_contract)
    if args.evaluator_backend == "graphcl_f1pr":
        if not isinstance(graphcl_contract, Mapping):
            raise ValueError("--graphcl-contract is required for GraphCL-F1PR.")
        required_graphcl = {
            "schema_version",
            "backend",
            "objective_json_path",
            "compatibility_objective_json_path",
            "encoder_bundle_sha256",
            "encoder_bundle_manifest_sha256",
            "encoder_checkpoints",
            "graphcl_runtime_sha256",
            "upstream_revision",
            "validation_collection_sha256",
            "validation_reference_file_sha256",
            "validation_split_fingerprint",
            "paths",
            "training_seeds",
            "checkpoint_count",
            "nearest_k",
            "test_access",
        }
        if set(graphcl_contract) != required_graphcl:
            raise ValueError("GraphCL backend contract fields differ from the frozen schema.")
        expected_exact = {
            "schema_version": "lobster-graphcl-f1pr-distributed-backend-v1",
            "backend": "graphcl_f1pr",
            "objective_json_path": "summary.f1_pr.mean",
            "compatibility_objective_json_path": OBJECTIVE_JSON_PATH,
            "training_seeds": [0, 1],
            "checkpoint_count": 5,
            "nearest_k": 5,
            "test_access": False,
        }
        for field, expected in expected_exact.items():
            if graphcl_contract.get(field) != expected:
                raise ValueError(f"GraphCL backend contract differs for {field}.")
        encoders = graphcl_contract["encoder_checkpoints"]
        if (
            not isinstance(encoders, list)
            or [item.get("seed") for item in encoders] != [101, 202, 303, 404, 505]
            or len({item.get("sha256") for item in encoders}) != 5
        ):
            raise ValueError("GraphCL backend encoder order or hashes differ.")
        if max_graphs != 10 or args.nearest_k != 5:
            raise ValueError("GraphCL-F1PR requires exactly 10 graphs and nearest-k 5.")
    elif graphcl_contract is not None:
        raise ValueError("--graphcl-contract is valid only with GraphCL-F1PR.")
    search_space = _search_space(args)
    reservation_plan = None
    if args.reservation_plan is not None:
        payload = _read_json(args.reservation_plan)
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "reservations",
        }:
            raise ValueError(
                "Reservation-plan JSON requires only schema_version and reservations."
            )
        if payload["schema_version"] != "graphvae-attr-f1pr-reservation-plan-v1":
            raise ValueError("Unsupported reservation-plan schema version.")
        reservation_plan = payload["reservations"]
        if search_space.get("fixed_parameters"):
            raise ValueError(
                "--reservation-plan cannot be combined with study-wide fixed parameters."
            )
    definition = build_study_definition(
        study_name=args.study_name,
        study_uuid=(None if existing_definition is None else existing_definition["study_uuid"]),
        base_config=base_config,
        base_config_sha256=sha256_file(config_path),
        ranges=search_space,
        reserved_trials=args.trials,
        seeds={
            "study_seed": args.sampler_seed,
            "sampler_seed": args.sampler_seed,
            "tpe_startup_trials": args.tpe_startup_trials,
            "split_seed": split_seed_value,
            "training_seed": args.training_seed,
            "generation_seed": args.generation_seed,
            "evaluator_seed": args.evaluator_seed,
            "evaluator_repeat_seeds": evaluator_seeds,
            "training_seeds": (
                [0, 1]
                if args.evaluator_backend == "graphcl_f1pr"
                else [args.training_seed]
            ),
        },
        evaluator={
            "backend": args.evaluator_backend,
            "mode": "decoded_node_edge",
            "split": "validation",
            "test_access": False,
            "repeat_count": args.evaluator_repeats,
            "max_graphs": max_graphs,
            "generation_batch_size": generation_batch_size,
            "nearest_k": args.nearest_k,
            "adjacency_threshold": args.adjacency_threshold,
            "backend_contract": graphcl_contract,
        },
        training={
            "epoch_number": int(flat["epoch_number"]),
            "training_timeout_seconds": None if training_timeout <= 0 else training_timeout,
            "evaluation_timeout_seconds": None if evaluation_timeout <= 0 else evaluation_timeout,
            "termination_grace_seconds": termination_grace,
            "mock": bool(args.mock),
        },
        source=source_manifest,
        environment=environment,
        dataset_cache=cache_manifest,
        feature_schemas=schemas,
        hardware_policy=hardware,
        heartbeat_interval=args.heartbeat_interval,
        grace_period=args.grace_period,
        max_parallel=args.max_parallel,
        reservation_plan=reservation_plan,
    )
    definition["storage"]["tls_policy"] = (
        "localhost-test-exception"
        if args.allow_insecure_local_postgres
        else (
            "require-documented-infrastructure-exception"
            if args.allow_sslmode_require_infrastructure_exception
            else "verify-full"
        )
    )
    return definition


def _publish_remote_study_inputs(
    output_dir: Path, definition: Mapping[str, Any]
) -> None:
    """Publish exact public manifests required by remote worker staging."""

    payloads = {
        "deployment_manifest.json": definition["source"],
        "dataset_cache_manifest.json": definition["dataset_cache"],
    }
    for filename, payload in payloads.items():
        path = output_dir / filename
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise DistributedContractError(
                    f"Existing remote study input differs: {filename}"
                )
            continue
        atomic_write_json(path, payload)


def command_init(args: argparse.Namespace) -> int:
    enforce_pinned_versions()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_url = storage_url_from_env(args.storage_env)
    with ControllerLocks(output_dir, storage_url, args.study_name) as locks:
        controller_uuid = _controller_uuid(output_dir)
        storage = _storage(args, storage_url)
        study = create_or_load_distributed_study(
            storage,
            study_name=args.study_name,
            sampler_seed_value=args.sampler_seed,
            startup_trials=args.tpe_startup_trials,
            create=True,
        )
        existing_definition = study.user_attrs.get(DEFINITION_ATTR)
        definition = _definition_for_init(args, existing_definition=existing_definition)
        locks.assert_alive()
        contract_hash = initialize_reserved_study(
            study,
            definition,
            controller_uuid=controller_uuid,
            output_root=output_dir,
            interrupt_after=args.interrupt_after_reservations,
        )
        locks.assert_alive()
        atomic_write_json(output_dir / "study_definition.json", definition)
        _publish_remote_study_inputs(output_dir, definition)
    print(
        f"Initialized {args.trials} reserved Attr-F1PR slots for {args.study_name!r}; "
        f"contract {contract_hash}."
    )
    return 0


def _load_mapping(path: Path) -> dict[str, str]:
    result = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        content = raw.split("#", 1)[0].strip()
        if not content:
            continue
        parts = content.split()
        if len(parts) != 2:
            raise ValueError(f"Malformed mapping row {line_number} in {path}.")
        host, value = parts
        validate_identifier(host, "mapping host")
        if host in result:
            raise ValueError(f"Duplicate host {host!r} in {path}.")
        result[host] = value
    return result


def render_worker_command(
    *,
    python_path: str,
    repo_path: str,
    study_name: str,
    base_config: str,
    artifact_root: str,
    contract_hash: str,
    worker_id: str,
    worker_run_id_value: str,
    sampler_seed_value: int,
    dispatch_sequence: int,
    physical_gpu: int | None,
    storage_env: str,
    heartbeat_interval: int,
    grace_period: int,
    tpe_startup_trials: int = DEFAULT_STARTUP_TRIALS,
    mock: bool = False,
    allow_sslmode_require_exception: bool = False,
) -> list[str]:
    command = [
        python_path,
        str(Path(repo_path) / "scripts" / "run_graphvae_attr_bo_worker.py"),
        "--study-name", study_name,
        "--base-config", base_config,
        "--artifact-root", artifact_root,
        "--study-contract-sha256", contract_hash,
        "--worker-id", worker_id,
        "--worker-run-id", worker_run_id_value,
        "--sampler-seed", str(sampler_seed_value),
        "--dispatch-sequence", str(dispatch_sequence),
        "--device", "cpu" if physical_gpu is None else "cuda:0",
        "--storage-env", storage_env,
        "--heartbeat-interval", str(heartbeat_interval),
        "--grace-period", str(grace_period),
        "--tpe-startup-trials", str(tpe_startup_trials),
    ]
    if physical_gpu is not None:
        command.extend(["--physical-gpu", str(physical_gpu)])
    if mock:
        command.append("--mock")
    if allow_sslmode_require_exception:
        command.append("--allow-sslmode-require-infrastructure-exception")
    return command


def render_remote_launch(
    command: Sequence[str],
    *,
    physical_gpu: int | None,
    lock_path: str,
    credential_env_file: str | None = None,
) -> str:
    # The storage URL remains solely in the named inherited environment variable.
    environment_prefix = ""
    if credential_env_file is not None:
        credential_path = Path(credential_env_file)
        if not credential_path.is_absolute():
            raise ValueError("Remote credential environment file must be absolute.")
        if any(
            character in credential_env_file
            for character in ("\n", "\r", "\x00")
        ):
            raise ValueError(
                "Remote credential environment file contains control characters."
            )
        environment_prefix = (
            f"set -a; . {shlex.quote(credential_env_file)}; set +a; "
        )
    device_prefix = (
        "" if physical_gpu is None else f"CUDA_VISIBLE_DEVICES={shlex.quote(str(physical_gpu))} "
    )
    inner = (
        environment_prefix
        + device_prefix
        + f"flock -n {shlex.quote(lock_path)} "
        + shlex.join(list(command))
    )
    return inner


def render_tmux_ssh_command(
    *, host: str, tmux_name: str, remote_shell: str
) -> list[str]:
    """Keep the complete worker shell as tmux's single shell-command argument."""

    validate_identifier(host, "SSH host")
    validate_identifier(tmux_name, "tmux session")
    return [
        "ssh",
        "-n",
        host,
        shlex.join(
            ["tmux", "new-session", "-d", "-s", tmux_name, remote_shell]
        ),
    ]


def _stage_remote_study_inputs(
    *, host: str, remote_root: str, output_dir: Path
) -> None:
    """Stage immutable public study inputs and verify their remote hashes."""

    local_paths = [output_dir / filename for filename in REMOTE_STUDY_INPUTS]
    missing = [str(path) for path in local_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Remote study input staging is missing: " + ", ".join(missing)
        )
    subprocess.run(
        ["ssh", "-n", host, shlex.join(["mkdir", "-p", remote_root])],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-azc",
            "--ignore-existing",
            "--chmod=F444",
            *(str(path) for path in local_paths),
            f"{host}:{remote_root.rstrip('/')}/",
        ],
        check=True,
    )
    remote_paths = [str(Path(remote_root) / path.name) for path in local_paths]
    result = subprocess.run(
        ["ssh", "-n", host, shlex.join(["sha256sum", *remote_paths])],
        check=True,
        capture_output=True,
        text=True,
    )
    remote_hashes = [
        line.split(maxsplit=1)[0]
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    local_hashes = [sha256_file(path) for path in local_paths]
    if remote_hashes != local_hashes:
        raise DistributedContractError(
            f"Remote immutable study inputs failed hash verification on {host}."
        )


def _preflight_inputs(args: argparse.Namespace):
    repositories = _load_mapping(args.repo_paths)
    pythons = _load_mapping(args.python_paths)
    if set(repositories) != set(pythons):
        raise ValueError("Repository and Python host mappings differ.")
    slots = parse_slots(args.slots, known_hosts=sorted(repositories))
    return repositories, pythons, slots


def _credential_environment_paths(
    args: argparse.Namespace,
    repositories: Mapping[str, str],
    *,
    required: bool,
) -> dict[str, str]:
    single = getattr(args, "credential_env_file", None)
    mapping_path = getattr(args, "credential_env_paths", None)
    if single and mapping_path is not None:
        raise ValueError(
            "Use either --credential-env-file or --credential-env-paths, not both."
        )
    if mapping_path is not None:
        mapped = _load_mapping(mapping_path)
        if set(mapped) != set(repositories):
            raise ValueError(
                "Credential environment mapping hosts differ from repository hosts."
            )
    elif single:
        mapped = {host: str(single) for host in repositories}
    else:
        if required:
            raise ValueError(
                "Remote dispatch requires a protected credential environment file or mapping."
            )
        return {}
    for host, value in mapped.items():
        if not Path(value).is_absolute() or any(
            character in value for character in ("\n", "\r", "\x00")
        ):
            raise ValueError(
                f"Credential environment path for {host} must be absolute and safe."
            )
    return mapped


def _load_ready_study(args: argparse.Namespace, *, require_ready: bool = True):
    output_dir = args.output_dir.expanduser().resolve()
    local_definition = json.loads(
        (output_dir / "study_definition.json").read_text(encoding="utf-8")
    )
    contract_hash = canonical_contract_hash(local_definition)
    storage_url = storage_url_from_env(args.storage_env)
    storage = _storage(args, storage_url)
    study = create_or_load_distributed_study(
        storage,
        study_name=args.study_name,
        sampler_seed_value=int(local_definition["seeds"]["study_seed"]),
        startup_trials=DEFAULT_STARTUP_TRIALS,
        create=False,
    )
    validate_study_contract(
        study,
        expected_contract_hash=contract_hash,
        local_definition=local_definition,
        require_ready=require_ready,
    )
    return storage_url, storage, study, local_definition, contract_hash


def _assert_controller_owner(study: Any, output_dir: Path, controller_uuid: str) -> None:
    if study.user_attrs.get(CONTROLLER_ATTR) != controller_uuid:
        raise DistributedContractError(
            "Controller identity differs from the initialized study owner."
        )
    if study.user_attrs.get("graphvae_bo_output_root_sha256") != output_root_fingerprint(
        output_dir
    ):
        raise DistributedContractError(
            "Controller output root differs from the initialized study owner."
        )


def _validate_mock_cpu_slots(
    definition: Mapping[str, Any], slots: Sequence[Mapping[str, Any]]
) -> None:
    if any(slot["physical_gpu"] is None for slot in slots) and not bool(
        definition["training"].get("mock", False)
    ):
        raise DistributedContractError("mock-cpu slots are forbidden for real studies.")


def _command_preflight_locked(args: argparse.Namespace, controller_uuid: str) -> int:
    repositories, pythons, slots = _preflight_inputs(args)
    _storage_url, _storage_instance, study, definition, contract_hash = _load_ready_study(args)
    _assert_controller_owner(
        study, args.output_dir.expanduser().resolve(), controller_uuid
    )
    _validate_mock_cpu_slots(definition, slots)
    if len(slots) < int(definition["scheduler"]["max_parallel"]):
        raise ValueError("Verified slot count is below the contract maximum concurrency.")
    payload = {
        "schema_version": "graphvae-attr-f1pr-preflight-v1",
        "study_name": study.study_name,
        "study_contract_sha256": contract_hash,
        "slot_count": len(slots),
        "hosts": sorted(repositories),
        "dry_run": bool(args.dry_run),
        "storage_backend": "PostgreSQL",
        "test_access": False,
    }
    atomic_write_json(args.output_dir / "preflight.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    storage_url = storage_url_from_env(args.storage_env)
    with ControllerLocks(output_dir, storage_url, args.study_name) as locks:
        controller_uuid = _controller_uuid(output_dir)
        result = _command_preflight_locked(args, controller_uuid)
        locks.assert_alive()
        return result


def _latest_launch_probe_records(output_dir: Path) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    probe_root = output_dir / "launch_probes"
    if not probe_root.is_dir():
        return latest
    for path in sorted(probe_root.glob("probe_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("launches", []):
            worker_run = record.get("worker_run_id")
            if worker_run:
                latest[str(worker_run)] = record
    return latest


def _assert_prior_launches_reconciled(output_dir: Path) -> None:
    """Refuse a new wave until every attempted prior launch has been probed."""

    latest = _latest_launch_probe_records(output_dir)
    launch_root = output_dir / "launch_manifests"
    if not launch_root.is_dir():
        return
    unresolved = []
    for path in sorted(launch_root.glob("wave_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dry_run") is True:
            continue
        for launch in payload.get("launches", []):
            state = launch.get("launch_state")
            if state == "PLANNED":
                # All public inputs are staged before any SSH/tmux launch call.
                # A PLANNED record therefore proves this identity was not started.
                continue
            worker_run = str(launch.get("worker_run_id"))
            probe = latest.get(worker_run)
            if (
                not probe
                or probe.get("retry_safe") is not True
                or probe.get("probe_status") not in RETRY_SAFE_LAUNCH_PROBE_STATES
            ):
                unresolved.append(worker_run)
    if unresolved:
        raise DistributedContractError(
            "Prior launch attempts require a safe probe before another wave: "
            + ", ".join(sorted(set(unresolved)))
        )


def retire_preclaim_study(
    study: Any,
    output_dir: Path,
    *,
    contract_hash: str,
    reason_code: str,
) -> dict[str, Any]:
    """Make an unused immutable study permanently non-dispatchable."""

    output_dir = output_dir.expanduser().resolve()
    definition = study.user_attrs.get(DEFINITION_ATTR) or {}
    expected_count = int(definition.get("reserved_trials", 0))
    audit = reservation_audit(study, expected_count)
    states = reserved_trial_states(study)
    if (
        expected_count <= 0
        or audit["missing_indexes"]
        or audit["duplicate_indexes"]
        or audit["invalid_trial_numbers"]
        or audit["unreserved_trials"]
        or states.get("RESERVED_TOTAL") != expected_count
        or states.get("UNRESERVED_GUARD") != 0
        or states.get("WAITING") != expected_count
        or any(states.get(name) != 0 for name in ("RUNNING", "COMPLETE", "FAIL", "OTHER"))
    ):
        raise DistributedContractError(
            "Preclaim retirement requires every exact reservation to remain WAITING."
        )
    if canonical_contract_hash(definition) != contract_hash:
        raise DistributedContractError("Preclaim retirement contract hash mismatch.")

    _assert_prior_launches_reconciled(output_dir)
    latest = _latest_launch_probe_records(output_dir)
    attempted = []
    for path in sorted((output_dir / "launch_manifests").glob("wave_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dry_run") is True:
            continue
        for launch in payload.get("launches", []):
            if launch.get("launch_state") == "PLANNED":
                continue
            worker_run = str(launch.get("worker_run_id"))
            probe = latest.get(worker_run)
            if (
                not probe
                or probe.get("probe_status") != "RECONCILED_PRETRIAL"
                or probe.get("retry_safe") is not True
                or probe.get("tmux_active") is not False
                or probe.get("db_trials") != []
            ):
                raise DistributedContractError(
                    f"Launch {worker_run} is not proven unclaimed and pretrial-terminal."
                )
            attempted.append(worker_run)

    lifecycle = study.user_attrs.get(LIFECYCLE_ATTR)
    if lifecycle not in {LIFECYCLE_READY, LIFECYCLE_RETIRED_PRECLAIM}:
        raise DistributedContractError(
            "Preclaim retirement requires READY or already RETIRED_PRECLAIM lifecycle."
        )
    marker = {
        "schema_version": "graphvae-attr-f1pr-retired-preclaim-v1",
        "study_name": study.study_name,
        "study_contract_sha256": contract_hash,
        "lifecycle": LIFECYCLE_RETIRED_PRECLAIM,
        "reason_code": reason_code,
        "reserved_waiting": expected_count,
        "reservation_consumed": False,
        "attempted_worker_runs": sorted(attempted),
    }
    marker_path = output_dir / "RETIRED_PRECLAIM.json"
    if marker_path.is_file():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if existing != marker:
            raise DistributedContractError("Preclaim retirement marker differs.")
    else:
        study.set_user_attr(LIFECYCLE_ATTR, LIFECYCLE_RETIRED_PRECLAIM)
        atomic_write_json(marker_path, marker)
    if study.user_attrs.get(LIFECYCLE_ATTR) != LIFECYCLE_RETIRED_PRECLAIM:
        study.set_user_attr(LIFECYCLE_ATTR, LIFECYCLE_RETIRED_PRECLAIM)
    return marker


def command_retire_preclaim(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    storage_url = storage_url_from_env(args.storage_env)
    with ControllerLocks(output_dir, storage_url, args.study_name) as locks:
        controller_uuid = _controller_uuid(output_dir)
        _url, _storage, study, _definition, contract_hash = _load_ready_study(
            args, require_ready=False
        )
        _assert_controller_owner(study, output_dir, controller_uuid)
        marker = retire_preclaim_study(
            study,
            output_dir,
            contract_hash=contract_hash,
            reason_code=args.reason_code,
        )
        locks.assert_alive()
    print(json.dumps(marker, sort_keys=True))
    return 0


def _classify_launch_probe(
    *,
    launch_state: str,
    remote_reachable: bool,
    tmux_active: bool,
    markers: Sequence[str],
    db_trials: Sequence[Mapping[str, Any]],
    marker_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, bool]:
    marker_set = set(markers)
    if len(marker_set) > 1:
        return "CONFLICT", False
    db_states = [str(record.get("state")) for record in db_trials]
    if launch_state == "PLANNED" and not db_trials:
        return "DEFINITE_PRELAUNCH", True
    if tmux_active or "RUNNING" in db_states:
        return "ACTIVE_AMBIGUOUS", False
    if not remote_reachable:
        return "UNREACHABLE_AMBIGUOUS", False
    marker = next(iter(marker_set), None)
    marker_payload = (marker_payloads or {}).get(str(marker), {})
    if marker and marker_payload.get("parse_ok") is not True:
        return "CONFLICT", False
    if (
        marker == "FAILED_PRETRIAL"
        and marker_payload.get("reservation_consumed") is False
        and not db_trials
    ):
        return "RECONCILED_PRETRIAL", True
    if (
        marker == "COMPLETED"
        and len(db_trials) == 1
        and db_states[0] in {"COMPLETE", "FAIL"}
        and db_trials[0].get("reserved") is True
        and marker_payload.get("trial_number") == db_trials[0].get("trial_number")
        and marker_payload.get("budget_index") == db_trials[0].get("budget_index")
        and marker_payload.get("db_state") == db_states[0]
    ):
        return "RECONCILED_TERMINAL", True
    if (
        marker == "RECONCILED_FAIL"
        and len(db_trials) == 1
        and db_states == ["FAIL"]
        and db_trials[0].get("reserved") is True
        and marker_payload.get("trial_number") == db_trials[0].get("trial_number")
        and marker_payload.get("budget_index") == db_trials[0].get("budget_index")
        and marker_payload.get("db_state") == "FAIL"
    ):
        return "RECONCILED_TERMINAL", True
    return "MISSING_AMBIGUOUS", False


def _probe_remote_launch(
    *,
    host: str,
    repo_path: str,
    python_path: str,
    study_name: str,
    worker_run_id_value: str,
    tmux_name: str,
    connect_timeout: int,
) -> dict[str, Any]:
    remote_root = Path(repo_path) / "runs" / "bayesian_optimization" / study_name
    run_dir = remote_root / "workers" / worker_run_id_value
    probe_code = """\
import json
import pathlib
import subprocess
import sys

run_dir = pathlib.Path(sys.argv[1])
tmux_name = sys.argv[2]
marker_payloads = {}
for marker in ("COMPLETED", "FAILED_PRETRIAL", "RECONCILED_FAIL"):
    path = run_dir / marker
    if not path.is_file():
        continue
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        marker_payloads[marker] = {"parse_ok": False}
        continue
    marker_payloads[marker] = {
        "parse_ok": True,
        "schema_version": value.get("schema_version"),
        "reservation_consumed": value.get("reservation_consumed"),
        "trial_number": value.get("trial_number"),
        "budget_index": value.get("budget_index"),
        "db_state": value.get("db_state"),
    }
payload = {
    "tmux_active": subprocess.run(
        ["tmux", "has-session", "-t", "=" + tmux_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0,
    "marker_payloads": marker_payloads,
    "run_info": (run_dir / "RUN_INFO.json").is_file(),
    "heartbeat": (run_dir / "HEARTBEAT.json").is_file(),
}
print(json.dumps(payload, sort_keys=True))
"""
    result = subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            f"ConnectTimeout={int(connect_timeout)}",
            host,
            shlex.join([python_path, "-c", probe_code, str(run_dir), tmux_name]),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=max(5, int(connect_timeout) + 5),
    )
    if result.returncode != 0:
        return {
            "remote_reachable": False,
            "ssh_returncode": result.returncode,
            "tmux_active": False,
            "markers": [],
            "marker_payloads": {},
            "run_info": False,
            "heartbeat": False,
        }
    values = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, Mapping):
            values = candidate
            break
    if values is None:
        return {
            "remote_reachable": True,
            "ssh_returncode": 0,
            "probe_parse_error": True,
            "tmux_active": False,
            "markers": [],
            "marker_payloads": {},
            "run_info": False,
            "heartbeat": False,
        }
    marker_payloads = dict(values.get("marker_payloads") or {})
    return {
        "remote_reachable": True,
        "ssh_returncode": 0,
        "tmux_active": values.get("tmux_active") is True,
        "markers": sorted(marker_payloads),
        "marker_payloads": marker_payloads,
        "run_info": values.get("run_info") is True,
        "heartbeat": values.get("heartbeat") is True,
    }


def _local_marker_probe_payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"parse_ok": False}
    return {
        "parse_ok": True,
        "schema_version": value.get("schema_version"),
        "reservation_consumed": value.get("reservation_consumed"),
        "trial_number": value.get("trial_number"),
        "budget_index": value.get("budget_index"),
        "db_state": value.get("db_state"),
    }


def _command_probe_locked(args: argparse.Namespace, controller_uuid: str) -> int:
    repositories = _load_mapping(args.repo_paths)
    pythons = _load_mapping(args.python_paths)
    if set(repositories) != set(pythons):
        raise ValueError("Repository and Python host mappings differ.")
    _url, _storage_instance, study, _definition, contract_hash = _load_ready_study(
        args, require_ready=False
    )
    output_dir = args.output_dir.expanduser().resolve()
    _assert_controller_owner(study, output_dir, controller_uuid)
    records = []
    launch_root = output_dir / "launch_manifests"
    for path in sorted(launch_root.glob("wave_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("dry_run") is True:
            continue
        for launch in manifest.get("launches", []):
            worker_run = str(launch["worker_run_id"])
            host = str(launch["host"])
            if host in repositories:
                remote = _probe_remote_launch(
                    host=host,
                    repo_path=repositories[host],
                    python_path=pythons[host],
                    study_name=args.study_name,
                    worker_run_id_value=worker_run,
                    tmux_name=str(launch["tmux_session"]),
                    connect_timeout=args.connect_timeout,
                )
            else:
                remote = {
                    "remote_reachable": False,
                    "ssh_returncode": None,
                    "tmux_active": False,
                    "markers": [],
                    "marker_payloads": {},
                    "run_info": False,
                    "heartbeat": False,
                }
            local_run_dir = output_dir / "workers" / worker_run
            local_markers = [
                marker
                for marker in ("COMPLETED", "FAILED_PRETRIAL", "RECONCILED_FAIL")
                if (local_run_dir / marker).is_file()
            ]
            marker_payloads = dict(remote["marker_payloads"])
            for marker in local_markers:
                local_payload = _local_marker_probe_payload(local_run_dir / marker)
                if marker in marker_payloads and marker_payloads[marker] != local_payload:
                    marker_payloads[marker] = {"parse_ok": False, "conflict": True}
                else:
                    marker_payloads[marker] = local_payload
            markers = sorted(set(remote["markers"]) | set(local_markers))
            db_trials = [
                {
                    "trial_number": trial.number,
                    "budget_index": trial.user_attrs.get(BUDGET_INDEX_ATTR),
                    "reserved": trial.user_attrs.get(RESERVED_ATTR) is True,
                    "state": trial.state.name,
                }
                for trial in study.get_trials(deepcopy=False)
                if trial.user_attrs.get("worker_run_id") == worker_run
            ]
            status, retry_safe = _classify_launch_probe(
                launch_state=str(launch.get("launch_state")),
                remote_reachable=bool(remote["remote_reachable"]),
                tmux_active=bool(remote["tmux_active"]),
                markers=markers,
                db_trials=db_trials,
                marker_payloads=marker_payloads,
            )
            records.append(
                {
                    "wave_index": manifest.get("wave_index"),
                    "host": host,
                    "worker_run_id": worker_run,
                    "launch_state": launch.get("launch_state"),
                    **remote,
                    "markers": markers,
                    "marker_payloads": marker_payloads,
                    "db_trials": db_trials,
                    "probe_status": status,
                    "retry_safe": retry_safe,
                }
            )
    probe_root = output_dir / "launch_probes"
    probe_index = len(list(probe_root.glob("probe_*.json"))) + 1
    payload = {
        "schema_version": "graphvae-attr-f1pr-launch-probe-v1",
        "study_name": args.study_name,
        "study_contract_sha256": contract_hash,
        "probe_index": probe_index,
        "probed_at_unix": time.time(),
        "test_access": False,
        "launches": records,
    }
    path = probe_root / f"probe_{probe_index:04d}.json"
    atomic_write_json(path, payload)
    if args.json is not None and args.json.expanduser().resolve() != path.resolve():
        atomic_write_json(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_probe(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    storage_url = storage_url_from_env(args.storage_env)
    with ControllerLocks(output_dir, storage_url, args.study_name) as locks:
        controller_uuid = _controller_uuid(output_dir)
        result = _command_probe_locked(args, controller_uuid)
        locks.assert_alive()
        return result


def _validate_test_faults(args: argparse.Namespace, wave_slots: Sequence[Mapping[str, Any]]) -> None:
    definite = args.test_inject_definite_prelaunch_host
    ambiguous = args.test_inject_ambiguous_after_ack_host
    if definite and ambiguous:
        raise ValueError("Only one launch fault may be injected per controller call.")
    selected = definite or ambiguous
    if selected is None:
        return
    validate_identifier(selected, "test fault host")
    if os.environ.get(TEST_FAULT_ENV) != "1":
        raise ValueError(f"Test launch faults require {TEST_FAULT_ENV}=1.")
    if selected not in {str(slot["host"]) for slot in wave_slots}:
        raise ValueError("Test launch fault host is not selected in this wave.")


def _waiting_reservations_require_adaptive_sampling(
    definition: Mapping[str, Any], waiting: Sequence[Any]
) -> bool:
    """Return whether any waiting reservation still needs a sampler decision."""

    search_space = definition.get("search_space") or {}
    sampled_names = {
        name
        for name in ("alpha_node_feat", "alpha_edge_feat", "alpha_motif_loss")
        if isinstance(search_space.get(name), Mapping)
    }
    for trial in waiting:
        fixed_parameters = trial.system_attrs.get("fixed_params") or {}
        if not isinstance(fixed_parameters, Mapping):
            return True
        if not sampled_names.issubset(fixed_parameters):
            return True
    return False


def _startup_aware_wave_limit(
    definition: Mapping[str, Any],
    waiting: Sequence[Any],
    *,
    usable_observations: int,
    contracted_parallelism: int,
) -> tuple[int, bool, int]:
    """Apply the startup barrier only while a waiting trial needs sampling."""

    startup_target = int(definition["sampler"]["n_startup_trials"])
    startup_remaining = max(0, startup_target - usable_observations)
    adaptive_sampling_waiting = _waiting_reservations_require_adaptive_sampling(
        definition, waiting
    )
    wave_limit = (
        min(contracted_parallelism, startup_remaining)
        if adaptive_sampling_waiting and startup_remaining
        else contracted_parallelism
    )
    return wave_limit, adaptive_sampling_waiting, startup_remaining


def _command_run_locked(args: argparse.Namespace, controller_uuid: str) -> int:
    repositories, pythons, slots = _preflight_inputs(args)
    credential_paths = _credential_environment_paths(
        args, repositories, required=not args.dry_run
    )
    storage_url, _storage_instance, study, definition, contract_hash = _load_ready_study(args)
    _assert_controller_owner(
        study, args.output_dir.expanduser().resolve(), controller_uuid
    )
    _validate_mock_cpu_slots(definition, slots)
    _assert_prior_launches_reconciled(args.output_dir.expanduser().resolve())
    contracted_parallelism = int(definition["scheduler"]["max_parallel"])
    if args.max_parallel != contracted_parallelism:
        raise DistributedContractError(
            "Requested maximum parallelism differs from the immutable contract."
        )
    if (
        args.heartbeat_interval
        != int(definition["storage"]["heartbeat_interval"])
        or args.grace_period
        != int(definition["storage"]["grace_period"])
    ):
        raise DistributedContractError(
            "Worker heartbeat/grace settings differ from the immutable contract."
        )
    if not args.dry_run and not args.execute_remote:
        raise ValueError("Remote dispatch requires the explicit --execute-remote acknowledgement.")
    waiting = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state.name == "WAITING" and trial.user_attrs.get(RESERVED_ATTR) is True
    ]
    usable_observations = sum(
        trial.state.name == "COMPLETE"
        and trial.value is not None
        and math.isfinite(float(trial.value))
        for trial in study.get_trials(deepcopy=False)
        if trial.user_attrs.get(RESERVED_ATTR) is True
    )
    wave_limit, adaptive_sampling_waiting, startup_remaining = (
        _startup_aware_wave_limit(
            definition,
            waiting,
            usable_observations=usable_observations,
            contracted_parallelism=contracted_parallelism,
        )
    )
    startup_target = int(definition["sampler"]["n_startup_trials"])
    wave_slots = slots[: min(wave_limit, len(waiting), len(slots))]
    _validate_test_faults(args, wave_slots)
    wave_index = len(list((args.output_dir / "launch_manifests").glob("wave_*.json"))) + 1
    launches = []
    for offset, slot in enumerate(wave_slots):
        dispatch_sequence = wave_index * 1_000_000 + offset
        seed = sampler_seed(int(definition["seeds"]["study_seed"]), dispatch_sequence)
        run_id = worker_run_id(slot["worker_id"], dispatch_sequence)
        repo_path = repositories[slot["host"]]
        remote_root = str(Path(repo_path) / "runs" / "bayesian_optimization" / args.study_name)
        remote_config = str(Path(repo_path) / args.base_config)
        worker_command = render_worker_command(
            python_path=pythons[slot["host"]],
            repo_path=repo_path,
            study_name=args.study_name,
            base_config=remote_config,
            artifact_root=remote_root,
            contract_hash=contract_hash,
            worker_id=slot["worker_id"],
            worker_run_id_value=run_id,
            sampler_seed_value=seed,
            dispatch_sequence=dispatch_sequence,
            physical_gpu=slot["physical_gpu"],
            storage_env=args.storage_env,
            heartbeat_interval=args.heartbeat_interval,
            grace_period=args.grace_period,
            tpe_startup_trials=int(definition["sampler"]["n_startup_trials"]),
            mock=bool(definition["training"].get("mock", False)),
            allow_sslmode_require_exception=(
                definition["storage"].get("tls_policy")
                == "require-documented-infrastructure-exception"
            ),
        )
        remote_shell = render_remote_launch(
            worker_command,
            physical_gpu=slot["physical_gpu"],
            lock_path=f"/tmp/graphvae-bo-{slot['worker_id']}.lock",
            credential_env_file=credential_paths.get(slot["host"]),
        )
        tmux_name = f"graphvae-bo-{run_id}"
        launch = {
            **slot,
            "dispatch_sequence": dispatch_sequence,
            "sampler_seed": seed,
            "worker_run_id": run_id,
            "tmux_session": tmux_name,
            "remote_command": remote_shell,
            "launch_state": "PLANNED",
        }
        launches.append(launch)
    manifest = {
        "schema_version": "graphvae-attr-f1pr-launch-wave-v1",
        "wave_index": wave_index,
        "bounded_synchronous": True,
        "study_contract_sha256": contract_hash,
        "launches": launches,
        "dry_run": bool(args.dry_run),
        "usable_complete_observations_before_wave": usable_observations,
        "startup_target": startup_target,
        "adaptive_sampling_waiting": adaptive_sampling_waiting,
        "startup_gating_applied": bool(
            adaptive_sampling_waiting and startup_remaining
        ),
        "controller_attempt": {
            "phase": "planned",
            "state": "PLANNED",
            "ambiguous": False,
        },
    }
    manifest_path = (
        args.output_dir / "launch_manifests" / f"wave_{wave_index:04d}.json"
    )
    atomic_write_json(manifest_path, manifest)
    if not args.dry_run:
        for host in sorted({slot["host"] for slot in wave_slots}):
            remote_root = str(
                Path(repositories[host])
                / "runs"
                / "bayesian_optimization"
                / args.study_name
            )
            try:
                if args.test_inject_definite_prelaunch_host == host:
                    raise DistributedContractError(
                        "Injected definite host failure before remote input staging."
                    )
                _stage_remote_study_inputs(
                    host=host,
                    remote_root=remote_root,
                    output_dir=args.output_dir.expanduser().resolve(),
                )
            except Exception as exc:
                manifest["controller_attempt"] = {
                    "phase": "remote_input_staging",
                    "state": "DEFINITE_PRELAUNCH_ERROR",
                    "host": host,
                    "exception_type": type(exc).__name__,
                    "ambiguous": False,
                    "failed_at_unix": time.time(),
                }
                atomic_write_json(manifest_path, manifest)
                raise
        for launch in launches:
            launch["launch_state"] = "ATTEMPTING"
            atomic_write_json(manifest_path, manifest)
            try:
                ssh_command = render_tmux_ssh_command(
                    host=launch["host"],
                    tmux_name=launch["tmux_session"],
                    remote_shell=launch["remote_command"],
                )
                subprocess.run(ssh_command, check=True)
                if args.test_inject_ambiguous_after_ack_host == launch["host"]:
                    launch["remote_ack_observed_before_injected_fault"] = True
                    raise subprocess.CalledProcessError(255, ssh_command)
            except subprocess.CalledProcessError as exc:
                launch["launch_state"] = "SSH_ERROR"
                launch["ssh_returncode"] = exc.returncode
                launch["launch_ambiguity"] = True
                manifest["controller_attempt"] = {
                    "phase": "remote_launch",
                    "state": "AMBIGUOUS_SSH_ERROR",
                    "host": launch["host"],
                    "worker_run_id": launch["worker_run_id"],
                    "exception_type": type(exc).__name__,
                    "ssh_returncode": exc.returncode,
                    "injected_after_remote_ack": bool(
                        launch.get("remote_ack_observed_before_injected_fault")
                    ),
                    "ambiguous": True,
                    "failed_at_unix": time.time(),
                }
                atomic_write_json(manifest_path, manifest)
                raise
            launch["launch_state"] = "SSH_ACKNOWLEDGED"
            atomic_write_json(manifest_path, manifest)
        manifest["controller_attempt"] = {
            "phase": "remote_launch",
            "state": "SSH_ACKNOWLEDGED",
            "ambiguous": False,
            "finished_at_unix": time.time(),
        }
        atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    storage_url = storage_url_from_env(args.storage_env)
    with ControllerLocks(output_dir, storage_url, args.study_name) as locks:
        controller_uuid = _controller_uuid(output_dir)
        result = _command_run_locked(args, controller_uuid)
        locks.assert_alive()
        return result


def status_payload(study: Any, output_dir: Path) -> dict[str, Any]:
    states = reserved_trial_states(study)
    workers = []
    worker_root = output_dir / "workers"
    if worker_root.is_dir():
        for run_dir in sorted(worker_root.iterdir()):
            if not run_dir.is_dir():
                continue
            heartbeat = run_dir / "HEARTBEAT.json"
            age = None
            if heartbeat.is_file():
                try:
                    updated = json.loads(heartbeat.read_text(encoding="utf-8"))["updated_at_unix"]
                    age = max(0.0, time.time() - float(updated))
                except Exception:
                    age = None
            workers.append(
                {
                    "worker_run_id": run_dir.name,
                    "worker_marker_age_seconds": age,
                    "worker_marker_age_source": "repository_atomic_marker",
                    "completed": (run_dir / "COMPLETED").exists(),
                    "failed_pretrial": (run_dir / "FAILED_PRETRIAL").exists(),
                }
            )
    return {
        "schema_version": "graphvae-attr-f1pr-status-v1",
        "study_name": study.study_name,
        "lifecycle": study.user_attrs.get(LIFECYCLE_ATTR),
        "objective_json_path": OBJECTIVE_JSON_PATH,
        "test_access": False,
        "reserved_states": states,
        "workers": workers,
        "proposal_order_replayable": (
            int(study.user_attrs[DEFINITION_ATTR]["scheduler"]["max_parallel"]) == 1
        ),
    }


def _command_status_locked(args: argparse.Namespace, controller_uuid: str) -> int:
    _url, _storage_instance, study, _definition, _contract_hash = _load_ready_study(
        args, require_ready=False
    )
    _assert_controller_owner(
        study, args.output_dir.expanduser().resolve(), controller_uuid
    )
    try:
        import optuna

        optuna.storages.fail_stale_trials(study)
    except Exception:
        # Status remains read-only and truthful when a stale sweep is unavailable.
        pass
    payload = status_payload(study, args.output_dir.expanduser().resolve())
    if args.json:
        atomic_write_json(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_status(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    storage_url = storage_url_from_env(args.storage_env)
    with ControllerLocks(output_dir, storage_url, args.study_name) as locks:
        controller_uuid = _controller_uuid(output_dir)
        result = _command_status_locked(args, controller_uuid)
        locks.assert_alive()
        return result


def _tree_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    digest = canonical_contract_hash({"files": files})
    return {"files": files, "sha256": digest}


def command_collect(args: argparse.Namespace) -> int:
    source = args.source_root.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Collection source does not exist: {source}")
    source_manifest = _tree_manifest(source)
    staging = destination / ".collection_staging" / f"collect-{uuid.uuid4()}"
    staging.mkdir(parents=True)
    # Local collection is the testable primitive; the shell transport stages remote bytes here.
    import shutil

    shutil.copytree(source, staging / "payload", dirs_exist_ok=True)
    if _tree_manifest(staging / "payload") != source_manifest:
        raise DistributedContractError("Staged collection hash verification failed.")
    for category in ("trials", "workers", "launch_manifests"):
        source_category = staging / "payload" / category
        if not source_category.is_dir():
            continue
        for source_item in sorted(source_category.iterdir()):
            target = destination / category / source_item.name
            if target.exists():
                same = (
                    _tree_manifest(target) == _tree_manifest(source_item)
                    if source_item.is_dir()
                    else sha256_file(target) == sha256_file(source_item)
                )
                if not same:
                    quarantine = (
                        destination
                        / ".collection_conflicts"
                        / f"{category}-{source_item.name}-{uuid.uuid4()}"
                    )
                    quarantine.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source_item), str(quarantine))
                    raise DistributedContractError(
                        f"Differing collection collision for {source_item.name}; "
                        "staged bytes quarantined."
                    )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(source_item), str(target))
    for filename in (
        "study_definition.json",
        "deployment_manifest.json",
        "dataset_cache_manifest.json",
    ):
        source_file = staging / "payload" / filename
        if not source_file.is_file():
            continue
        target = destination / filename
        if target.exists() and sha256_file(target) != sha256_file(source_file):
            raise DistributedContractError(f"Differing collection collision for {filename}.")
        if not target.exists():
            os.replace(str(source_file), str(target))
    atomic_write_json(destination / "last_collection_manifest.json", source_manifest)
    shutil.rmtree(staging)
    return 0


def command_collect_with_locks(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    storage_url = storage_url_from_env(args.storage_env)
    with ControllerLocks(output_dir, storage_url, args.study_name) as locks:
        controller_uuid = _controller_uuid(output_dir)
        _url, _storage_instance, study, _definition, _contract = _load_ready_study(
            args, require_ready=False
        )
        _assert_controller_owner(study, output_dir, controller_uuid)
        result = command_collect(args)
        locks.assert_alive()
        return result


FINAL_CSV_FIELDS = (
    "trial_number", "budget_index", "reserved", "unreserved_guard", "state",
    "validation_attr_f1pr", "alpha_node_feat", "alpha_edge_feat", "alpha_motif_loss",
    "validation_precision", "validation_recall", "accepted_validation_graphs",
    "worker_run_id", "sampler_seed", "training_seed", "failure_phase", "failure_reason",
    "completion_order", "datetime_complete",
)


def _final_outputs(
    study: Any,
    definition: Mapping[str, Any],
    output_dir: Path,
    *,
    artifact_root: Path | None = None,
) -> Any | None:
    artifact_root = output_dir if artifact_root is None else artifact_root
    audit = reservation_audit(study, int(definition["reserved_trials"]))
    rows = []
    selectable = []
    completed_in_order = sorted(
        (
            trial
            for trial in study.get_trials(deepcopy=False)
            if trial.datetime_complete is not None
        ),
        key=lambda trial: (trial.datetime_complete, trial.number),
    )
    completion_order = {
        trial.number: index for index, trial in enumerate(completed_in_order)
    }
    for trial in study.get_trials(deepcopy=False):
        attrs = trial.user_attrs
        reserved = attrs.get(RESERVED_ATTR) is True
        if reserved:
            result = audit_trial_result(
                trial,
                study_root=artifact_root,
                definition=definition,
            )
            if trial.state.name == "COMPLETE":
                selectable.append((trial, result))
        rows.append(
            {
                "trial_number": trial.number,
                "budget_index": attrs.get(BUDGET_INDEX_ATTR),
                "reserved": reserved,
                "unreserved_guard": attrs.get(UNRESERVED_GUARD_ATTR) is True,
                "state": trial.state.name,
                "validation_attr_f1pr": trial.value,
                "alpha_node_feat": trial.params.get("alpha_node_feat"),
                "alpha_edge_feat": trial.params.get("alpha_edge_feat"),
                "alpha_motif_loss": trial.params.get("alpha_motif_loss"),
                "validation_precision": attrs.get("validation_precision"),
                "validation_recall": attrs.get("validation_recall"),
                "accepted_validation_graphs": attrs.get("accepted_validation_graphs"),
                "worker_run_id": attrs.get("worker_run_id"),
                "sampler_seed": attrs.get("sampler_seed"),
                "training_seed": attrs.get("training_seed"),
                "failure_phase": attrs.get("failure_phase"),
                "failure_reason": attrs.get("failure_reason"),
                "completion_order": completion_order.get(trial.number),
                "datetime_complete": (
                    None
                    if trial.datetime_complete is None
                    else trial.datetime_complete.isoformat()
                ),
            }
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FINAL_CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(output_dir / "trials.csv", buffer.getvalue().encode("utf-8"))
    if not selectable:
        atomic_write_text(
            output_dir / "SUMMARY.md",
            "# Distributed GraphVAE Attr-F1PR Bayesian Optimization\n\n"
            "All reserved scientific trials failed; no best trial exists.\n\n"
            f"Objective path: `{OBJECTIVE_JSON_PATH}`. Test access: `no`.\n",
        )
        return None
    best_trial, best_result = max(selectable, key=lambda item: float(item[0].value))
    grouped_graphcl = definition.get("evaluator", {}).get("backend") == "graphcl_f1pr"
    common_best = {
        "schema_version": "graphvae-attr-f1pr-bo-best-v2",
        "distributed": True,
        "study_name": study.study_name,
        "study_contract_sha256": canonical_contract_hash(definition),
        "objective": OBJECTIVE_NAME,
        "objective_json_path": (
            "summary.f1_pr.mean" if grouped_graphcl else OBJECTIVE_JSON_PATH
        ),
        "compatibility_objective_json_path": OBJECTIVE_JSON_PATH,
        "selection_split": "validation",
        "test_access_during_optimization": False,
        "trial_number": best_trial.number,
        "budget_index": best_trial.user_attrs[BUDGET_INDEX_ATTR],
        "sampled_weights": dict(best_trial.params),
        "validation_attr_f1pr": float(best_trial.value),
        "best_config": "best_config.yaml",
        "generation_seed": best_result["generation_seed"],
    }
    if grouped_graphcl:
        descriptor = {
            "schema_version": "lobster-graphcl-f1pr-selected-candidate-v1",
            "sampled_weights": dict(best_trial.params),
            "training_seeds": best_result["training_seeds"],
            "aggregation": best_result["aggregation"],
            "replicate_resolved_configs": [
                replicate["resolved_config"]
                for replicate in best_result["replicates"]
            ],
            "selection_split": "validation",
            "test_access": False,
        }
        atomic_write_yaml(output_dir / "best_config.yaml", descriptor)
        best_payload = {
            **common_best,
            "schema_version": "lobster-graphcl-f1pr-grouped-best-v1",
            "evaluator_backend": "graphcl_f1pr",
            "aggregation": best_result["aggregation"],
            "training_seeds": best_result["training_seeds"],
            "replicates": best_result["replicates"],
        }
    else:
        config_path = resolve_artifact_path(
            artifact_root, best_result["resolved_config"]
        )
        config = load_yaml_mapping(config_path)
        atomic_write_yaml(output_dir / "best_config.yaml", config)
        best_payload = {
            **common_best,
            "validation_precision": best_result["validation_precision"],
            "validation_recall": best_result["validation_recall"],
            "accepted_validation_graphs": best_result["accepted_validation_graphs"],
            "resolved_config": best_result["resolved_config"],
            "checkpoint": best_result["checkpoint"],
            "checkpoint_sha256": best_result["checkpoint_sha256"],
            "training_seed": best_result["training_seed"],
            "evaluator_seed": best_result["evaluator_seed"],
            "evaluator_repeats": best_result["evaluator_repeats"],
        }
    atomic_write_json(output_dir / "best_trial.json", best_payload)
    guard_count = len(audit["unreserved_trials"])
    atomic_write_text(
        output_dir / "SUMMARY.md",
        "# Distributed GraphVAE Attr-F1PR Bayesian Optimization\n\n"
        f"- Study: `{study.study_name}`\n"
        f"- Best reserved trial: `{best_trial.number}`\n"
        f"- Validation Attr-F1PR: `{float(best_trial.value):.6f}`\n"
        f"- Objective path: `{'summary.f1_pr.mean' if grouped_graphcl else OBJECTIVE_JSON_PATH}`\n"
        f"- Compatibility objective path: `{OBJECTIVE_JSON_PATH}`\n"
        f"- Reserved scientific slots: `{definition['reserved_trials']}`\n"
        f"- Audited unreserved guard rows: `{guard_count}`\n"
        "- Test split evaluated during optimization: `no`\n",
    )
    return best_trial


def _assert_workers_reconciled(study: Any, output_dir: Path) -> None:
    worker_root = output_dir / "workers"
    if worker_root.is_dir():
        for run_dir in worker_root.iterdir():
            if not run_dir.is_dir():
                continue
            marker_paths = [
                run_dir / "COMPLETED",
                run_dir / "FAILED_PRETRIAL",
                run_dir / "RECONCILED_FAIL",
            ]
            present = [path for path in marker_paths if path.exists()]
            if len(present) != 1:
                raise DistributedContractError(
                    f"Worker {run_dir.name} has no unique reconciled terminal marker."
                )
            completed = run_dir / "COMPLETED"
            if completed.exists():
                marker = json.loads(completed.read_text(encoding="utf-8"))
                matches = [
                    trial
                    for trial in study.get_trials(deepcopy=False)
                    if trial.number == marker.get("trial_number")
                ]
                if len(matches) != 1 or matches[0].state.name != marker.get("db_state"):
                    raise DistributedContractError(
                        f"Worker {run_dir.name} marker differs from PostgreSQL."
                    )
    for trial in study.get_trials(deepcopy=False):
        worker_run = trial.user_attrs.get("worker_run_id")
        if worker_run and not any(
            (worker_root / worker_run / marker).is_file()
            for marker in ("COMPLETED", "RECONCILED_FAIL")
        ):
            raise DistributedContractError(
                f"Trial {trial.number} worker marker is missing or uncollected."
            )


def _reconcile_terminal_failures_without_results(
    study: Any,
    output_dir: Path,
    definition: Mapping[str, Any],
) -> None:
    """Publish tombstones for DB-terminal failures without a terminal result."""

    contract_hash = canonical_contract_hash(definition)
    worker_root = output_dir / "workers"
    for trial in study.get_trials(deepcopy=False):
        attrs = trial.user_attrs
        if attrs.get(RESERVED_ATTR) is not True or trial.state.name != "FAIL":
            continue
        trial_result = attrs.get("trial_result") or (
            f"trials/trial_{trial.number:05d}/trial_result.json"
        )
        result_path = resolve_artifact_path(output_dir, trial_result)
        tombstone_path = resolve_artifact_path(
            output_dir,
            attrs.get("failure_tombstone")
            or f"trials/trial_{trial.number:05d}/trial_failure_tombstone.json",
        )
        if tombstone_path.is_file():
            continue
        budget_index = int(attrs[BUDGET_INDEX_ATTR])
        retained_evidence = []
        interrupted_path = result_path.with_name(
            f"{result_path.stem}.interrupted{result_path.suffix}"
        )
        partial_path = result_path if result_path.is_file() else interrupted_path
        if partial_path.is_file():
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            if partial.get("status") == "FAIL" and partial_path == result_path:
                if interrupted_path.exists():
                    raise DistributedContractError(
                        f"Trial {trial.number} has conflicting terminal and interrupted results."
                    )
                continue
            expected_partial = {
                "schema_version": "graphvae-attr-f1pr-bo-trial-v2",
                "status": "RUNNING",
                "trial_number": trial.number,
                "budget_index": budget_index,
                "study_contract_sha256": contract_hash,
                "worker_run_id": attrs.get("worker_run_id"),
                "sampled_weights": dict(trial.params),
            }
            if any(
                partial.get(field) != expected
                for field, expected in expected_partial.items()
            ) or partial.get("finished_at_unix") is not None:
                raise DistributedContractError(
                    f"Trial {trial.number} partial result identity is invalid."
                )
            if partial_path == result_path:
                if interrupted_path.exists():
                    raise DistributedContractError(
                        f"Trial {trial.number} interrupted-result path already exists."
                    )
                os.replace(str(result_path), str(interrupted_path))
                directory_descriptor = os.open(
                    str(interrupted_path.parent), os.O_RDONLY
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            retained_evidence.append(
                {
                    "kind": "interrupted_trial_result",
                    "path": relative_artifact_path(output_dir, interrupted_path),
                    "sha256": sha256_file(interrupted_path),
                    "recorded_status": "RUNNING",
                }
            )
        tombstone_path = write_failure_tombstone(
            study_root=output_dir,
            trial_number=trial.number,
            budget_index=budget_index,
            contract_hash=contract_hash,
            worker_id=attrs.get("worker_id"),
            worker_run_id_value=attrs.get("worker_run_id"),
            failure_category=(
                "postgresql_stale_worker_with_interrupted_result"
                if retained_evidence
                else "postgresql_terminal_fail_without_result_artifact"
            ),
            missing_artifacts=(
                {
                    "kind": "trial_result",
                    "path": trial_result,
                    "verified_absent": True,
                },
            ),
            retained_evidence=retained_evidence,
        )
        worker_run = attrs.get("worker_run_id")
        if not worker_run:
            continue
        run_dir = worker_root / worker_run
        run_dir.mkdir(parents=True, exist_ok=True)
        terminal_markers = [
            run_dir / marker
            for marker in ("COMPLETED", "FAILED_PRETRIAL", "RECONCILED_FAIL")
            if (run_dir / marker).is_file()
        ]
        if terminal_markers:
            continue
        atomic_write_json(
            run_dir / "RECONCILED_FAIL",
            {
                "schema_version": "graphvae-attr-f1pr-reconciled-fail-v1",
                "trial_number": trial.number,
                "budget_index": budget_index,
                "db_state": "FAIL",
                "failure_tombstone": relative_artifact_path(
                    output_dir, tombstone_path
                ),
                "reconciled_at_unix": time.time(),
            },
        )


def command_finalize(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    storage_url = storage_url_from_env(args.storage_env)
    with ControllerLocks(output_dir, storage_url, args.study_name) as locks:
        controller_uuid = _controller_uuid(output_dir)
        storage = _storage(args, storage_url)
        local_definition = json.loads(
            (output_dir / "study_definition.json").read_text(encoding="utf-8")
        )
        contract_hash = canonical_contract_hash(local_definition)
        study = create_or_load_distributed_study(
            storage,
            study_name=args.study_name,
            sampler_seed_value=int(local_definition["seeds"]["study_seed"]),
            create=False,
        )
        validate_study_contract(
            study,
            expected_contract_hash=contract_hash,
            local_definition=local_definition,
            require_ready=False,
        )
        _assert_controller_owner(study, output_dir, controller_uuid)
        if study.user_attrs.get(LIFECYCLE_ATTR) not in {LIFECYCLE_READY, LIFECYCLE_FROZEN}:
            raise DistributedContractError("Finalization requires READY or already FROZEN lifecycle.")
        assert_quiescent_reserved_study(study)
        _reconcile_terminal_failures_without_results(
            study, output_dir, local_definition
        )
        _assert_workers_reconciled(study, output_dir)
        best = _final_outputs(study, local_definition, output_dir)
        locks.assert_alive()
        study.set_user_attr(LIFECYCLE_ATTR, LIFECYCLE_FROZEN)
        frozen_study = create_or_load_distributed_study(
            storage,
            study_name=args.study_name,
            sampler_seed_value=int(local_definition["seeds"]["study_seed"]),
            create=False,
        )
        assert_quiescent_reserved_study(frozen_study)
        _assert_workers_reconciled(frozen_study, output_dir)
        snapshot = create_portable_snapshot(
            frozen_study,
            source_storage=storage,
            snapshot_path=output_dir / "study_snapshot.sqlite3",
        )
        assert_quiescent_reserved_study(frozen_study)
        locks.assert_alive()
        atomic_write_json(
            output_dir / "FROZEN.json",
            {
                "schema_version": "graphvae-attr-f1pr-frozen-v1",
                "study_name": args.study_name,
                "study_contract_sha256": contract_hash,
                "lifecycle": LIFECYCLE_FROZEN,
                "snapshot": snapshot.name,
                "best_trial_number": None if best is None else best.number,
                "frozen_at_unix": time.time(),
            },
        )
    return 0 if best is not None else 2


def build_hardware_repeatability_report(output_dir: Path) -> dict[str, Any]:
    """Audit a frozen, fixed-parameter qualification across distinct GPU slots."""

    root = output_dir.expanduser().resolve()
    definition = json.loads((root / "study_definition.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "FROZEN.json").read_text(encoding="utf-8"))
    contract_hash = canonical_contract_hash(definition)
    if (
        frozen.get("lifecycle") != LIFECYCLE_FROZEN
        or frozen.get("study_name") != definition.get("study_name")
        or frozen.get("study_contract_sha256") != contract_hash
    ):
        raise DistributedContractError("Hardware audit requires the matching frozen contract.")
    fixed = definition.get("search_space", {}).get("fixed_parameters")
    if not isinstance(fixed, Mapping) or not {
        "alpha_node_feat",
        "alpha_edge_feat",
    }.issubset(fixed):
        raise DistributedContractError(
            "Hardware audit requires contracted fixed attribute-loss parameters."
        )
    expected_trials = int(definition["reserved_trials"])
    result_paths = sorted(root.glob("trials/trial_*/trial_result.json"))
    if len(result_paths) != expected_trials:
        raise DistributedContractError(
            "Hardware audit requires one COMPLETE result for every reservation."
        )

    records = []
    trial_numbers = set()
    budget_indexes = set()
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        trial_number = int(result["trial_number"])
        budget_index = int(result["budget_index"])
        trial_numbers.add(trial_number)
        budget_indexes.add(budget_index)
        if (
            result.get("status") != "COMPLETE"
            or result.get("study_contract_sha256") != contract_hash
            or result.get("objective_json_path") != OBJECTIVE_JSON_PATH
            or result.get("sampled_weights") != dict(fixed)
        ):
            raise DistributedContractError(
                f"Trial {trial_number} does not match the fixed hardware contract."
            )
        value = float(result["validation_attr_f1pr"])
        if not math.isfinite(value):
            raise DistributedContractError("Hardware audit objective must be finite.")
        expected_hashes = {
            "cache_sha256": definition["dataset_cache"].get("sha256"),
            "split_fingerprint": definition["dataset_cache"].get("split_fingerprint"),
            "node_schema_fingerprint": definition["feature_schemas"].get("node_sha256"),
            "edge_schema_fingerprint": definition["feature_schemas"].get("edge_sha256"),
            "source_tree_sha256": definition["source"].get("tree_sha256"),
            "environment_sha256": definition["environment"].get("sha256"),
        }
        for name, expected in expected_hashes.items():
            if expected is not None and result.get("hashes", {}).get(name) != expected:
                raise DistributedContractError(
                    f"Trial {trial_number} hardware audit hash mismatch for {name}."
                )
        checkpoint = resolve_artifact_path(root, result["checkpoint"])
        if sha256_file(checkpoint) != result.get("checkpoint_sha256"):
            raise DistributedContractError("Hardware audit checkpoint hash mismatch.")
        evaluator_path = resolve_artifact_path(root, result["evaluator_output"])
        evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
        if (
            evaluator.get("split") != "validation"
            or evaluator.get("primary_mode") != "decoded_node_edge"
            or evaluator.get("feature_source", {}).get("generated")
            != "GraphVAE node_feature_decoder and edge_feature_decoder"
        ):
            raise DistributedContractError("Hardware audit evaluator contract mismatch.")
        hostname = str(result.get("hostname") or "")
        gpu_model = str(result.get("gpu_model") or "")
        physical_gpu = result.get("physical_gpu")
        gpu_vram_bytes = result.get("gpu_vram_bytes")
        if (
            not hostname
            or not gpu_model
            or not isinstance(physical_gpu, int)
            or not isinstance(gpu_vram_bytes, int)
            or gpu_vram_bytes <= 0
        ):
            raise DistributedContractError("Hardware audit GPU identity is incomplete.")
        records.append(
            {
                "trial_number": trial_number,
                "budget_index": budget_index,
                "slot": f"{hostname}:gpu{physical_gpu}",
                "hostname": hostname,
                "physical_gpu": physical_gpu,
                "gpu_model": gpu_model,
                "gpu_vram_bytes": gpu_vram_bytes,
                "validation_attr_f1pr": value,
                "checkpoint_sha256": result["checkpoint_sha256"],
                "final_training_loss": result.get("final_training_loss"),
            }
        )
    if len(trial_numbers) != expected_trials or budget_indexes != set(range(expected_trials)):
        raise DistributedContractError("Hardware audit trial identities are not exact.")
    if len({record["slot"] for record in records}) < 2:
        raise DistributedContractError("Hardware audit requires at least two distinct GPU slots.")

    objective_tolerance = float(
        definition["hardware_policy"]["attr_f1pr_abs_tolerance"]
    )
    objective_pairs = []
    objective_passed = True
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            difference = abs(
                left["validation_attr_f1pr"] - right["validation_attr_f1pr"]
            )
            passed = difference <= objective_tolerance
            objective_passed = objective_passed and passed
            objective_pairs.append(
                {
                    "left_slot": left["slot"],
                    "right_slot": right["slot"],
                    "absolute_difference": difference,
                    "tolerance": objective_tolerance,
                    "passed": passed,
                }
            )

    losses = [record["final_training_loss"] for record in records]
    if all(value is None for value in losses):
        training_loss = {"status": "not_recorded", "pairs": [], "passed": True}
    elif any(value is None for value in losses):
        raise DistributedContractError(
            "Hardware audit final training loss is present for only some trials."
        )
    else:
        loss_policy = definition["hardware_policy"]["training_loss_tolerance"]
        floor = float(loss_policy["absolute_floor"])
        relative = float(loss_policy["relative"])
        loss_pairs = []
        loss_passed = True
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                left_loss = float(left["final_training_loss"])
                right_loss = float(right["final_training_loss"])
                tolerance = max(floor, relative * max(abs(left_loss), abs(right_loss)))
                difference = abs(left_loss - right_loss)
                passed = difference <= tolerance
                loss_passed = loss_passed and passed
                loss_pairs.append(
                    {
                        "left_slot": left["slot"],
                        "right_slot": right["slot"],
                        "absolute_difference": difference,
                        "tolerance": tolerance,
                        "passed": passed,
                    }
                )
        training_loss = {"status": "compared", "pairs": loss_pairs, "passed": loss_passed}

    passed = objective_passed and bool(training_loss["passed"])
    return {
        "schema_version": "graphvae-attr-f1pr-hardware-repeatability-v1",
        "study_name": definition["study_name"],
        "study_contract_sha256": contract_hash,
        "lifecycle": LIFECYCLE_FROZEN,
        "objective_json_path": OBJECTIVE_JSON_PATH,
        "split": "validation",
        "test_access": False,
        "fixed_parameters": dict(fixed),
        "checkpoint_byte_equality_expected": False,
        "records": records,
        "objective_comparison": {
            "absolute_tolerance": objective_tolerance,
            "pairs": objective_pairs,
            "passed": objective_passed,
        },
        "training_loss_comparison": training_loss,
        "passed": passed,
        "eligible_slots": sorted({record["slot"] for record in records}) if passed else [],
    }


def command_hardware_audit(args: argparse.Namespace) -> int:
    report = build_hardware_repeatability_report(args.output_dir)
    if report["study_name"] != args.study_name:
        raise DistributedContractError("Hardware audit study name mismatch.")
    output = args.output_dir.expanduser().resolve() / "hardware_repeatability.json"
    atomic_write_json(output, report)
    print(
        f"Hardware repeatability {'passed' if report['passed'] else 'failed'} for "
        f"{len(report['records'])} fixed-parameter trials."
    )
    return 0 if report["passed"] else 2


def restore_frozen_study(
    source_output_dir: Path,
    restore_output_dir: Path,
    *,
    study_name: str,
) -> dict[str, Any]:
    """Restore aggregate outputs without PostgreSQL access or source mutation."""

    import shutil

    source = source_output_dir.expanduser().resolve()
    destination = restore_output_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Frozen source output directory not found: {source}")
    if destination == source:
        raise DistributedContractError("Restore output must differ from the frozen source.")
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise DistributedContractError("Restore output may not be nested in the frozen source.")
    if destination.exists():
        raise DistributedContractError("Restore output must be a fresh absent path.")

    definition_path = source / "study_definition.json"
    frozen_path = source / "FROZEN.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    contract_hash = canonical_contract_hash(definition)
    if (
        definition.get("study_name") != study_name
        or frozen.get("study_name") != study_name
        or frozen.get("lifecycle") != LIFECYCLE_FROZEN
        or frozen.get("study_contract_sha256") != contract_hash
    ):
        raise DistributedContractError("Restore source is not the matching frozen study.")
    environment = runtime_dependency_fingerprint()
    if environment.get("sha256") != definition.get("environment", {}).get("sha256"):
        raise DistributedContractError(
            "Restore runtime fingerprint differs from the frozen contract."
        )
    snapshot_path = source / str(frozen.get("snapshot"))
    if not snapshot_path.is_file():
        raise FileNotFoundError("Frozen portable SQLite snapshot is missing.")
    source_snapshot_sha = sha256_file(snapshot_path)

    aggregate_names = ("trials.csv", "best_trial.json", "best_config.yaml", "SUMMARY.md")
    for name in aggregate_names:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Frozen aggregate output is missing: {name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4()}"
    staging.mkdir()
    try:
        shutil.copy2(definition_path, staging / definition_path.name)
        shutil.copy2(frozen_path, staging / frozen_path.name)
        restored_snapshot_path = staging / snapshot_path.name
        shutil.copy2(snapshot_path, restored_snapshot_path)
        import optuna

        restored = optuna.load_study(
            study_name=study_name,
            storage="sqlite:///" + restored_snapshot_path.as_posix(),
        )
        validate_study_contract(
            restored,
            expected_contract_hash=contract_hash,
            local_definition=definition,
            require_ready=False,
        )
        if restored.user_attrs.get(LIFECYCLE_ATTR) != LIFECYCLE_FROZEN:
            raise DistributedContractError("Restored snapshot lifecycle is not FROZEN.")
        assert_quiescent_reserved_study(restored)
        best = _final_outputs(
            restored,
            definition,
            staging,
            artifact_root=source,
        )
        if best is None:
            raise DistributedContractError("R09 restoration requires a selected best trial.")
        aggregate_hashes = {}
        for name in aggregate_names:
            original_sha = sha256_file(source / name)
            restored_sha = sha256_file(staging / name)
            if restored_sha != original_sha:
                raise DistributedContractError(
                    f"Restored aggregate differs from frozen source: {name}"
                )
            aggregate_hashes[name] = restored_sha
        if sha256_file(restored_snapshot_path) != source_snapshot_sha:
            raise DistributedContractError("Reopened snapshot bytes changed during restore.")
        if sha256_file(snapshot_path) != source_snapshot_sha:
            raise DistributedContractError("Frozen source snapshot changed during restore.")
        best_payload = json.loads((staging / "best_trial.json").read_text(encoding="utf-8"))
        report = {
            "schema_version": "graphvae-attr-f1pr-restored-v1",
            "study_name": study_name,
            "study_contract_sha256": contract_hash,
            "source_lifecycle": LIFECYCLE_FROZEN,
            "restored_snapshot": snapshot_path.name,
            "snapshot_sha256": source_snapshot_sha,
            "semantic_fingerprint": trial_semantic_fingerprint(restored),
            "aggregate_sha256": aggregate_hashes,
            "aggregate_outputs_match": True,
            "best_trial_number": best.number,
            "best_validation_attr_f1pr": best_payload["validation_attr_f1pr"],
            "objective_json_path": OBJECTIVE_JSON_PATH,
            "selection_split": "validation",
            "runtime_fingerprint": environment["sha256"],
            "postgresql_access": False,
            "test_access": False,
        }
        atomic_write_json(staging / "RESTORED.json", report)
        directory_fd = os.open(str(staging), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(str(staging), str(destination))
        parent_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def command_restore(args: argparse.Namespace) -> int:
    report = restore_frozen_study(
        args.source_output_dir,
        args.restore_output_dir,
        study_name=args.study_name,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_identifier(args.study_name, "study name")
    return {
        "init": command_init,
        "preflight": command_preflight,
        "run": command_run,
        "probe": command_probe,
        "retire-preclaim": command_retire_preclaim,
        "status": command_status,
        "collect": command_collect_with_locks,
        "finalize": command_finalize,
        "hardware-audit": command_hardware_audit,
        "restore": command_restore,
    }[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        message = f"ERROR: {type(exc).__name__}: {exc}"
        for environment_name in (
            "GRAPHVAE_BO_STORAGE_URL",
            "GRAPHVAE_BO_TEST_STORAGE_URL",
        ):
            value = os.environ.get(environment_name)
            if value:
                message = redact_secret(message, storage_url=value)
        print(message, file=sys.stderr)
        raise SystemExit(2)
