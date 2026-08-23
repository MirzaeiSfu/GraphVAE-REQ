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
    init.add_argument("--tune-alpha-motif", action="store_true")
    init.add_argument("--alpha-motif-min", type=float, default=1e-3)
    init.add_argument("--alpha-motif-max", type=float, default=1e2)
    init.add_argument("--split-seed", type=int, default=None)
    init.add_argument("--training-seed", type=int, default=0)
    init.add_argument("--generation-seed", type=int, default=123)
    init.add_argument("--evaluator-seed", type=int, default=0)
    init.add_argument("--evaluator-repeats", type=int, default=5)
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
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--execute-remote",
        action="store_true",
        help="Required safety acknowledgement before SSH/tmux dispatch.",
    )

    for name in ("status", "collect", "finalize"):
        command = subparsers.add_parser(name)
        _add_storage_options(command)
        if name == "status":
            command.add_argument("--json", type=Path, default=None)
        if name == "collect":
            command.add_argument("--source-root", type=Path, required=True)

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

    return {
        "alpha_node_feat": entry(ranges.alpha_node_feat),
        "alpha_edge_feat": entry(ranges.alpha_edge_feat),
        "alpha_motif_loss": entry(ranges.alpha_motif_loss),
        "motif_opt_in": bool(args.tune_alpha_motif),
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
    definition = build_study_definition(
        study_name=args.study_name,
        study_uuid=(None if existing_definition is None else existing_definition["study_uuid"]),
        base_config=base_config,
        base_config_sha256=sha256_file(config_path),
        ranges=_search_space(args),
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
        },
        evaluator={
            "mode": "decoded_node_edge",
            "split": "validation",
            "test_access": False,
            "repeat_count": args.evaluator_repeats,
            "max_graphs": max_graphs,
            "generation_batch_size": generation_batch_size,
            "nearest_k": args.nearest_k,
            "adjacency_threshold": args.adjacency_threshold,
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
    physical_gpu: int,
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
        "--physical-gpu", str(physical_gpu),
        "--device", "cuda:0",
        "--storage-env", storage_env,
        "--heartbeat-interval", str(heartbeat_interval),
        "--grace-period", str(grace_period),
        "--tpe-startup-trials", str(tpe_startup_trials),
    ]
    if mock:
        command.append("--mock")
    if allow_sslmode_require_exception:
        command.append("--allow-sslmode-require-infrastructure-exception")
    return command


def render_remote_launch(command: Sequence[str], *, physical_gpu: int, lock_path: str) -> str:
    # The storage URL remains solely in the named inherited environment variable.
    inner = (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(str(physical_gpu))} "
        f"flock -n {shlex.quote(lock_path)} "
        + shlex.join(list(command))
    )
    return inner


def _preflight_inputs(args: argparse.Namespace):
    repositories = _load_mapping(args.repo_paths)
    pythons = _load_mapping(args.python_paths)
    if set(repositories) != set(pythons):
        raise ValueError("Repository and Python host mappings differ.")
    slots = parse_slots(args.slots, known_hosts=sorted(repositories))
    return repositories, pythons, slots


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


def _command_preflight_locked(args: argparse.Namespace, controller_uuid: str) -> int:
    repositories, pythons, slots = _preflight_inputs(args)
    _storage_url, _storage_instance, study, definition, contract_hash = _load_ready_study(args)
    _assert_controller_owner(
        study, args.output_dir.expanduser().resolve(), controller_uuid
    )
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


def _command_run_locked(args: argparse.Namespace, controller_uuid: str) -> int:
    repositories, pythons, slots = _preflight_inputs(args)
    storage_url, _storage_instance, study, definition, contract_hash = _load_ready_study(args)
    _assert_controller_owner(
        study, args.output_dir.expanduser().resolve(), controller_uuid
    )
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
    startup_target = int(definition["sampler"]["n_startup_trials"])
    startup_remaining = max(0, startup_target - usable_observations)
    wave_limit = (
        min(contracted_parallelism, startup_remaining)
        if startup_remaining
        else contracted_parallelism
    )
    wave_slots = slots[: min(wave_limit, len(waiting), len(slots))]
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
        )
        tmux_name = f"graphvae-bo-{run_id}"
        launch = {
            **slot,
            "dispatch_sequence": dispatch_sequence,
            "sampler_seed": seed,
            "worker_run_id": run_id,
            "tmux_session": tmux_name,
            "remote_command": remote_shell,
        }
        launches.append(launch)
        if not args.dry_run:
            subprocess.run(
                ["ssh", "-n", slot["host"], "tmux", "new-session", "-d", "-s", tmux_name, remote_shell],
                check=True,
            )
    manifest = {
        "schema_version": "graphvae-attr-f1pr-launch-wave-v1",
        "wave_index": wave_index,
        "bounded_synchronous": True,
        "study_contract_sha256": contract_hash,
        "launches": launches,
        "dry_run": bool(args.dry_run),
        "usable_complete_observations_before_wave": usable_observations,
        "startup_target": startup_target,
    }
    atomic_write_json(
        args.output_dir / "launch_manifests" / f"wave_{wave_index:04d}.json", manifest
    )
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
    "worker_run_id", "sampler_seed", "failure_phase", "failure_reason",
    "completion_order", "datetime_complete",
)


def _final_outputs(study: Any, definition: Mapping[str, Any], output_dir: Path) -> Any | None:
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
            result = audit_trial_result(trial, study_root=output_dir, definition=definition)
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
    config_path = resolve_artifact_path(output_dir, best_result["resolved_config"])
    config = load_yaml_mapping(config_path)
    atomic_write_yaml(output_dir / "best_config.yaml", config)
    best_payload = {
        "schema_version": "graphvae-attr-f1pr-bo-best-v2",
        "distributed": True,
        "study_name": study.study_name,
        "study_contract_sha256": canonical_contract_hash(definition),
        "objective": OBJECTIVE_NAME,
        "objective_json_path": OBJECTIVE_JSON_PATH,
        "selection_split": "validation",
        "test_access_during_optimization": False,
        "trial_number": best_trial.number,
        "budget_index": best_trial.user_attrs[BUDGET_INDEX_ATTR],
        "sampled_weights": dict(best_trial.params),
        "validation_attr_f1pr": float(best_trial.value),
        "validation_precision": best_result["validation_precision"],
        "validation_recall": best_result["validation_recall"],
        "accepted_validation_graphs": best_result["accepted_validation_graphs"],
        "resolved_config": best_result["resolved_config"],
        "best_config": "best_config.yaml",
        "checkpoint": best_result["checkpoint"],
        "checkpoint_sha256": best_result["checkpoint_sha256"],
        "training_seed": best_result["training_seed"],
        "generation_seed": best_result["generation_seed"],
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
        f"- Objective path: `{OBJECTIVE_JSON_PATH}`\n"
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
    """Publish tombstones only for DB-terminal reserved failures with no result."""

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
        if result_path.is_file() or tombstone_path.is_file():
            continue
        budget_index = int(attrs[BUDGET_INDEX_ATTR])
        tombstone_path = write_failure_tombstone(
            study_root=output_dir,
            trial_number=trial.number,
            budget_index=budget_index,
            contract_hash=contract_hash,
            worker_id=attrs.get("worker_id"),
            worker_run_id_value=attrs.get("worker_run_id"),
            failure_category="postgresql_terminal_fail_without_result_artifact",
            missing_artifacts=(
                {
                    "kind": "trial_result",
                    "path": trial_result,
                    "verified_absent": True,
                },
            ),
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_identifier(args.study_name, "study name")
    return {
        "init": command_init,
        "preflight": command_preflight,
        "run": command_run,
        "status": command_status,
        "collect": command_collect_with_locks,
        "finalize": command_finalize,
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
