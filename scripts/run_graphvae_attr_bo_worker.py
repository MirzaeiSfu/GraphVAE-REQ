#!/usr/bin/env python3
"""Consume exactly one reserved PostgreSQL GraphVAE Attr-F1PR trial."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from graphvae_attr_bo_distributed import (  # noqa: E402
    BUDGET_INDEX_ATTR,
    CONTRACT_ATTR,
    DEFAULT_STARTUP_TRIALS,
    LIFECYCLE_READY,
    RESERVED_ATTR,
    TRIAL_CONTRACT_ATTR,
    UNRESERVED_GUARD_ATTR,
    DistributedContractError,
    atomic_write_json,
    atomic_write_text,
    canonical_contract_hash,
    create_or_load_distributed_study,
    create_postgresql_storage,
    enforce_pinned_versions,
    guard_reserved_trial,
    runtime_dependency_fingerprint,
    sampler_seed,
    redact_secret,
    sha256_file,
    storage_url_from_env,
    validate_identifier,
    validate_study_contract,
    verify_deployment_manifest,
    worker_run_info,
)
from tune_graphvae_attribute_weights import (  # noqa: E402
    DEFAULT_EVALUATOR_REPEATS,
    SearchRanges,
    execute_trial,
    flatten_config,
    load_yaml_mapping,
    validate_base_config,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--study-contract-sha256", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--worker-run-id", required=True)
    parser.add_argument("--sampler-seed", type=int, required=True)
    parser.add_argument("--dispatch-sequence", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, default=None)
    parser.add_argument("--storage-env", default="GRAPHVAE_BO_STORAGE_URL")
    parser.add_argument("--heartbeat-interval", type=int, default=60)
    parser.add_argument("--grace-period", type=int, default=600)
    parser.add_argument("--connect-timeout", type=int, default=15)
    parser.add_argument("--tpe-startup-trials", type=int, default=DEFAULT_STARTUP_TRIALS)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--allow-insecure-local-postgres",
        action="store_true",
        help="Test-only: allow a localhost PostgreSQL URL without verify-full TLS.",
    )
    parser.add_argument(
        "--allow-sslmode-require-infrastructure-exception",
        action="store_true",
        help="Use only when the immutable contract records this TLS exception.",
    )
    parser.add_argument(
        "--mock-fail-trial", action="append", type=int, default=[], help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def _failure_marker(
    run_dir: Path,
    phase: str,
    exc: BaseException,
    *,
    storage_url: str | None = None,
) -> None:
    atomic_write_json(
        run_dir / "FAILED_PRETRIAL",
        {
            "schema_version": "graphvae-attr-f1pr-worker-pretrial-failure-v1",
            "phase": phase,
            "exception_type": type(exc).__name__,
            "message": redact_secret(str(exc), storage_url=storage_url),
            "traceback": redact_secret(traceback.format_exc(), storage_url=storage_url),
            "time_unix": time.time(),
            "reservation_consumed": False,
        },
    )


def _load_local_definition(artifact_root: Path, expected_hash: str) -> dict[str, Any]:
    path = artifact_root / "study_definition.json"
    if not path.is_file():
        raise FileNotFoundError(f"Study definition not found: {path}")
    definition = json.loads(path.read_text(encoding="utf-8"))
    if canonical_contract_hash(definition) != expected_hash:
        raise DistributedContractError("Local study definition hash mismatch.")
    objective = definition.get("objective") or {}
    if (
        objective.get("json_path")
        != "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
        or objective.get("primary_mode") != "decoded_node_edge"
        or objective.get("split") != "validation"
        or objective.get("test_access") is not False
    ):
        raise DistributedContractError("Local study definition changes the Attr-F1PR objective.")
    return definition


def local_preflight(
    args: argparse.Namespace, definition: Mapping[str, Any]
) -> tuple[dict[str, Any], Path | None]:
    base_config_path = args.base_config.expanduser().resolve()
    if sha256_file(base_config_path) != definition["base_config_sha256"]:
        raise DistributedContractError("Base configuration SHA-256 mismatch.")
    base_config = load_yaml_mapping(base_config_path)
    if base_config != definition["resolved_fixed_configuration"]:
        raise DistributedContractError("Resolved base configuration mismatch.")
    if not args.mock:
        validate_base_config(
            base_config,
            tune_alpha_motif=definition["search_space"].get("alpha_motif_loss") is not None,
        )

    dependency = runtime_dependency_fingerprint()
    expected_dependency = definition.get("environment", {}).get("sha256")
    if expected_dependency and dependency["sha256"] != expected_dependency:
        raise DistributedContractError("Runtime dependency fingerprint mismatch.")
    expected_source = definition.get("source", {})
    if expected_source.get("tree_sha256"):
        verify_deployment_manifest(REPO_ROOT, expected_source)

    cache_definition = definition.get("dataset_cache") or {}
    cache_path = None
    relative_cache = cache_definition.get("relative_path")
    expected_cache_hash = cache_definition.get("sha256")
    if relative_cache:
        cache_path = (REPO_ROOT / relative_cache).resolve()
        try:
            cache_path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise DistributedContractError("Dataset cache path escapes the repository.") from exc
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Required distributed dataset cache is missing: {cache_path}"
            )
        if expected_cache_hash and sha256_file(cache_path) != expected_cache_hash:
            raise DistributedContractError("Dataset cache SHA-256 mismatch.")
    elif not args.mock:
        raise DistributedContractError("Distributed study definition has no cache path.")
    return base_config, cache_path


def _ranges(definition: Mapping[str, Any]) -> SearchRanges:
    def bounds(name: str):
        value = definition["search_space"].get(name)
        if value is None:
            return None
        if isinstance(value, Mapping):
            if value.get("log") is not True:
                raise DistributedContractError(f"{name} must use log-scale sampling.")
            return (float(value["low"]), float(value["high"]))
        return (float(value[0]), float(value[1]))

    return SearchRanges(
        alpha_node_feat=bounds("alpha_node_feat"),
        alpha_edge_feat=bounds("alpha_edge_feat"),
        alpha_motif_loss=bounds("alpha_motif_loss"),
    )


def _execution_args(
    cli: argparse.Namespace,
    definition: Mapping[str, Any],
    *,
    budget_index: int,
) -> argparse.Namespace:
    seeds = definition["seeds"]
    evaluator = definition["evaluator"]
    training = definition["training"]
    cache = definition.get("dataset_cache") or {}
    schemas = definition.get("feature_schemas") or {}
    import optuna
    import psycopg2
    expected_count = cache.get("expected_validation_graphs")
    return argparse.Namespace(
        distributed=True,
        study_contract_sha256=cli.study_contract_sha256,
        budget_index=budget_index,
        worker_id=cli.worker_id,
        worker_run_id=cli.worker_run_id,
        hostname=socket.gethostname(),
        physical_gpu=cli.physical_gpu,
        gpu_model=None,
        gpu_vram_bytes=None,
        dispatch_sequence=cli.dispatch_sequence,
        sampler_constant_liar=True,
        sampler_seed=cli.sampler_seed,
        optuna_version=optuna.__version__,
        db_driver_version=str(psycopg2.__version__).split()[0],
        tpe_startup_trials=cli.tpe_startup_trials,
        training_seed=int(seeds["training_seed"]),
        generation_seed=int(seeds["generation_seed"]),
        evaluator_seed=int(seeds["evaluator_seed"]),
        evaluator_repeats=int(evaluator.get("repeat_count", DEFAULT_EVALUATOR_REPEATS)),
        max_graphs=int(evaluator.get("max_graphs", 0)),
        generation_batch_size=int(evaluator.get("generation_batch_size", 16)),
        nearest_k=int(evaluator.get("nearest_k", 5)),
        adjacency_threshold=float(evaluator.get("adjacency_threshold", 0.5)),
        device=cli.device,
        python_bin=cli.python_bin,
        training_timeout=training.get("training_timeout_seconds"),
        evaluation_timeout=training.get("evaluation_timeout_seconds"),
        process_termination_grace=training.get("termination_grace_seconds", 10.0),
        mock=cli.mock,
        mock_fail_trial=list(cli.mock_fail_trial),
        expected_validation_graph_count=(
            None if expected_count is None else int(expected_count)
        ),
        expected_node_feature_dimension=cache.get("node_feature_dimension"),
        expected_edge_feature_dimension=cache.get("edge_feature_dimension"),
        integrity={
            "cache_sha256": cache.get("sha256"),
            "split_fingerprint": cache.get("split_fingerprint"),
            "node_schema_fingerprint": schemas.get("node_sha256"),
            "edge_schema_fingerprint": schemas.get("edge_sha256"),
            "source_tree_sha256": definition.get("source", {}).get("tree_sha256"),
            "environment_sha256": definition.get("environment", {}).get("sha256"),
        },
        secret_environment_names=(
            cli.storage_env,
            "GRAPHVAE_BO_STORAGE_URL",
            "GRAPHVAE_BO_TEST_STORAGE_URL",
            "PGPASSWORD",
            "PGPASSFILE",
        ),
    )


class HeartbeatMarker:
    def __init__(self, path: Path, run_id: str, interval: float = 30.0):
        self.path = path
        self.run_id = run_id
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _write(self):
        atomic_write_json(
            self.path,
            {
                "schema_version": "graphvae-attr-f1pr-worker-heartbeat-v1",
                "worker_run_id": self.run_id,
                "hostname": socket.gethostname(),
                "updated_at_unix": time.time(),
            },
        )

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self._write()

    def __enter__(self):
        self._write()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback_value):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self._write()
        return False


def run_worker(args: argparse.Namespace) -> int:
    validate_identifier(args.study_name, "study name")
    validate_identifier(args.worker_id, "worker ID")
    validate_identifier(args.worker_run_id, "worker run ID")
    artifact_root = args.artifact_root.expanduser().resolve()
    run_dir = artifact_root / "workers" / args.worker_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    info = worker_run_info(
        worker_id=args.worker_id,
        worker_run_id_value=args.worker_run_id,
        sampler_seed_value=args.sampler_seed,
        device=args.device,
    )
    info.update(
        {
            "study_name": args.study_name,
            "study_contract_sha256": args.study_contract_sha256,
            "dispatch_sequence": args.dispatch_sequence,
            "physical_gpu": args.physical_gpu,
        }
    )
    atomic_write_json(run_dir / "RUN_INFO.json", info)

    phase = "local_preflight"
    storage_url = None
    try:
        enforce_pinned_versions()
        definition = _load_local_definition(artifact_root, args.study_contract_sha256)
        expected_tls_policy = definition.get("storage", {}).get(
            "tls_policy", "verify-full"
        )
        if args.allow_sslmode_require_infrastructure_exception != (
            expected_tls_policy == "require-documented-infrastructure-exception"
        ):
            raise DistributedContractError("Worker PostgreSQL TLS policy differs from contract.")
        if args.allow_insecure_local_postgres != (
            expected_tls_policy == "localhost-test-exception"
        ):
            raise DistributedContractError("Worker localhost test TLS policy differs from contract.")
        expected_sampler_seed = sampler_seed(
            int(definition["sampler"]["study_seed"]), args.dispatch_sequence
        )
        if args.sampler_seed != expected_sampler_seed:
            raise DistributedContractError(
                "Worker sampler seed does not match the immutable dispatch sequence."
            )
        if args.tpe_startup_trials != int(definition["sampler"]["n_startup_trials"]):
            raise DistributedContractError("Worker TPE startup setting differs from contract.")
        base_config, _cache_path = local_preflight(args, definition)
        storage_url = storage_url_from_env(args.storage_env)
        phase = "postgresql_preflight"
        storage = create_postgresql_storage(
            storage_url,
            heartbeat_interval=args.heartbeat_interval,
            grace_period=args.grace_period,
            connect_timeout=args.connect_timeout,
            allow_insecure_local_test=args.allow_insecure_local_postgres,
            allow_sslmode_require_exception=(
                args.allow_sslmode_require_infrastructure_exception
            ),
        )
        study = create_or_load_distributed_study(
            storage,
            study_name=args.study_name,
            sampler_seed_value=args.sampler_seed,
            startup_trials=args.tpe_startup_trials,
            create=False,
        )
        validate_study_contract(
            study,
            expected_contract_hash=args.study_contract_sha256,
            local_definition=definition,
        )
    except Exception as exc:
        _failure_marker(run_dir, phase, exc, storage_url=storage_url)
        return 2

    claimed: list[tuple[int, int]] = []

    def objective(trial):
        trial.set_user_attr("worker_id", args.worker_id)
        trial.set_user_attr("worker_run_id", args.worker_run_id)
        trial.set_user_attr("sampler_seed", int(args.sampler_seed))
        trial.set_user_attr("dispatch_sequence", int(args.dispatch_sequence))
        trial.set_user_attr("sampler_constant_liar", True)
        trial.set_user_attr("tpe_startup_trials", int(args.tpe_startup_trials))
        import optuna
        import psycopg2

        trial.set_user_attr("optuna_version", optuna.__version__)
        trial.set_user_attr(
            "db_driver_version", str(psycopg2.__version__).split()[0]
        )
        budget_index = guard_reserved_trial(
            trial, study, expected_contract_hash=args.study_contract_sha256
        )
        claimed.append((trial.number, budget_index))
        execution_args = _execution_args(args, definition, budget_index=budget_index)
        return execute_trial(
            trial,
            args=execution_args,
            base_config=base_config,
            ranges=_ranges(definition),
            output_dir=artifact_root,
            split_seed=int(definition["seeds"]["split_seed"]),
        )

    with HeartbeatMarker(run_dir / "HEARTBEAT.json", args.worker_run_id):
        study.optimize(objective, n_trials=1, catch=(Exception,))

    trials = study.get_trials(deepcopy=False)
    if claimed:
        trial_number, budget_index = claimed[0]
    else:
        guarded = [
            trial
            for trial in trials
            if trial.user_attrs.get(UNRESERVED_GUARD_ATTR) is True
            and trial.user_attrs.get("worker_run_id") in {None, args.worker_run_id}
        ]
        if not guarded:
            raise DistributedContractError("One-trial optimize returned without a claimed trial.")
        trial_number = guarded[-1].number
        budget_index = None
    terminal = next(trial for trial in trials if trial.number == trial_number)
    if terminal.state.name not in {"COMPLETE", "FAIL"}:
        raise DistributedContractError(
            f"Claimed trial {trial_number} is not terminal: {terminal.state.name}."
        )
    atomic_write_json(
        run_dir / "COMPLETED",
        {
            "schema_version": "graphvae-attr-f1pr-worker-complete-v1",
            "trial_number": trial_number,
            "budget_index": budget_index,
            "db_state": terminal.state.name,
            "unreserved_guard": terminal.user_attrs.get(UNRESERVED_GUARD_ATTR) is True,
            "finished_at_unix": time.time(),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_worker(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
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
