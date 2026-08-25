#!/usr/bin/env python3
"""Consume exactly one reserved PostgreSQL GraphVAE Attr-F1PR trial."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
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
    reservation_plan_entry,
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
    execute_grouped_graphcl_trial,
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


def _probe_gpu_identity(physical_gpu: int) -> tuple[str, int]:
    """Return the physical GPU model and VRAM bytes or fail before trial claim."""

    if physical_gpu < 0:
        raise DistributedContractError("Physical GPU index must be non-negative.")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                str(physical_gpu),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
        if len(rows) != 1:
            raise ValueError("GPU probe did not return exactly one row.")
        model, memory_mib_text = (part.strip() for part in rows[0].rsplit(",", 1))
        memory_mib = float(memory_mib_text)
        if not model or memory_mib <= 0:
            raise ValueError("GPU probe returned an empty model or non-positive VRAM.")
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as exc:
        raise DistributedContractError(
            f"Could not verify physical GPU {physical_gpu} identity."
        ) from exc
    return model, int(memory_mib * 1024 * 1024)


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
    evaluator = definition.get("evaluator") or {}
    if evaluator.get("backend") == "graphcl_f1pr" and not args.mock:
        graphcl = evaluator.get("backend_contract") or {}
        paths = graphcl.get("paths") or {}
        if set(paths) != {
            "reference",
            "encoder_bundle_manifest",
            "campaign_root",
            "dependency_root",
            "upstream_repo",
        }:
            raise DistributedContractError("GraphCL backend paths differ from contract.")

        def resolved_path(name: str) -> Path:
            path = Path(str(paths[name])).expanduser()
            return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()

        reference_path = resolved_path("reference")
        manifest_path = resolved_path("encoder_bundle_manifest")
        if not reference_path.is_file() or sha256_file(reference_path) != graphcl.get(
            "validation_reference_file_sha256"
        ):
            raise DistributedContractError("Frozen GraphCL validation reference differs.")
        if not manifest_path.is_file() or sha256_file(manifest_path) != graphcl.get(
            "encoder_bundle_manifest_sha256"
        ):
            raise DistributedContractError("Frozen GraphCL bundle manifest differs.")
        for name in ("campaign_root", "dependency_root", "upstream_repo"):
            if not resolved_path(name).is_dir():
                raise DistributedContractError(f"GraphCL {name} directory is missing.")
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
    planned = reservation_plan_entry(definition, budget_index)
    training_seed = (
        int(planned["training_seed"])
        if planned is not None
        else int(seeds["training_seed"])
    )
    graphcl = evaluator.get("backend_contract") or {}
    graphcl_paths = graphcl.get("paths") or {}

    def graphcl_path(name: str) -> str | None:
        value = graphcl_paths.get(name)
        if value is None:
            return None
        path = Path(str(value)).expanduser()
        return str(path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve())

    return argparse.Namespace(
        distributed=True,
        study_contract_sha256=cli.study_contract_sha256,
        budget_index=budget_index,
        worker_id=cli.worker_id,
        worker_run_id=cli.worker_run_id,
        hostname=socket.gethostname(),
        physical_gpu=cli.physical_gpu,
        gpu_model=getattr(cli, "gpu_model", None),
        gpu_vram_bytes=getattr(cli, "gpu_vram_bytes", None),
        dispatch_sequence=cli.dispatch_sequence,
        sampler_constant_liar=True,
        sampler_seed=cli.sampler_seed,
        optuna_version=optuna.__version__,
        db_driver_version=str(psycopg2.__version__).split()[0],
        tpe_startup_trials=cli.tpe_startup_trials,
        training_seed=training_seed,
        training_seeds=tuple(int(seed) for seed in seeds.get("training_seeds", [training_seed])),
        generation_seed=int(seeds["generation_seed"]),
        evaluator_seed=int(seeds["evaluator_seed"]),
        evaluator_repeats=int(evaluator.get("repeat_count", DEFAULT_EVALUATOR_REPEATS)),
        evaluator_backend=evaluator.get("backend", "random_gin"),
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
        mock_fail_training_seed=[],
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
        graphcl_cache_path=(None if cache.get("relative_path") is None else str((REPO_ROOT / cache["relative_path"]).resolve())),
        graphcl_reference_path=graphcl_path("reference"),
        graphcl_encoder_bundle_manifest=graphcl_path("encoder_bundle_manifest"),
        graphcl_encoder_bundle_manifest_sha256=graphcl.get("encoder_bundle_manifest_sha256"),
        graphcl_campaign_root=graphcl_path("campaign_root"),
        graphcl_dependency_root=graphcl_path("dependency_root"),
        graphcl_runtime_sha256=graphcl.get("graphcl_runtime_sha256"),
        graphcl_upstream_repo=graphcl_path("upstream_repo"),
        graphcl_bundle_sha256=graphcl.get("encoder_bundle_sha256"),
        graphcl_encoder_checkpoints=graphcl.get("encoder_checkpoints"),
        graphcl_validation_collection_sha256=graphcl.get("validation_collection_sha256"),
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
            "gpu_model": None,
            "gpu_vram_bytes": None,
        }
    )
    atomic_write_json(run_dir / "RUN_INFO.json", info)

    phase = "local_preflight"
    storage_url = None
    try:
        enforce_pinned_versions()
        args.gpu_model = None
        args.gpu_vram_bytes = None
        if args.physical_gpu is not None:
            phase = "hardware_preflight"
            args.gpu_model, args.gpu_vram_bytes = _probe_gpu_identity(
                args.physical_gpu
            )
            info.update(
                {
                    "gpu_model": args.gpu_model,
                    "gpu_vram_bytes": args.gpu_vram_bytes,
                }
            )
            atomic_write_json(run_dir / "RUN_INFO.json", info)
            phase = "local_preflight"
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
        if args.mock != bool(definition["training"].get("mock", False)):
            raise DistributedContractError(
                "Worker mock/real execution mode differs from contract."
            )
        if (
            args.heartbeat_interval
            != int(definition["storage"]["heartbeat_interval"])
            or args.grace_period
            != int(definition["storage"]["grace_period"])
        ):
            raise DistributedContractError(
                "Worker heartbeat/grace settings differ from contract."
            )
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
        trial.set_user_attr("physical_gpu", args.physical_gpu)
        trial.set_user_attr("gpu_model", args.gpu_model)
        trial.set_user_attr("gpu_vram_bytes", args.gpu_vram_bytes)
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
        runner = (
            execute_grouped_graphcl_trial
            if definition["evaluator"].get("backend") == "graphcl_f1pr"
            else execute_trial
        )
        return runner(
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
