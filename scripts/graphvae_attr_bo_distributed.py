#!/usr/bin/env python3
"""Shared contracts for PostgreSQL-backed distributed Attr-F1PR optimization.

This module intentionally leaves trial allocation and state transitions to
Optuna.  It supplies the repository-specific exact-budget reservation guard,
immutable study contract, redaction, audit, locking, and portable snapshot
logic used by the controller and one-trial worker.
"""

from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
import platform
import re
import socket
import sys
import tempfile
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for GraphVAE BO.") from exc

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:  # pragma: no cover - diagnosed by require_distributed_dependencies
    optuna = None
    TrialState = None

try:
    from graphvae_attr_bo_fingerprints import (
        canonical_json_bytes,
        framed_sha256,
        output_root_fingerprint,
        sampler_seed,
        sha256_file,
        verify_deployment_manifest,
    )
except ImportError:  # imported as scripts.graphvae_attr_bo_distributed
    from scripts.graphvae_attr_bo_fingerprints import (
        canonical_json_bytes,
        framed_sha256,
        output_root_fingerprint,
        sampler_seed,
        sha256_file,
        verify_deployment_manifest,
    )


SCHEMA_VERSION = "graphvae-attr-f1pr-distributed-study-v1"
OBJECTIVE_NAME = "Attr-F1PR"
OBJECTIVE_JSON_PATH = "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
PRIMARY_MODE = "decoded_node_edge"
DIRECTION = "maximize"
LIFECYCLE_ATTR = "graphvae_bo_lifecycle"
DEFINITION_ATTR = "graphvae_bo_study_definition"
CONTRACT_ATTR = "graphvae_bo_study_contract_sha256"
UUID_ATTR = "graphvae_bo_study_uuid"
CONTROLLER_ATTR = "graphvae_bo_controller_uuid"
OUTPUT_ROOT_ATTR = "graphvae_bo_output_root_sha256"
RESERVED_ATTR = "graphvae_bo_reserved"
BUDGET_INDEX_ATTR = "budget_index"
TRIAL_CONTRACT_ATTR = "study_contract_sha256"
UNRESERVED_GUARD_ATTR = "unreserved_guard"
LIFECYCLE_INITIALIZING = "INITIALIZING"
LIFECYCLE_READY = "READY"
LIFECYCLE_FROZEN = "FROZEN"
DEFAULT_HEARTBEAT_INTERVAL = 60
DEFAULT_GRACE_PERIOD = 600
DEFAULT_CONNECT_TIMEOUT = 15
DEFAULT_STARTUP_TRIALS = 5
EXPECTED_OPTUNA_VERSION = "4.2.1"
EXPECTED_PSYCOPG2_VERSION = "2.9.10"
EXPECTED_SQLALCHEMY_VERSION = "2.0.52"
EXPECTED_ALEMBIC_VERSION = "1.14.1"
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DistributedContractError(RuntimeError):
    pass


class UnreservedTrialError(DistributedContractError):
    pass


def require_distributed_dependencies() -> None:
    if optuna is None:
        raise RuntimeError(
            "Distributed GraphVAE BO requires the pinned requirements-bo-py38.txt."
        )


def validate_identifier(value: str, label: str) -> str:
    if not SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"Unsafe {label}: use 1-128 letters, digits, dots, underscores, or dashes."
        )
    return value


def canonical_contract_hash(definition: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(definition)).hexdigest()


def atomic_write_bytes(path: Path, content: bytes, *, mode: int = 0o644) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), mode)
        os.replace(str(temporary), str(destination))
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(
        path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    )


def atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path, yaml.safe_dump(dict(payload), sort_keys=False).encode("utf-8")
    )


def atomic_write_text(path: Path, content: str) -> None:
    atomic_write_bytes(path, content.encode("utf-8"))


def _storage_parts(storage_url: str) -> tuple[str, Any]:
    parsed = urlsplit(storage_url)
    dialect = parsed.scheme.lower()
    if dialect not in {"postgresql", "postgresql+psycopg2"}:
        raise ValueError(
            "Distributed mode requires PostgreSQL via postgresql+psycopg2 storage; SQLite, "
            "JournalStorage, and copied databases are rejected."
        )
    if not parsed.hostname or not parsed.path.strip("/"):
        raise ValueError("PostgreSQL storage requires a hostname and database name.")
    return dialect, parsed


def redacted_storage_identity(storage_url: str) -> dict[str, Any]:
    dialect, parsed = _storage_parts(storage_url)
    return {
        "dialect": dialect,
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "database": unquote(parsed.path.lstrip("/")),
    }


def redact_secret(text: Any, *, storage_url: str | None = None) -> str:
    value = str(text)
    secrets = []
    if storage_url:
        secrets.append(storage_url)
        try:
            parsed = urlsplit(storage_url)
            if parsed.password:
                secrets.extend((parsed.password, unquote(parsed.password)))
            if parsed.username and parsed.password:
                secrets.append(f"{parsed.username}:{parsed.password}")
        except Exception:
            pass
    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        value = value.replace(secret, "<redacted>")
    value = re.sub(r"(postgresql(?:\+psycopg2)?://[^:/\s]+:)[^@\s]+@", r"\1<redacted>@", value)
    return value


def storage_url_from_env(environment_name: str) -> str:
    validate_identifier(environment_name, "storage environment variable")
    value = os.environ.get(environment_name)
    if not value:
        raise RuntimeError(f"Storage environment variable {environment_name!r} is unset.")
    _storage_parts(value)
    validate_credential_environment(value)
    return value


def validate_credential_environment(storage_url: str) -> None:
    """Validate libpq password-file permissions without exposing its content."""

    parsed = urlsplit(storage_url)
    password_file_value = os.environ.get("PGPASSFILE")
    if not password_file_value:
        # A credential-bearing URL is allowed only via the protected environment
        # variable and is redacted everywhere else. libpq may also use ~/.pgpass.
        return
    password_file = Path(password_file_value)
    if not password_file.is_absolute() or not password_file.is_file():
        raise RuntimeError("PGPASSFILE must name an existing absolute regular file.")
    if password_file.is_symlink():
        raise RuntimeError("PGPASSFILE may not be a symlink.")
    if password_file.stat().st_mode & 0o077:
        raise RuntimeError("PGPASSFILE permissions must be 0600.")
    parent = password_file.parent
    if parent.stat().st_mode & 0o077:
        raise RuntimeError("PGPASSFILE parent directory must not allow group/other access.")


def _sslmode(storage_url: str) -> str | None:
    parsed = urlsplit(storage_url)
    for item in parsed.query.split("&"):
        key, separator, value = item.partition("=")
        if separator and unquote(key).lower() == "sslmode":
            return unquote(value).lower()
    return None


def create_postgresql_storage(
    storage_url: str,
    *,
    heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
    grace_period: int = DEFAULT_GRACE_PERIOD,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
    allow_insecure_local_test: bool = False,
    allow_sslmode_require_exception: bool = False,
):
    require_distributed_dependencies()
    _, parsed = _storage_parts(storage_url)
    if heartbeat_interval < 1 or grace_period < heartbeat_interval:
        raise ValueError("Heartbeat/grace values require 1 <= heartbeat <= grace.")
    if connect_timeout < 1:
        raise ValueError("PostgreSQL connection timeout must be positive.")
    sslmode = _sslmode(storage_url)
    is_local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    documented_require_exception = (
        sslmode == "require" and allow_sslmode_require_exception
    )
    if (
        sslmode != "verify-full"
        and not documented_require_exception
        and not (allow_insecure_local_test and is_local)
    ):
        raise ValueError(
            "Distributed PostgreSQL requires sslmode=verify-full. Only explicit "
            "localhost test storage may opt out."
        )
    try:
        return optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=int(heartbeat_interval),
            grace_period=int(grace_period),
            failed_trial_callback=None,
            engine_kwargs={
                "pool_pre_ping": True,
                "connect_args": {"connect_timeout": int(connect_timeout)},
            },
        )
    except Exception as exc:
        identity = redacted_storage_identity(storage_url)
        raise RuntimeError(
            "PostgreSQL connection failed for "
            f"{identity['host']}:{identity['port']}/{identity['database']}: "
            f"{redact_secret(type(exc).__name__, storage_url=storage_url)}"
        ) from None


def runtime_dependency_fingerprint() -> dict[str, Any]:
    packages = (
        ("optuna", "optuna"),
        ("psycopg2", "psycopg2"),
        ("sqlalchemy", "sqlalchemy"),
        ("alembic", "alembic"),
        ("numpy", "numpy"),
        ("yaml", "PyYAML"),
        ("torch", "torch"),
        ("dgl", "dgl"),
    )
    imports = []
    for module_name, distribution_name in packages:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None)
            if module_name == "yaml":
                version = getattr(module, "__version__", version)
            imports.append(
                {
                    "module": module_name,
                    "distribution": distribution_name,
                    "version": None if version is None else str(version).split()[0],
                    "path": str(Path(module.__file__).resolve()),
                    "module_file_sha256": sha256_file(Path(module.__file__).resolve()),
                }
            )
        except Exception as exc:
            imports.append(
                {
                    "module": module_name,
                    "distribution": distribution_name,
                    "error": type(exc).__name__,
                }
            )
    payload = {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "imports": imports,
    }
    semantic = {
        "python_version": payload["python_version"],
        "imports": [
            {
                key: record.get(key)
                for key in (
                    "module",
                    "distribution",
                    "version",
                    "module_file_sha256",
                    "error",
                )
                if key in record
            }
            for record in imports
        ],
    }
    payload["sha256"] = framed_sha256(
        "runtime-dependencies", (("payload", canonical_json_bytes(semantic)),)
    )
    payload["provenance_sha256"] = framed_sha256(
        "runtime-dependencies-provenance",
        (("payload", canonical_json_bytes(payload)),),
    )
    return payload


def enforce_pinned_versions() -> None:
    require_distributed_dependencies()
    if optuna.__version__ != EXPECTED_OPTUNA_VERSION:
        raise RuntimeError(
            f"Distributed BO requires optuna=={EXPECTED_OPTUNA_VERSION}, got {optuna.__version__}."
        )
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("Distributed BO requires psycopg2-binary==2.9.10.") from exc
    actual = str(psycopg2.__version__).split()[0]
    if actual != EXPECTED_PSYCOPG2_VERSION:
        raise RuntimeError(
            f"Distributed BO requires psycopg2-binary=={EXPECTED_PSYCOPG2_VERSION}, got {actual}."
        )
    import sqlalchemy
    import alembic

    for name, actual_version, expected_version in (
        ("SQLAlchemy", sqlalchemy.__version__, EXPECTED_SQLALCHEMY_VERSION),
        ("alembic", alembic.__version__, EXPECTED_ALEMBIC_VERSION),
    ):
        if str(actual_version) != expected_version:
            raise RuntimeError(
                f"Distributed BO requires {name}=={expected_version}, got {actual_version}."
            )


def build_worker_sampler(seed: int, startup_trials: int = DEFAULT_STARTUP_TRIALS):
    require_distributed_dependencies()
    return optuna.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=int(startup_trials),
        constant_liar=True,
    )


def create_or_load_distributed_study(
    storage: Any,
    *,
    study_name: str,
    sampler_seed_value: int,
    startup_trials: int = DEFAULT_STARTUP_TRIALS,
    create: bool = False,
):
    validate_identifier(study_name, "study name")
    sampler = build_worker_sampler(sampler_seed_value, startup_trials)
    if create:
        return optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction=DIRECTION,
            sampler=sampler,
            load_if_exists=True,
        )
    return optuna.load_study(
        storage=storage,
        study_name=study_name,
        sampler=sampler,
    )


def build_study_definition(
    *,
    study_name: str,
    base_config: Mapping[str, Any],
    base_config_sha256: str,
    ranges: Mapping[str, Any],
    reserved_trials: int,
    seeds: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    training: Mapping[str, Any],
    source: Mapping[str, Any],
    environment: Mapping[str, Any],
    dataset_cache: Mapping[str, Any],
    feature_schemas: Mapping[str, Any],
    hardware_policy: Mapping[str, Any],
    heartbeat_interval: int,
    grace_period: int,
    max_parallel: int,
    study_uuid: str | None = None,
) -> dict[str, Any]:
    if reserved_trials < 1:
        raise ValueError("The reserved scientific trial count must be positive.")
    if max_parallel < 1 or max_parallel > reserved_trials:
        raise ValueError("Maximum concurrency must be between one and the trial budget.")
    return {
        "schema_version": SCHEMA_VERSION,
        "study_name": validate_identifier(study_name, "study name"),
        "study_uuid": study_uuid or str(uuid.uuid4()),
        "objective": {
            "public_name": OBJECTIVE_NAME,
            "json_path": OBJECTIVE_JSON_PATH,
            "direction": DIRECTION,
            "primary_mode": PRIMARY_MODE,
            "split": "validation",
            "test_access": False,
        },
        "base_config_sha256": str(base_config_sha256),
        "resolved_fixed_configuration": copy.deepcopy(dict(base_config)),
        "search_space": copy.deepcopy(dict(ranges)),
        "reserved_trials": int(reserved_trials),
        "seeds": copy.deepcopy(dict(seeds)),
        "sampler": {
            "name": "TPESampler",
            "constant_liar": True,
            "n_startup_trials": int(seeds.get("tpe_startup_trials", DEFAULT_STARTUP_TRIALS)),
            "study_seed": int(seeds.get("study_seed", seeds.get("sampler_seed", 0))),
            "worker_seed_derivation": "graphvae-attr-f1pr-sampler-v1-sha256-first4-big-endian",
        },
        "evaluator": copy.deepcopy(dict(evaluator)),
        "training": copy.deepcopy(dict(training)),
        "source": copy.deepcopy(dict(source)),
        "environment": copy.deepcopy(dict(environment)),
        "dataset_cache": copy.deepcopy(dict(dataset_cache)),
        "feature_schemas": copy.deepcopy(dict(feature_schemas)),
        "hardware_policy": copy.deepcopy(dict(hardware_policy)),
        "storage": {
            "backend": "PostgreSQL",
            "heartbeat_interval": int(heartbeat_interval),
            "grace_period": int(grace_period),
            "stale_retry": False,
        },
        "scheduler": {
            "mode": "bounded_synchronous_waves",
            "max_parallel": int(max_parallel),
            "study_path_replay": max_parallel == 1,
        },
        "retention": {
            "canonical_automatic_deletion": False,
            "failed_slots_replaced": False,
        },
    }


def _study_attrs(study: Any) -> Mapping[str, Any]:
    return study.user_attrs


def reservation_audit(study: Any, expected_count: int) -> dict[str, Any]:
    reserved: dict[int, Any] = {}
    unreserved = []
    duplicates = []
    invalid = []
    for trial in study.get_trials(deepcopy=False):
        attrs = trial.user_attrs
        if attrs.get(RESERVED_ATTR) is True:
            try:
                index = int(attrs[BUDGET_INDEX_ATTR])
            except (KeyError, TypeError, ValueError):
                invalid.append(trial.number)
                continue
            if index < 0 or index >= expected_count:
                invalid.append(trial.number)
            elif index in reserved:
                duplicates.append(index)
            else:
                reserved[index] = trial
        else:
            unreserved.append(trial)
    missing = sorted(set(range(expected_count)) - set(reserved))
    return {
        "reserved_by_index": reserved,
        "missing_indexes": missing,
        "duplicate_indexes": sorted(set(duplicates)),
        "invalid_trial_numbers": sorted(invalid),
        "unreserved_trials": unreserved,
    }


def validate_guard_rows(trials: Sequence[Any]) -> None:
    for trial in trials:
        if (
            trial.state != TrialState.FAIL
            or trial.user_attrs.get(UNRESERVED_GUARD_ATTR) is not True
            or trial.params
            or trial.value is not None
        ):
            raise DistributedContractError(
                f"Unreserved trial {trial.number} is not a parameter-free FAIL guard row."
            )


def initialize_reserved_study(
    study: Any,
    definition: Mapping[str, Any],
    *,
    controller_uuid: str,
    output_root: Path,
    interrupt_after: int | None = None,
) -> str:
    """Idempotently create missing empty WAITING reservations through public APIs."""

    contract_hash = canonical_contract_hash(definition)
    expected_count = int(definition["reserved_trials"])
    attrs = _study_attrs(study)
    recorded_definition = attrs.get(DEFINITION_ATTR)
    if recorded_definition is not None and recorded_definition != definition:
        raise DistributedContractError("Study definition differs from the immutable PostgreSQL contract.")
    if attrs.get(CONTRACT_ATTR) not in {None, contract_hash}:
        raise DistributedContractError("Study contract SHA-256 differs from the requested definition.")
    if attrs.get(OUTPUT_ROOT_ATTR) not in {None, output_root_fingerprint(output_root)}:
        raise DistributedContractError("Study is owned by a different controller output root.")
    if attrs.get(CONTROLLER_ATTR) not in {None, controller_uuid}:
        raise DistributedContractError(
            "Study is owned by a different controller identity; explicit recovery is required."
        )

    if recorded_definition is None:
        if study.get_trials(deepcopy=False):
            raise DistributedContractError(
                "Existing trials have no immutable study definition; refusing adoption."
            )
        study.set_user_attr(LIFECYCLE_ATTR, LIFECYCLE_INITIALIZING)
        study.set_user_attr(DEFINITION_ATTR, dict(definition))
        study.set_user_attr(CONTRACT_ATTR, contract_hash)
        study.set_user_attr(UUID_ATTR, definition["study_uuid"])
        study.set_user_attr(CONTROLLER_ATTR, controller_uuid)
        study.set_user_attr(OUTPUT_ROOT_ATTR, output_root_fingerprint(output_root))
    elif attrs.get(LIFECYCLE_ATTR) == LIFECYCLE_FROZEN:
        raise DistributedContractError("A frozen study cannot be initialized or extended.")

    audit = reservation_audit(study, expected_count)
    if audit["duplicate_indexes"] or audit["invalid_trial_numbers"]:
        raise DistributedContractError("Reservation indexes are duplicate, invalid, or out of range.")
    if audit["unreserved_trials"]:
        if attrs.get(LIFECYCLE_ATTR) != LIFECYCLE_READY:
            raise DistributedContractError("Unmarked trials exist during reservation initialization.")
        validate_guard_rows(audit["unreserved_trials"])
    if attrs.get(LIFECYCLE_ATTR) == LIFECYCLE_READY and audit["missing_indexes"]:
        raise DistributedContractError(
            "A READY study is missing reserved indexes; ordinary resume may not append slots."
        )

    fixed_parameters = definition.get("search_space", {}).get("fixed_parameters")
    if fixed_parameters is None:
        fixed_parameters = {}
    if not isinstance(fixed_parameters, Mapping):
        raise DistributedContractError("Fixed qualification parameters must be a mapping.")
    allowed_fixed = {"alpha_node_feat", "alpha_edge_feat", "alpha_motif_loss"}
    if set(fixed_parameters) - allowed_fixed:
        raise DistributedContractError("Fixed qualification parameters contain an unknown key.")
    for name, value in fixed_parameters.items():
        search_entry = definition["search_space"].get(name)
        if not isinstance(search_entry, Mapping):
            raise DistributedContractError(
                f"Fixed qualification parameter {name} is outside the search contract."
            )
        numeric = float(value)
        if (
            not math.isfinite(numeric)
            or numeric < float(search_entry["low"])
            or numeric > float(search_entry["high"])
        ):
            raise DistributedContractError(
                f"Fixed qualification parameter {name} is outside its contracted range."
            )

    created = 0
    for index in audit["missing_indexes"]:
        study.enqueue_trial(
            dict(fixed_parameters),
            user_attrs={
                RESERVED_ATTR: True,
                BUDGET_INDEX_ATTR: index,
                TRIAL_CONTRACT_ATTR: contract_hash,
            },
        )
        created += 1
        if interrupt_after is not None and created >= interrupt_after:
            raise RuntimeError("Injected reservation initialization interruption.")

    final = reservation_audit(study, expected_count)
    if (
        final["missing_indexes"]
        or final["duplicate_indexes"]
        or final["invalid_trial_numbers"]
    ):
        raise DistributedContractError("Could not prove exact reservation indexes 0..N-1.")
    validate_guard_rows(final["unreserved_trials"])
    study.set_user_attr(LIFECYCLE_ATTR, LIFECYCLE_READY)
    return contract_hash


def validate_study_contract(
    study: Any,
    *,
    expected_contract_hash: str,
    local_definition: Mapping[str, Any] | None = None,
    require_ready: bool = True,
) -> Mapping[str, Any]:
    attrs = _study_attrs(study)
    if attrs.get(CONTRACT_ATTR) != expected_contract_hash:
        raise DistributedContractError("PostgreSQL study contract hash mismatch.")
    definition = attrs.get(DEFINITION_ATTR)
    if not isinstance(definition, Mapping):
        raise DistributedContractError("PostgreSQL study definition is missing.")
    if canonical_contract_hash(definition) != expected_contract_hash:
        raise DistributedContractError("PostgreSQL study definition is internally inconsistent.")
    if local_definition is not None and dict(definition) != dict(local_definition):
        raise DistributedContractError("Local study definition differs from PostgreSQL.")
    lifecycle = attrs.get(LIFECYCLE_ATTR)
    if require_ready and lifecycle != LIFECYCLE_READY:
        raise DistributedContractError(
            f"Workers require lifecycle READY, got {lifecycle!r}."
        )
    return definition


def guard_reserved_trial(
    trial: Any,
    study: Any,
    *,
    expected_contract_hash: str,
) -> int:
    """Validate a claimed reservation before suggesting any parameter."""

    lifecycle = study.user_attrs.get(LIFECYCLE_ATTR)
    attrs = trial.user_attrs
    if (
        lifecycle != LIFECYCLE_READY
        or attrs.get(RESERVED_ATTR) is not True
        or attrs.get(TRIAL_CONTRACT_ATTR) != expected_contract_hash
    ):
        trial.set_user_attr(UNRESERVED_GUARD_ATTR, True)
        trial.set_user_attr("failure_phase", "reservation_guard")
        raise UnreservedTrialError(
            "Trial is not an authorized reservation for this READY study."
        )
    try:
        budget_index = int(attrs[BUDGET_INDEX_ATTR])
    except (KeyError, TypeError, ValueError) as exc:
        raise DistributedContractError("Reserved trial has no valid budget index.") from exc
    expected_count = int(study.user_attrs[DEFINITION_ATTR]["reserved_trials"])
    if not 0 <= budget_index < expected_count:
        raise DistributedContractError("Reserved trial budget index is out of range.")
    trial.set_user_attr("objective_name", OBJECTIVE_NAME)
    trial.set_user_attr("objective_json_path", OBJECTIVE_JSON_PATH)
    return budget_index


def reserved_trial_states(study: Any) -> dict[str, int]:
    definition = study.user_attrs.get(DEFINITION_ATTR) or {}
    count = int(definition.get("reserved_trials", 0))
    audit = reservation_audit(study, count)
    result = {name: 0 for name in ("WAITING", "RUNNING", "COMPLETE", "FAIL", "OTHER")}
    for trial in audit["reserved_by_index"].values():
        name = trial.state.name
        result[name if name in result else "OTHER"] += 1
    result["UNRESERVED_GUARD"] = len(audit["unreserved_trials"])
    result["RESERVED_TOTAL"] = len(audit["reserved_by_index"])
    return result


def assert_quiescent_reserved_study(study: Any) -> None:
    definition = validate_study_contract(
        study,
        expected_contract_hash=study.user_attrs[CONTRACT_ATTR],
        require_ready=False,
    )
    audit = reservation_audit(study, int(definition["reserved_trials"]))
    if audit["missing_indexes"] or audit["duplicate_indexes"] or audit["invalid_trial_numbers"]:
        raise DistributedContractError("Reserved budget cardinality is invalid.")
    validate_guard_rows(audit["unreserved_trials"])
    active = [
        trial.number
        for trial in audit["reserved_by_index"].values()
        if trial.state in {TrialState.WAITING, TrialState.RUNNING}
    ]
    if active:
        raise DistributedContractError(
            "Finalization refuses WAITING/RUNNING reservations: "
            + ", ".join(map(str, active))
        )


def relative_artifact_path(root: Path, path: Path) -> str:
    root_resolved = Path(root).resolve()
    path_resolved = Path(path).resolve()
    try:
        return path_resolved.relative_to(root_resolved).as_posix()
    except ValueError as exc:
        raise DistributedContractError(f"Artifact escapes study root: {path}") from exc


def resolve_artifact_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise DistributedContractError("Portable artifact paths must be relative.")
    candidate = (Path(root).resolve() / relative).resolve()
    try:
        candidate.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise DistributedContractError("Artifact path traverses outside study root.") from exc
    return candidate


def audit_trial_result(
    trial: Any,
    *,
    study_root: Path,
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    attrs = trial.user_attrs
    if attrs.get(RESERVED_ATTR) is not True:
        raise DistributedContractError("Only reserved scientific trials are auditable.")
    budget_index = int(attrs[BUDGET_INDEX_ATTR])
    contract_hash = canonical_contract_hash(definition)
    if attrs.get(TRIAL_CONTRACT_ATTR) != contract_hash:
        raise DistributedContractError("Trial contract hash mismatch.")
    tombstone_relative = attrs.get("failure_tombstone")
    default_tombstone = f"trials/trial_{trial.number:05d}/trial_failure_tombstone.json"
    if trial.state == TrialState.FAIL and not tombstone_relative:
        if resolve_artifact_path(study_root, default_tombstone).is_file():
            tombstone_relative = default_tombstone
    if trial.state == TrialState.FAIL and tombstone_relative:
        tombstone_path = resolve_artifact_path(study_root, tombstone_relative)
        tombstone = json.loads(tombstone_path.read_text(encoding="utf-8"))
        if (
            tombstone.get("schema_version")
            != "graphvae-attr-f1pr-failure-tombstone-v1"
            or not tombstone.get("failure_category")
            or not isinstance(tombstone.get("missing_artifacts"), list)
            or not tombstone.get("missing_artifacts")
            or tombstone.get("trial_number") != trial.number
            or tombstone.get("budget_index") != budget_index
            or tombstone.get("db_state") != "FAIL"
            or tombstone.get("study_contract_sha256") != contract_hash
        ):
            raise DistributedContractError("Failure tombstone identity mismatch.")
        for missing in tombstone["missing_artifacts"]:
            if (
                not isinstance(missing, Mapping)
                or missing.get("verified_absent") is not True
                or not missing.get("path")
            ):
                raise DistributedContractError(
                    "Failure tombstone has an invalid missing-artifact record."
                )
            if resolve_artifact_path(study_root, str(missing["path"])).exists():
                raise DistributedContractError(
                    "Failure tombstone names an artifact that now exists."
                )
        retained_evidence = tombstone.get("retained_evidence") or []
        if not isinstance(retained_evidence, list):
            raise DistributedContractError("Failure tombstone retained evidence is invalid.")
        for retained in retained_evidence:
            if (
                not isinstance(retained, Mapping)
                or not retained.get("path")
                or not retained.get("sha256")
            ):
                raise DistributedContractError(
                    "Failure tombstone retained evidence is invalid."
                )
            retained_path = resolve_artifact_path(study_root, str(retained["path"]))
            if (
                not retained_path.is_file()
                or sha256_file(retained_path) != retained["sha256"]
            ):
                raise DistributedContractError(
                    "Failure tombstone retained evidence hash mismatch."
                )
        return tombstone
    result_relative = attrs.get("trial_result") or f"trials/trial_{trial.number:05d}/trial_result.json"
    result_path = resolve_artifact_path(study_root, result_relative)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_pairs = {
        "trial_number": trial.number,
        "budget_index": budget_index,
        "study_contract_sha256": contract_hash,
        "sampled_weights": dict(trial.params),
    }
    for field, expected in expected_pairs.items():
        if result.get(field) != expected:
            raise DistributedContractError(f"Trial result mismatch for {field}.")
    expected_status = "COMPLETE" if trial.state == TrialState.COMPLETE else "FAIL"
    if result.get("status") != expected_status:
        raise DistributedContractError("Trial result status differs from PostgreSQL.")
    hashes = result.get("hashes") or {}
    contract_hashes = {
        "cache_sha256": definition["dataset_cache"].get("sha256"),
        "split_fingerprint": definition["dataset_cache"].get("split_fingerprint"),
        "node_schema_fingerprint": definition["feature_schemas"].get("node_sha256"),
        "edge_schema_fingerprint": definition["feature_schemas"].get("edge_sha256"),
        "source_tree_sha256": definition["source"].get("tree_sha256"),
        "environment_sha256": definition["environment"].get("sha256"),
    }
    for field, expected in contract_hashes.items():
        if expected is not None and hashes.get(field) != expected:
            raise DistributedContractError(f"Trial integrity mismatch for {field}.")
    if trial.state == TrialState.COMPLETE:
        if trial.value is None or not math.isfinite(float(trial.value)):
            raise DistributedContractError("COMPLETE trial objective is non-finite.")
        if float(result.get("validation_attr_f1pr")) != float(trial.value):
            raise DistributedContractError("Trial result objective differs from PostgreSQL.")
        if result.get("physical_gpu") is not None:
            gpu_model = result.get("gpu_model")
            gpu_vram_bytes = result.get("gpu_vram_bytes")
            if (
                not isinstance(gpu_model, str)
                or not gpu_model.strip()
                or not isinstance(gpu_vram_bytes, int)
                or gpu_vram_bytes <= 0
            ):
                raise DistributedContractError(
                    "GPU trial result is missing verified model or VRAM metadata."
                )
        expected_metadata = {
            "training_seed": definition["seeds"].get("training_seed"),
            "split_seed": definition["seeds"].get("split_seed"),
            "generation_seed": definition["seeds"].get("generation_seed"),
            "evaluator_seed": definition["seeds"].get("evaluator_seed"),
            "evaluator_repeats": definition["evaluator"].get("repeat_count"),
            "fixed_generated_graph_limit": definition["evaluator"].get("max_graphs"),
        }
        for field, expected in expected_metadata.items():
            if expected is not None and result.get(field) != expected:
                raise DistributedContractError(f"Trial result mismatch for {field}.")
        for path_field, hash_field in (
            ("resolved_config", "resolved_config_sha256"),
            ("checkpoint", "checkpoint_sha256"),
            ("evaluator_output", "evaluator_output_sha256"),
        ):
            path = resolve_artifact_path(study_root, result[path_field])
            if sha256_file(path) != result[hash_field]:
                raise DistributedContractError(f"Artifact hash mismatch for {path_field}.")
        evaluator_path = resolve_artifact_path(study_root, result["evaluator_output"])
        try:
            try:
                from tune_graphvae_attribute_weights import parse_attr_f1pr_file
            except ImportError:
                from scripts.tune_graphvae_attribute_weights import parse_attr_f1pr_file
            evaluator_settings = definition["evaluator"]
            cache = definition["dataset_cache"]
            schemas = definition["feature_schemas"]
            metrics = parse_attr_f1pr_file(
                evaluator_path,
                expected_split="validation",
                expected_graph_count=cache.get("expected_validation_graphs"),
                expected_cache_sha256=cache.get("sha256"),
                expected_split_fingerprint=cache.get("split_fingerprint"),
                expected_node_schema_fingerprint=schemas.get("node_sha256"),
                expected_edge_schema_fingerprint=schemas.get("edge_sha256"),
                expected_node_feature_dimension=cache.get("node_feature_dimension"),
                expected_edge_feature_dimension=cache.get("edge_feature_dimension"),
                expected_generation_seed=definition["seeds"].get("generation_seed"),
                expected_evaluator_seed=definition["seeds"].get("evaluator_seed"),
                expected_repeats=evaluator_settings.get("repeat_count"),
            )
        except Exception as exc:
            raise DistributedContractError(f"Evaluator contract audit failed: {exc}") from exc
        if float(metrics.f1_pr) != float(trial.value):
            raise DistributedContractError(
                "PostgreSQL objective is not the exact evaluator Attr-F1PR value."
            )
    return result


def write_failure_tombstone(
    *,
    study_root: Path,
    trial_number: int,
    budget_index: int,
    contract_hash: str,
    worker_id: str | None,
    worker_run_id_value: str | None,
    failure_category: str,
    missing_artifacts: Sequence[Mapping[str, Any]],
    retained_evidence: Sequence[Mapping[str, Any]] = (),
) -> Path:
    path = (
        Path(study_root).resolve()
        / "trials"
        / f"trial_{int(trial_number):05d}"
        / "trial_failure_tombstone.json"
    )
    atomic_write_json(
        path,
        {
            "schema_version": "graphvae-attr-f1pr-failure-tombstone-v1",
            "trial_number": int(trial_number),
            "budget_index": int(budget_index),
            "study_contract_sha256": contract_hash,
            "worker_id": worker_id,
            "worker_run_id": worker_run_id_value,
            "db_state": "FAIL",
            "reconciled_at_unix": time.time(),
            "failure_category": failure_category,
            "missing_artifacts": [dict(item) for item in missing_artifacts],
            "retained_evidence": [dict(item) for item in retained_evidence],
        },
    )
    return path


def trial_semantic_fingerprint(study: Any) -> str:
    trials = []
    for trial in study.get_trials(deepcopy=False):
        trials.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "params": dict(trial.params),
                "distributions": {
                    key: str(value) for key, value in sorted(trial.distributions.items())
                },
                "value": trial.value,
                "user_attrs": dict(trial.user_attrs),
            }
        )
    payload = {
        "study_name": study.study_name,
        "directions": [direction.name for direction in study.directions],
        "user_attrs": dict(study.user_attrs),
        "trials": trials,
    }
    return framed_sha256(
        "optuna-study-semantic", (("payload", canonical_json_bytes(payload)),)
    )


def create_portable_snapshot(
    study: Any,
    *,
    source_storage: Any,
    snapshot_path: Path,
) -> Path:
    assert_quiescent_reserved_study(study)
    destination = Path(snapshot_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_fingerprint = trial_semantic_fingerprint(study)
    if destination.exists():
        existing = optuna.load_study(
            study_name=study.study_name,
            storage="sqlite:///" + destination.as_posix(),
        )
        if trial_semantic_fingerprint(existing) != source_fingerprint:
            raise DistributedContractError("Existing portable snapshot does not match PostgreSQL.")
        return destination
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.sqlite3", dir=str(destination.parent)
    )
    os.close(fd)
    temporary = Path(name)
    temporary.unlink()
    try:
        target_url = "sqlite:///" + temporary.as_posix()
        optuna.copy_study(
            from_study_name=study.study_name,
            from_storage=source_storage,
            to_storage=target_url,
            to_study_name=study.study_name,
        )
        copied = optuna.load_study(study_name=study.study_name, storage=target_url)
        if trial_semantic_fingerprint(copied) != source_fingerprint:
            raise DistributedContractError("Portable snapshot semantic verification failed.")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()


def _advisory_key(identity: Mapping[str, Any], study_name: str) -> int:
    material = canonical_json_bytes({"database": dict(identity), "study": study_name})
    raw = int.from_bytes(
        hashlib.sha256(b"graphvae-attr-bo-controller-lock-v1\0" + material).digest()[:8],
        "big",
    )
    return raw - (1 << 64) if raw >= (1 << 63) else raw


def _psycopg_connection(storage_url: str):
    import psycopg2

    parsed = urlsplit(storage_url)
    kwargs: dict[str, Any] = {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "dbname": unquote(parsed.path.lstrip("/")),
    }
    if parsed.username:
        kwargs["user"] = unquote(parsed.username)
    if parsed.password:
        kwargs["password"] = unquote(parsed.password)
    for item in parsed.query.split("&"):
        key, separator, value = item.partition("=")
        if separator:
            kwargs[unquote(key)] = unquote(value)
    return psycopg2.connect(**kwargs)


class ControllerLocks(AbstractContextManager):
    """Hold one output-root flock and one PostgreSQL advisory lock."""

    def __init__(self, output_dir: Path, storage_url: str, study_name: str):
        self.output_dir = Path(output_dir).resolve()
        self.storage_url = storage_url
        self.study_name = validate_identifier(study_name, "study name")
        self.file_handle = None
        self.connection = None
        self.key = _advisory_key(redacted_storage_identity(storage_url), study_name)

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_handle = (self.output_dir / ".controller.lock").open("a+")
        try:
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file_handle.close()
            raise DistributedContractError("Another controller holds the output-root lock.") from exc
        try:
            self.connection = _psycopg_connection(self.storage_url)
            self.connection.autocommit = True
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT pg_try_advisory_lock(%s)", (self.key,))
                acquired = bool(cursor.fetchone()[0])
            if not acquired:
                raise DistributedContractError("Another controller holds the PostgreSQL study lock.")
        except Exception:
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
            self.file_handle.close()
            self.file_handle = None
            if self.connection is not None:
                self.connection.close()
            raise
        return self

    def assert_alive(self) -> None:
        if self.connection is None or self.connection.closed:
            raise DistributedContractError("PostgreSQL advisory-lock connection was lost.")
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        except Exception as exc:
            raise DistributedContractError(
                "PostgreSQL advisory-lock connection was lost; stop mutation and reconcile."
            ) from exc

    def __exit__(self, exc_type, exc, traceback_value):
        if self.connection is not None:
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_unlock(%s)", (self.key,))
            finally:
                self.connection.close()
        if self.file_handle is not None:
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
            self.file_handle.close()
        return False


def parse_slots(path: Path, *, known_hosts: Sequence[str] | None = None) -> list[dict[str, Any]]:
    slots = []
    seen_gpu = set()
    seen_workers = set()
    allowed = None if known_hosts is None else set(known_hosts)
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        content = raw.split("#", 1)[0].strip()
        if not content:
            continue
        parts = content.split()
        if len(parts) != 3:
            raise ValueError(f"Malformed slot row {line_number}; expected HOST GPU WORKER_ID.")
        host, raw_gpu, worker_id = parts
        validate_identifier(host, "slot host")
        validate_identifier(worker_id, "worker ID")
        try:
            gpu = int(raw_gpu)
        except ValueError as exc:
            raise ValueError(f"Invalid GPU index on slot row {line_number}.") from exc
        if gpu < 0:
            raise ValueError(f"Invalid GPU index on slot row {line_number}.")
        if allowed is not None and host not in allowed:
            raise ValueError(f"Unknown host {host!r} on slot row {line_number}.")
        if (host, gpu) in seen_gpu:
            raise ValueError(f"Duplicate host/GPU slot: {host}:{gpu}.")
        if worker_id in seen_workers:
            raise ValueError(f"Duplicate worker ID: {worker_id}.")
        seen_gpu.add((host, gpu))
        seen_workers.add(worker_id)
        slots.append({"host": host, "physical_gpu": gpu, "worker_id": worker_id})
    if not slots:
        raise ValueError("Slot file contains no worker slots.")
    return slots


def worker_run_id(worker_id: str, dispatch_sequence: int) -> str:
    validate_identifier(worker_id, "worker ID")
    if dispatch_sequence < 0:
        raise ValueError("Dispatch sequence must be non-negative.")
    return f"{worker_id}-dispatch-{dispatch_sequence:06d}"


def worker_run_info(
    *, worker_id: str, worker_run_id_value: str, sampler_seed_value: int, device: str
) -> dict[str, Any]:
    return {
        "schema_version": "graphvae-attr-f1pr-worker-run-v1",
        "worker_id": worker_id,
        "worker_run_id": worker_run_id_value,
        "hostname": socket.gethostname(),
        "device": device,
        "sampler_seed": int(sampler_seed_value),
        "optuna_version": None if optuna is None else optuna.__version__,
        "dependency_fingerprint": runtime_dependency_fingerprint(),
        "started_at_unix": time.time(),
    }


__all__ = [name for name in globals() if not name.startswith("_")]
