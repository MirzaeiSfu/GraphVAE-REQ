#!/usr/bin/env python3
"""Fail-closed unattended supervision for bounded distributed BO waves.

This process never evaluates a trial itself.  It invokes the supported
``probe``, ``status``, ``collect``, and ``run`` controller operations, and it
launches another wave only after every prior attempted launch has a reconciled
terminal marker.  Any other probe state stops the supervisor for operator
review instead of risking duplicate work.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ACTIVE_PROBE_STATUS = "ACTIVE_AMBIGUOUS"
TERMINAL_PROBE_STATUS = "RECONCILED_TERMINAL"


class SupervisorError(RuntimeError):
    """A condition that requires operator review."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decide_next_action(
    probe: Mapping[str, Any], status: Mapping[str, Any]
) -> str:
    """Return WAIT, LAUNCH, COLLECT_AND_LAUNCH, or COLLECT_AND_FINISH.

    The decision is deliberately conservative.  A live launch is called
    ``ACTIVE_AMBIGUOUS`` by the controller because it must never be duplicated;
    that state means wait here, not launch.  Every other non-terminal probe
    state is an error requiring operator review.
    """

    if probe.get("study_name") != status.get("study_name"):
        raise SupervisorError("Probe and status study identities differ.")
    if probe.get("test_access") is not False or status.get("test_access") is not False:
        raise SupervisorError("Controller evidence does not prove test_access=false.")

    states = status.get("reserved_states")
    if not isinstance(states, Mapping):
        raise SupervisorError("Status lacks reserved trial states.")
    required = (
        "RESERVED_TOTAL",
        "WAITING",
        "RUNNING",
        "COMPLETE",
        "FAIL",
        "OTHER",
        "UNRESERVED_GUARD",
    )
    if any(not isinstance(states.get(name), int) for name in required):
        raise SupervisorError("Reserved trial states are incomplete or non-integral.")
    if states["OTHER"] != 0 or states["UNRESERVED_GUARD"] != 0:
        raise SupervisorError("Unexpected trial state or unreserved guard row detected.")
    consumed_or_pending = sum(
        states[name] for name in ("WAITING", "RUNNING", "COMPLETE", "FAIL")
    )
    if consumed_or_pending != states["RESERVED_TOTAL"]:
        raise SupervisorError("Reserved trial accounting is not exact.")

    launches = probe.get("launches")
    if not isinstance(launches, list):
        raise SupervisorError("Probe launch records are missing.")
    probe_statuses = [str(record.get("probe_status")) for record in launches]
    unsafe = sorted(
        {
            value
            for value in probe_statuses
            if value not in {ACTIVE_PROBE_STATUS, TERMINAL_PROBE_STATUS}
        }
    )
    if unsafe:
        raise SupervisorError(
            "Launch probe requires operator review: " + ", ".join(unsafe)
        )

    active_count = probe_statuses.count(ACTIVE_PROBE_STATUS)
    running = states["RUNNING"]
    waiting = states["WAITING"]
    if running:
        if not launches or active_count == 0:
            raise SupervisorError("Database has RUNNING work without an active launch probe.")
        return "WAIT"
    if active_count:
        # A worker may become terminal between the probe and status snapshots.
        # Waiting and probing again is safe; dispatching is not.
        return "WAIT"
    if any(value != TERMINAL_PROBE_STATUS for value in probe_statuses):
        raise SupervisorError("Prior attempted launches are not terminally reconciled.")
    if waiting:
        return "LAUNCH" if not launches else "COLLECT_AND_LAUNCH"
    return "COLLECT_AND_FINISH"


def _controller_command(args: argparse.Namespace, operation: str) -> list[str]:
    common = [
        str(args.controller_python),
        str(args.controller_script),
        operation,
        "--study-name",
        args.study_name,
        "--output-dir",
        str(args.output_dir),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--grace-period",
        str(args.grace_period),
    ]
    if operation == "probe":
        return common + [
            "--repo-paths",
            str(args.repo_paths),
            "--python-paths",
            str(args.python_paths),
            "--json",
            str(args.state_dir / "current_probe.json"),
        ]
    if operation == "status":
        return common + ["--json", str(args.state_dir / "current_status.json")]
    if operation == "collect":
        if args.collector_script is not None:
            return [
                str(args.collector_script),
                "--repo-paths",
                str(args.repo_paths),
                "--remote-run-root",
                args.remote_run_root,
                "--exact-destination",
                str(args.output_dir),
                "--verify-manifests",
            ]
        return common + ["--source-root", str(args.source_root)]
    if operation == "run":
        return common + [
            "--base-config",
            str(args.base_config),
            "--repo-paths",
            str(args.repo_paths),
            "--python-paths",
            str(args.python_paths),
            "--slots",
            str(args.slots),
            "--max-parallel",
            str(args.max_parallel),
            "--credential-env-file",
            str(args.credential_env_file),
            "--execute-remote",
        ]
    raise ValueError(f"Unsupported controller operation: {operation}")


def _invoke(args: argparse.Namespace, operation: str) -> None:
    result = subprocess.run(
        _controller_command(args, operation),
        cwd=str(args.repository_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        raise SupervisorError(
            f"Supported controller operation {operation!r} returned "
            f"exit code {result.returncode}."
        )


def _event(args: argparse.Namespace, kind: str, **values: Any) -> None:
    event_root = args.state_dir / "events"
    index = len(list(event_root.glob("event_*.json"))) + 1
    payload = {
        "schema_version": "graphvae-attr-f1pr-supervisor-event-v1",
        "event_index": index,
        "recorded_at_unix": time.time(),
        "study_name": args.study_name,
        "kind": kind,
        "test_access": False,
        **values,
    }
    _atomic_json(event_root / f"event_{index:06d}.json", payload)
    _atomic_json(args.state_dir / "latest_event.json", payload)


def supervise(args: argparse.Namespace) -> int:
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_dir / "supervisor.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SupervisorError("Another supervisor already owns this study.") from exc

    _event(args, "SUPERVISOR_STARTED", poll_seconds=args.poll_seconds)
    while True:
        _invoke(args, "probe")
        _invoke(args, "status")
        probe = _read_json(args.state_dir / "current_probe.json")
        status = _read_json(args.state_dir / "current_status.json")
        try:
            action = decide_next_action(probe, status)
        except SupervisorError as exc:
            _event(args, "STOPPED_REVIEW_REQUIRED", reason=str(exc))
            raise
        states = status["reserved_states"]
        _event(
            args,
            "OBSERVED",
            action=action,
            reserved_states={name: states[name] for name in sorted(states)},
        )
        if action == "WAIT":
            time.sleep(args.poll_seconds)
            continue
        if action in {"COLLECT_AND_LAUNCH", "COLLECT_AND_FINISH"}:
            _invoke(args, "collect")
            _event(args, "COLLECTED", action=action)
        if action == "COLLECT_AND_FINISH":
            _event(args, "SUPERVISOR_COMPLETE")
            return 0
        _invoke(args, "run")
        _event(args, "WAVE_LAUNCHED")
        time.sleep(min(10, args.poll_seconds))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--controller-python", type=Path, required=True)
    parser.add_argument("--controller-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    collection = parser.add_mutually_exclusive_group(required=True)
    collection.add_argument("--source-root", type=Path)
    collection.add_argument("--collector-script", type=Path)
    parser.add_argument(
        "--remote-run-root",
        help=(
            "Study-relative run root collected from every repo-path host. "
            "Required with --collector-script."
        ),
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--repo-paths", type=Path, required=True)
    parser.add_argument("--python-paths", type=Path, required=True)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--credential-env-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, required=True)
    parser.add_argument("--heartbeat-interval", type=int, default=60)
    parser.add_argument("--grace-period", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if args.max_parallel <= 0:
        parser.error("--max-parallel must be positive")
    if args.poll_seconds < 30:
        parser.error("--poll-seconds must be at least 30")
    if args.collector_script is not None:
        if not args.remote_run_root:
            parser.error("--remote-run-root is required with --collector-script")
        remote = PurePosixPath(args.remote_run_root)
        if (
            remote.is_absolute()
            or not remote.parts
            or any(part in {"", ".", ".."} for part in remote.parts)
        ):
            parser.error("--remote-run-root must be a safe relative path")
    elif args.remote_run_root is not None:
        parser.error("--remote-run-root requires --collector-script")
    for name in (
        "repository_root",
        "controller_python",
        "controller_script",
        "output_dir",
        "base_config",
        "repo_paths",
        "python_paths",
        "slots",
        "credential_env_file",
        "state_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for name in ("source_root", "collector_script"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.collector_script is not None and not args.collector_script.is_file():
        parser.error("--collector-script must name an existing regular file")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return supervise(args)
    except Exception as exc:
        # Arguments and subprocess output are intentionally omitted: either may
        # contain protected paths or a storage URL supplied by the environment.
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
