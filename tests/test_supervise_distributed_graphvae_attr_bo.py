import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.supervise_distributed_graphvae_attr_bo import (
    SupervisorError,
    _controller_command,
    decide_next_action,
)


def _status(**updates):
    states = {
        "RESERVED_TOTAL": 3,
        "WAITING": 2,
        "RUNNING": 1,
        "COMPLETE": 0,
        "FAIL": 0,
        "OTHER": 0,
        "UNRESERVED_GUARD": 0,
    }
    states.update(updates)
    return {
        "study_name": "study",
        "test_access": False,
        "reserved_states": states,
    }


def _probe(*statuses):
    return {
        "study_name": "study",
        "test_access": False,
        "launches": [
            {"probe_status": status, "worker_run_id": f"worker-{index}"}
            for index, status in enumerate(statuses)
        ],
    }


def _reviewed_pretrial_record(worker_run_id="reviewed-worker"):
    return {
        "worker_run_id": worker_run_id,
        "probe_status": "RECONCILED_PRETRIAL",
        "retry_safe": True,
        "db_trials": [],
        "heartbeat": False,
        "tmux_active": False,
        "marker_payloads": {
            "FAILED_PRETRIAL": {
                "parse_ok": True,
                "reservation_consumed": False,
                "trial_number": None,
                "budget_index": None,
            }
        },
    }


def test_active_trial_waits_without_dispatching_duplicate():
    assert decide_next_action(_probe("ACTIVE_AMBIGUOUS"), _status()) == "WAIT"


def test_probe_status_race_waits_conservatively():
    status = _status(WAITING=2, RUNNING=0, COMPLETE=1)
    assert decide_next_action(_probe("ACTIVE_AMBIGUOUS"), status) == "WAIT"


def test_initial_unlaunched_budget_can_launch_once():
    status = _status(WAITING=3, RUNNING=0)
    assert decide_next_action(_probe(), status) == "LAUNCH"


@pytest.mark.parametrize("failed", [0, 1])
def test_terminal_consumption_collects_before_next_wave(failed):
    status = _status(
        WAITING=2,
        RUNNING=0,
        COMPLETE=1 - failed,
        FAIL=failed,
    )
    assert (
        decide_next_action(_probe("RECONCILED_TERMINAL"), status)
        == "COLLECT_AND_LAUNCH"
    )


def test_exact_consumed_budget_collects_and_finishes():
    status = _status(WAITING=0, RUNNING=0, COMPLETE=2, FAIL=1)
    probe = _probe(
        "RECONCILED_TERMINAL",
        "RECONCILED_TERMINAL",
        "RECONCILED_TERMINAL",
    )
    assert decide_next_action(probe, status) == "COLLECT_AND_FINISH"


@pytest.mark.parametrize(
    "unsafe",
    [
        "UNREACHABLE_AMBIGUOUS",
        "MISSING_AMBIGUOUS",
        "CONFLICT",
        "RECONCILED_PRETRIAL",
        "DEFINITE_PRELAUNCH",
    ],
)
def test_any_nonactive_nonterminal_probe_fails_closed(unsafe):
    status = _status(WAITING=3, RUNNING=0)
    with pytest.raises(SupervisorError, match="operator review"):
        decide_next_action(_probe(unsafe), status)


def test_exact_reviewed_pretrial_can_collect_and_launch_without_consumption():
    status = _status(WAITING=2, RUNNING=0, COMPLETE=1)
    probe = _probe("RECONCILED_TERMINAL")
    probe["launches"].append(_reviewed_pretrial_record())
    assert (
        decide_next_action(
            probe,
            status,
            reviewed_pretrial_worker_run_ids=["reviewed-worker"],
        )
        == "COLLECT_AND_LAUNCH"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update({"retry_safe": False}),
        lambda record: record.update({"db_trials": [{"number": 1}]}),
        lambda record: record.update({"heartbeat": True}),
        lambda record: record.update({"tmux_active": True}),
        lambda record: record["marker_payloads"]["FAILED_PRETRIAL"].update(
            {"parse_ok": False}
        ),
        lambda record: record["marker_payloads"]["FAILED_PRETRIAL"].update(
            {"reservation_consumed": True}
        ),
        lambda record: record["marker_payloads"]["FAILED_PRETRIAL"].update(
            {"trial_number": 1}
        ),
        lambda record: record["marker_payloads"]["FAILED_PRETRIAL"].update(
            {"budget_index": 1}
        ),
    ],
)
def test_inexact_reviewed_pretrial_still_fails_closed(mutation):
    status = _status(WAITING=3, RUNNING=0)
    record = _reviewed_pretrial_record()
    mutation(record)
    probe = _probe()
    probe["launches"].append(record)
    with pytest.raises(SupervisorError, match="zero reservation consumption"):
        decide_next_action(
            probe,
            status,
            reviewed_pretrial_worker_run_ids=["reviewed-worker"],
        )


def test_unknown_reviewed_pretrial_identity_fails_closed():
    status = _status(WAITING=3, RUNNING=0)
    with pytest.raises(SupervisorError, match="absent or no longer pretrial"):
        decide_next_action(
            _probe(),
            status,
            reviewed_pretrial_worker_run_ids=["unknown-worker"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda status: status.update({"test_access": True}),
        lambda status: status["reserved_states"].update({"OTHER": 1, "WAITING": 1}),
        lambda status: status["reserved_states"].update(
            {"UNRESERVED_GUARD": 1, "WAITING": 1}
        ),
        lambda status: status["reserved_states"].update({"RESERVED_TOTAL": 4}),
    ],
)
def test_unsafe_status_fails_closed(mutation):
    status = _status()
    mutation(status)
    with pytest.raises(SupervisorError):
        decide_next_action(_probe("ACTIVE_AMBIGUOUS"), status)


def test_untracked_running_trial_fails_closed():
    with pytest.raises(SupervisorError, match="without an active launch probe"):
        decide_next_action(_probe(), _status())


def test_multi_host_collection_command_is_exact_staged_and_verified(tmp_path):
    collector = tmp_path / "collect.sh"
    collector.write_text("#!/bin/sh\n", encoding="utf-8")
    args = Namespace(
        controller_python=tmp_path / "python",
        controller_script=tmp_path / "controller.py",
        study_name="study",
        output_dir=tmp_path / "output",
        heartbeat_interval=60,
        grace_period=600,
        collector_script=collector,
        repo_paths=tmp_path / "repos.txt",
        remote_run_root="runs/bayesian_optimization/study",
        source_root=None,
    )
    assert _controller_command(args, "collect") == [
        str(collector),
        "--repo-paths",
        str(args.repo_paths),
        "--remote-run-root",
        "runs/bayesian_optimization/study",
        "--exact-destination",
        str(args.output_dir),
        "--verify-manifests",
    ]


def test_run_command_passes_repository_relative_base_config(tmp_path):
    repository_root = tmp_path / "repo"
    args = Namespace(
        controller_python=tmp_path / "python",
        controller_script=tmp_path / "controller.py",
        study_name="study",
        output_dir=tmp_path / "output",
        heartbeat_interval=60,
        grace_period=600,
        base_config=repository_root / "configs" / "search.yaml",
        repository_root=repository_root,
        repo_paths=repository_root / "repos.txt",
        python_paths=repository_root / "pythons.txt",
        slots=repository_root / "slots.txt",
        max_parallel=3,
        credential_env_file=repository_root / "credentials.txt",
    )
    command = _controller_command(args, "run")
    assert command[command.index("--base-config") + 1] == "configs/search.yaml"


def test_run_command_rejects_base_config_outside_repository(tmp_path):
    repository_root = tmp_path / "repo"
    args = Namespace(
        controller_python=tmp_path / "python",
        controller_script=tmp_path / "controller.py",
        study_name="study",
        output_dir=tmp_path / "output",
        heartbeat_interval=60,
        grace_period=600,
        base_config=tmp_path / "outside.yaml",
        repository_root=repository_root,
    )
    with pytest.raises(SupervisorError, match="inside the controller repository"):
        _controller_command(args, "run")


def test_phase_b_supervisor_launch_evidence_is_fail_closed_and_test_free():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (
            root
            / "configs/bayesian_optimization"
            / "lobster_graphcl_f1pr_gate5_phase_b_supervisor_launch.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["implementation"] == {
        "commit": "2b65e5bd91c12a908869c5cd4bb23953f8f53c26",
        "script_sha256": (
            "59e55a00359c8ad82fb39d12888c5bd03bb218c87dd34292a5d7c6faf531d926"
        ),
        "staged_outside_immutable_source_root": True,
        "staged_mode": "0500",
        "focused_tests_passed": 90,
    }
    assert evidence["first_observation"]["reserved_states"] == {
        "RESERVED_TOTAL": 3,
        "WAITING": 2,
        "RUNNING": 1,
        "COMPLETE": 0,
        "FAIL": 0,
        "OTHER": 0,
        "UNRESERVED_GUARD": 0,
    }
    assert evidence["first_observation"]["test_access"] is False
    contract = evidence["automation_contract"]
    assert contract["active_work_is_never_duplicated"] is True
    assert contract["consumed_failure_is_never_replaced"] is True
    assert contract["held_out_or_test_evaluation"] is False
    assert contract["finalize_or_freeze_automatic"] is False
    assert evidence["launch"]["credential_contents_recorded"] is False
