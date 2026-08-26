import pytest

from scripts.supervise_distributed_graphvae_attr_bo import (
    SupervisorError,
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
        "launches": [{"probe_status": status} for status in statuses],
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
