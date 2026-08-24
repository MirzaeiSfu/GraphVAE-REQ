import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import yaml

from scripts.graphvae_attr_bo_distributed import (
    BUDGET_INDEX_ATTR,
    DistributedContractError as GraphDistributedContractError,
    RESERVED_ATTR,
    TRIAL_CONTRACT_ATTR,
    atomic_write_json,
    audit_trial_result,
    build_study_definition,
    canonical_contract_hash,
    sha256_file,
)
from scripts.run_distributed_graphvae_attr_bo import (
    DistributedContractError,
    _assert_prior_launches_reconciled,
    _classify_launch_probe,
    _preflight_inputs,
    _reconcile_terminal_failures_without_results,
    _validate_test_faults,
    command_collect,
)


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[1]
MICRO_PYTHON = Path("/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python")


def _base_config(path):
    payload = {
        "experiment": {"epoch_number": 1, "task": "graphGeneration"},
        "loss": {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return payload


def _definition(name, config, config_path):
    return build_study_definition(
        study_name=name,
        study_uuid=str(uuid.uuid4()),
        base_config=config,
        base_config_sha256=sha256_file(config_path),
        ranges={
            "alpha_node_feat": {"low": 1e-3, "high": 1e2, "log": True},
            "alpha_edge_feat": {"low": 1e-3, "high": 1e2, "log": True},
            "alpha_motif_loss": None,
        },
        reserved_trials=1,
        seeds={
            "study_seed": 0, "split_seed": 123, "training_seed": 0,
            "generation_seed": 123, "evaluator_seed": 0,
        },
        evaluator={
            "mode": "decoded_node_edge", "split": "validation", "test_access": False,
            "repeat_count": 5, "max_graphs": 8, "generation_batch_size": 4,
            "nearest_k": 5, "adjacency_threshold": 0.5,
        },
        training={
            "epoch_number": 1, "training_timeout_seconds": 5,
            "evaluation_timeout_seconds": 5, "mock": True,
        },
        source={},
        environment={},
        dataset_cache={
            "sha256": "cache", "split_fingerprint": "split",
            "expected_validation_graphs": 8,
        },
        feature_schemas={"node_sha256": "node", "edge_sha256": "edge"},
        hardware_policy={"attr_f1pr_abs_tolerance": 0.02},
        heartbeat_interval=60,
        grace_period=600,
        max_parallel=1,
    )


def test_l03_actual_worker_preflight_failure_creates_marker_without_database(tmp_path):
    if not MICRO_PYTHON.is_file():
        pytest.fail(f"Required qualified Python is missing: {MICRO_PYTHON}")
    root = tmp_path / "study"
    root.mkdir()
    config_path = tmp_path / "config.yaml"
    config = _base_config(config_path)
    definition = _definition("preflight-test", config, config_path)
    contract = canonical_contract_hash(definition)
    atomic_write_json(root / "study_definition.json", definition)
    environment = os.environ.copy()
    environment.pop("GRAPHVAE_BO_TEST_UNSET_URL", None)
    result = subprocess.run(
        [
            str(MICRO_PYTHON),
            str(REPO_ROOT / "scripts" / "run_graphvae_attr_bo_worker.py"),
            "--study-name", "preflight-test",
            "--base-config", str(config_path),
            "--artifact-root", str(root),
            "--study-contract-sha256", contract,
            "--worker-id", "local-worker",
            "--worker-run-id", "local-worker-run-1",
            "--sampler-seed", "1",
            "--device", "cpu",
            "--storage-env", "GRAPHVAE_BO_TEST_UNSET_URL",
            "--mock",
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 2
    run_dir = root / "workers" / "local-worker-run-1"
    failure = json.loads((run_dir / "FAILED_PRETRIAL").read_text(encoding="utf-8"))
    assert failure["reservation_consumed"] is False
    assert failure["phase"] == "local_preflight"
    assert not (root / "trials").exists()


@pytest.mark.parametrize(
    "mode,returncode",
    [
        ("success", 0),
        ("training-failure", 21),
        ("evaluation-failure", 22),
        ("malformed-json", 0),
        ("non-finite", 0),
        ("wrong-split", 0),
        ("topology-only", 0),
        ("missing-node", 0),
        ("missing-edge", 0),
        ("post-write-corruption", 0),
    ],
)
def test_l02_fake_objective_supports_required_failures(tmp_path, mode, returncode):
    output = tmp_path / mode
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tests" / "fakes" / "fake_graphvae_attr_trial.py"),
            "--output-dir", str(output), "--mode", mode,
        ],
        cwd=REPO_ROOT,
        timeout=10,
    )
    assert result.returncode == returncode


def _trial_tree(root, content=b"same"):
    trial = root / "trials" / "trial_00000"
    trial.mkdir(parents=True)
    (trial / "trial_result.json").write_bytes(content)


def test_d07_local_collector_is_idempotent_and_quarantines_collision(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _trial_tree(source)
    args = argparse.Namespace(source_root=source, output_dir=destination)
    assert command_collect(args) == 0
    assert command_collect(args) == 0
    (source / "trials" / "trial_00000" / "trial_result.json").write_bytes(b"different")
    with pytest.raises(Exception, match="collision"):
        command_collect(args)
    assert (destination / "trials" / "trial_00000" / "trial_result.json").read_bytes() == b"same"
    assert list((destination / ".collection_conflicts").iterdir())


def test_shell_launchers_parse_and_offer_safe_staging_modes():
    for script in (
        REPO_ROOT / "scripts" / "cluster_distribute_code.sh",
        REPO_ROOT / "scripts" / "cluster_collect_results.sh",
    ):
        subprocess.run(["bash", "-n", str(script)], check=True)
    distribute_help = subprocess.check_output(
        ["bash", str(REPO_ROOT / "scripts" / "cluster_distribute_code.sh"), "--help"],
        text=True,
    )
    collect_help = subprocess.check_output(
        ["bash", str(REPO_ROOT / "scripts" / "cluster_collect_results.sh"), "--help"],
        text=True,
    )
    assert "--bo-cache-manifest" in distribute_help
    assert "--host HOST" in distribute_help
    assert "--local-python PATH" in distribute_help
    assert "--exact-destination" in collect_help
    distribute_source = (
        REPO_ROOT / "scripts" / "cluster_distribute_code.sh"
    ).read_text(encoding="utf-8")
    assert "--exclude .runtime/" in distribute_source


def test_code_distributor_dry_run_can_target_one_host(tmp_path):
    repo_paths = tmp_path / "repo_paths.txt"
    repo_paths.write_text(
        "worker-a /srv/graphvae-a\nworker-b /srv/graphvae-b\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "cluster_distribute_code.sh"),
            "--repo-paths",
            str(repo_paths),
            "--host",
            "worker-a",
            "--code-source",
            str(REPO_ROOT),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert f"[code] {REPO_ROOT} -> worker-a:/srv/graphvae-a" in result.stdout
    assert "worker-b:/srv/graphvae-b" not in result.stdout


def test_code_distributor_rejects_unknown_selected_host(tmp_path):
    repo_paths = tmp_path / "repo_paths.txt"
    repo_paths.write_text("worker-a /srv/graphvae-a\n", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "cluster_distribute_code.sh"),
            "--repo-paths",
            str(repo_paths),
            "--host",
            "missing-worker",
            "--code-source",
            str(REPO_ROOT),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "No repo-path entry found for selected host: missing-worker" in result.stderr


def test_code_distributor_stops_before_transfer_when_manifest_build_fails(tmp_path):
    source = tmp_path / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "graphvae_attr_bo_fingerprints.py").write_text(
        "raise SystemExit(9)\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test User"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    repo_paths = tmp_path / "repo_paths.txt"
    repo_paths.write_text("worker-a /srv/graphvae-a\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "cluster_distribute_code.sh"),
            "--repo-paths",
            str(repo_paths),
            "--code-source",
            str(source),
            "--local-python",
            "/bin/false",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "nothing was transferred" in result.stderr
    assert "-> worker-a:" not in result.stdout


def test_gate4_mappings_select_exactly_one_approved_slot():
    args = argparse.Namespace(
        repo_paths=REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_GATE4_REPO_PATHS.txt",
        python_paths=REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_GATE4_PYTHON_PATHS.txt",
        slots=REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_GATE4_SLOTS.txt",
    )
    repositories, pythons, slots = _preflight_inputs(args)

    assert repositories == {
        "cs-cl-13": "/local-scratch/graphvae-req-work/GraphVAE-REQ-gate4-lobster"
    }
    assert pythons == {
        "cs-cl-13": "/localhome/mirzaei/miniconda3/envs/micro/bin/python"
    }
    assert slots == [
        {
            "host": "cs-cl-13",
            "physical_gpu": 0,
            "worker_id": "cs-cl-13-gate4-gpu0",
        }
    ]


def test_gate5_mappings_select_exactly_two_cross_host_slots():
    args = argparse.Namespace(
        repo_paths=REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_GATE5_REPO_PATHS.txt",
        python_paths=REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_GATE5_PYTHON_PATHS.txt",
        slots=REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_GATE5_SLOTS.txt",
    )
    repositories, pythons, slots = _preflight_inputs(args)

    assert repositories == {
        "cs-cl-13": "/local-scratch/graphvae-req-work/GraphVAE-REQ-gate5-lobster",
        "cs-cl-17": "/local-scratch/graphvae-req-work/GraphVAE-REQ-gate5-lobster",
    }
    assert pythons == {
        "cs-cl-13": "/localhome/mirzaei/miniconda3/envs/micro/bin/python",
        "cs-cl-17": "/localhome/mirzaei/miniconda3/envs/micro/bin/python",
    }
    assert slots == [
        {
            "host": "cs-cl-13",
            "physical_gpu": 0,
            "worker_id": "cs-cl-13-gate5-gpu0",
        },
        {
            "host": "cs-cl-17",
            "physical_gpu": 0,
            "worker_id": "cs-cl-17-gate5-gpu0",
        },
    ]


@pytest.mark.parametrize(
    "launch_state,reachable,tmux_active,markers,db_trials,marker_payloads,expected",
    [
        ("PLANNED", False, False, [], [], {}, ("DEFINITE_PRELAUNCH", True)),
        ("SSH_ERROR", True, True, [], [], {}, ("ACTIVE_AMBIGUOUS", False)),
        (
            "SSH_ERROR",
            True,
            False,
            ["COMPLETED"],
            [
                {
                    "trial_number": 0,
                    "budget_index": 0,
                    "reserved": True,
                    "state": "COMPLETE",
                }
            ],
            {
                "COMPLETED": {
                    "parse_ok": True,
                    "trial_number": 0,
                    "budget_index": 0,
                    "db_state": "COMPLETE",
                }
            },
            ("RECONCILED_TERMINAL", True),
        ),
        (
            "SSH_ACKNOWLEDGED",
            True,
            False,
            ["COMPLETED"],
            [
                {
                    "trial_number": 0,
                    "budget_index": 0,
                    "reserved": True,
                    "state": "FAIL",
                }
            ],
            {
                "COMPLETED": {
                    "parse_ok": True,
                    "trial_number": 0,
                    "budget_index": 0,
                    "db_state": "FAIL",
                }
            },
            ("RECONCILED_TERMINAL", True),
        ),
        (
            "SSH_ERROR",
            True,
            False,
            ["FAILED_PRETRIAL"],
            [],
            {"FAILED_PRETRIAL": {"parse_ok": True, "reservation_consumed": False}},
            ("RECONCILED_PRETRIAL", True),
        ),
        ("SSH_ERROR", False, False, [], [], {}, ("UNREACHABLE_AMBIGUOUS", False)),
        ("SSH_ERROR", True, False, [], [], {}, ("MISSING_AMBIGUOUS", False)),
    ],
)
def test_r06_launch_probe_classification_is_fail_closed(
    launch_state, reachable, tmux_active, markers, db_trials, marker_payloads, expected
):
    assert _classify_launch_probe(
        launch_state=launch_state,
        remote_reachable=reachable,
        tmux_active=tmux_active,
        markers=markers,
        db_trials=db_trials,
        marker_payloads=marker_payloads,
    ) == expected


def test_r06_launch_probe_rejects_terminal_identity_mismatch():
    status = _classify_launch_probe(
        launch_state="SSH_ERROR",
        remote_reachable=True,
        tmux_active=False,
        markers=["COMPLETED"],
        db_trials=[
            {
                "trial_number": 0,
                "budget_index": 0,
                "reserved": True,
                "state": "COMPLETE",
            }
        ],
        marker_payloads={
            "COMPLETED": {
                "parse_ok": True,
                "trial_number": 0,
                "budget_index": 1,
                "db_state": "COMPLETE",
            }
        },
    )
    assert status == ("MISSING_AMBIGUOUS", False)


def test_r06_new_wave_requires_safe_probe_for_attempted_launch(tmp_path):
    launch_root = tmp_path / "launch_manifests"
    launch_root.mkdir()
    worker_run = "worker-a-dispatch-1000000"
    atomic_write_json(
        launch_root / "wave_0001.json",
        {
            "dry_run": False,
            "launches": [
                {"worker_run_id": worker_run, "launch_state": "SSH_ERROR"}
            ],
        },
    )

    with pytest.raises(DistributedContractError, match="safe probe"):
        _assert_prior_launches_reconciled(tmp_path)

    probe_root = tmp_path / "launch_probes"
    atomic_write_json(
        probe_root / "probe_0001.json",
        {
            "launches": [
                {
                    "worker_run_id": worker_run,
                    "probe_status": "ACTIVE_AMBIGUOUS",
                    "retry_safe": False,
                }
            ]
        },
    )
    with pytest.raises(DistributedContractError, match=worker_run):
        _assert_prior_launches_reconciled(tmp_path)

    atomic_write_json(
        probe_root / "probe_0002.json",
        {
            "launches": [
                {
                    "worker_run_id": worker_run,
                    "probe_status": "RECONCILED_TERMINAL",
                    "retry_safe": True,
                }
            ]
        },
    )
    _assert_prior_launches_reconciled(tmp_path)


def test_r06_planned_prelaunch_identity_is_retryable_without_probe(tmp_path):
    launch_root = tmp_path / "launch_manifests"
    launch_root.mkdir()
    atomic_write_json(
        launch_root / "wave_0001.json",
        {
            "dry_run": False,
            "launches": [
                {
                    "worker_run_id": "worker-a-dispatch-1000000",
                    "launch_state": "PLANNED",
                }
            ],
        },
    )
    _assert_prior_launches_reconciled(tmp_path)


def test_r06_launch_fault_injection_requires_explicit_test_environment(monkeypatch):
    args = argparse.Namespace(
        test_inject_definite_prelaunch_host="worker-a",
        test_inject_ambiguous_after_ack_host=None,
    )
    slots = [{"host": "worker-a"}]
    monkeypatch.delenv("GRAPHVAE_BO_ENABLE_TEST_FAULTS", raising=False)
    with pytest.raises(ValueError, match="GRAPHVAE_BO_ENABLE_TEST_FAULTS=1"):
        _validate_test_faults(args, slots)

    monkeypatch.setenv("GRAPHVAE_BO_ENABLE_TEST_FAULTS", "1")
    _validate_test_faults(args, slots)


def test_r07_interrupted_result_is_retained_and_bound_to_failure_tombstone(tmp_path):
    definition = {"schema_version": "test-contract", "study_name": "r07"}
    contract_hash = canonical_contract_hash(definition)
    worker_run = "worker-run"
    trial = SimpleNamespace(
        number=0,
        state=optuna.trial.TrialState.FAIL,
        value=None,
        params={},
        user_attrs={
            RESERVED_ATTR: True,
            BUDGET_INDEX_ATTR: 0,
            TRIAL_CONTRACT_ATTR: contract_hash,
            "worker_id": "worker",
            "worker_run_id": worker_run,
        },
    )

    class Study:
        def get_trials(self, deepcopy=False):
            assert deepcopy is False
            return [trial]

    trial_dir = tmp_path / "trials" / "trial_00000"
    worker_dir = tmp_path / "workers" / worker_run
    worker_dir.mkdir(parents=True)
    atomic_write_json(
        trial_dir / "trial_result.json",
        {
            "schema_version": "graphvae-attr-f1pr-bo-trial-v2",
            "status": "RUNNING",
            "trial_number": 0,
            "budget_index": 0,
            "study_contract_sha256": contract_hash,
            "worker_run_id": worker_run,
            "sampled_weights": {},
            "finished_at_unix": None,
        },
    )

    _reconcile_terminal_failures_without_results(Study(), tmp_path, definition)
    assert not (trial_dir / "trial_result.json").exists()
    interrupted = trial_dir / "trial_result.interrupted.json"
    tombstone_path = trial_dir / "trial_failure_tombstone.json"
    marker_path = worker_dir / "RECONCILED_FAIL"
    assert interrupted.is_file()
    assert marker_path.is_file()
    tombstone = json.loads(tombstone_path.read_text(encoding="utf-8"))
    assert tombstone["failure_category"] == (
        "postgresql_stale_worker_with_interrupted_result"
    )
    assert tombstone["retained_evidence"] == [
        {
            "kind": "interrupted_trial_result",
            "path": "trials/trial_00000/trial_result.interrupted.json",
            "recorded_status": "RUNNING",
            "sha256": sha256_file(interrupted),
        }
    ]
    assert audit_trial_result(
        trial, study_root=tmp_path, definition=definition
    ) == tombstone

    before = tombstone_path.read_bytes()
    _reconcile_terminal_failures_without_results(Study(), tmp_path, definition)
    assert tombstone_path.read_bytes() == before

    interrupted.write_bytes(interrupted.read_bytes() + b"tamper")
    with pytest.raises(GraphDistributedContractError, match="retained evidence"):
        audit_trial_result(trial, study_root=tmp_path, definition=definition)
