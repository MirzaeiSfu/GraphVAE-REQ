import copy
import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pytest
import yaml

from scripts.graphvae_attr_bo_distributed import (
    DistributedContractError,
    LIFECYCLE_ATTR,
    LIFECYCLE_RETIRED_PRECLAIM,
    PLANNED_TRAINING_SEED_ATTR,
    atomic_write_json,
    assert_quiescent_reserved_study,
    audit_trial_result,
    build_study_definition,
    canonical_contract_hash,
    create_postgresql_storage,
    initialize_reserved_study,
    parse_slots,
    redact_secret,
    relative_artifact_path,
    resolve_artifact_path,
    sampler_seed,
    trial_semantic_fingerprint,
    validate_reservation_plan,
    validate_identifier,
)
from scripts.graphvae_attr_bo_fingerprints import (
    array_fingerprint,
    feature_schema_fingerprint,
    feature_schema_payload,
    graph_fingerprint,
    split_fingerprint,
)
from scripts.run_distributed_graphvae_attr_bo import (
    _credential_environment_paths,
    _search_space,
    _validated_mock_child_seconds,
    _validated_mock_hold_seconds,
    _validate_mock_cpu_slots,
    build_hardware_repeatability_report,
    render_remote_launch,
    render_tmux_ssh_command,
    render_worker_command,
    retire_preclaim_study,
    restore_frozen_study,
    _final_outputs,
)
from scripts.run_graphvae_attr_bo_worker import _execution_args, _probe_gpu_identity
from scripts.recover_graphvae_attr_bo_process import (
    _validated_paths as validated_recovery_paths,
)
from scripts.tune_graphvae_attribute_weights import (
    PRIMARY_MODE,
    SearchRanges,
    TrialExecutionError,
    _pid_start_ticks,
    inspect_recorded_process_group,
    inject_sampled_parameters,
    parse_attr_f1pr_payload,
    recover_recorded_process_group,
    run_logged_command,
    execute_trial,
    sample_search_space,
)


pytestmark = pytest.mark.unit


class RecordingTrial:
    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, *, log):
        self.calls.append((name, low, high, log))
        return (low * high) ** 0.5


def evaluator_payload(**overrides):
    payload = {
        "split": "validation",
        "primary_mode": PRIMARY_MODE,
        "generation_seed": 123,
        "evaluator_seed": 0,
        "graph_counts": {
            "accepted_per_collection": 8,
            "generated_accepted": 8,
            "reference_accepted": 8,
        },
        "feature_source": {
            "generated": "GraphVAE node_feature_decoder and edge_feature_decoder",
            "reference": "cached dataset node and edge one-hot attributes",
            "hand_made_topology_features": False,
        },
        "integrity": {
            "cache_sha256": "cache",
            "split_fingerprint": "split",
            "node_schema_fingerprint": "node",
            "edge_schema_fingerprint": "edge",
        },
        "evaluation": {
            "feature_dimensions": {"node": 4, "edge": 3},
            "repeats": 5,
            "modes": {
                PRIMARY_MODE: {
                    "summary": {
                        "f1_pr": {"mean": 0.75},
                        "precision": {"mean": 0.8},
                        "recall": {"mean": 0.7},
                    }
                }
            },
        },
    }
    payload.update(overrides)
    return payload


def strict_parse(payload):
    return parse_attr_f1pr_payload(
        payload,
        expected_split="validation",
        expected_graph_count=8,
        expected_cache_sha256="cache",
        expected_split_fingerprint="split",
        expected_node_schema_fingerprint="node",
        expected_edge_schema_fingerprint="edge",
        expected_generation_seed=123,
        expected_evaluator_seed=0,
        expected_repeats=5,
    )


def minimal_definition(study_name="unit-study"):
    return build_study_definition(
        study_name=study_name,
        base_config={"epoch_number": 2},
        base_config_sha256="base",
        ranges={
            "alpha_node_feat": {"low": 1e-3, "high": 1e2, "log": True},
            "alpha_edge_feat": {"low": 1e-3, "high": 1e2, "log": True},
            "alpha_motif_loss": None,
        },
        reserved_trials=4,
        seeds={"study_seed": 0, "split_seed": 123},
        evaluator={"mode": PRIMARY_MODE, "split": "validation", "test_access": False},
        training={"epoch_number": 2},
        source={"tree_sha256": "source"},
        environment={"sha256": "environment"},
        dataset_cache={"sha256": "cache", "split_fingerprint": "split"},
        feature_schemas={"node_sha256": "node", "edge_sha256": "edge"},
        hardware_policy={"attr_f1pr_abs_tolerance": 0.02},
        heartbeat_interval=60,
        grace_period=600,
        max_parallel=2,
        study_uuid="00000000-0000-0000-0000-000000000001",
    )


def test_u01_only_attribute_weights_change_and_motif_is_opt_in():
    config = {
        "data": {"split": "fixed"},
        "loss": {
            "alpha_node_feat": 1.0,
            "alpha_edge_feat": 1.0,
            "alpha_motif_loss": 0.25,
            "alpha_adj_recon": 7.0,
            "beta": 0.5,
        },
        "experiment": {"epoch_number": 10},
    }
    original = copy.deepcopy(config)
    trial = RecordingTrial()
    sampled = sample_search_space(trial, SearchRanges((1e-3, 1e2), (1e-4, 1e1)))
    resolved = inject_sampled_parameters(config, sampled)
    assert config == original
    assert resolved["data"] == original["data"]
    assert resolved["experiment"] == original["experiment"]
    assert resolved["loss"]["alpha_motif_loss"] == 0.25
    assert resolved["loss"]["alpha_adj_recon"] == 7.0
    assert resolved["loss"]["beta"] == 0.5
    motif = sample_search_space(
        RecordingTrial(), SearchRanges((1e-3, 1e2), (1e-4, 1e1), (1e-2, 1.0))
    )
    assert set(motif) == {"alpha_node_feat", "alpha_edge_feat", "alpha_motif_loss"}


def test_u02_contract_is_canonical_and_every_field_mutation_changes_hash():
    definition = minimal_definition()
    assert canonical_contract_hash(definition) == canonical_contract_hash(copy.deepcopy(definition))
    for key in (
        "objective", "resolved_fixed_configuration", "search_space", "reserved_trials",
        "seeds", "sampler", "evaluator", "training", "source", "environment", "dataset_cache",
        "feature_schemas", "hardware_policy", "storage", "scheduler",
    ):
        changed = copy.deepcopy(definition)
        changed[key] = {"changed": True} if isinstance(changed[key], dict) else 99
        assert canonical_contract_hash(changed) != canonical_contract_hash(definition), key


def test_u03_u04_only_exact_finite_validation_attr_f1pr_is_accepted():
    assert strict_parse(evaluator_payload()).f1_pr == 0.75
    adversarial = []
    wrong_split = evaluator_payload(split="test")
    adversarial.append(wrong_split)
    topology = evaluator_payload(primary_mode="topology_control")
    adversarial.append(topology)
    node_only = evaluator_payload()
    node_only["evaluation"]["feature_dimensions"]["edge"] = 0
    adversarial.append(node_only)
    hand_made = evaluator_payload()
    hand_made["feature_source"]["hand_made_topology_features"] = True
    adversarial.append(hand_made)
    missing_decoder = evaluator_payload()
    missing_decoder["feature_source"]["generated"] = "degree features"
    adversarial.append(missing_decoder)
    nonfinite = evaluator_payload()
    nonfinite["evaluation"]["modes"][PRIMARY_MODE]["summary"]["f1_pr"]["mean"] = float("nan")
    adversarial.append(nonfinite)
    wrong_count = evaluator_payload()
    wrong_count["graph_counts"]["generated_accepted"] = 7
    adversarial.append(wrong_count)
    wrong_integrity = evaluator_payload()
    wrong_integrity["integrity"]["cache_sha256"] = "changed"
    adversarial.append(wrong_integrity)
    for payload in adversarial:
        with pytest.raises(TrialExecutionError):
            strict_parse(payload)


class FakeTrial:
    def __init__(self, state, contract):
        self.number = 0
        self.state = state
        self.value = 0.75
        self.params = {"alpha_node_feat": 2.0, "alpha_edge_feat": 3.0}
        self.user_attrs = {
            "graphvae_bo_reserved": True,
            "budget_index": 0,
            "study_contract_sha256": contract,
            "trial_result": "trials/trial_00000/trial_result.json",
        }


def _auditable_tree(root, definition):
    from optuna.trial import TrialState

    contract = canonical_contract_hash(definition)
    trial = FakeTrial(TrialState.COMPLETE, contract)
    trial_dir = root / "trials" / "trial_00000"
    trial_dir.mkdir(parents=True)
    config = trial_dir / "resolved_config.yaml"
    checkpoint = trial_dir / "checkpoint"
    evaluator = trial_dir / "attributed_random_gin.json"
    config.write_text("epoch_number: 2\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    atomic_write_json(evaluator, evaluator_payload())
    result = {
        "trial_number": 0,
        "budget_index": 0,
        "study_contract_sha256": contract,
        "sampled_weights": dict(trial.params),
        "status": "COMPLETE",
        "validation_attr_f1pr": 0.75,
        "validation_precision": 0.8,
        "validation_recall": 0.7,
        "accepted_validation_graphs": 8,
        "split_seed": 123,
        "generation_seed": 123,
        "evaluator_seed": 0,
        "evaluator_repeats": 5,
        "resolved_config": "trials/trial_00000/resolved_config.yaml",
        "resolved_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint": "trials/trial_00000/checkpoint",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluator_output": "trials/trial_00000/attributed_random_gin.json",
        "evaluator_output_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
        "hashes": {
            "cache_sha256": "cache",
            "split_fingerprint": "split",
            "node_schema_fingerprint": "node",
            "edge_schema_fingerprint": "edge",
            "source_tree_sha256": "source",
            "environment_sha256": "environment",
        },
    }
    atomic_write_json(trial_dir / "trial_result.json", result)
    return trial, result


def test_u05_every_trial_identity_and_integrity_mismatch_fails_audit(tmp_path):
    definition = minimal_definition()
    definition["dataset_cache"]["expected_validation_graphs"] = 8
    definition["seeds"].update({"generation_seed": 123, "evaluator_seed": 0})
    definition["evaluator"]["repeat_count"] = 5
    trial, original = _auditable_tree(tmp_path, definition)
    assert audit_trial_result(trial, study_root=tmp_path, definition=definition)[
        "validation_attr_f1pr"
    ] == 0.75
    path = tmp_path / trial.user_attrs["trial_result"]
    mutations = {
        "trial_number": 1,
        "budget_index": 1,
        "study_contract_sha256": "bad",
        "sampled_weights": {"alpha_node_feat": 9.0},
    }
    for field, value in mutations.items():
        changed = copy.deepcopy(original)
        changed[field] = value
        atomic_write_json(path, changed)
        with pytest.raises(DistributedContractError):
            audit_trial_result(trial, study_root=tmp_path, definition=definition)
    missing_gpu_identity = copy.deepcopy(original)
    missing_gpu_identity.update(
        {"physical_gpu": 0, "gpu_model": None, "gpu_vram_bytes": None}
    )
    atomic_write_json(path, missing_gpu_identity)
    with pytest.raises(DistributedContractError, match="GPU trial result"):
        audit_trial_result(trial, study_root=tmp_path, definition=definition)
    atomic_write_json(path, original)
    checkpoint = tmp_path / original["checkpoint"]
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(DistributedContractError, match="hash"):
        audit_trial_result(trial, study_root=tmp_path, definition=definition)


def test_u06_collected_trial_tree_is_portable_after_root_move(tmp_path):
    definition = minimal_definition()
    definition["dataset_cache"]["expected_validation_graphs"] = 8
    definition["seeds"].update({"generation_seed": 123, "evaluator_seed": 0})
    definition["evaluator"]["repeat_count"] = 5
    first_root = tmp_path / "first"
    trial, _result = _auditable_tree(first_root, definition)
    second_root = tmp_path / "collected"
    first_root.rename(second_root)
    assert audit_trial_result(trial, study_root=second_root, definition=definition)[
        "status"
    ] == "COMPLETE"


def test_audit_enforces_planned_parameters_and_training_seed(tmp_path):
    definition = minimal_definition("planned-audit")
    definition["dataset_cache"]["expected_validation_graphs"] = 8
    definition["seeds"].update(
        {"training_seed": 99, "generation_seed": 123, "evaluator_seed": 0}
    )
    definition["evaluator"]["repeat_count"] = 5
    definition["reservation_plan"] = [
        {
            "budget_index": 0,
            "parameters": {"alpha_node_feat": 2.0, "alpha_edge_feat": 3.0},
            "training_seed": 7,
        },
        *[
            {"budget_index": index, "parameters": {}, "training_seed": index + 7}
            for index in range(1, 4)
        ],
    ]
    trial, result = _auditable_tree(tmp_path, definition)
    trial.user_attrs[PLANNED_TRAINING_SEED_ATTR] = 7
    result["training_seed"] = 7
    result_path = tmp_path / trial.user_attrs["trial_result"]
    atomic_write_json(result_path, result)
    assert audit_trial_result(trial, study_root=tmp_path, definition=definition)[
        "training_seed"
    ] == 7

    trial.user_attrs[PLANNED_TRAINING_SEED_ATTR] = 8
    with pytest.raises(DistributedContractError, match="immutable reservation plan"):
        audit_trial_result(trial, study_root=tmp_path, definition=definition)
    trial.user_attrs[PLANNED_TRAINING_SEED_ATTR] = 7
    trial.params["alpha_node_feat"] = 9.0
    result["sampled_weights"] = dict(trial.params)
    atomic_write_json(result_path, result)
    with pytest.raises(DistributedContractError, match="parameter alpha_node_feat"):
        audit_trial_result(trial, study_root=tmp_path, definition=definition)


def test_u07_atomic_json_never_publishes_partial_file(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    atomic_write_json(path, {"old": True})
    old = path.read_bytes()

    def interrupted(_source, _destination):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_json(path, {"new": True})
    assert path.read_bytes() == old
    assert not list(tmp_path.glob("*.tmp"))


def test_u08_paths_and_identifiers_are_confined(tmp_path):
    artifact = tmp_path / "trials" / "trial_00000" / "result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    relative = relative_artifact_path(tmp_path, artifact)
    assert resolve_artifact_path(tmp_path, relative) == artifact
    for unsafe in ("../study", "a/b", "$(touch pwned)", "semi;colon", ""):
        with pytest.raises(ValueError):
            validate_identifier(unsafe, "test")
    with pytest.raises(DistributedContractError):
        resolve_artifact_path(tmp_path, "../escape")


def test_u09_storage_rejection_and_credential_redaction():
    with pytest.raises(ValueError, match="PostgreSQL"):
        create_postgresql_storage("sqlite:///tmp/study.sqlite3")
    sentinel = "sentinel-secret-password"
    url = f"postgresql+psycopg2://sentinel-user:{sentinel}@localhost/db"
    rendered = redact_secret(f"connection failed: {url} and {sentinel}", storage_url=url)
    assert sentinel not in rendered
    assert url not in rendered


def test_u10_slot_validation_fails_before_external_access(tmp_path):
    valid = tmp_path / "valid.txt"
    valid.write_text("host-a 0 worker-a\nhost-b 1 worker-b\n", encoding="utf-8")
    assert len(parse_slots(valid, known_hosts=["host-a", "host-b"])) == 2
    bad_rows = (
        "host-a 0 worker-a\nhost-a 0 worker-b\n",
        "host-a 0 worker-a\nhost-b 1 worker-a\n",
        "unknown 0 worker-a\n",
        "host-a nope worker-a\n",
        "host-a 0\n",
    )
    for index, content in enumerate(bad_rows):
        path = tmp_path / f"bad-{index}.txt"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            parse_slots(path, known_hosts=["host-a", "host-b"])


def test_mock_cpu_slots_allow_concurrent_lifecycle_workers_on_one_host(tmp_path):
    path = tmp_path / "mock-cpu-slots.txt"
    path.write_text(
        "host-a mock-cpu worker-a\nhost-a mock-cpu worker-b\nhost-a mock-cpu worker-c\n",
        encoding="utf-8",
    )

    slots = parse_slots(path, known_hosts=["host-a"])

    assert [slot["worker_id"] for slot in slots] == ["worker-a", "worker-b", "worker-c"]
    assert all(slot["physical_gpu"] is None for slot in slots)


def test_mock_cpu_slots_are_fail_closed_for_real_studies():
    definition = minimal_definition()
    slots = [{"host": "host-a", "physical_gpu": None, "worker_id": "worker-a"}]

    with pytest.raises(RuntimeError, match="forbidden for real studies"):
        _validate_mock_cpu_slots(definition, slots)

    definition["training"]["mock"] = True
    _validate_mock_cpu_slots(definition, slots)


def test_mock_hold_is_bounded_and_forbidden_for_real_studies():
    assert _validated_mock_hold_seconds(
        argparse.Namespace(mock=True, mock_hold_seconds=2.0)
    ) == 2.0
    for value in (-0.01, 30.01):
        with pytest.raises(ValueError, match="between 0 and 30"):
            _validated_mock_hold_seconds(
                argparse.Namespace(mock=True, mock_hold_seconds=value)
            )
    with pytest.raises(ValueError, match="forbidden for real studies"):
        _validated_mock_hold_seconds(
            argparse.Namespace(mock=False, mock_hold_seconds=2.0)
        )


def test_mock_child_is_bounded_and_forbidden_for_real_studies():
    assert _validated_mock_child_seconds(
        argparse.Namespace(mock=True, mock_child_seconds=120.0)
    ) == 120.0
    for value in (-0.01, 300.01):
        with pytest.raises(ValueError, match="between 0 and 300"):
            _validated_mock_child_seconds(
                argparse.Namespace(mock=True, mock_child_seconds=value)
            )
    with pytest.raises(ValueError, match="forbidden for real studies"):
        _validated_mock_child_seconds(
            argparse.Namespace(mock=False, mock_child_seconds=1.0)
        )


def test_u11_finalization_refuses_waiting_or_running_reservations():
    from optuna.trial import TrialState

    definition = minimal_definition()
    definition["reserved_trials"] = 1
    contract = canonical_contract_hash(definition)

    class FakeStudy:
        user_attrs = {
            "graphvae_bo_study_definition": definition,
            "graphvae_bo_study_contract_sha256": contract,
            "graphvae_bo_lifecycle": "READY",
        }

        def __init__(self, state):
            trial = FakeTrial(state, contract)
            trial.params = {}
            trial.value = None
            self.trial = trial

        def get_trials(self, deepcopy=False):
            return [self.trial]

    for state in (TrialState.WAITING, TrialState.RUNNING):
        with pytest.raises(DistributedContractError, match="WAITING/RUNNING"):
            assert_quiescent_reserved_study(FakeStudy(state))


def test_u12_optimization_helpers_never_access_a_test_object():
    class Poison(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("held-out test data was accessed")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key == "test":
                raise AssertionError("held-out test data was accessed")
            return super().get(key, default)

    config = Poison({"loss": {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0}})
    resolved = inject_sampled_parameters(
        config, {"alpha_node_feat": 2.0, "alpha_edge_feat": 3.0}
    )
    assert resolved["loss"] == {"alpha_node_feat": 2.0, "alpha_edge_feat": 3.0}


def test_u13_sampler_seed_matches_specified_sha256_formula():
    expected = int.from_bytes(
        hashlib.sha256(b"graphvae-attr-f1pr-sampler-v1\x000\x0017").digest()[:4], "big"
    )
    assert sampler_seed(0, 17) == expected
    assert sampler_seed(0, 17) == sampler_seed(0, 17)
    assert len({sampler_seed(0, index) for index in range(20)}) == 20


def test_fingerprint_framing_order_and_schema_meanings_are_sensitive():
    array = np.arange(6, dtype=np.int32).reshape(2, 3)
    assert array_fingerprint(array) == array_fingerprint(array.copy(order="F"))
    assert array_fingerprint(array) != array_fingerprint(array.astype(np.int64))
    graph_a = graph_fingerprint(array, np.ones((2, 2)), np.ones((1, 2, 2)))
    graph_b = graph_fingerprint(array, np.zeros((2, 2)), np.ones((1, 2, 2)))
    assert graph_a != graph_b
    assert split_fingerprint([graph_a, graph_b]) != split_fingerprint([graph_b, graph_a])
    first = feature_schema_payload(
        {0: {"feature_name": "atom", "value": "C"}}, total_dimension=1, dtype="float32"
    )
    second = copy.deepcopy(first)
    second["channels"][0]["meaning"] = "N"
    assert feature_schema_fingerprint(first) != feature_schema_fingerprint(second)


def test_launcher_uses_physical_visibility_logical_cuda_and_no_secret_or_test():
    command = render_worker_command(
        python_path="/env/bin/python",
        repo_path="/repo",
        study_name="study",
        base_config="/repo/config.yaml",
        artifact_root="/repo/runs/study",
        contract_hash="a" * 64,
        worker_id="host-gpu1",
        worker_run_id_value="host-gpu1-dispatch-000001",
        sampler_seed_value=7,
        dispatch_sequence=1,
        physical_gpu=3,
        storage_env="GRAPHVAE_BO_STORAGE_URL",
        heartbeat_interval=60,
        grace_period=600,
        mock=False,
    )
    rendered = render_remote_launch(command, physical_gpu=3, lock_path="/tmp/slot.lock")
    assert "CUDA_VISIBLE_DEVICES=3" in rendered
    assert "--device cuda:0" in rendered
    assert "GRAPHVAE_BO_STORAGE_URL" in rendered
    assert "password" not in rendered.lower()
    assert "--split test" not in rendered


def test_mock_cpu_launcher_omits_physical_gpu_and_cuda_visibility():
    command = render_worker_command(
        python_path="/env/bin/python",
        repo_path="/repo",
        study_name="mock-study",
        base_config="/repo/config.yaml",
        artifact_root="/repo/runs/mock-study",
        contract_hash="a" * 64,
        worker_id="host-cpu0",
        worker_run_id_value="host-cpu0-dispatch-000001",
        sampler_seed_value=7,
        dispatch_sequence=1,
        physical_gpu=None,
        storage_env="GRAPHVAE_BO_STORAGE_URL",
        heartbeat_interval=60,
        grace_period=600,
        mock=True,
    )
    rendered = render_remote_launch(command, physical_gpu=None, lock_path="/tmp/slot.lock")

    assert "CUDA_VISIBLE_DEVICES" not in rendered
    assert "--physical-gpu" not in rendered
    assert "--device cpu" in rendered
    assert "--mock" in rendered


def test_launcher_sources_only_protected_credential_file_path():
    command = [
        "/env/bin/python",
        "/repo/scripts/run_graphvae_attr_bo_worker.py",
        "--storage-env",
        "GRAPHVAE_BO_STORAGE_URL",
    ]
    credential_path = "/local-scratch/graphvae-bo-credentials/gate4/worker.env"
    rendered = render_remote_launch(
        command,
        physical_gpu=0,
        lock_path="/tmp/slot.lock",
        credential_env_file=credential_path,
    )

    assert rendered.startswith(f"set -a; . {credential_path}; set +a; ")
    assert "GRAPHVAE_BO_STORAGE_URL" in rendered
    assert "postgresql" not in rendered.lower()
    assert "password" not in rendered.lower()
    with pytest.raises(ValueError, match="must be absolute"):
        render_remote_launch(
            command,
            physical_gpu=0,
            lock_path="/tmp/slot.lock",
            credential_env_file="relative/worker.env",
        )


def test_launcher_preserves_worker_shell_as_one_tmux_argument():
    worker_shell = (
        "set -a; . /protected/worker.env; set +a; "
        "CUDA_VISIBLE_DEVICES=0 flock -n /tmp/slot.lock /env/bin/python worker.py"
    )
    command = render_tmux_ssh_command(
        host="cs-cl-13",
        tmux_name="graphvae-bo-worker-run",
        remote_shell=worker_shell,
    )

    assert command[:3] == ["ssh", "-n", "cs-cl-13"]
    assert shlex.split(command[3]) == [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "graphvae-bo-worker-run",
        worker_shell,
    ]


def test_gpu_probe_records_model_and_vram_bytes(monkeypatch):
    def fake_run(command, **kwargs):
        assert command[-2:] == ["-i", "3"]
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 10,
        }
        return subprocess.CompletedProcess(
            command, 0, stdout="NVIDIA TITAN RTX, 24576\n"
        )

    monkeypatch.setattr(
        "scripts.run_graphvae_attr_bo_worker.subprocess.run", fake_run
    )
    assert _probe_gpu_identity(3) == (
        "NVIDIA TITAN RTX",
        24576 * 1024 * 1024,
    )


def test_l06_timeout_kills_exact_child_group_and_unrelated_process_survives(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(TrialExecutionError, match="timeout"):
            run_logged_command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                log_path=tmp_path / "timeout.log",
                environment=os.environ.copy(),
                timeout_seconds=0.2,
                termination_grace_seconds=0.1,
            )
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_r07_recorded_process_recovery_checks_identity_and_spares_unrelated(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    target = subprocess.Popen(command, cwd=tmp_path, start_new_session=True)
    unrelated = subprocess.Popen(command, cwd=tmp_path, start_new_session=True)
    identity_path = tmp_path / "training_subprocess.log.process.json"
    contract_hash = "c" * 64
    atomic_write_json(
        identity_path,
        {
            "schema_version": "graphvae-attr-f1pr-process-v1",
            "pid": target.pid,
            "process_group_id": target.pid,
            "pid_start_ticks": _pid_start_ticks(target.pid),
            "command": command,
            "cwd": str(tmp_path.resolve()),
            "study_contract_sha256": contract_hash,
            "worker_run_id": "worker-run",
            "trial_number": 0,
            "phase": "training",
        },
    )
    try:
        inspected = inspect_recorded_process_group(
            identity_path,
            expected_cwd=tmp_path,
            expected_study_contract_sha256=contract_hash,
            expected_worker_run_id="worker-run",
            expected_trial_number=0,
            expected_phase="training",
        )
        assert inspected["status"] == "MATCHING_LIVE"
        assert inspected["process_group_id"] == target.pid

        with pytest.raises(TrialExecutionError, match="identity contract"):
            recover_recorded_process_group(
                identity_path,
                expected_cwd=tmp_path,
                expected_study_contract_sha256=contract_hash,
                expected_worker_run_id="different-worker-run",
                expected_trial_number=0,
                expected_phase="training",
                grace_seconds=0.1,
            )
        assert target.poll() is None

        assert recover_recorded_process_group(
            identity_path,
            expected_cwd=tmp_path,
            expected_study_contract_sha256=contract_hash,
            expected_worker_run_id="worker-run",
            expected_trial_number=0,
            expected_phase="training",
            grace_seconds=0.1,
        )
        target.wait(timeout=5)
        assert unrelated.poll() is None
        assert inspect_recorded_process_group(
            identity_path,
            expected_cwd=tmp_path,
            expected_study_contract_sha256=contract_hash,
            expected_worker_run_id="worker-run",
            expected_trial_number=0,
            expected_phase="training",
        )["status"] == "ABSENT"
    finally:
        if target.poll() is None:
            os.killpg(target.pid, signal.SIGKILL)
            target.wait(timeout=5)
        if unrelated.poll() is None:
            os.killpg(unrelated.pid, signal.SIGKILL)
            unrelated.wait(timeout=5)


def test_r07_grouped_recovery_selects_only_the_contracted_seed_identity(tmp_path):
    study_root = tmp_path / "grouped-recovery"
    study_root.mkdir()
    definition = {
        "study_name": study_root.name,
        "objective": {
            "json_path": "evaluation.modes.decoded_node_edge.summary.f1_pr.mean",
            "split": "validation",
            "test_access": False,
        },
        "evaluator": {"backend": "graphcl_f1pr"},
        "seeds": {"training_seeds": [0, 1]},
    }
    atomic_write_json(study_root / "study_definition.json", definition)
    contract_hash = canonical_contract_hash(definition)
    args = argparse.Namespace(
        repo_root=tmp_path,
        study_root=study_root,
        worker_run_id="worker-run",
        trial_number=0,
        training_seed=0,
        phase="training",
        study_contract_sha256=contract_hash,
        grace_seconds=1.0,
        output=None,
    )

    _repo, _study, identity, output = validated_recovery_paths(args)

    assert identity == (
        study_root
        / "trials"
        / "trial_00000"
        / "replicates"
        / "seed_0"
        / "training_subprocess.log.process.json"
    )
    assert output == study_root / "workers" / "worker-run" / "PROCESS_RECOVERY.json"
    args.training_seed = 2
    with pytest.raises(RuntimeError, match="contracted training seed"):
        validated_recovery_paths(args)


def test_l05_fixed_mock_parameters_and_seeds_are_reproducible(tmp_path, monkeypatch):
    class FixedTrial:
        number = 0

        def __init__(self):
            self.user_attrs = {"budget_index": 0}

        def suggest_float(self, name, low, high, *, log):
            return {"alpha_node_feat": 2.0, "alpha_edge_feat": 3.0}[name]

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    args = argparse.Namespace(
        distributed=True,
        study_contract_sha256="c" * 64,
        budget_index=0,
        worker_id="worker",
        worker_run_id="run",
        hostname="host",
        physical_gpu=None,
        gpu_model=None,
        gpu_vram_bytes=None,
        dispatch_sequence=0,
        sampler_constant_liar=True,
        sampler_seed=1,
        tpe_startup_trials=5,
        optuna_version="4.2.1",
        db_driver_version="2.9.10",
        training_seed=0,
        generation_seed=123,
        evaluator_seed=0,
        evaluator_repeats=5,
        max_graphs=8,
        generation_batch_size=4,
        nearest_k=5,
        adjacency_threshold=0.5,
        device="cpu",
        python_bin=sys.executable,
        training_timeout=5,
        evaluation_timeout=5,
        process_termination_grace=1,
        mock=True,
        mock_hold_seconds=0.25,
        mock_child_seconds=0.5,
        mock_fail_trial=[],
        expected_validation_graph_count=8,
        expected_node_feature_dimension=14,
        expected_edge_feature_dimension=11,
        integrity={
            "cache_sha256": "cache",
            "split_fingerprint": "split",
            "node_schema_fingerprint": "node",
            "edge_schema_fingerprint": "edge",
        },
    )
    base = {"experiment": {"epoch_number": 1}, "loss": {}}
    held = []
    monkeypatch.setattr(
        "scripts.tune_graphvae_attribute_weights.time.sleep", held.append
    )
    child_calls = []

    def fake_child(command, **kwargs):
        child_calls.append((command, kwargs))
        return 0.5

    monkeypatch.setattr(
        "scripts.tune_graphvae_attribute_weights.run_logged_command", fake_child
    )
    records = []
    configs = []
    for name in ("first", "second"):
        root = tmp_path / name
        execute_trial(
            FixedTrial(),
            args=args,
            base_config=base,
            ranges=SearchRanges((1e-3, 1e2), (1e-3, 1e2)),
            output_dir=root,
            split_seed=123,
        )
        record = json.loads(
            (root / "trials" / "trial_00000" / "trial_result.json").read_text(
                encoding="utf-8"
            )
        )
        for key in (
            "started_at_unix", "finished_at_unix", "training_elapsed_seconds",
            "evaluation_elapsed_seconds", "total_elapsed_seconds",
            "host_local_trial_directory", "hostname", "physical_gpu", "gpu_model",
            "gpu_vram_bytes", "worker_run_id",
        ):
            record.pop(key, None)
        records.append(record)
        evaluator = json.loads(
            (
                root
                / "trials"
                / "trial_00000"
                / "validation_evaluation"
                / "attributed_random_gin.json"
            ).read_text(encoding="utf-8")
        )
        assert evaluator["evaluation"]["feature_dimensions"] == {
            "node": 14,
            "edge": 11,
        }
        assert evaluator["evaluation"]["actual_decoder_output_dimensions"] == {
            "node": 14,
            "edge": 11,
        }
        config = yaml.safe_load(
            (root / "trials" / "trial_00000" / "resolved_config.yaml").read_text(
                encoding="utf-8"
            )
        )
        config["runtime"]["graph_save_path"] = "<host-local-path>"
        configs.append(config)
    assert records[0] == records[1]
    assert configs[0] == configs[1]
    assert held == [0.25, 0.25]
    assert len(child_calls) == 2
    assert all("time.sleep(0.5)" in call[0][2] for call in child_calls)
    assert all(
        set(call[1]["environment"]) == {"LANG", "PATH"} for call in child_calls
    )


def test_r08_fixed_parameters_are_contracted_and_enqueued_exactly(tmp_path):
    args = argparse.Namespace(
        alpha_node_feat_min=1e-3,
        alpha_node_feat_max=1e2,
        alpha_edge_feat_min=1e-3,
        alpha_edge_feat_max=1e2,
        tune_alpha_motif=False,
        alpha_motif_min=1e-3,
        alpha_motif_max=1e2,
        fixed_alpha_node_feat=2.0,
        fixed_alpha_edge_feat=3.0,
    )
    search_space = _search_space(args)
    assert search_space["fixed_parameters"] == {
        "alpha_node_feat": 2.0,
        "alpha_edge_feat": 3.0,
    }
    definition = minimal_definition("fixed-r08")
    definition["reserved_trials"] = 2
    definition["search_space"] = search_space
    study = optuna.create_study(study_name="fixed-r08", direction="maximize")
    initialize_reserved_study(
        study,
        definition,
        controller_uuid="controller",
        output_root=tmp_path,
    )
    trials = study.get_trials(deepcopy=False)
    assert len(trials) == 2
    assert all(
        trial.system_attrs["fixed_params"] == search_space["fixed_parameters"]
        for trial in trials
    )

    args.fixed_alpha_edge_feat = None
    with pytest.raises(ValueError, match="requires both"):
        _search_space(args)
    args.fixed_alpha_edge_feat = 3.0
    args.fixed_alpha_node_feat = 1e4
    with pytest.raises(ValueError, match="must be finite and within"):
        _search_space(args)


def test_mixed_reservation_plan_enqueues_exact_parameters_and_seeds(tmp_path):
    definition = minimal_definition("mixed-plan")
    definition["reserved_trials"] = 3
    definition["reservation_plan"] = [
        {
            "budget_index": 0,
            "parameters": {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0},
            "training_seed": 0,
        },
        {"budget_index": 1, "parameters": {}, "training_seed": 0},
        {
            "budget_index": 2,
            "parameters": {"alpha_node_feat": 0.25, "alpha_edge_feat": 4.0},
            "training_seed": 2,
        },
    ]
    study = optuna.create_study(study_name="mixed-plan", direction="maximize")
    initialize_reserved_study(
        study,
        definition,
        controller_uuid="controller",
        output_root=tmp_path,
    )
    trials = study.get_trials(deepcopy=False)
    assert [trial.system_attrs["fixed_params"] for trial in trials] == [
        {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0},
        {},
        {"alpha_node_feat": 0.25, "alpha_edge_feat": 4.0},
    ]
    assert [
        trial.user_attrs[PLANNED_TRAINING_SEED_ATTR] for trial in trials
    ] == [0, 0, 2]


def test_per_host_credential_environment_paths_are_exact_and_fail_closed(tmp_path):
    repositories = {"worker-a": "/repo/a", "worker-b": "/repo/b"}
    mapping = tmp_path / "credential_paths.txt"
    mapping.write_text(
        "worker-a /protected/a.env\nworker-b /protected/b.env\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        credential_env_file=None,
        credential_env_paths=mapping,
    )
    assert _credential_environment_paths(args, repositories, required=True) == {
        "worker-a": "/protected/a.env",
        "worker-b": "/protected/b.env",
    }

    args.credential_env_file = "/protected/all.env"
    with pytest.raises(ValueError, match="either"):
        _credential_environment_paths(args, repositories, required=True)
    args.credential_env_file = None
    mapping.write_text("worker-a /protected/a.env\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hosts differ"):
        _credential_environment_paths(args, repositories, required=True)
    mapping.write_text(
        "worker-a relative.env\nworker-b /protected/b.env\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="absolute and safe"):
        _credential_environment_paths(args, repositories, required=True)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda plan: plan.pop(), "exactly one entry"),
        (lambda plan: plan[1].update(budget_index=0), "unique and exactly cover"),
        (
            lambda plan: plan[0]["parameters"].update(alpha_node_feat=1e9),
            "outside its contracted range",
        ),
        (lambda plan: plan[0].update(training_seed=-1), r"in \[0, 2\^32-1\]"),
    ],
)
def test_reservation_plan_fails_closed(mutation, message):
    definition = minimal_definition("bad-plan")
    plan = [
        {"budget_index": index, "parameters": {}, "training_seed": 0}
        for index in range(4)
    ]
    mutation(plan)
    with pytest.raises(DistributedContractError, match=message):
        validate_reservation_plan(
            plan,
            expected_count=4,
            search_space=definition["search_space"],
        )


def test_worker_execution_uses_planned_training_seed():
    definition = minimal_definition("planned-worker-seed")
    definition["seeds"].update(
        {
            "training_seed": 99,
            "generation_seed": 123,
            "evaluator_seed": 0,
        }
    )
    definition["reservation_plan"] = [
        {"budget_index": index, "parameters": {}, "training_seed": index + 7}
        for index in range(4)
    ]
    cli = argparse.Namespace(
        study_contract_sha256="contract",
        worker_id="worker",
        worker_run_id="run",
        physical_gpu=0,
        gpu_model="GPU",
        gpu_vram_bytes=1,
        dispatch_sequence=1,
        sampler_seed=2,
        tpe_startup_trials=5,
        device="cuda:0",
        python_bin="python",
        mock=True,
        mock_fail_trial=[],
        storage_env="GRAPHVAE_BO_STORAGE_URL",
    )
    assert _execution_args(cli, definition, budget_index=2).training_seed == 9


def test_preclaim_retirement_requires_safe_probe_and_consumes_no_reservation(tmp_path):
    definition = minimal_definition("retire-preclaim")
    definition["reserved_trials"] = 1
    study = optuna.create_study(study_name="retire-preclaim", direction="maximize")
    contract = initialize_reserved_study(
        study,
        definition,
        controller_uuid="controller",
        output_root=tmp_path,
    )
    launch = {
        "dry_run": False,
        "launches": [
            {
                "launch_state": "SSH_ACKNOWLEDGED",
                "worker_run_id": "worker-dispatch-1000000",
            }
        ],
    }
    atomic_write_json(tmp_path / "launch_manifests" / "wave_0001.json", launch)
    with pytest.raises(RuntimeError, match="probe before"):
        retire_preclaim_study(
            study,
            tmp_path,
            contract_hash=contract,
            reason_code="source-contract-superseded",
        )
    probe = {
        "launches": [
            {
                "worker_run_id": "worker-dispatch-1000000",
                "probe_status": "RECONCILED_PRETRIAL",
                "retry_safe": True,
                "tmux_active": False,
                "db_trials": [],
            }
        ]
    }
    atomic_write_json(tmp_path / "launch_probes" / "probe_0001.json", probe)
    marker = retire_preclaim_study(
        study,
        tmp_path,
        contract_hash=contract,
        reason_code="source-contract-superseded",
    )
    assert marker["reservation_consumed"] is False
    assert marker["reserved_waiting"] == 1
    assert study.user_attrs[LIFECYCLE_ATTR] == LIFECYCLE_RETIRED_PRECLAIM
    trial = study.get_trials(deepcopy=False)[0]
    assert trial.state.name == "WAITING"
    with pytest.raises(DistributedContractError, match="cannot be initialized"):
        initialize_reserved_study(
            study,
            definition,
            controller_uuid="controller",
            output_root=tmp_path,
        )
    assert retire_preclaim_study(
        study,
        tmp_path,
        contract_hash=contract,
        reason_code="source-contract-superseded",
    ) == marker


def test_r09_clean_snapshot_restore_regenerates_identical_aggregates(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "frozen"
    source.mkdir()
    definition = minimal_definition("restore-r09")
    definition["reserved_trials"] = 1
    definition["dataset_cache"]["expected_validation_graphs"] = 8
    definition["seeds"].update(
        {
            "training_seed": 0,
            "generation_seed": 123,
            "evaluator_seed": 0,
        }
    )
    definition["evaluator"]["repeat_count"] = 5
    contract = canonical_contract_hash(definition)
    atomic_write_json(source / "study_definition.json", definition)
    snapshot_path = source / "study_snapshot.sqlite3"
    study = optuna.create_study(
        study_name="restore-r09",
        direction="maximize",
        storage="sqlite:///" + snapshot_path.as_posix(),
    )
    initialize_reserved_study(
        study,
        definition,
        controller_uuid="controller",
        output_root=source,
    )
    claimed = study.ask()
    sampled = {
        "alpha_node_feat": claimed.suggest_float("alpha_node_feat", 1e-3, 1e2, log=True),
        "alpha_edge_feat": claimed.suggest_float("alpha_edge_feat", 1e-3, 1e2, log=True),
    }
    trial_dir = source / "trials" / "trial_00000"
    trial_dir.mkdir(parents=True)
    config = trial_dir / "resolved_config.yaml"
    checkpoint = trial_dir / "checkpoint"
    evaluator = trial_dir / "attributed_random_gin.json"
    config.write_text("epoch_number: 2\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    atomic_write_json(evaluator, evaluator_payload())
    result = {
        "trial_number": 0,
        "budget_index": 0,
        "study_contract_sha256": contract,
        "sampled_weights": sampled,
        "status": "COMPLETE",
        "validation_attr_f1pr": 0.75,
        "validation_precision": 0.8,
        "validation_recall": 0.7,
        "accepted_validation_graphs": 8,
        "training_seed": 0,
        "split_seed": 123,
        "generation_seed": 123,
        "evaluator_seed": 0,
        "evaluator_repeats": 5,
        "resolved_config": "trials/trial_00000/resolved_config.yaml",
        "resolved_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint": "trials/trial_00000/checkpoint",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluator_output": "trials/trial_00000/attributed_random_gin.json",
        "evaluator_output_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
        "hashes": {
            "cache_sha256": "cache",
            "split_fingerprint": "split",
            "node_schema_fingerprint": "node",
            "edge_schema_fingerprint": "edge",
            "source_tree_sha256": "source",
            "environment_sha256": "environment",
        },
    }
    atomic_write_json(trial_dir / "trial_result.json", result)
    for key, value in {
        "trial_result": "trials/trial_00000/trial_result.json",
        "validation_precision": 0.8,
        "validation_recall": 0.7,
        "accepted_validation_graphs": 8,
    }.items():
        claimed.set_user_attr(key, value)
    study.tell(claimed, 0.75)
    study.set_user_attr(LIFECYCLE_ATTR, "FROZEN")
    atomic_write_json(
        source / "FROZEN.json",
        {
            "study_name": "restore-r09",
            "study_contract_sha256": contract,
            "lifecycle": "FROZEN",
            "snapshot": snapshot_path.name,
            "best_trial_number": 0,
        },
    )
    _final_outputs(study, definition, source)
    monkeypatch.setattr(
        "scripts.run_distributed_graphvae_attr_bo.runtime_dependency_fingerprint",
        lambda: {"sha256": "environment"},
    )
    source_snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    source_semantic = trial_semantic_fingerprint(study)
    destination = tmp_path / "restored"
    report = restore_frozen_study(
        source,
        destination,
        study_name="restore-r09",
    )
    assert report["postgresql_access"] is False
    assert report["test_access"] is False
    assert report["semantic_fingerprint"] == source_semantic
    assert report["snapshot_sha256"] == source_snapshot_sha
    for name in ("trials.csv", "best_trial.json", "best_config.yaml", "SUMMARY.md"):
        assert (destination / name).read_bytes() == (source / name).read_bytes()
    with pytest.raises(RuntimeError, match="fresh absent"):
        restore_frozen_study(
            source,
            destination,
            study_name="restore-r09",
        )


def test_r08_hardware_report_enforces_fixed_objective_tolerance(tmp_path):
    definition = minimal_definition("hardware-r08")
    definition["reserved_trials"] = 2
    definition["search_space"]["fixed_parameters"] = {
        "alpha_node_feat": 2.0,
        "alpha_edge_feat": 3.0,
    }
    contract = canonical_contract_hash(definition)
    atomic_write_json(tmp_path / "study_definition.json", definition)
    atomic_write_json(
        tmp_path / "FROZEN.json",
        {
            "lifecycle": "FROZEN",
            "study_name": "hardware-r08",
            "study_contract_sha256": contract,
        },
    )
    result_paths = []
    for index, (host, gpu, value) in enumerate(
        (("host-a", 0, 0.50), ("host-b", 1, 0.51))
    ):
        trial_dir = tmp_path / "trials" / f"trial_{index:05d}"
        trial_dir.mkdir(parents=True)
        checkpoint = trial_dir / "checkpoint.pt"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        evaluator = trial_dir / "evaluation.json"
        atomic_write_json(
            evaluator,
            {
                "split": "validation",
                "primary_mode": "decoded_node_edge",
                "feature_source": {
                    "generated": "GraphVAE node_feature_decoder and edge_feature_decoder"
                },
            },
        )
        result_path = trial_dir / "trial_result.json"
        atomic_write_json(
            result_path,
            {
                "trial_number": index,
                "budget_index": index,
                "status": "COMPLETE",
                "study_contract_sha256": contract,
                "objective_json_path": (
                    "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
                ),
                "sampled_weights": {
                    "alpha_node_feat": 2.0,
                    "alpha_edge_feat": 3.0,
                },
                "validation_attr_f1pr": value,
                "hostname": host,
                "physical_gpu": gpu,
                "gpu_model": "NVIDIA TITAN RTX",
                "gpu_vram_bytes": 25769803776,
                "checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "evaluator_output": evaluator.relative_to(tmp_path).as_posix(),
                "hashes": {
                    "cache_sha256": "cache",
                    "split_fingerprint": "split",
                    "node_schema_fingerprint": "node",
                    "edge_schema_fingerprint": "edge",
                    "source_tree_sha256": "source",
                    "environment_sha256": "environment",
                },
            },
        )
        result_paths.append(result_path)

    report = build_hardware_repeatability_report(tmp_path)
    assert report["passed"] is True
    assert report["eligible_slots"] == ["host-a:gpu0", "host-b:gpu1"]
    assert report["objective_comparison"]["pairs"][0]["absolute_difference"] == pytest.approx(0.01)
    assert report["training_loss_comparison"]["status"] == "not_recorded"

    changed = json.loads(result_paths[1].read_text(encoding="utf-8"))
    changed["validation_attr_f1pr"] = 0.53
    atomic_write_json(result_paths[1], changed)
    report = build_hardware_repeatability_report(tmp_path)
    assert report["passed"] is False
    assert report["eligible_slots"] == []
