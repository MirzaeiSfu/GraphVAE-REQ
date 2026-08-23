import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from scripts.graphvae_attr_bo_distributed import (
    atomic_write_json,
    build_study_definition,
    canonical_contract_hash,
    sha256_file,
)
from scripts.run_distributed_graphvae_attr_bo import command_collect


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
    assert "--exact-destination" in collect_help


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
