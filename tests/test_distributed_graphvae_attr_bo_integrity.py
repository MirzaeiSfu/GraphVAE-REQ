"""Gate 3 data, cache, and collection integrity acceptance cases."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
import yaml

import scripts.prepare_graphvae_attr_bo_cache as cache_module
import scripts.run_graphvae_attr_bo_worker as worker_module
from scripts.graphvae_attr_bo_distributed import (
    atomic_write_json,
    build_study_definition,
    canonical_contract_hash,
    sampler_seed,
    sha256_file,
)
from scripts.run_distributed_graphvae_attr_bo import command_collect


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[1]
MICRO_PYTHON = Path("/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python")
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "distributed_attr_f1pr_bo"
CANONICAL_CACHE = FIXTURE_ROOT / "qm9_tiny_cache.pkl"
CANONICAL_MANIFEST = FIXTURE_ROOT / "dataset_cache_manifest.json"


def _manifest():
    return json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))


def _base_config(path: Path):
    payload = {
        "experiment": {"epoch_number": 1, "task": "graphGeneration"},
        "loss": {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return payload


def _definition(name: str, config: dict, config_path: Path, manifest: dict):
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
            "study_seed": 0,
            "split_seed": 123,
            "training_seed": 0,
            "generation_seed": 123,
            "evaluator_seed": 0,
        },
        evaluator={
            "mode": "decoded_node_edge",
            "split": "validation",
            "test_access": False,
            "repeat_count": 5,
            "max_graphs": 3,
        },
        training={"epoch_number": 1, "mock": True},
        source={},
        environment={},
        dataset_cache=manifest,
        feature_schemas={
            "node_sha256": manifest["node_schema_fingerprint"],
            "edge_sha256": manifest["edge_schema_fingerprint"],
            "node": manifest["node_schema"],
            "edge": manifest["edge_schema"],
        },
        hardware_policy={"attr_f1pr_abs_tolerance": 0.02},
        heartbeat_interval=60,
        grace_period=600,
        max_parallel=1,
    )


def _stage_cache(worker_root: Path, manifest: dict) -> Path:
    destination = worker_root / manifest["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CANONICAL_CACHE, destination)
    return destination


def test_d01_missing_cache_fails_before_database_or_reservation(tmp_path):
    config_path = tmp_path / "config.yaml"
    config = _base_config(config_path)
    manifest = _manifest()
    manifest["relative_path"] = (
        "tests/fixtures/distributed_attr_f1pr_bo/definitely_missing_cache.pkl"
    )
    definition = _definition("d01-missing-cache", config, config_path, manifest)
    artifact_root = tmp_path / "study"
    artifact_root.mkdir()
    contract = canonical_contract_hash(definition)
    atomic_write_json(artifact_root / "study_definition.json", definition)
    missing_path = REPO_ROOT / manifest["relative_path"]
    assert not missing_path.exists()
    environment = os.environ.copy()
    environment.pop("GRAPHVAE_BO_D01_UNSET_STORAGE", None)
    result = subprocess.run(
        [
            str(MICRO_PYTHON),
            str(REPO_ROOT / "scripts" / "run_graphvae_attr_bo_worker.py"),
            "--study-name", "d01-missing-cache",
            "--base-config", str(config_path),
            "--artifact-root", str(artifact_root),
            "--study-contract-sha256", contract,
            "--worker-id", "d01-worker",
            "--worker-run-id", "d01-worker-run",
            "--sampler-seed", str(sampler_seed(0, 0)),
            "--dispatch-sequence", "0",
            "--device", "cpu",
            "--storage-env", "GRAPHVAE_BO_D01_UNSET_STORAGE",
            "--mock",
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 2
    failure = json.loads(
        (
            artifact_root
            / "workers"
            / "d01-worker-run"
            / "FAILED_PRETRIAL"
        ).read_text(encoding="utf-8")
    )
    assert failure["phase"] == "local_preflight"
    assert failure["exception_type"] == "FileNotFoundError"
    assert failure["reservation_consumed"] is False
    assert not (artifact_root / "trials").exists()
    assert not missing_path.exists()


def test_d02_one_byte_cache_change_is_rejected_before_claim(tmp_path, monkeypatch):
    worker_root = tmp_path / "worker"
    config_path = worker_root / "config.yaml"
    config = _base_config(config_path)
    manifest = _manifest()
    cache_path = _stage_cache(worker_root, manifest)
    definition = _definition("d02-cache-byte", config, config_path, manifest)
    content = bytearray(cache_path.read_bytes())
    content[-1] ^= 0x01
    cache_path.write_bytes(content)
    monkeypatch.setattr(worker_module, "REPO_ROOT", worker_root)
    args = argparse.Namespace(base_config=config_path, mock=True)
    with pytest.raises(worker_module.DistributedContractError, match="cache SHA-256"):
        worker_module.local_preflight(args, definition)


def test_d03_two_staged_workers_recompute_identical_fingerprints(
    tmp_path, monkeypatch
):
    expected = _manifest()
    verified = []
    for worker_name in ("worker-a", "worker-b"):
        worker_root = tmp_path / worker_name
        cache_path = _stage_cache(worker_root, expected)
        monkeypatch.setattr(cache_module, "REPO_ROOT", worker_root)
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        verified.append(
            cache_module.verify_cache_manifest(
                cache_path, payload, copy.deepcopy(expected)
            )
        )
    assert verified[0] == verified[1] == expected
    for manifest in verified:
        assert manifest["sha256"] == expected["sha256"]
        assert manifest["split_fingerprint"] == expected["split_fingerprint"]
        assert manifest["splits"]["validation"]["graph_count"] == 3
        assert manifest["node_schema_fingerprint"] == expected[
            "node_schema_fingerprint"
        ]
        assert manifest["edge_schema_fingerprint"] == expected[
            "edge_schema_fingerprint"
        ]


def test_d04_equal_dimension_changed_channel_meaning_is_rejected(
    tmp_path, monkeypatch
):
    worker_root = tmp_path / "worker"
    expected = _manifest()
    cache_path = _stage_cache(worker_root, expected)
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["node_onehot_info"][0]["value"] = "Si"
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    monkeypatch.setattr(cache_module, "REPO_ROOT", worker_root)
    actual = cache_module.build_cache_manifest(
        cache_path, payload, max_graphs=expected["expected_validation_graphs"]
    )
    plausible = copy.deepcopy(expected)
    for field in (
        "byte_length",
        "sha256",
        "splits",
        "split_fingerprint",
    ):
        plausible[field] = actual[field]
    assert actual["node_feature_dimension"] == expected["node_feature_dimension"]
    with pytest.raises(cache_module.DistributedContractError, match="node_schema"):
        cache_module.verify_cache_manifest(cache_path, payload, plausible)


def test_d06_partial_collection_is_not_promoted_and_retry_is_safe(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    trial = source / "trials" / "trial_00000"
    trial.mkdir(parents=True)
    (trial / "trial_result.json").write_bytes(b"complete-result")
    (trial / "checkpoint").write_bytes(b"complete-checkpoint")
    destination = tmp_path / "destination"
    original_copytree = shutil.copytree

    def interrupted_copytree(source_path, destination_path, *args, **kwargs):
        destination_path = Path(destination_path)
        destination_path.mkdir(parents=True, exist_ok=True)
        for source_item in Path(source_path).rglob("*"):
            target = destination_path / source_item.relative_to(source_path)
            if source_item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_item, target)
        partial = Path(destination_path) / "trials" / "trial_00000" / "checkpoint"
        partial.write_bytes(b"partial")
        return destination_path

    monkeypatch.setattr(shutil, "copytree", interrupted_copytree)
    args = argparse.Namespace(source_root=source, output_dir=destination)
    with pytest.raises(Exception, match="hash verification"):
        command_collect(args)
    assert not (destination / "trials" / "trial_00000").exists()
    assert list((destination / ".collection_staging").glob("*/payload"))

    monkeypatch.setattr(shutil, "copytree", original_copytree)
    assert command_collect(args) == 0
    assert (
        destination / "trials" / "trial_00000" / "checkpoint"
    ).read_bytes() == b"complete-checkpoint"
