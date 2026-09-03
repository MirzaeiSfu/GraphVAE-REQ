import json
from pathlib import Path

import pytest

from baselines.defog.frozen_eval.aggregate import aggregate_dataset
from baselines.defog.frozen_eval.run_defog_job import (
    executable_path,
    quoted_override,
    train,
)
from baselines.defog.frozen_eval.verify_campaign import load_yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "baselines" / "defog" / "frozen_eval" / "manifest.yaml"


def test_manifest_freezes_all_primary_datasets_and_seeds():
    manifest = load_yaml(MANIFEST)

    assert set(manifest["datasets"]) == {
        "MUTAG",
        "PROTEINS",
        "GRID",
        "LOBSTER",
        "TRIANGULAR_GRID",
    }
    assert manifest["protocol"]["training_seeds"] == [0, 1, 2]
    assert manifest["protocol"]["generation_seed"] == 12345
    assert manifest["evaluator"]["evaluator_seeds"] == list(range(10))
    assert manifest["evaluator"]["nearest_k"] == 5
    assert "TO_FREEZE" not in MANIFEST.read_text(encoding="utf-8")
    assert len(manifest["repositories"]["defog_benchmark_commit"]) == 40


def test_campaign_has_a_schedule_for_every_frozen_dataset():
    campaign = load_yaml(MANIFEST.with_name("campaign.yaml"))
    manifest = load_yaml(MANIFEST)

    assert set(campaign["datasets"]) == set(manifest["datasets"])
    assert campaign["defog_commit"] == manifest["repositories"]["defog_benchmark_commit"]
    assert quoted_override("dataset.feature_schema", "a|export=b") == (
        'dataset.feature_schema="a|export=b"'
    )


def test_train_preserves_best_and_final_checkpoints(tmp_path, monkeypatch):
    def fake_run_command(command, *, cwd, env):
        checkpoint_dir = tmp_path / "job" / "training" / "checkpoints" / "run"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "best-00001-1.000000.ckpt").write_bytes(b"best")
        (checkpoint_dir / "last.ckpt").write_bytes(b"final")

    monkeypatch.setattr(
        "baselines.defog.frozen_eval.run_defog_job.run_command", fake_run_command
    )
    job_root = tmp_path / "job"
    best, final = train(
        python=Path("/usr/bin/python3"),
        defog_root=tmp_path,
        job_root=job_root,
        overrides=[],
        schedule={"epochs": 1, "validate_every_epochs": 1},
    )

    assert best.read_bytes() == b"best"
    assert final.read_bytes() == b"final"
    assert (job_root / "best_validation.ckpt").resolve() == best
    assert (job_root / "final_epoch.ckpt").resolve() == final


def test_executable_path_keeps_virtual_environment_symlink(tmp_path):
    interpreter = tmp_path / "base-python"
    interpreter.touch()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(interpreter)

    assert executable_path(venv_python) == venv_python.absolute()
    assert executable_path(venv_python) != venv_python.resolve()


def test_manifest_records_proteins_rejections_and_feature_contract():
    manifest = load_yaml(MANIFEST)
    proteins = manifest["datasets"]["PROTEINS"]

    assert proteins["raw_counts"] == {
        "train": 731,
        "validation": 104,
        "reference": 210,
    }
    assert proteins["accepted_counts"] == {
        "train": 728,
        "validation": 104,
        "reference": 209,
    }
    assert proteins["node_feature_dim"] == 3
    assert proteins["edge_feature_dim"] == 0


def _write_seed_result(root: Path, seed: int, value: float):
    metrics = {
        name: {"mean": value, "std": 0.0, "min": value, "max": value}
        for name in ("f1_pr", "precision", "recall", "mmd_rbf", "mmd_linear")
    }
    payload = {
        "campaign": {
            "dataset": "MUTAG",
            "training_seed": seed,
            "evaluator_seeds": list(range(10)),
        },
        "reference_sha256": "a" * 64,
        "evaluation": {
            "repeats": 10,
            "base_seed": 0,
            "modes": {"decoded_node": {"summary": metrics}},
        },
    }
    path = root / "mutag" / f"seed_{seed}" / "evaluation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregator_uses_sample_sd_across_training_seeds(tmp_path):
    for seed, value in enumerate((1.0, 2.0, 3.0)):
        _write_seed_result(tmp_path, seed, value)

    result = aggregate_dataset(tmp_path, "MUTAG")
    summary = result["aggregate"]["decoded_node"]["f1_pr"]

    assert summary["mean"] == pytest.approx(2.0)
    assert summary["sample_sd"] == pytest.approx(1.0)
    assert summary["training_seed_values"] == [1.0, 2.0, 3.0]
