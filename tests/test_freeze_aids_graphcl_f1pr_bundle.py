import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import freeze_aids_graphcl_f1pr_bundle as freezer


def _write_seed(root: Path, seed: int, *, test_access: bool = False) -> Path:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    checkpoint_path = seed_dir / "checkpoint.pt"
    metadata = {
        "dataset": "AIDS",
        "feature_mode": "decoded_node_edge",
        "feature_schema": freezer.EXPECTED_FEATURE_SCHEMA,
        "node_feature_dimension": 56,
        "edge_feature_dimension": 3,
        "split": "train",
        "split_fingerprint": freezer.EXPECTED_SPLIT_FINGERPRINT,
        "test_access": test_access,
    }
    torch.save(
        {
            "format": "ggm-eval-upstream-gconv",
            "version": 1,
            "encoder": "graphcl",
            "feature_mode": "decoded_node_edge",
            "seed": seed,
            "model": freezer.EXPECTED_MODEL,
            "training": {"epochs": 100, "trained": True},
            "training_metadata": metadata,
            "upstream_revision": freezer.EXPECTED_UPSTREAM_REVISION,
            "state_dict": {"layer.weight": torch.ones(2, 2) * seed},
        },
        checkpoint_path,
    )
    training = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_format": "ggm-eval-upstream-gconv",
        "checkpoint_version": 1,
        "elapsed_seconds": float(seed),
        "encoder": "graphcl",
        "feature_mode": "decoded_node_edge",
        "model": freezer.EXPECTED_MODEL,
        "seed": seed,
        "training": {"epochs": 100, "trained": True},
        "training_collection_sha256": freezer.EXPECTED_COLLECTION_SHA256,
        "training_graphs": freezer.EXPECTED_TRAINING_GRAPHS,
        "training_loss": 0.5,
        "training_metadata": metadata,
        "upstream": {"revision": freezer.EXPECTED_UPSTREAM_REVISION},
        "versions": {"torch": "2.1.2", "pygcl": "0.1.2"},
    }
    (seed_dir / "training.json").write_text(json.dumps(training), encoding="utf-8")
    return seed_dir


def _roots(tmp_path: Path):
    first = tmp_path / "gpu0"
    second = tmp_path / "gpu1"
    for seed in freezer.EXPECTED_SEEDS[:5]:
        _write_seed(first, seed)
    for seed in freezer.EXPECTED_SEEDS[5:]:
        _write_seed(second, seed)
    return first, second


def test_bundle_manifest_requires_ten_exact_train_only_encoders(tmp_path, monkeypatch):
    first, second = _roots(tmp_path)
    monkeypatch.setattr(
        freezer,
        "validate_contrastive_upstream",
        lambda _path: {
            "revision": freezer.EXPECTED_UPSTREAM_REVISION,
            "worktree_dirty": False,
        },
    )
    monkeypatch.setattr(
        freezer,
        "graphcl_runtime_fingerprint",
        lambda _path: {"sha256": "a" * 64, "dependency_tree": {"file_count": 125}},
    )

    manifest = freezer.build_bundle_manifest(
        campaign_root=tmp_path,
        training_roots=[first, second],
        upstream_repo=tmp_path,
        dependency_root=tmp_path,
    )

    assert manifest["seeds"] == list(freezer.EXPECTED_SEEDS)
    assert manifest["checkpoint_count"] == 10
    assert manifest["training_split"] == "train"
    assert manifest["test_access"] is False
    unhashed = dict(manifest)
    digest = unhashed.pop("bundle_sha256")
    assert digest == hashlib.sha256(freezer.canonical_json_bytes(unhashed)).hexdigest()


def test_bundle_manifest_rejects_missing_or_duplicate_seed(tmp_path):
    first, second = _roots(tmp_path)
    missing = second / f"seed_{freezer.EXPECTED_SEEDS[-1]}"
    missing.rename(second / "seed_1111")
    with pytest.raises(freezer.DistributedContractError, match="exactly one"):
        freezer._find_seed_directory([first, second], freezer.EXPECTED_SEEDS[-1])

    duplicate = second / f"seed_{freezer.EXPECTED_SEEDS[0]}"
    _write_seed(second, freezer.EXPECTED_SEEDS[0])
    assert duplicate.is_dir()
    with pytest.raises(freezer.DistributedContractError, match="exactly one"):
        freezer._find_seed_directory([first, second], freezer.EXPECTED_SEEDS[0])


def test_bundle_rejects_test_access_and_nonfinite_tensor(tmp_path):
    seed_dir = _write_seed(tmp_path, 101, test_access=True)
    with pytest.raises(freezer.DistributedContractError, match="test_access"):
        freezer._assert_checkpoint(seed_dir, 101)

    clean_dir = _write_seed(tmp_path / "clean", 101)
    checkpoint_path = clean_dir / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["state_dict"]["layer.weight"][0, 0] = float("nan")
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(freezer.DistributedContractError, match="invalid tensor"):
        freezer._assert_checkpoint(clean_dir, 101)


def test_read_only_freeze_covers_every_encoder_file(tmp_path):
    root = tmp_path / "bundle"
    (root / "seed_101").mkdir(parents=True)
    (root / "seed_101" / "checkpoint.pt").write_bytes(b"checkpoint")
    freezer._make_tree_read_only(root)
    assert root.stat().st_mode & 0o777 == 0o555
    assert (root / "seed_101").stat().st_mode & 0o777 == 0o555
    assert (root / "seed_101" / "checkpoint.pt").stat().st_mode & 0o777 == 0o444
