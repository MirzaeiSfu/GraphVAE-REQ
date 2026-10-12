"""Tests for safe, versioned PyG tensor serialization."""

import json

import pytest
import torch

from ggm_eval.io import (
    load_pyg_collection,
    load_pyg_collection_with_metadata,
    save_pyg_collection,
)
from test_contract import make_path


def test_safe_collection_roundtrip_and_manifest(tmp_path):
    destination = tmp_path / "graphs.pt"
    original = [
        make_path(node_count=3, offset=0.0),
        make_path(node_count=4, offset=0.5),
    ]
    original[1].source_node_ids = torch.tensor(
        [10, 12, 15, 20], dtype=torch.int64
    )

    manifest = save_pyg_collection(
        destination,
        original,
        metadata={"generator": "test", "split": "generated"},
    )
    loaded = load_pyg_collection(destination)
    disk_manifest = json.loads(
        (tmp_path / "graphs.pt.json").read_text(encoding="utf-8")
    )

    assert len(loaded) == 2
    assert manifest == disk_manifest
    assert disk_manifest["summary"]["graph_count"] == 2
    assert len(disk_manifest["collection_sha256"]) == 64
    torch.testing.assert_close(loaded[1].x, original[1].x)
    torch.testing.assert_close(
        loaded[1].edge_index, original[1].edge_index
    )
    torch.testing.assert_close(loaded[1].edge_attr, original[1].edge_attr)
    torch.testing.assert_close(
        loaded[1].source_node_ids, original[1].source_node_ids
    )
    _, metadata = load_pyg_collection_with_metadata(destination)
    assert metadata == {"generator": "test", "split": "generated"}


def test_raw_pickled_pyg_requires_explicit_trust(tmp_path):
    destination = tmp_path / "raw.pt"
    original = [make_path()]
    torch.save(original, destination)

    with pytest.raises(ValueError, match="trusted=True"):
        load_pyg_collection(destination)

    loaded = load_pyg_collection(destination, trusted=True)
    assert len(loaded) == 1
    torch.testing.assert_close(loaded[0].x, original[0].x)


def test_empty_collection_is_rejected_before_serialization(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        save_pyg_collection(tmp_path / "empty.pt", [])


def test_metadata_rejects_custom_pickle_objects(tmp_path):
    with pytest.raises(TypeError, match="JSON-serializable"):
        save_pyg_collection(
            tmp_path / "unsafe-metadata.pt",
            [make_path()],
            metadata={"path_object": tmp_path},
        )


def test_safe_collection_detects_tensor_tampering(tmp_path):
    destination = tmp_path / "tampered.pt"
    save_pyg_collection(destination, [make_path()])
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    payload["graphs"][0]["x"][0, 0] += 1.0
    torch.save(payload, destination)

    with pytest.raises(ValueError, match="digest mismatch"):
        load_pyg_collection(destination)
