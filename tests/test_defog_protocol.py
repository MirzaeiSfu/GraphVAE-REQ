import json
from pathlib import Path

import pytest

from baselines.defog.verify_protocol import (
    ProtocolError,
    artifact_manifest,
    load_yaml,
    nested_value,
    split_digest,
    split_payload,
    verify_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "baselines" / "defog" / "protocol.yaml"


def test_frozen_proteins_split_digest_and_counts():
    protocol = load_yaml(PROTOCOL_PATH)
    payload = split_payload(protocol)

    assert {name: len(indices) for name, indices in payload.items()} == {
        "train": 731,
        "validation": 104,
        "test": 210,
    }
    assert split_digest(protocol) == protocol["split"]["index_sha256"]
    assert sorted(payload["train"] + payload["validation"] + payload["test"]) == list(
        range(1045)
    )


def test_protocol_points_to_history_preserving_defog_commit():
    protocol = load_yaml(PROTOCOL_PATH)

    assert nested_value(protocol, "repositories.defog.url") == (
        "https://github.com/MirzaeiSfu/defog"
    )
    assert nested_value(protocol, "repositories.defog.required_head") == (
        "474f9405bdcdddc2d96cfedc3305172dffbe8fbd"
    )
    assert nested_value(protocol, "checkpoint_selection.held_out_test_access") == (
        "forbidden"
    )


def write_artifact(tmp_path: Path, *, split: str, graph_count: int = 210) -> Path:
    artifact = tmp_path / f"{split}.pt"
    artifact.write_bytes(b"placeholder")
    manifest = {
        "format": "ggm-eval-pyg-tensors",
        "version": 1,
        "collection_sha256": "a" * 64,
        "metadata": {
            "dataset": "PROTEINS",
            "feature_schema": "default|export=decoded_node",
            "split": split,
        },
        "summary": {
            "graph_count": graph_count,
            "node_feature_dim": 3,
            "edge_feature_dim": 0,
            "total_nodes": 1,
            "directed_edge_count": 2,
        },
    }
    artifact.with_suffix(".pt.json").write_text(json.dumps(manifest), encoding="utf-8")
    return artifact


def test_artifact_manifest_contract_accepts_matching_metadata(tmp_path):
    protocol = load_yaml(PROTOCOL_PATH)
    artifact = write_artifact(tmp_path, split="generated")

    assert artifact_manifest(artifact)["collection_sha256"] == "a" * 64
    assert verify_artifact(artifact, protocol, split="generated")[
        "collection_sha256"
    ] == "a" * 64


def test_artifact_manifest_contract_rejects_wrong_graph_count(tmp_path):
    protocol = load_yaml(PROTOCOL_PATH)
    artifact = write_artifact(tmp_path, split="generated", graph_count=209)

    with pytest.raises(ProtocolError, match="graph_count"):
        verify_artifact(artifact, protocol, split="generated")
