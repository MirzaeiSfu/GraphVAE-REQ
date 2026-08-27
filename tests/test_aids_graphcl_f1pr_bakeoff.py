import json
from pathlib import Path

import pytest

from graph_evaluation.src.ggm_eval.io import load_pyg_collection_with_metadata
from scripts import export_aids_graphcl_f1pr_splits as exporter


REPO_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_evaluator_bakeoff_split_qualification.json"
)


def test_frozen_aids_export_is_exact_deterministic_and_test_free(tmp_path):
    cache_path = REPO_ROOT / exporter.CACHE_RELATIVE_PATH
    first = exporter.export_frozen_splits(cache_path, tmp_path / "first")
    second = exporter.export_frozen_splits(cache_path, tmp_path / "second")

    assert first["test_access"] is False
    assert first["exported_splits"] == ["train", "validation"]
    assert first["split_overlap_count"] == 0
    assert first["cache"] == {
        "byte_length": exporter.CACHE_BYTE_LENGTH,
        "mode": "0444",
        "sha256": exporter.CACHE_SHA256,
    }
    for split, count in (("train", 1294), ("validation", 184)):
        first_artifact = first["artifacts"][split]
        second_artifact = second["artifacts"][split]
        assert first_artifact["collection_sha256"] == second_artifact[
            "collection_sha256"
        ]
        assert first_artifact["split_fingerprint"] == second_artifact[
            "split_fingerprint"
        ]
        graphs, metadata = load_pyg_collection_with_metadata(first_artifact["path"])
        assert len(graphs) == count
        assert metadata["split"] == split
        assert metadata["test_access"] is False
        assert metadata["feature_schema"] == exporter.FEATURE_SCHEMA
        assert graphs[0].x.shape[1] == 56
        assert graphs[0].edge_attr.shape[1] == 3
    assert not list((tmp_path / "first").glob("*test*"))


def test_aids_test_export_guard_precedes_cache_access(tmp_path):
    with pytest.raises(
        exporter.AidsGraphCLExportError,
        match="Held-out/test export is forbidden",
    ):
        exporter.export_frozen_splits(
            tmp_path / "missing.pkl", tmp_path / "output", include_test=True
        )


def test_aids_cache_hash_guard_precedes_pickle_loading(tmp_path, monkeypatch):
    relative = Path("cache_datasets/frozen.pkl")
    cache_path = tmp_path / relative
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"not a trusted pickle")
    cache_path.chmod(0o444)
    monkeypatch.setattr(exporter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exporter, "CACHE_RELATIVE_PATH", relative)
    monkeypatch.setattr(exporter, "CACHE_BYTE_LENGTH", cache_path.stat().st_size)
    monkeypatch.setattr(exporter, "CACHE_SHA256", "0" * 64)

    with pytest.raises(exporter.AidsGraphCLExportError, match="SHA-256"):
        exporter.export_frozen_splits(cache_path, tmp_path / "output")


def test_aids_split_contract_rejects_overlap_and_test_name():
    with pytest.raises(exporter.AidsGraphCLExportError, match="overlap"):
        exporter.assert_disjoint_splits(["same"], ["same"])
    with pytest.raises(exporter.AidsGraphCLExportError, match="Only training"):
        exporter._split_values({}, "test")


def test_aids_split_qualification_records_exact_test_free_inputs():
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    assert qualification["contract"] == {
        "dataset": "AIDS",
        "feature_schema": exporter.FEATURE_SCHEMA,
        "node_feature_dimension": 56,
        "edge_feature_dimension": 3,
        "exported_splits": ["train", "validation"],
        "test_access": False,
        "split_overlap_count": 0,
    }
    assert qualification["artifacts"]["train"]["collection_sha256"] == (
        "3d979eb9fec89967832df54547fff6a9dfa2d6f9da8a5d357200dc5963f41650"
    )
    assert qualification["artifacts"]["validation"]["collection_sha256"] == (
        "c0eaa2e545525e4a7a321eaa1937f6c68db76738906e2aa7f514b46dcd907e74"
    )
    assert qualification["verification"]["generated_collections_committed"] is False
