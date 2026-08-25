import json
from pathlib import Path

import pytest

from graph_evaluation.src.ggm_eval.io import load_pyg_collection_with_metadata
from scripts import export_lobster_graphcl_f1pr_splits as exporter


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "bayesian_optimization"
QUALIFICATION = CONFIG_ROOT / "lobster_graphcl_f1pr_prerequisite_qualification.json"
SPLIT_QUALIFICATION = CONFIG_ROOT / "lobster_graphcl_f1pr_split_qualification.json"
REPO_PATHS = REPO_ROOT / "CLUSTER_GRAPHVAE_GRAPHCL_F1PR_LOBSTER_REPO_PATHS.txt"
PYTHON_PATHS = REPO_ROOT / "CLUSTER_GRAPHVAE_GRAPHCL_F1PR_LOBSTER_PYTHON_PATHS.txt"
SLOTS = REPO_ROOT / "CLUSTER_GRAPHVAE_GRAPHCL_F1PR_LOBSTER_SLOTS.txt"
CREDENTIAL_PATHS = (
    REPO_ROOT / "CLUSTER_GRAPHVAE_GRAPHCL_F1PR_LOBSTER_CREDENTIAL_ENV_PATHS.txt"
)


def _data_rows(path):
    return [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_graphcl_f1pr_prerequisite_contract_is_test_free():
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))

    assert qualification["dataset_cache"] == {
        "relative_path": (
            "cache_datasets/LOBSTER_split-paper_70_10_20_train0p7_val0p1_"
            "test0p2_seed123_loaderseed-0_bfs-legacy_first_component_"
            "features-lobster-optimal_v2.pkl"
        ),
        "byte_length": 59295793,
        "sha256": (
            "928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660"
        ),
        "mode": "0444",
        "split_counts": {"train": 70, "validation": 10, "held_out_test": 20},
        "node_feature_dimension": 14,
        "edge_feature_dimension": 11,
        "test_access": False,
    }
    assert qualification["contrastive_upstream"]["revision"] == (
        "fb6bc26237eb21d7617fd41b22b4bb26ab29bf95"
    )
    assert qualification["selected_host"]["host"] == "cs-cl-09"
    assert qualification["concurrency"]["candidate_max_parallel"] == 2
    assert qualification["concurrency"]["hardware_qualified_for_new_objective"] is False
    assert qualification["checkpoint_reuse"] == {
        "bundled_lobster_graphcl_checkpoint_exists": False,
        "new_training_only_encoder_bundle_required": True,
    }
    assert qualification["execution"] == {
        "graphcl_f1pr_study_created": False,
        "graphcl_encoder_training_started": False,
        "held_out_or_test_access": False,
    }


def test_graphcl_f1pr_cluster_mappings_are_dedicated_and_exact():
    assert _data_rows(REPO_PATHS) == [[
        "cs-cl-09",
        "/local-scratch2/graphvae-req-work/GraphVAE-REQ-lobster-graphcl-f1pr",
    ]]
    assert _data_rows(PYTHON_PATHS) == [[
        "cs-cl-09",
        "/localhome/mirzaei/miniconda3/envs/micro/bin/python",
    ]]
    assert _data_rows(SLOTS) == [
        ["cs-cl-09", "0", "cs-cl-09-lobster-graphcl-gpu0"],
        ["cs-cl-09", "1", "cs-cl-09-lobster-graphcl-gpu1"],
    ]
    assert _data_rows(CREDENTIAL_PATHS) == [[
        "cs-cl-09",
        "/localhome/mirzaei/.graphvae-bo-credentials/lobster-production/worker.env",
    ]]
    credential_path = Path(_data_rows(CREDENTIAL_PATHS)[0][1])
    repository_path = Path(_data_rows(REPO_PATHS)[0][1])
    try:
        credential_path.relative_to(repository_path)
    except ValueError:
        pass
    else:
        raise AssertionError("Credential path must remain outside the repository root")


def test_frozen_lobster_export_is_deterministic_and_test_free(tmp_path):
    cache_path = REPO_ROOT / exporter.CACHE_RELATIVE_PATH
    first = exporter.export_frozen_splits(cache_path, tmp_path / "first")
    second = exporter.export_frozen_splits(cache_path, tmp_path / "second")

    assert first["test_access"] is False
    assert first["exported_splits"] == ["train", "validation"]
    assert first["split_overlap_count"] == 0
    assert first["cache"] == {
        "byte_length": 59295793,
        "mode": "0444",
        "sha256": (
            "928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660"
        ),
    }
    for split, count in (("train", 70), ("validation", 10)):
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
        assert metadata["feature_schema"] == (
            "lobster-optimal_v2|export=decoded_node_edge"
        )
        assert graphs[0].x.shape[1] == 14
        assert graphs[0].edge_attr.shape[1] == 11
    assert not list((tmp_path / "first").glob("*test*"))


def test_test_export_guard_runs_before_cache_access(tmp_path):
    with pytest.raises(
        exporter.LobsterGraphCLExportError,
        match="Held-out/test export is forbidden",
    ):
        exporter.export_frozen_splits(
            tmp_path / "does-not-exist.pkl",
            tmp_path / "output",
            include_test=True,
        )


def test_cache_hash_guard_precedes_pickle_loading(tmp_path, monkeypatch):
    cache_relative = Path("cache_datasets/frozen.pkl")
    cache_path = tmp_path / cache_relative
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"this is not a trusted pickle")
    cache_path.chmod(0o444)
    monkeypatch.setattr(exporter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exporter, "CACHE_RELATIVE_PATH", cache_relative)
    monkeypatch.setattr(exporter, "CACHE_BYTE_LENGTH", cache_path.stat().st_size)
    monkeypatch.setattr(exporter, "CACHE_SHA256", "0" * 64)

    with pytest.raises(exporter.LobsterGraphCLExportError, match="SHA-256"):
        exporter.export_frozen_splits(cache_path, tmp_path / "output")


def test_split_contract_rejects_overlap_and_missing_attributes():
    with pytest.raises(exporter.LobsterGraphCLExportError, match="overlap"):
        exporter.assert_disjoint_splits(["train", "same"], ["same", "validation"])
    with pytest.raises(exporter.LobsterGraphCLExportError, match="missing"):
        exporter._split_values(
            {
                "list_graphs": [object()] * 70,
                "list_noh_train": [object()] * 70,
            },
            "train",
        )


def test_frozen_split_qualification_records_exact_collections():
    qualification = json.loads(SPLIT_QUALIFICATION.read_text(encoding="utf-8"))

    assert qualification["contract"] == {
        "feature_schema": "lobster-optimal_v2|export=decoded_node_edge",
        "node_feature_dimension": 14,
        "edge_feature_dimension": 11,
        "exported_splits": ["train", "validation"],
        "test_access": False,
        "split_overlap_count": 0,
    }
    assert qualification["artifacts"]["train"]["graph_count"] == 70
    assert qualification["artifacts"]["train"]["collection_sha256"] == (
        "8de6ccf86bb2ae994f0a7401217d57a814d5e71c6e49732e345ae2b242f569e4"
    )
    assert qualification["artifacts"]["validation"]["graph_count"] == 10
    assert qualification["artifacts"]["validation"]["collection_sha256"] == (
        "0a5ad40ab717440f1739f0b203df3df253a6318089202aa467dd4fc6ee5c1832"
    )
    assert qualification["verification"] == {
        "deterministic_repeated_collection_digests": True,
        "atomic_manifest_publication": True,
        "focused_tests_passed": 7,
        "failed_tests": 0,
    }
    assert qualification["generated_artifacts_committed"] is False
