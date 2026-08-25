import json
from pathlib import Path

import pytest
import yaml

from graph_evaluation.src.ggm_eval.io import load_pyg_collection_with_metadata
from scripts import export_lobster_graphcl_f1pr_splits as exporter


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "bayesian_optimization"
QUALIFICATION = CONFIG_ROOT / "lobster_graphcl_f1pr_prerequisite_qualification.json"
SPLIT_QUALIFICATION = CONFIG_ROOT / "lobster_graphcl_f1pr_split_qualification.json"
ENCODER_QUALIFICATION = CONFIG_ROOT / "lobster_graphcl_f1pr_encoder_qualification.json"
EVALUATOR_QUALIFICATION = CONFIG_ROOT / "lobster_graphcl_f1pr_evaluator_qualification.json"
GATE4_EXIT_QUALIFICATION = (
    CONFIG_ROOT / "lobster_graphcl_f1pr_gate4_exit_qualification.json"
)
GATE5_POLICY = CONFIG_ROOT / "lobster_graphcl_f1pr_gate5_policy.json"
GATE5_RESERVATIONS = (
    CONFIG_ROOT / "lobster_graphcl_f1pr_anchor_reservations_6.json"
)
GATE5_HARDWARE = (
    CONFIG_ROOT / "lobster_graphcl_f1pr_gate5_hardware_policy.json"
)
GATE5_CONFIG = CONFIG_ROOT / "lobster_graphcl_f1pr_signal.yaml"
GATE5_SLOTS = REPO_ROOT / "CLUSTER_GRAPHVAE_GRAPHCL_F1PR_LOBSTER_GATE5_SLOTS.txt"
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


def test_frozen_encoder_qualification_records_exact_train_only_bundle():
    qualification = json.loads(ENCODER_QUALIFICATION.read_text(encoding="utf-8"))

    assert qualification["training_input"]["split"] == "train"
    assert qualification["training_input"]["test_access"] is False
    assert qualification["encoder_contract"]["seeds"] == [101, 202, 303, 404, 505]
    assert qualification["frozen_bundle"] == {
        "checkpoint_count": 5,
        "bundle_sha256": (
            "4cf22a9b204b3638ce6f63fc691ff9556986af1334b64b0994411f5bfd7ac8be"
        ),
        "manifest_byte_length": 4477,
        "manifest_file_sha256": (
            "638b51662d64d484e8d63e59372d27e4f71be1c73285f8101908a7696c321451"
        ),
        "manifest_mode": "0444",
        "checkpoint_modes": "0444",
        "directory_modes": "0555",
        "independent_restore_verified": True,
    }
    assert qualification["dependency_resolution"]["initial_attempt_preserved"] is True
    assert qualification["verification"]["cache_unchanged"] is True
    assert qualification["verification"]["binary_checkpoints_committed"] is False


def test_real_graphcl_evaluator_qualification_is_validation_only():
    qualification = json.loads(EVALUATOR_QUALIFICATION.read_text(encoding="utf-8"))

    assert qualification["qualification_source_checkpoint"]["new_training_performed"] is False
    assert qualification["qualification_source_checkpoint"]["search_or_selection_evidence"] is False
    assert qualification["evaluator_contract"]["split"] == "validation"
    assert qualification["evaluator_contract"]["test_access"] is False
    assert qualification["evaluator_contract"]["checkpoint_count"] == 5
    assert qualification["result"]["objective_json_path"] == "summary.f1_pr.mean"
    assert qualification["result"]["objective"] == 0.6968191868090722
    assert [row["seed"] for row in qualification["result"]["per_encoder"]] == [
        101,
        202,
        303,
        404,
        505,
    ]
    assert qualification["integrity"]["frozen_file_count"] == 20
    assert qualification["integrity"]["independent_parse_and_reopen"] is True
    assert qualification["verification"]["postgresql_access"] is False
    assert qualification["verification"]["held_out_evaluation"] is False


def test_gate4_exit_is_complete_test_free_and_non_scientific():
    qualification = json.loads(
        GATE4_EXIT_QUALIFICATION.read_text(encoding="utf-8")
    )

    assert qualification["objective_contract"] == {
        "primary_path": "summary.f1_pr.mean",
        "compatibility_path": (
            "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
        ),
        "aggregation": (
            "arithmetic_mean_only_after_graphvae_training_seeds_0_and_1_complete"
        ),
        "selection_split": "validation",
        "test_access": False,
        "node_feature_decoder_required": True,
        "edge_feature_decoder_required": True,
    }
    studies = qualification["lifecycle_studies"]
    assert set(studies) == {
        "two_worker",
        "three_worker",
        "ambiguous_after_ack",
        "stale_worker",
    }
    assert all(item["lifecycle"] == "FROZEN" for item in studies.values())
    assert studies["three_worker"]["postgresql_running_peak"] == 3
    assert studies["ambiguous_after_ack"]["duplicate_dispatch_count"] == 0
    assert studies["stale_worker"]["failed"] == 1
    assert studies["stale_worker"]["replacement_count"] == 0
    assert qualification["tests"] == {
        "non_postgresql_distributed_attribute_and_graphcl": 151,
        "isolated_postgresql": 19,
        "failed": 0,
        "skipped": 0,
        "residual_isolated_postgresql_studies": 0,
    }
    assert qualification["conclusion"] == {
        "gate4_passed": True,
        "scientific_weight_claim": (
            "none; all Gate 4 objective values are synthetic lifecycle mocks"
        ),
        "next_authorized_gate": "Gate 5 fixed real LOBSTER anchors",
        "adaptive_bo_authorized": False,
        "held_out_or_test_evaluation_authorized": False,
    }


def test_gate5_fixed_anchor_plan_is_exact_and_validation_only():
    reservations = json.loads(GATE5_RESERVATIONS.read_text(encoding="utf-8"))
    assert reservations["schema_version"] == "graphvae-attr-f1pr-reservation-plan-v1"
    assert reservations["reservations"] == [
        {
            "budget_index": 0,
            "parameters": {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0},
            "training_seed": 0,
        },
        {
            "budget_index": 1,
            "parameters": {
                "alpha_node_feat": 5.229045672,
                "alpha_edge_feat": 0.0538641483,
            },
            "training_seed": 0,
        },
        {
            "budget_index": 2,
            "parameters": {"alpha_node_feat": 0.25, "alpha_edge_feat": 0.25},
            "training_seed": 0,
        },
        {
            "budget_index": 3,
            "parameters": {"alpha_node_feat": 4.0, "alpha_edge_feat": 4.0},
            "training_seed": 0,
        },
        {
            "budget_index": 4,
            "parameters": {"alpha_node_feat": 4.0, "alpha_edge_feat": 0.25},
            "training_seed": 0,
        },
        {
            "budget_index": 5,
            "parameters": {"alpha_node_feat": 0.25, "alpha_edge_feat": 4.0},
            "training_seed": 0,
        },
    ]

    policy = json.loads(GATE5_POLICY.read_text(encoding="utf-8"))
    assert policy["phase_a"]["reserved_candidates"] == 6
    assert policy["phase_a"]["graphvae_training_seeds_per_candidate"] == [0, 1]
    assert policy["phase_a"]["study_seed"] == 13034
    assert policy["phase_a"]["tpe_startup_trials"] == 6
    assert policy["phase_a"]["heartbeat_interval_seconds"] == 60
    assert policy["phase_a"]["grace_period_seconds"] == 600
    assert policy["phase_a"]["alpha_node_feat_range"] == [0.001, 100.0]
    assert policy["phase_a"]["alpha_edge_feat_range"] == [0.001, 100.0]
    assert policy["phase_a"]["evaluator_repeat_count"] == 5
    assert policy["phase_a"]["nearest_k"] == 5
    assert policy["phase_a"]["adjacency_threshold"] == 0.5
    assert policy["phase_a"]["max_parallel"] == 1
    assert policy["phase_b"]["epoch_number"] == 10000
    assert policy["phase_b"]["promoted_candidates"] == 3
    assert policy["generation_stability"]["generation_seeds"] == [123, 124, 125]
    assert (
        policy["pass_thresholds"]["training_seed_coefficient_of_variation_max"]
        == 0.2
    )
    assert policy["pass_thresholds"]["minimum_anchors_passing_cv"] == 4
    assert policy["pass_thresholds"]["phase_a_phase_b_spearman_min"] == 0.5
    assert policy["objective_contract"]["selection_split"] == "validation"
    assert policy["objective_contract"]["test_access"] is False
    assert policy["objective_contract"]["partial_candidate_score_forbidden"] is True
    config = yaml.safe_load(GATE5_CONFIG.read_text(encoding="utf-8"))
    assert config["data"]["dataset"] == "LOBSTER"
    assert config["experiment"]["epoch_number"] == 2000
    assert config["runtime"]["skip_final_evaluation"] is True
    assert config["bayesian_optimization_qualification"] == {
        "expected_total_graphs": 100,
        "max_graphs": 10,
        "generation_batch_size": 10,
        "training_timeout_seconds": 7200,
        "evaluation_timeout_seconds": 1200,
        "termination_grace_seconds": 10,
    }
    hardware = json.loads(GATE5_HARDWARE.read_text(encoding="utf-8"))
    assert hardware["homogeneous_production_pool"] == ["cs-cl-09:cuda:1"]
    assert _data_rows(GATE5_SLOTS) == [
        ["cs-cl-09", "1", "cs-cl-09-lobster-graphcl-gate5-gpu1"]
    ]
