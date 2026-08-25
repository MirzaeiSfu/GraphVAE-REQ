import json
from pathlib import Path

from factorbase_motif_pipeline.tu_dataset_to_db import TU_DATASET_SPECS
from scripts.graphvae_attr_bo_distributed import parse_slots
from scripts.resample_grid_checkpoints import (
    build_dataset_cache_metadata,
    build_dataset_cache_name,
)
from scripts.run_distributed_graphvae_attr_bo import _load_mapping
from scripts.tune_graphvae_attribute_weights import (
    flatten_config,
    load_yaml_mapping,
    validate_base_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_graphvae_attr_f1pr_calibration.yaml"
)
PROVENANCE_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_dataset_provenance.json"
)
CACHE_MANIFEST_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_cache_manifest.json"
)
DEPLOYMENT_QUALIFICATION_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_deployment_qualification.json"
)
TIMING_QUALIFICATION_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_timing_qualification.json"
)
SEARCH_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_graphvae_attr_f1pr_search.yaml"
)
SEARCH_RESERVATIONS_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_search_reservations_14.json"
)
SEARCH_POLICY_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_search_policy.json"
)
SEARCH_QUALIFICATION_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_search_qualification.json"
)
CONFIRMATION_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_graphvae_attr_f1pr_confirmation.yaml"
)
CONFIRMATION_RESERVATIONS_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_confirmation_reservations_6.json"
)
CONFIRMATION_POLICY_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_confirmation_analysis_policy.json"
)
CONFIRMATION_QUALIFICATION_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_confirmation_qualification.json"
)
FINAL_QUALIFICATION_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_final_qualification.json"
)
REPO_PATHS = REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_AIDS_REPO_PATHS.txt"
PYTHON_PATHS = REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_AIDS_PYTHON_PATHS.txt"
CREDENTIAL_PATHS = (
    REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_AIDS_CREDENTIAL_ENV_PATHS.txt"
)
SLOTS = REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_AIDS_SLOTS.txt"


def test_aids_calibration_config_is_validation_only_and_bounded():
    config = load_yaml_mapping(CONFIG_PATH)
    validate_base_config(config, tune_alpha_motif=False)
    flat = flatten_config(config)

    assert flat["dataset"] == "AIDS"
    assert flat["split_mode"] == "paper_70_10_20"
    assert flat["train_fraction"] == 0.7
    assert flat["val_fraction"] == 0.1
    assert flat["split_seed"] == 123
    assert flat["dataset_loader_seed"] == 0
    assert flat["bfs_strategy"] == "all_components"
    assert flat["tu_attribute_bins"] == 8
    assert flat["tu_max_nodes"] == 40
    assert flat["data_dir"] is None
    assert flat["expected_total_graphs"] == 1849
    assert flat["max_graphs"] == 0
    assert flat["alpha_node_feat"] == 1.0
    assert flat["alpha_edge_feat"] == 1.0
    assert flat["require_existing_dataset_cache"] is True
    assert flat["skip_final_evaluation"] is True
    assert flat["third_party_eval"] is False
    assert flat["ideal_Evalaution"] is False
    assert flat["tiny_overfit"] is False


def test_aids_cache_identity_is_exact():
    flat = flatten_config(load_yaml_mapping(CONFIG_PATH))
    metadata = build_dataset_cache_metadata(flat)

    assert metadata == {
        "cache_schema_version": "dataset-cache-v4",
        "dataset": "AIDS",
        "split_mode": "paper_70_10_20",
        "bfs_strategy": "all_components",
        "split_kind": "three_way",
        "train_fraction": 0.7,
        "val_fraction": 0.1,
        "test_fraction": 0.20000000000000004,
        "split_seed": 123,
        "dataset_loader_seed": 0,
        "feature_schema": "tu-quantile8-max40",
    }
    assert build_dataset_cache_name(metadata) == (
        "AIDS_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_"
        "loaderseed-0_bfs-all_components_features-tu-quantile8-max40.pkl"
    )


def test_aids_provenance_requires_real_edge_labels_and_exact_splits():
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert TU_DATASET_SPECS["AIDS"].has_edge_labels is True
    assert provenance["source"]["dataset"] == "AIDS"
    assert provenance["expected_contract"]["source_graphs"] == 2000
    assert provenance["expected_contract"]["retained_graphs"] == 1849
    assert provenance["expected_contract"]["split_counts"] == {
        "train": 1294,
        "validation": 184,
        "test": 371,
    }
    assert provenance["expected_contract"]["edge_feature_fields"] == {
        "edge_label": [0, 1, 2]
    }
    assert len(provenance["archive"]["sha256"]) == 64
    assert set(provenance["files"]) == {
        "AIDS_A.txt",
        "AIDS_edge_labels.txt",
        "AIDS_graph_indicator.txt",
        "AIDS_graph_labels.txt",
        "AIDS_node_attributes.txt",
        "AIDS_node_labels.txt",
    }
    assert all(len(value) == 64 for value in provenance["files"].values())


def test_aids_cache_manifest_freezes_full_attributed_validation_contract():
    manifest = json.loads(CACHE_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["relative_path"] == (
        "cache_datasets/AIDS_split-paper_70_10_20_train0p7_val0p1_"
        "test0p2_seed123_loaderseed-0_bfs-all_components_"
        "features-tu-quantile8-max40.pkl"
    )
    assert manifest["byte_length"] == 73822456
    assert manifest["sha256"] == (
        "6edcc3309fb1c3d366b0f87065aa1b2e2c7d23cbff92bc729053f44e874909bb"
    )
    assert {
        split: manifest["splits"][split]["graph_count"]
        for split in ("train", "validation", "test")
    } == {"train": 1294, "validation": 184, "test": 371}
    assert manifest["expected_validation_graphs"] == 184
    assert manifest["node_feature_dimension"] == 56
    assert manifest["edge_feature_dimension"] == 3
    assert manifest["node_schema_fingerprint"] == (
        "630400b38c74e0a51e505e57b7e41ee986deea1eb06e27099ea792cf6876c2c9"
    )
    assert manifest["edge_schema_fingerprint"] == (
        "1fee16532276d512d7143669f2d3c80a0140ee3c2ded035fa206c323320bf772"
    )
    assert len(manifest["splits"]["validation"]["graph_fingerprints"]) == 184


def test_aids_deployment_mappings_are_dedicated_and_homogeneous():
    repositories = _load_mapping(REPO_PATHS)
    pythons = _load_mapping(PYTHON_PATHS)
    credentials = _load_mapping(CREDENTIAL_PATHS)
    slots = parse_slots(SLOTS, known_hosts=sorted(repositories))

    assert set(repositories) == {"cs-cl-13", "cs-cl-17"}
    assert set(repositories) == set(pythons) == set(credentials)
    assert all(
        path.endswith("GraphVAE-REQ-aids-attr-bo")
        for path in repositories.values()
    )
    assert all("lobster" not in path.lower() for path in repositories.values())
    assert all(path.endswith("/envs/micro/bin/python") for path in pythons.values())
    assert all(
        path.endswith("/aids-production/worker.env")
        for path in credentials.values()
    )
    assert [(slot["host"], slot["physical_gpu"]) for slot in slots] == [
        ("cs-cl-13", 0),
        ("cs-cl-17", 0),
        ("cs-cl-17", 1),
    ]
    assert all("aids-prod" in slot["worker_id"] for slot in slots)


def test_aids_deployment_qualification_matches_frozen_contract():
    qualification = json.loads(
        DEPLOYMENT_QUALIFICATION_PATH.read_text(encoding="utf-8")
    )

    assert qualification["controller"] == {
        "protected_postgresql_authentication": "verify-full",
        "runtime_fingerprint": (
            "e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1"
        ),
    }
    assert qualification["cache"] == {
        "byte_length": 73822456,
        "mode": "0444",
        "sha256": (
            "6edcc3309fb1c3d366b0f87065aa1b2e2c7d23cbff92bc729053f44e874909bb"
        ),
        "verified_after_deployment": True,
    }
    assert qualification["source"]["clean_worktree"] is True
    assert qualification["source"]["git_commit"] == (
        "7fd1fec871c47a398135ad5cc4fbe4abbbd18c87"
    )
    assert qualification["source"]["verified_on_every_host"] is True
    assert len(qualification["source"]["tree_sha256"]) == 64
    workers = qualification["workers"]
    assert len(workers) == 3
    assert {worker["worker_id"] for worker in workers} == {
        "cs-cl-13-aids-prod-gpu0",
        "cs-cl-17-aids-prod-gpu0",
        "cs-cl-17-aids-prod-gpu1",
    }
    assert all(worker["model"] == "NVIDIA TITAN RTX" for worker in workers)
    assert all(worker["reported_vram_mib"] == 24576 for worker in workers)
    assert all(worker["logical_device_count"] == 1 for worker in workers)
    assert all(
        worker["protected_postgresql_authentication"] == "verify-full"
        for worker in workers
    )


def test_aids_timing_qualification_is_exact_and_validation_only():
    qualification = json.loads(
        TIMING_QUALIFICATION_PATH.read_text(encoding="utf-8")
    )

    assert qualification["study"] == {
        "contract_sha256": (
            "76e42209b65ea492334234369c6c39237ced9f03e1656ae684b35a41fe74390a"
        ),
        "lifecycle": "FROZEN",
        "name": "aids_attr_f1pr_hw_timing_20260824a",
        "training_epochs": 250,
        "training_seed": 0,
        "weights": {"alpha_edge_feat": 1.0, "alpha_node_feat": 1.0},
    }
    assert qualification["reservations"] == {
        "complete": 3,
        "fail": 0,
        "max_parallel": 3,
        "reserved_total": 3,
        "running": 0,
        "unreserved_guard": 0,
        "waiting": 0,
    }
    objective = qualification["objective"]
    assert objective["json_path"] == (
        "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
    )
    assert objective["selection_split"] == "validation"
    assert objective["test_access"] is False
    assert objective["node_decoder_required"] is True
    assert objective["edge_decoder_required"] is True
    assert objective["accepted_validation_graphs"] == 184
    assert objective["evaluator_repeats"] == 5
    assert objective["validation_attr_f1pr"] == 0.718249786584885
    assert qualification["hardware_repeatability"]["passed"] is True
    assert qualification["hardware_repeatability"][
        "maximum_objective_difference"
    ] == 0.0
    assert qualification["dispatch_audit"] == {
        "initial_attempt": "DEFINITE_PRELAUNCH_ERROR",
        "initial_attempt_claimed_reservations": 0,
        "initial_worker_identities_reused": False,
        "probe_status": "DEFINITE_PRELAUNCH",
        "retry_safe": True,
        "successful_wave": 2,
    }
    assert qualification["artifact_audit"]["credential_matches"] == 0
    assert qualification["artifact_audit"][
        "forbidden_storage_or_test_access_matches"
    ] == 0
    assert qualification["artifact_audit"][
        "aggregate_outputs_match_after_restore"
    ] is True
    assert qualification["cache"]["verified_on_controller_and_both_hosts"] is True
    assert [row["trial_number"] for row in qualification["timing_seconds"]] == [
        0,
        1,
        2,
    ]
    assert all(row["total"] < 7200 for row in qualification["timing_seconds"])


def test_aids_search_contract_is_bounded_and_confirmation_gated():
    config = load_yaml_mapping(SEARCH_CONFIG_PATH)
    validate_base_config(config, tune_alpha_motif=False)
    flat = flatten_config(config)
    plan = json.loads(SEARCH_RESERVATIONS_PATH.read_text(encoding="utf-8"))
    policy = json.loads(SEARCH_POLICY_PATH.read_text(encoding="utf-8"))

    assert flat["dataset"] == "AIDS"
    assert flat["expected_total_graphs"] == 1849
    assert flat["epoch_number"] == 100
    assert flat["max_graphs"] == 0
    assert flat["require_existing_dataset_cache"] is True
    assert flat["skip_final_evaluation"] is True
    assert flat["third_party_eval"] is False
    assert flat["ideal_Evalaution"] is False
    assert flat["training_timeout_seconds"] == 3600
    assert flat["evaluation_timeout_seconds"] == 1800

    reservations = plan["reservations"]
    assert plan["schema_version"] == "graphvae-attr-f1pr-reservation-plan-v1"
    assert [entry["budget_index"] for entry in reservations] == list(range(14))
    assert all(entry["training_seed"] == 0 for entry in reservations)
    assert [entry["parameters"] for entry in reservations[:5]] == [
        {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0},
        {"alpha_node_feat": 0.5, "alpha_edge_feat": 0.5},
        {"alpha_node_feat": 2.0, "alpha_edge_feat": 2.0},
        {"alpha_node_feat": 2.0, "alpha_edge_feat": 0.5},
        {"alpha_node_feat": 0.5, "alpha_edge_feat": 2.0},
    ]
    assert all(entry["parameters"] == {} for entry in reservations[5:])

    assert policy["study_name"] == "aids_attr_f1pr_search14_20260824a"
    assert policy["search"]["total_reservations"] == 14
    assert policy["search"]["anchor_reservations"] == 5
    assert policy["search"]["adaptive_reservations"] == 9
    assert policy["scheduler"]["exact_wave_sizes"] == [3, 2, 3, 3, 3]
    assert sum(policy["scheduler"]["exact_wave_sizes"]) == 14
    assert policy["sampler"] == {
        "name": "TPESampler",
        "seed": 83,
        "startup_trials": 5,
    }
    assert policy["objective"] == {
        "direction": "maximize",
        "evaluator_repeats": 5,
        "json_path": "evaluation.modes.decoded_node_edge.summary.f1_pr.mean",
        "max_graphs": 0,
        "selection_split": "validation",
        "test_access": False,
    }
    assert policy["selection"]["selected_candidates"] == 1
    assert policy["selection"]["selection_occurs_after_freeze"] is True
    assert policy["confirmation_required"] == {
        "alternate_candidate_fallback": False,
        "comparison": "single selected candidate versus uniform",
        "epochs": 250,
        "matched_training_seeds": [0, 1, 2],
    }


def test_aids_search_qualification_freezes_validation_only_winner():
    qualification = json.loads(
        SEARCH_QUALIFICATION_PATH.read_text(encoding="utf-8")
    )

    assert qualification["study"] == {
        "name": "aids_attr_f1pr_search14_20260824a",
        "contract_sha256": (
            "5d4d8c8157916f4e29610d1e9aebadc7fc7079f7b5fe077e118f95d8390e5221"
        ),
        "lifecycle": "FROZEN",
        "training_epochs": 100,
        "training_seed": 0,
    }
    assert qualification["reservations"] == {
        "reserved_total": 14,
        "complete": 14,
        "fail": 0,
        "running": 0,
        "waiting": 0,
        "unreserved_guard": 0,
        "max_parallel": 3,
    }
    objective = qualification["objective"]
    assert objective["json_path"] == (
        "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
    )
    assert objective["selection_split"] == "validation"
    assert objective["test_access"] is False
    assert objective["node_decoder_required"] is True
    assert objective["edge_decoder_required"] is True
    assert objective["accepted_validation_graphs"] == 184
    assert objective["evaluator_repeats"] == 5

    results = qualification["results"]
    assert [result["trial_number"] for result in results] == list(range(14))
    assert qualification["selection"] == {
        "rule": "maximum exact validation objective after freeze",
        "selected_trial_number": 12,
        "selected_weights": {
            "alpha_node_feat": 1.4240488736039931,
            "alpha_edge_feat": 2.468932652132638,
        },
        "validation_attr_f1pr": 0.43905872826081566,
        "validation_precision": 0.2891304347826087,
        "validation_recall": 0.9391304347826086,
        "uniform_search_validation_attr_f1pr": 0.3794140016611583,
        "search_fidelity_selected_minus_uniform": 0.05964472659965736,
        "confirmation_required": True,
        "alternate_candidate_fallback": False,
    }
    assert max(
        results, key=lambda result: result["validation_attr_f1pr"]
    )["trial_number"] == 12
    assert qualification["scheduler"]["exact_wave_sizes"] == [3, 2, 3, 3, 3]
    assert qualification["scheduler"]["launches_reconciled_terminal"] == 14
    assert qualification["scheduler"]["duplicate_or_replacement_trials"] == 0
    assert qualification["portable_restore"]["aggregate_outputs_match"] is True
    assert qualification["portable_restore"]["postgresql_access"] is False
    assert qualification["portable_restore"]["test_access"] is False
    assert qualification["cache"]["verified_on_controller_and_both_hosts"] is True
    assert qualification["artifact_audit"]["credential_matches"] == 0
    assert qualification["artifact_audit"][
        "unredacted_storage_url_matches"
    ] == 0
    assert qualification["artifact_audit"]["test_access_matches"] == 0


def test_aids_confirmation_contract_is_paired_and_test_free():
    config = load_yaml_mapping(CONFIRMATION_CONFIG_PATH)
    validate_base_config(config, tune_alpha_motif=False)
    flat = flatten_config(config)
    plan = json.loads(CONFIRMATION_RESERVATIONS_PATH.read_text(encoding="utf-8"))
    policy = json.loads(CONFIRMATION_POLICY_PATH.read_text(encoding="utf-8"))

    assert flat["dataset"] == "AIDS"
    assert flat["epoch_number"] == 250
    assert flat["expected_total_graphs"] == 1849
    assert flat["max_graphs"] == 0
    assert flat["require_existing_dataset_cache"] is True
    assert flat["skip_final_evaluation"] is True
    assert flat["third_party_eval"] is False
    assert flat["ideal_Evalaution"] is False
    assert flat["training_timeout_seconds"] == 7200
    assert flat["evaluation_timeout_seconds"] == 1800

    selected = {
        "alpha_node_feat": 1.4240488736039931,
        "alpha_edge_feat": 2.468932652132638,
    }
    uniform = {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0}
    reservations = plan["reservations"]
    assert [entry["budget_index"] for entry in reservations] == list(range(6))
    assert [entry["training_seed"] for entry in reservations] == [0, 0, 1, 1, 2, 2]
    assert [entry["parameters"] for entry in reservations] == [
        selected,
        uniform,
        selected,
        uniform,
        selected,
        uniform,
    ]

    assert policy["study_name"] == "aids_attr_f1pr_confirmation6_20260824a"
    assert policy["selection_source_study"] == (
        "aids_attr_f1pr_search14_20260824a"
    )
    assert policy["selection_source_trial_number"] == 12
    assert policy["selection_split"] == "validation"
    assert policy["objective_json_path"] == (
        "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
    )
    assert policy["test_access_during_confirmation"] is False
    assert policy["training_epoch_number"] == 250
    assert policy["paired_training_seeds"] == [0, 1, 2]
    assert policy["selected_weights"] == selected
    assert policy["uniform_weights"] == uniform
    assert policy["scheduler"] == {
        "max_parallel": 3,
        "exact_wave_sizes": [3, 3],
    }
    assert policy["no_reranking"] is True
    assert policy["alternate_candidate_fallback"] is False
    assert policy["held_out_access_before_freeze"] is False


def test_aids_confirmation_qualification_concludes_no_improvement():
    qualification = json.loads(
        CONFIRMATION_QUALIFICATION_PATH.read_text(encoding="utf-8")
    )

    assert qualification["study"] == {
        "name": "aids_attr_f1pr_confirmation6_20260824a",
        "contract_sha256": (
            "c009d609308f429b570a401c714384270702732fd5bb7209047fd3fcb00c22b5"
        ),
        "lifecycle": "FROZEN",
        "training_epochs": 250,
    }
    assert qualification["reservations"] == {
        "reserved_total": 6,
        "complete": 6,
        "fail": 0,
        "running": 0,
        "waiting": 0,
        "unreserved_guard": 0,
        "duplicate_or_replacement_trials": 0,
    }
    assert qualification["objective"] == {
        "json_path": "evaluation.modes.decoded_node_edge.summary.f1_pr.mean",
        "selection_split": "validation",
        "test_access": False,
        "held_out_access": False,
        "node_decoder_required": True,
        "edge_decoder_required": True,
        "accepted_validation_graphs": 184,
        "evaluator_repeats": 5,
    }
    pairs = qualification["pairs"]
    assert [pair["training_seed"] for pair in pairs] == [0, 1, 2]
    assert [pair["paired_difference"] for pair in pairs] == [
        -0.13488603117285514,
        0.01110061131044604,
        -0.0696959442374615,
    ]
    analysis = qualification["analysis"]
    assert analysis["selected_mean_validation_attr_f1pr"] == 0.5810227228423
    assert analysis["uniform_mean_validation_attr_f1pr"] == 0.6455165108755903
    assert analysis["mean_paired_difference"] == -0.0644937880332902
    assert analysis["confidence_interval_95"] == [
        -0.2461642964902339,
        0.11717672042365351,
    ]
    assert analysis["superiority_rule_met"] is False
    assert analysis["conclusion"] == "no_improvement"
    assert analysis["no_reranking"] is True
    assert analysis["alternate_candidate_fallback"] is False

    scheduler = qualification["scheduler_audit"]
    assert scheduler["predeclared_wave_sizes"] == [3, 3]
    assert scheduler["actual_wave_sizes"] == [3, 2, 1]
    assert scheduler["deviation"] is True
    assert scheduler["scientific_inputs_changed"] is False
    assert scheduler["reservation_count_or_identity_changed"] is False
    assert scheduler["paired_analysis_valid"] is True
    assert scheduler["controller_fix_required"] is True
    assert qualification["portable_restore"]["aggregate_outputs_match"] is True
    assert qualification["portable_restore"]["postgresql_access"] is False
    assert qualification["portable_restore"]["test_access"] is False
    assert qualification["cache"]["verified_on_controller_and_both_hosts"] is True
    assert qualification["artifact_audit"]["credential_matches"] == 0
    assert qualification["artifact_audit"][
        "unredacted_storage_url_matches"
    ] == 0
    assert qualification["artifact_audit"]["test_access_matches"] == 0


def test_aids_final_qualification_is_complete_test_free_and_leak_free():
    qualification = json.loads(FINAL_QUALIFICATION_PATH.read_text(encoding="utf-8"))

    assert qualification["code_head_before_evidence"] == (
        "d667d9fde8131d6c2d7e670591b3ebec292229e8"
    )
    assert qualification["studies"]["search"] == {
        "name": "aids_attr_f1pr_search14_20260824a",
        "contract_sha256": (
            "5d4d8c8157916f4e29610d1e9aebadc7fc7079f7b5fe077e118f95d8390e5221"
        ),
        "lifecycle": "FROZEN",
        "complete": 14,
        "fail": 0,
        "waiting": 0,
        "running": 0,
        "unreserved_guard": 0,
    }
    assert qualification["studies"]["confirmation"]["lifecycle"] == "FROZEN"
    assert qualification["studies"]["confirmation"]["complete"] == 6
    assert qualification["objective"] == {
        "json_path": "evaluation.modes.decoded_node_edge.summary.f1_pr.mean",
        "selection_split": "validation",
        "test_access": False,
        "held_out_access": False,
        "node_decoder_required": True,
        "edge_decoder_required": True,
    }
    assert qualification["tests"] == {
        "non_postgresql_distributed_and_attribute_bo": 117,
        "isolated_postgresql": 19,
        "failed": 0,
        "skipped": 0,
        "residual_isolated_postgresql_schemas": 0,
    }
    remediation = qualification["scheduler_remediation"]
    assert remediation["fully_fixed_reservations_use_contracted_parallelism"] is True
    assert remediation["mixed_or_adaptive_reservations_retain_startup_gating"] is True
    assert remediation["historical_confirmation_rerun"] is False
    assert remediation["scientific_result_changed"] is False
    assert qualification["cache"]["verified_on_controller_and_both_hosts"] is True
    assert qualification["security"]["tracked_credential_material_matches"] == 0
    assert qualification["security"][
        "unredacted_storage_url_matches_in_study_artifacts"
    ] == 0
    assert qualification["security"]["test_access_matches_in_study_artifacts"] == 0
    assert qualification["conclusion"]["decision"] == "no_improvement"
    assert qualification["conclusion"]["supported_default"] == "uniform_(1,1)"
