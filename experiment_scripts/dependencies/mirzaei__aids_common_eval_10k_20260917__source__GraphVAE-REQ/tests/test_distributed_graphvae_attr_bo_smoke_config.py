import json
from pathlib import Path

import pytest

from scripts.tune_graphvae_attribute_weights import (
    flatten_config,
    load_yaml_mapping,
    resolve_trial_config,
    validate_base_config,
)
from scripts.prepare_graphvae_attr_bo_cache import (
    DistributedContractError,
    validate_expected_total_graphs,
)
from scripts.graphvae_attr_bo_distributed import validate_reservation_plan


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "lobster_graphvae_attr_f1pr_smoke.yaml"
)
GRAPHCL_MOCK_CONFIG = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "lobster_graphcl_f1pr_mock.yaml"
)
SIGNAL_CONFIG = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "lobster_graphvae_attr_f1pr_signal.yaml"
)
SEARCH_PLAN = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "lobster_attr_f1pr_search_reservations_30.json"
)
HARDWARE_POLICY = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "lobster_attr_f1pr_candidate_hardware_policy.json"
)


def test_gate4_smoke_config_is_bounded_and_feature_complete():
    config = load_yaml_mapping(SMOKE_CONFIG)
    validate_base_config(config, tune_alpha_motif=False)
    flat = flatten_config(config)

    assert flat["dataset"] == "LOBSTER"
    assert flat["lobster_feature_schema"] == "optimal_v2"
    assert flat["split_mode"] == "paper_70_10_20"
    assert flat["train_fraction"] == 0.7
    assert flat["val_fraction"] == 0.1
    assert flat["split_seed"] == 123
    assert flat["dataset_loader_seed"] == 0
    assert flat["epoch_number"] == 2
    assert flat["train_batch_size"] == 8
    assert flat["graphEmDim"] == 64
    assert flat["require_existing_dataset_cache"] is True
    assert flat["skip_final_evaluation"] is True
    assert flat["third_party_eval"] is False
    assert flat["ideal_Evalaution"] is False
    assert flat["tiny_overfit"] is False

    qualification = config["bayesian_optimization_qualification"]
    assert qualification == {
        "expected_total_graphs": 100,
        "max_graphs": 8,
        "generation_batch_size": 4,
        "training_timeout_seconds": 600,
        "evaluation_timeout_seconds": 600,
        "termination_grace_seconds": 10,
    }


def test_gate4_smoke_config_renders_only_main_arguments(tmp_path):
    config = load_yaml_mapping(SMOKE_CONFIG)
    resolved = resolve_trial_config(
        config,
        {"alpha_node_feat": 0.25, "alpha_edge_feat": 4.0},
        trial_number=0,
        trial_directory=tmp_path / "trial_00000",
        training_seed=17,
        split_seed=123,
        device="cuda:0",
        require_existing_dataset_cache=True,
    )
    flat = flatten_config(resolved)

    assert "bayesian_optimization_qualification" not in resolved
    assert flat["dataset"] == "LOBSTER"
    assert flat["alpha_node_feat"] == 0.25
    assert flat["alpha_edge_feat"] == 4.0
    assert flat["seed"] == 17
    assert flat["split_seed"] == 123
    assert flat["device"] == "cuda:0"
    assert flat["require_existing_dataset_cache"] is True
    assert flat["skip_final_evaluation"] is True
    assert flat["third_party_eval"] is False


def test_graphcl_mock_config_preserves_exact_validation_cardinality():
    config = load_yaml_mapping(GRAPHCL_MOCK_CONFIG)
    validate_base_config(config, tune_alpha_motif=False)
    flat = flatten_config(config)
    qualification = config["bayesian_optimization_qualification"]

    assert flat["dataset"] == "LOBSTER"
    assert flat["split_mode"] == "paper_70_10_20"
    assert flat["skip_final_evaluation"] is True
    assert flat["third_party_eval"] is False
    assert qualification["max_graphs"] == 10
    assert qualification["generation_batch_size"] == 4


def test_gate4_smoke_cache_size_contract_is_enforced():
    config = flatten_config(load_yaml_mapping(SMOKE_CONFIG))
    manifest = {
        "splits": {
            "train": {"graph_count": 70},
            "validation": {"graph_count": 10},
            "test": {"graph_count": 20},
        }
    }

    validate_expected_total_graphs(config, manifest)
    manifest["splits"]["train"]["graph_count"] = 69
    with pytest.raises(DistributedContractError, match="expected 100, found 99"):
        validate_expected_total_graphs(config, manifest)


def test_lobster_signal_config_preserves_scientific_and_bounded_contract():
    config = load_yaml_mapping(SIGNAL_CONFIG)
    validate_base_config(config, tune_alpha_motif=False)
    flat = flatten_config(config)

    assert flat["dataset"] == "LOBSTER"
    assert flat["lobster_feature_schema"] == "optimal_v2"
    assert flat["split_mode"] == "paper_70_10_20"
    assert flat["split_seed"] == 123
    assert flat["dataset_loader_seed"] == 0
    assert flat["epoch_number"] == 2000
    assert flat["Vis_step"] == 400
    assert flat["train_batch_size"] == 200
    assert flat["graphEmDim"] == 1024
    assert flat["require_existing_dataset_cache"] is True
    assert flat["skip_final_evaluation"] is True
    assert flat["third_party_eval"] is False
    assert flat["ideal_Evalaution"] is False
    assert flat["tiny_overfit"] is False
    assert flat["alpha_node_feat"] == 1.0
    assert flat["alpha_edge_feat"] == 1.0

    assert config["bayesian_optimization_qualification"] == {
        "expected_total_graphs": 100,
        "max_graphs": 0,
        "generation_batch_size": 10,
        "training_timeout_seconds": 7200,
        "evaluation_timeout_seconds": 1200,
        "termination_grace_seconds": 10,
    }


def test_lobster_signal_config_renders_validation_only_trial(tmp_path):
    config = load_yaml_mapping(SIGNAL_CONFIG)
    resolved = resolve_trial_config(
        config,
        {"alpha_node_feat": 0.1, "alpha_edge_feat": 10.0},
        trial_number=0,
        trial_directory=tmp_path / "trial_00000",
        training_seed=0,
        split_seed=123,
        device="cuda:0",
        require_existing_dataset_cache=True,
    )
    flat = flatten_config(resolved)

    assert "bayesian_optimization_qualification" not in resolved
    assert flat["alpha_node_feat"] == 0.1
    assert flat["alpha_edge_feat"] == 10.0
    assert flat["seed"] == 0
    assert flat["split_seed"] == 123
    assert flat["skip_final_evaluation"] is True
    assert flat["third_party_eval"] is False


def test_lobster_search_plan_has_one_uniform_and_29_tpe_reservations():
    payload = json.loads(SEARCH_PLAN.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "graphvae-attr-f1pr-reservation-plan-v1"
    config = flatten_config(load_yaml_mapping(SIGNAL_CONFIG))
    search_space = {
        "alpha_node_feat": {"low": 0.01, "high": 10.0, "log": True},
        "alpha_edge_feat": {"low": 0.01, "high": 10.0, "log": True},
        "alpha_motif_loss": None,
    }
    plan = validate_reservation_plan(
        payload["reservations"],
        expected_count=30,
        search_space=search_space,
    )

    assert config["dataset"] == "LOBSTER"
    assert plan[0]["parameters"] == {
        "alpha_node_feat": 1.0,
        "alpha_edge_feat": 1.0,
    }
    assert all(entry["parameters"] == {} for entry in plan[1:])
    assert [entry["budget_index"] for entry in plan] == list(range(30))
    assert {entry["training_seed"] for entry in plan} == {0}


def test_lobster_candidate_hardware_policy_is_exact():
    policy = json.loads(HARDWARE_POLICY.read_text(encoding="utf-8"))
    assert policy["schema_version"] == "graphvae-attr-f1pr-hardware-policy-v1"
    assert policy["attr_f1pr_abs_tolerance"] == 0.02
    assert policy["checkpoint_byte_equality_expected"] is False
    assert policy["required_runtime_fingerprint"] == (
        "e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1"
    )
    assert policy["candidate_pool"] == [
        "cs-cl-09:cuda:1",
        "cs-cl-13:cuda:0",
        "cs-cl-16:cuda:0",
        "cs-cl-17:cuda:0",
        "cs-cl-17:cuda:1",
        "cs-cl-19:cuda:0",
        "cs-cl-19:cuda:1",
        "cs-cl-26:cuda:0",
        "cs-cl-26:cuda:1",
    ]
