from pathlib import Path

import pytest

from scripts.tune_graphvae_attribute_weights import (
    flatten_config,
    load_yaml_mapping,
    resolve_trial_config,
    validate_base_config,
)


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "lobster_graphvae_attr_f1pr_smoke.yaml"
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
