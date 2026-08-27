import json
from pathlib import Path

import yaml

from scripts.graphvae_attr_bo_distributed import validate_reservation_plan
from scripts.tune_graphvae_attribute_weights import flatten_config, validate_base_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "bayesian_optimization"


def _yaml(name):
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def _json(name):
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_aids_kl_configs_are_direct_weighted_validation_only_and_decoder_complete():
    expected_epochs = {
        "aids_graphvae_attr_f1pr_kl_smoke.yaml": 5,
        "aids_graphvae_attr_f1pr_kl_search.yaml": 250,
        "aids_graphvae_attr_f1pr_kl_confirmation.yaml": 250,
    }
    for name, epochs in expected_epochs.items():
        config = _yaml(name)
        flat = flatten_config(config)
        validate_base_config(config, tune_alpha_motif=False, tune_beta=True)
        assert flat["dataset"] == "AIDS"
        assert flat["split_mode"] == "paper_70_10_20"
        assert flat["split_seed"] == 123
        assert flat["train_fraction"] == 0.7
        assert flat["val_fraction"] == 0.1
        assert flat["epoch_number"] == epochs
        assert flat["beta"] is None
        assert flat["alpha_node_feat"] == 1.0
        assert flat["alpha_edge_feat"] == 1.0
        assert flat["use_graphvae_mm_bce_kl_weights"] is False
        assert flat["motif_loss"] is False
        assert flat["skip_final_evaluation"] is True
        assert flat["ideal_Evalaution"] is False
        assert flat["third_party_eval"] is False
        assert flat["max_graphs"] == 0
        assert flat["expected_total_graphs"] == 1849


def test_aids_kl_search_policy_and_exact_reservations_are_frozen():
    policy = _json("aids_attr_f1pr_kl_search_policy.json")
    plan = _json("aids_attr_f1pr_kl_search_reservations_15.json")["reservations"]
    assert policy["prerequisites"]["selected_primary_evaluator"] == "random_gin"
    assert policy["prerequisites"]["current_launch_authorized"] is False
    assert policy["objective"] == {
        "json_path": "evaluation.modes.decoded_node_edge.summary.f1_pr.mean",
        "direction": "maximize",
        "selection_split": "validation",
        "test_access": False,
        "skip_final_evaluation": True,
        "max_graphs": 0,
        "validation_graphs": 184,
        "evaluator_seeds": list(range(1000, 1010)),
    }
    assert policy["search"]["total_reservations"] == 15
    assert policy["search"]["anchor_reservations"] == 6
    assert policy["search"]["adaptive_reservations"] == 9
    assert policy["search"]["epoch_number"] == 250
    assert policy["scheduler"]["exact_wave_sizes"] == [3, 3, 3, 3, 3]
    assert policy["sampler"] == {
        "name": "TPESampler",
        "seed": 83,
        "startup_trials": 6,
        "constant_liar": True,
    }

    search_space = {
        name: bounds for name, bounds in policy["search"]["ranges"].items()
    }
    normalized = validate_reservation_plan(
        plan, expected_count=15, search_space=search_space
    )
    assert normalized == plan
    assert [row["budget_index"] for row in plan] == list(range(15))
    assert [row["parameters"] for row in plan[:6]] == [
        {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0, "beta": 1.0},
        {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0, "beta": 0.5},
        {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0, "beta": 2.0},
        {"alpha_node_feat": 0.5, "alpha_edge_feat": 0.5, "beta": 1.0},
        {"alpha_node_feat": 2.0, "alpha_edge_feat": 2.0, "beta": 1.0},
        {"alpha_node_feat": 1.0, "alpha_edge_feat": 2.0, "beta": 0.5},
    ]
    assert all(row["parameters"] == {} for row in plan[6:])
    assert all(row["training_seed"] == 0 for row in plan)


def test_aids_kl_confirmation_is_disjoint_and_cannot_be_launched_from_template():
    template = _json("aids_attr_f1pr_kl_confirmation_policy_template.json")
    search = _json("aids_attr_f1pr_kl_search_policy.json")
    assert template["reservation_plan"] is None
    assert template["status"] == "template_only_until_search_winner_is_frozen"
    assert template["uniform"] == {
        "alpha_node_feat": 1.0,
        "alpha_edge_feat": 1.0,
        "beta": 1.0,
    }
    assert template["training_seeds"] == [1, 2, 3]
    assert template["total_reservations"] == 6
    assert template["objective"]["test_access"] is False
    assert template["objective"]["skip_final_evaluation"] is True
    assert template["objective"]["evaluator_seeds"] == list(range(2000, 2010))
    assert set(template["objective"]["evaluator_seeds"]).isdisjoint(
        search["objective"]["evaluator_seeds"]
    )
    assert template["decision"]["alternate_candidate_fallback"] is False
    assert template["decision"]["extra_trials_after_result"] is False
