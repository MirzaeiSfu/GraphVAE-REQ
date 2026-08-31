import json
from pathlib import Path

import yaml

from loss_weight_utils import apply_kia_bce_kl_weights
from scripts.graphvae_attr_bo_distributed import validate_reservation_plan
from scripts.tune_graphvae_attribute_weights import flatten_config, validate_base_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "bayesian_optimization"


def _json(name):
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_aids_kia_comparison_is_statistics_free_and_matches_last_aids_contract():
    config = yaml.safe_load(
        (CONFIG_ROOT / "aids_graphvae_attr_f1pr_kia_bce_kl_comparison.yaml").read_text(
            encoding="utf-8"
        )
    )
    flat = flatten_config(config)
    validate_base_config(config, tune_alpha_motif=False, tune_beta=False)
    assert flat["dataset"] == "AIDS"
    assert flat["model"] == "GraphVAE"
    assert flat["motif_loss"] is False
    assert flat["use_graphvae_mm_bce_kl_weights"] is True
    assert flat["beta"] is None
    assert flat["epoch_number"] == 250
    assert flat["split_mode"] == "paper_70_10_20"
    assert flat["split_seed"] == 123
    assert flat["dataset_loader_seed"] == 0
    assert flat["train_batch_size"] == 200
    assert flat["skip_final_evaluation"] is True
    assert flat["ideal_Evalaution"] is False
    assert apply_kia_bce_kl_weights([1.0, 1.0], "AIDS", True) == [50.0, 2000.0]


def test_aids_kia_comparison_has_exact_three_fixed_candidates_and_uniform_reuse():
    policy = _json("aids_kia_bce_kl_comparison_policy.json")
    reservations = _json("aids_kia_bce_kl_comparison_reservations_3.json")[
        "reservations"
    ]
    search_space = {
        "alpha_node_feat": {"low": 0.1, "high": 1.0, "log": True},
        "alpha_edge_feat": {"low": 0.1, "high": 1.0, "log": True},
        "alpha_motif_loss": None,
        "motif_opt_in": False,
        "fixed_parameters": None,
    }
    assert validate_reservation_plan(
        reservations, expected_count=3, search_space=search_space
    ) == reservations
    assert [row["parameters"] for row in reservations] == [
        {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0},
        {"alpha_node_feat": 1.0, "alpha_edge_feat": 0.1},
        {"alpha_node_feat": 0.1, "alpha_edge_feat": 1.0},
    ]
    assert all(row["training_seed"] == 0 for row in reservations)
    assert policy["scientific_contract"]["graph_statistics_enabled"] is False
    assert policy["scientific_contract"]["generation_seeds"] == [123, 124, 125]
    assert policy["scientific_contract"]["evaluator_seeds"] == list(range(1000, 1010))
    assert policy["study"]["total_reservations"] == 3
    assert policy["study"]["exact_wave_sizes"] == [3]
    assert policy["uniform_reuse"]["checkpoint_sha256"] == (
        "d0ba5db16be2a4800675a2aa7b238bc8ae6d8ea02b868afdd509ba8eddca915b"
    )
    assert policy["uniform_reuse"]["retrain_uniform"] is False
    assert policy["decision"]["no_test_or_held_out_evaluation"] is True


def test_aids_kia_prelaunch_is_exact_ready_and_unclaimed():
    policy = _json("aids_kia_bce_kl_comparison_policy.json")
    prelaunch = _json("aids_kia_bce_kl_comparison_prelaunch.json")
    assert policy["status"] == "ready_launch_authorized"
    assert prelaunch["study_contract_sha256"] == (
        "262ac59a6d3b5ea96b12b4c8e2130ca98f2cca4c3ed14d06cd13de384006da0c"
    )
    assert prelaunch["lifecycle"] == "READY"
    assert prelaunch["scientific_contract"]["graph_statistics_enabled"] is False
    assert prelaunch["scientific_contract"]["test_access"] is False
    assert prelaunch["worker_qualification"][
        "postgresql_verify_full_load_study_on_both_hosts"
    ] is True
    assert prelaunch["uniform_reuse_proof"]["byte_identical"] is True
    assert prelaunch["reservation_budget"]["total"] == 3
    assert prelaunch["reservation_budget"]["WAITING"] == 3
    assert prelaunch["reservation_budget"]["RUNNING"] == 0
    assert prelaunch["reservation_budget"]["UNRESERVED_GUARD"] == 0
    assert prelaunch["authorization"]["study_launched"] is False
    assert prelaunch["authorization"]["reservation_claims"] == 0
