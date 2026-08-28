import json
from pathlib import Path

import yaml

from scripts.graphvae_attr_bo_distributed import validate_reservation_plan
from scripts.tune_graphvae_attribute_weights import flatten_config, validate_base_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "bayesian_optimization"
KL_CREDENTIAL_PATHS = (
    REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_AIDS_KL_GATE5_CREDENTIAL_ENV_PATHS.txt"
)


def _yaml(name):
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def _json(name):
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def _mapping(path):
    rows = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            host, value = line.split()
            rows[host] = value
    return rows


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
    assert policy["prerequisites"]["current_launch_authorized"] is True
    assert policy["prerequisites"]["credential_env_paths"] == (
        "CLUSTER_GRAPHVAE_ATTR_BO_AIDS_KL_GATE5_CREDENTIAL_ENV_PATHS.txt"
    )
    credentials = _mapping(KL_CREDENTIAL_PATHS)
    assert credentials == {
        "cs-cl-13": (
            "/local-scratch/graphvae-req-work/"
            ".graphvae-bo-credentials/gate5/worker.env"
        ),
        "cs-cl-17": (
            "/local-scratch/graphvae-req-work/"
            ".graphvae-bo-credentials/gate5/worker.env"
        ),
    }
    assert all("production" not in path for path in credentials.values())
    hardware = _json("aids_attr_f1pr_kl_hardware_policy.json")
    assert hardware["homogeneous_production_pool"] == [
        "cs-cl-13:cuda:0",
        "cs-cl-17:cuda:0",
        "cs-cl-17:cuda:1",
    ]
    assert hardware["attr_f1pr_abs_tolerance"] == 0.02
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


def test_aids_kl_search_prelaunch_is_exact_ready_and_unclaimed():
    policy = _json("aids_attr_f1pr_kl_search_policy.json")
    prelaunch = _json("aids_attr_f1pr_kl_search_prelaunch.json")
    assert policy["prerequisites"]["prelaunch_record"] == (
        "configs/bayesian_optimization/aids_attr_f1pr_kl_search_prelaunch.json"
    )
    assert prelaunch["study_name"] == policy["study_name"]
    assert prelaunch["contract_sha256"] == (
        "cf2defdc577ca878208f3777ada534c071d2551fc0bcc978d22aa1fd402a2e78"
    )
    assert prelaunch["immutable_inputs"]["source_commit"].startswith("df81ff2")
    assert prelaunch["scientific_contract"]["objective_json_path"] == (
        "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
    )
    assert prelaunch["scientific_contract"]["selection_split"] == "validation"
    assert prelaunch["scientific_contract"]["test_access"] is False
    assert prelaunch["scientific_contract"]["held_out_access"] is False
    assert prelaunch["scientific_contract"]["skip_final_evaluation"] is True
    assert prelaunch["scientific_contract"]["evaluator_seeds"] == list(
        range(1000, 1010)
    )
    assert prelaunch["reservation_budget"] == {
        "total": 15,
        "fixed_anchors": 6,
        "adaptive": 9,
        "max_parallel": 3,
        "exact_wave_sizes": [3, 3, 3, 3, 3],
        "sampler": "TPESampler",
        "sampler_seed": 83,
        "startup_trials": 6,
        "failed_reservations_replaced": False,
        "ambiguous_launches_duplicated": False,
    }
    assert prelaunch["postgresql_prelaunch"]["lifecycle"] == "READY"
    assert prelaunch["postgresql_prelaunch"]["reserved_states"] == {
        "WAITING": 15,
        "RUNNING": 0,
        "COMPLETE": 0,
        "FAIL": 0,
        "OTHER": 0,
        "UNRESERVED_GUARD": 0,
        "RESERVED_TOTAL": 15,
    }
    assert prelaunch["workers"]["preflight_slot_count"] == 3
    assert prelaunch["workers"]["active_worker_processes"] == 0
    assert prelaunch["workers"]["study_tmux_sessions"] == 0
    assert prelaunch["workers"]["launch_manifests"] == 0
    assert prelaunch["authorization"] == {
        "first_wave_budget_indexes": [0, 1, 2],
        "first_wave_parameters_fixed": True,
        "launch_authorized": True,
        "search_launched": False,
        "reservation_claims": 0,
        "reason": (
            "immutable study, exact budget, multi-host preflight, and definite "
            "no-launch state independently verified"
        ),
    }


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


def test_aids_kl_smoke_qualification_preserves_failure_and_is_validation_only():
    config = _yaml("aids_graphvae_attr_f1pr_kl_smoke.yaml")
    qualification = _json("aids_attr_f1pr_kl_smoke_qualification.json")
    smoke = qualification["real_smoke"]
    failed, completed = smoke["preserved_attempts"]

    assert config["bayesian_optimization_qualification"][
        "training_timeout_seconds"
    ] == 1200
    assert qualification["status"] == "i3_qualified_search_not_initialized"
    assert qualification["postgresql"]["controller_authentication"] is True
    assert qualification["postgresql"]["worker_authentication"] is True
    assert qualification["postgresql"]["qualification_passed"] is True
    assert qualification["postgresql"]["qualification_schema_created"] is False
    assert qualification["postgresql"]["search_study_created"] is False
    assert qualification["postgresql"]["isolated_suite"] == {
        "tests_passed": 19,
        "tests_failed": 0,
        "disposable_studies_only": True,
        "residual_test_studies": 0,
    }
    assert qualification["interpretation"]["i3_complete"] is True
    assert qualification["interpretation"]["scientific_weight_claim"] is False
    assert qualification["interpretation"]["search_launched"] is False
    assert qualification["interpretation"]["remaining_i3_blocker"] is None

    assert smoke["selection_split"] == "validation"
    assert smoke["validation_graphs"] == 184
    assert smoke["test_access"] is False
    assert smoke["held_out_access"] is False
    assert smoke["skip_final_evaluation"] is True
    assert smoke["objective_json_path"] == (
        "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
    )
    assert smoke["node_feature_decoder_required"] is True
    assert smoke["edge_feature_decoder_required"] is True
    assert smoke["evaluator_seeds"] == list(range(1000, 1010))
    assert smoke["resolved_parameter_scopes"] == {
        "loss.alpha_node_feat": smoke["parameters"]["alpha_node_feat"],
        "loss.alpha_edge_feat": smoke["parameters"]["alpha_edge_feat"],
        "model.beta": smoke["parameters"]["beta"],
        "loss.use_graphvae_mm_bce_kl_weights": False,
    }

    assert failed["status"] == "FAIL"
    assert failed["training_timeout_seconds"] == 600
    assert failed["replacement"] is False
    assert failed["orphan_processes"] == 0
    assert completed["status"] == "COMPLETE"
    assert completed["training_timeout_seconds"] == 1200
    assert completed["accepted_graphs"] == 184
    assert completed["objective"] == 0.00001999301714791553
    assert completed["artifacts_read_only"] is True
    assert qualification["integrity"]["credential_marker_files"] == 0
    assert qualification["integrity"]["test_or_held_out_true_files"] == 0
