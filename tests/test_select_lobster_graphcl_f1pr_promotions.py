import copy
import json

import pytest
import yaml

from scripts.select_lobster_graphcl_f1pr_promotions import (
    CONFIG_ROOT,
    DEFAULT_COMPLETION,
    DEFAULT_CONFIG,
    DEFAULT_CONTRACT,
    DEFAULT_POLICY,
    DEFAULT_RESERVATIONS,
    PromotionContractError,
    build_outputs,
    select_unique_promotions,
)


def _payload(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_phase_b_promotions_are_reproducible_and_unique():
    completion = _payload(DEFAULT_COMPLETION)
    policy = _payload(DEFAULT_POLICY)
    contract, reservations = build_outputs(
        completion,
        policy,
        completion_sha256="9a09d1592b1aa7c3a0ed672964112ad8ad2ea8c1911b63005f4a29c19abf18a7",
        policy_sha256="c02afa10bcd17a172c5fbdece3ab4acdd4d727b522f2cf92f29e177bd4659bb2",
        phase_b_config_sha256="placeholder",
    )
    committed = _payload(DEFAULT_CONTRACT)
    assert contract["promotions"] == committed["promotions"]
    assert contract["clarification"] == committed["clarification"]
    assert reservations == _payload(DEFAULT_RESERVATIONS)

    promotions = committed["promotions"]
    assert [record["source_budget_index"] for record in promotions] == [0, 1, 5]
    assert [record["promotion_role"] for record in promotions] == [
        "uniform",
        "best_nonuniform",
        "contrasting_anchor",
    ]
    weights = [tuple(record["weights"].values()) for record in promotions]
    assert len(set(weights)) == 3
    assert committed["execution"] == {
        "phase_b_study_created": False,
        "phase_b_training_started": False,
        "adaptive_bo": False,
        "held_out_or_test_evaluation": False,
    }
    assert committed["objective_contract"]["selection_split"] == "validation"
    assert committed["objective_contract"]["test_access"] is False


def test_best_nonuniform_tie_break_uses_lowest_budget_index():
    completion = _payload(DEFAULT_COMPLETION)
    policy = _payload(DEFAULT_POLICY)
    completion = copy.deepcopy(completion)
    completion["results"][4]["mean"] = completion["results"][1]["mean"]
    selected = select_unique_promotions(completion, policy)
    assert selected[1]["source_budget_index"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["study"].update({"lifecycle": "READY"}),
        lambda payload: payload["objective_contract"].update({"test_access": True}),
        lambda payload: payload["phase_a_threshold_audit"].update(
            {"phase_a_checks_passed": False}
        ),
        lambda payload: payload["results"][1].update(
            {"weights": {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0}}
        ),
    ],
)
def test_selector_fails_closed_on_unsafe_phase_a_evidence(mutation):
    completion = copy.deepcopy(_payload(DEFAULT_COMPLETION))
    mutation(completion)
    with pytest.raises(PromotionContractError):
        select_unique_promotions(completion, _payload(DEFAULT_POLICY))


def test_phase_b_config_is_exact_and_test_free():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert DEFAULT_CONFIG.parent == CONFIG_ROOT
    assert config["data"]["dataset"] == "LOBSTER"
    assert config["data"]["split_mode"] == "paper_70_10_20"
    assert config["data"]["split_seed"] == 123
    assert config["experiment"]["epoch_number"] == 10000
    assert config["runtime"]["skip_final_evaluation"] is True
    assert config["runtime"]["save_validation_checkpoints"] is False
    assert config["bayesian_optimization_qualification"] == {
        "expected_total_graphs": 100,
        "max_graphs": 10,
        "generation_batch_size": 10,
        "training_timeout_seconds": 7200,
        "evaluation_timeout_seconds": 1200,
        "termination_grace_seconds": 10,
    }
