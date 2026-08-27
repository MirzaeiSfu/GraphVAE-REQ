import copy
import json
import os
from pathlib import Path

import pytest

from scripts.run_lobster_graphcl_f1pr_generation_stability import (
    StabilityContractError,
    aggregate_records,
    build_evaluator_environment,
    build_new_tasks,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "configs/bayesian_optimization"
    / "lobster_graphcl_f1pr_gate5_generation_stability_contract.json"
)


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _records(uniform, edge):
    records = []
    for label, values in (
        ("uniform", uniform),
        ("phase_b_best_edge_emphasis", edge),
    ):
        for generation_seed, pair in zip((123, 124, 125), values):
            for training_seed, value in enumerate(pair):
                records.append(
                    {
                        "candidate_label": label,
                        "generation_seed": generation_seed,
                        "training_seed": training_seed,
                        "value": value,
                    }
                )
    return records


def test_committed_contract_deduplicates_policy_and_is_test_free():
    contract = _contract()
    validate_contract(contract)
    assert contract["clarification"]["phase_a_best_is_uniform"] is True
    assert contract["clarification"]["deduplicated_policy_candidates"] == [
        "uniform"
    ]
    assert contract["clarification"]["additional_candidate"] == (
        "phase_b_best_edge_emphasis"
    )
    assert contract["clarification"]["changes_gate5_no_go_decision"] is False
    assert contract["evaluator"]["compatibility_objective_json_path"] == (
        "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
    )
    assert contract["evaluator"]["selection_split"] == "validation"
    assert contract["evaluator"]["test_access"] is False
    assert contract["evaluator"]["skip_final_evaluation"] is True
    assert contract["execution"]["held_out_or_test_evaluation"] is False
    assert contract["execution"]["adaptive_bo"] is False


def test_exactly_eight_new_evaluations_reuse_seed_123():
    tasks = build_new_tasks(_contract())
    assert len(tasks) == 8
    assert {task["generation_seed"] for task in tasks} == {124, 125}
    assert {
        (task["candidate_label"], task["training_seed"])
        for task in tasks
    } == {
        ("uniform", 0),
        ("uniform", 1),
        ("phase_b_best_edge_emphasis", 0),
        ("phase_b_best_edge_emphasis", 1),
    }


def test_evaluator_environment_mirrors_frozen_worker_pythonpath():
    environment = build_evaluator_environment(
        ROOT,
        ROOT / ".graphcl_deps",
        {"PYTHONPATH": "/existing/pythonpath", "SENTINEL": "safe"},
    )
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str((ROOT / ".graphcl_deps").resolve()),
        str((ROOT / "graph_evaluation/src").resolve()),
        "/existing/pythonpath",
    ]
    assert environment["SENTINEL"] == "safe"


def test_failed_a_root_is_preserved_empty_and_never_reused():
    contract = _contract()
    assert contract["execution"]["output_root"].endswith("20260826b")
    assert contract["precreation_attempts"] == [
        {
            "output_root": "runs/lobster_graphcl_f1pr_generation_stability_20260826a",
            "status": "empty_unusable_preserved",
            "failure_phase": "graphcl_runtime_import_before_generation",
            "failure_type": "ModuleNotFoundError",
            "missing_module": "GCL",
            "runner_omitted_dependency_pythonpath": True,
            "evaluation_attempt_roots": 2,
            "files_below_evaluation_attempt_roots": 0,
            "generated_graph_files": 0,
            "objective_files": 0,
            "scientific_evaluations_consumed": 0,
            "reuse_forbidden": True,
        }
    ]


def test_dominance_rule_passes_at_equality_and_fails_above_threshold():
    contract = _contract()
    threshold = contract["dominance_rule"][
        "phase_b_best_minus_uniform_absolute"
    ]
    passing = aggregate_records(
        contract,
        _records(
            uniform=((0.5, 0.5), (0.5 + threshold, 0.5 + threshold), (0.5, 0.5)),
            edge=((0.6, 0.6), (0.6, 0.6), (0.6, 0.6)),
        ),
    )
    assert passing["maximum_within_candidate_range"] == pytest.approx(threshold)
    assert passing["dominance_rule_passed"] is True
    failing = aggregate_records(
        contract,
        _records(
            uniform=(
                (0.5, 0.5),
                (0.5 + threshold + 0.001, 0.5 + threshold + 0.001),
                (0.5, 0.5),
            ),
            edge=((0.6, 0.6), (0.6, 0.6), (0.6, 0.6)),
        ),
    )
    assert failing["dominance_rule_passed"] is False
    assert failing["generation_seed_variation_dominates_candidate_difference"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"generation_seeds": [123, 124]}),
        lambda value: value.update({"new_evaluation_count": 9}),
        lambda value: value["clarification"].update(
            {"changes_gate5_no_go_decision": True}
        ),
        lambda value: value["evaluator"].update({"test_access": True}),
        lambda value: value["evaluator"].update({"selection_split": "test"}),
        lambda value: value["execution"].update({"max_parallel": 2}),
    ],
)
def test_contract_mutations_fail_closed(mutation):
    contract = copy.deepcopy(_contract())
    mutation(contract)
    with pytest.raises(StabilityContractError):
        validate_contract(contract)
