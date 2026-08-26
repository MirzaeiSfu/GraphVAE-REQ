#!/usr/bin/env python3
"""Select the three unique, fixed Gate 5 Phase B fidelity candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "bayesian_optimization"
DEFAULT_COMPLETION = CONFIG_ROOT / "lobster_graphcl_f1pr_gate5_phase_a_completion.json"
DEFAULT_POLICY = CONFIG_ROOT / "lobster_graphcl_f1pr_gate5_policy.json"
DEFAULT_CONFIG = CONFIG_ROOT / "lobster_graphcl_f1pr_fidelity.yaml"
DEFAULT_CONTRACT = CONFIG_ROOT / "lobster_graphcl_f1pr_gate5_phase_b_contract.json"
DEFAULT_RESERVATIONS = (
    CONFIG_ROOT / "lobster_graphcl_f1pr_promoted_reservations_3.json"
)

EXPECTED_SELECTION_POLICY = [
    "uniform_(1,1)",
    "maximum_finite_phase_a_mean_with_lowest_budget_index_tie_break",
    "first_nonduplicate_from_edge_emphasis_node_emphasis_common_weak_"
    "common_strong_previous_random_gin",
]
CONTRAST_PRIORITY = (
    "edge_emphasis",
    "node_emphasis",
    "common_weak",
    "common_strong",
    "previous_random_gin",
)
UNIFORM_WEIGHTS = (1.0, 1.0)
PHASE_B_STUDY_NAME = "lobster_graphcl_f1pr_promoted10000_20260826b"
FAILED_PRECREATION_STUDY_NAME = "lobster_graphcl_f1pr_promoted10000_20260826a"


class PromotionContractError(ValueError):
    """Raised when Phase A evidence cannot safely determine Phase B."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PromotionContractError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _weights(result: Mapping[str, Any]) -> tuple[float, float]:
    raw = result.get("weights")
    if not isinstance(raw, Mapping):
        raise PromotionContractError("Every Phase A result requires weights.")
    try:
        node = float(raw["alpha_node_feat"])
        edge = float(raw["alpha_edge_feat"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionContractError("Candidate weights must be numeric.") from exc
    if not math.isfinite(node) or not math.isfinite(edge) or node <= 0 or edge <= 0:
        raise PromotionContractError("Candidate weights must be finite and positive.")
    return node, edge


def _validated_results(completion: Mapping[str, Any]) -> list[dict[str, Any]]:
    study = completion.get("study", {})
    states = study.get("reserved_states", {}) if isinstance(study, Mapping) else {}
    if study.get("lifecycle") != "FROZEN" or states != {
        "RESERVED_TOTAL": 6,
        "WAITING": 0,
        "RUNNING": 0,
        "COMPLETE": 6,
        "FAIL": 0,
        "OTHER": 0,
        "UNRESERVED_GUARD": 0,
    }:
        raise PromotionContractError("Phase A must be frozen with exactly six COMPLETE rows.")
    objective = completion.get("objective_contract", {})
    if (
        objective.get("selection_split") != "validation"
        or objective.get("test_access") is not False
        or objective.get("node_feature_decoder_required") is not True
        or objective.get("edge_feature_decoder_required") is not True
    ):
        raise PromotionContractError("Phase A objective contract is not promotion-safe.")
    threshold = completion.get("phase_a_threshold_audit", {})
    if (
        threshold.get("phase_a_checks_passed") is not True
        or threshold.get("adaptive_bo_authorized") is not False
    ):
        raise PromotionContractError("Phase A stability checks must pass before promotion.")
    results = completion.get("results")
    if not isinstance(results, list) or len(results) != 6:
        raise PromotionContractError("Phase A must contain exactly six result records.")
    normalized: list[dict[str, Any]] = []
    seen_budget_indices: set[int] = set()
    seen_labels: set[str] = set()
    seen_weights: set[tuple[float, float]] = set()
    for raw in results:
        if not isinstance(raw, Mapping):
            raise PromotionContractError("Phase A result records must be objects.")
        budget_index = raw.get("budget_index")
        label = raw.get("label")
        mean = raw.get("mean")
        if isinstance(budget_index, bool) or not isinstance(budget_index, int):
            raise PromotionContractError("Budget indices must be integers.")
        if not isinstance(label, str) or not label:
            raise PromotionContractError("Candidate labels must be nonempty strings.")
        if isinstance(mean, bool) or not isinstance(mean, (int, float)):
            raise PromotionContractError("Candidate means must be numeric.")
        mean = float(mean)
        weights = _weights(raw)
        if not math.isfinite(mean):
            raise PromotionContractError("Candidate means must be finite.")
        if budget_index in seen_budget_indices or label in seen_labels:
            raise PromotionContractError("Phase A candidate identities must be unique.")
        if weights in seen_weights:
            raise PromotionContractError("Phase A candidate weights must be unique.")
        seen_budget_indices.add(budget_index)
        seen_labels.add(label)
        seen_weights.add(weights)
        normalized.append(
            {
                "source_budget_index": budget_index,
                "label": label,
                "weights": {
                    "alpha_node_feat": weights[0],
                    "alpha_edge_feat": weights[1],
                },
                "phase_a_mean": mean,
                "phase_a_rank": int(raw["rank"]),
            }
        )
    if seen_budget_indices != set(range(6)):
        raise PromotionContractError("Phase A budget indices must be exactly 0 through 5.")
    return normalized


def select_unique_promotions(
    completion: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return uniform, best nonuniform, and the first contrasting nonduplicate."""

    if policy.get("phase_b", {}).get("selection") != EXPECTED_SELECTION_POLICY:
        raise PromotionContractError("The frozen Phase B selection policy changed.")
    results = _validated_results(completion)
    uniform = [
        result
        for result in results
        if result["label"] == "uniform"
        and _weights(result) == UNIFORM_WEIGHTS
    ]
    if len(uniform) != 1:
        raise PromotionContractError("Exactly one explicit uniform candidate is required.")

    selected: list[dict[str, Any]] = []
    selected_weights: set[tuple[float, float]] = set()

    def add(result: Mapping[str, Any], role: str) -> None:
        weights = _weights(result)
        if weights in selected_weights:
            raise PromotionContractError("Promotion candidates must be weight-unique.")
        selected_weights.add(weights)
        record = dict(result)
        record["promotion_role"] = role
        record["promotion_index"] = len(selected)
        selected.append(record)

    add(uniform[0], "uniform")
    nonuniform = sorted(
        (result for result in results if _weights(result) != UNIFORM_WEIGHTS),
        key=lambda result: (-float(result["phase_a_mean"]), result["source_budget_index"]),
    )
    if not nonuniform:
        raise PromotionContractError("No finite nonuniform candidate is available.")
    add(nonuniform[0], "best_nonuniform")

    by_label = {result["label"]: result for result in results}
    contrast = next(
        (
            by_label[label]
            for label in CONTRAST_PRIORITY
            if label in by_label and _weights(by_label[label]) not in selected_weights
        ),
        None,
    )
    if contrast is None:
        raise PromotionContractError("No predeclared contrasting nonduplicate is available.")
    add(contrast, "contrasting_anchor")
    if len(selected) != 3 or len(selected_weights) != 3:
        raise PromotionContractError("Phase B requires exactly three unique candidates.")
    return selected


def build_outputs(
    completion: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    completion_sha256: str,
    policy_sha256: str,
    phase_b_config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    promotions = select_unique_promotions(completion, policy)
    reservations = {
        "schema_version": "graphvae-attr-f1pr-reservation-plan-v1",
        "reservations": [
            {
                "budget_index": record["promotion_index"],
                "parameters": record["weights"],
                "training_seed": 0,
            }
            for record in promotions
        ],
    }
    contract = {
        "schema_version": "lobster-graphcl-f1pr-gate5-phase-b-contract-v1",
        "study": {
            "name": PHASE_B_STUDY_NAME,
            "controller_output_root": f"runs/{PHASE_B_STUDY_NAME}",
            "worker_artifact_root": f"runs/bayesian_optimization/{PHASE_B_STUDY_NAME}",
            "reserved_candidates": 3,
            "epoch_number": 10000,
            "graphvae_training_seeds": [0, 1],
            "generation_seed": 123,
            "study_seed": 23034,
            "tpe_startup_trials": 3,
            "heartbeat_interval_seconds": 60,
            "grace_period_seconds": 600,
            "max_parallel": 1,
            "phase_b_config_sha256": phase_b_config_sha256,
        },
        "source_evidence": {
            "phase_a_completion_sha256": completion_sha256,
            "frozen_policy_sha256": policy_sha256,
            "phase_a_study_contract_sha256": completion["study"][
                "study_contract_sha256"
            ],
            "phase_a_snapshot_sha256": completion["study"]["snapshot_sha256"],
        },
        "clarification": {
            "original_policy_preserved": True,
            "reason": "Uniform is both explicitly required and the Phase A maximum.",
            "deterministic_resolution": [
                "select the single exact uniform candidate",
                "select the maximum finite nonuniform mean with lowest budget-index tie break",
                "select the first weight-unique candidate in the frozen contrast priority",
            ],
            "contrast_priority": list(CONTRAST_PRIORITY),
            "post_result_parameter_discretion": False,
        },
        "promotions": promotions,
        "objective_contract": {
            "primary_path": "summary.f1_pr.mean",
            "compatibility_path": "evaluation.modes.decoded_node_edge.summary.f1_pr.mean",
            "selection_split": "validation",
            "test_access": False,
            "node_feature_decoder_required": True,
            "edge_feature_decoder_required": True,
            "partial_candidate_score_forbidden": True,
        },
        "execution": {
            "phase_b_study_created": False,
            "phase_b_training_started": False,
            "adaptive_bo": False,
            "held_out_or_test_evaluation": False,
        },
        "precreation_attempts": [
            {
                "study_name": FAILED_PRECREATION_STUDY_NAME,
                "status": "empty_unusable_preserved",
                "failure_phase": "cache_manifest_resolution_after_database_create",
                "failure_type": "FileNotFoundError",
                "database_trial_count": 0,
                "immutable_definition_created": False,
                "reservations_created": 0,
                "workers_launched": 0,
                "reservation_consumed": False,
                "reuse_forbidden": True,
            }
        ],
    }
    return contract, reservations


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, default=DEFAULT_COMPLETION)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--phase-b-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--reservation-output", type=Path, default=DEFAULT_RESERVATIONS)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    completion = _load_json(args.completion)
    policy = _load_json(args.policy)
    contract, reservations = build_outputs(
        completion,
        policy,
        completion_sha256=_sha256(args.completion),
        policy_sha256=_sha256(args.policy),
        phase_b_config_sha256=_sha256(args.phase_b_config),
    )
    if args.check:
        if _load_json(args.contract_output) != contract:
            raise PromotionContractError("Committed Phase B contract is not reproducible.")
        if _load_json(args.reservation_output) != reservations:
            raise PromotionContractError("Committed Phase B reservations are not reproducible.")
        print("Phase B promotion contract is reproducible.")
        return 0
    _atomic_write_json(args.contract_output, contract)
    _atomic_write_json(args.reservation_output, reservations)
    print("Wrote exactly three unique Gate 5 Phase B promotions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
