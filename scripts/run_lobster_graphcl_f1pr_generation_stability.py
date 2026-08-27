#!/usr/bin/env python3
"""Run the frozen Gate-5 LOBSTER GraphCL generation-seed characterization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


class StabilityContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def resolve_under(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise StabilityContractError(f"Unsafe relative path: {value!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise StabilityContractError(f"Path escapes repository root: {value!r}") from exc
    return resolved


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != (
        "lobster-graphcl-f1pr-gate5-generation-stability-contract-v1"
    ):
        raise StabilityContractError("Unsupported stability contract schema.")
    if contract.get("generation_seeds") != [123, 124, 125]:
        raise StabilityContractError("Generation seeds differ from frozen policy.")
    if contract.get("reuse_generation_seed_123") is not True:
        raise StabilityContractError("Seed 123 must reuse the exact Phase-B evaluation.")
    if contract.get("new_evaluation_count") != 8:
        raise StabilityContractError("The exact new-evaluation budget must be eight.")
    clarification = contract.get("clarification") or {}
    if (
        clarification.get("phase_a_best_is_uniform") is not True
        or clarification.get("duplicate_evaluation_forbidden") is not True
        or clarification.get("deduplicated_policy_candidates") != ["uniform"]
        or clarification.get("additional_candidate")
        != "phase_b_best_edge_emphasis"
        or clarification.get("changes_gate5_no_go_decision") is not False
    ):
        raise StabilityContractError("Candidate de-duplication contract differs.")
    candidates = contract.get("candidates") or []
    if [record.get("label") for record in candidates] != [
        "uniform",
        "phase_b_best_edge_emphasis",
    ]:
        raise StabilityContractError("Stability candidate identities differ.")
    for candidate in candidates:
        replicates = candidate.get("replicates") or []
        if [item.get("training_seed") for item in replicates] != [0, 1]:
            raise StabilityContractError("Training-seed identities differ.")
    evaluator = contract.get("evaluator") or {}
    if (
        evaluator.get("selection_split") != "validation"
        or evaluator.get("test_access") is not False
        or evaluator.get("skip_final_evaluation") is not True
        or evaluator.get("node_feature_decoder_required") is not True
        or evaluator.get("edge_feature_decoder_required") is not True
        or evaluator.get("compatibility_objective_json_path")
        != "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
    ):
        raise StabilityContractError("Evaluator scientific contract differs.")
    execution = contract.get("execution") or {}
    if (
        execution.get("max_parallel") != 1
        or execution.get("physical_gpu") != 1
        or execution.get("held_out_or_test_evaluation") is not False
        or execution.get("adaptive_bo") is not False
    ):
        raise StabilityContractError("Execution contract differs.")


def validate_evaluation(
    path: Path,
    *,
    expected_seed: int,
    expected_checkpoint_sha256: str,
) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        value = float(payload["summary"]["f1_pr"]["mean"])
        compatibility = float(
            payload["evaluation"]["modes"]["decoded_node_edge"]["summary"]
            ["f1_pr"]["mean"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StabilityContractError("Evaluation objective path is missing.") from exc
    if (
        payload.get("schema_version") != "lobster-graphcl-f1pr-evaluation-v1"
        or payload.get("split") != "validation"
        or payload.get("test_access") is not False
        or payload.get("skip_final_evaluation") is not True
        or payload.get("generation_seed") != expected_seed
        or payload.get("objective_json_path") != "summary.f1_pr.mean"
        or payload.get("compatibility_objective_json_path")
        != "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
        or payload.get("feature_dimensions") != {"node": 14, "edge": 11}
        or payload.get("integrity", {}).get("graphvae_checkpoint_sha256")
        != expected_checkpoint_sha256
        or not math.isfinite(value)
        or value != compatibility
    ):
        raise StabilityContractError("Evaluation violates the frozen objective contract.")
    return value


def build_new_tasks(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_label": candidate["label"],
            "training_seed": replicate["training_seed"],
            "generation_seed": generation_seed,
            "checkpoint": replicate["checkpoint"],
            "checkpoint_sha256": replicate["checkpoint_sha256"],
            "resolved_config": replicate["resolved_config"],
            "resolved_config_sha256": replicate["resolved_config_sha256"],
        }
        for candidate in contract["candidates"]
        for replicate in candidate["replicates"]
        for generation_seed in contract["generation_seeds"]
        if generation_seed != 123
    ]


def aggregate_records(
    contract: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    candidate_summaries = []
    for candidate in contract["candidates"]:
        label = candidate["label"]
        per_generation_seed = []
        for generation_seed in contract["generation_seeds"]:
            selected = sorted(
                (
                    record
                    for record in records
                    if record["candidate_label"] == label
                    and record["generation_seed"] == generation_seed
                ),
                key=lambda record: record["training_seed"],
            )
            if [record["training_seed"] for record in selected] != [0, 1]:
                raise StabilityContractError("A candidate/seed cell is incomplete.")
            values = [float(record["value"]) for record in selected]
            per_generation_seed.append(
                {
                    "generation_seed": generation_seed,
                    "training_seed_values": values,
                    "mean": statistics.mean(values),
                }
            )
        means = [record["mean"] for record in per_generation_seed]
        candidate_summaries.append(
            {
                "label": label,
                "weights": candidate["weights"],
                "per_generation_seed": per_generation_seed,
                "minimum_mean": min(means),
                "maximum_mean": max(means),
                "within_candidate_range": max(means) - min(means),
            }
        )
    maximum_range = max(
        candidate["within_candidate_range"] for candidate in candidate_summaries
    )
    threshold = float(
        contract["dominance_rule"]["phase_b_best_minus_uniform_absolute"]
    )
    return {
        "candidates": candidate_summaries,
        "maximum_within_candidate_range": maximum_range,
        "phase_b_best_minus_uniform_absolute": threshold,
        "generation_seed_variation_dominates_candidate_difference": (
            maximum_range > threshold
        ),
        "dominance_rule_passed": maximum_range <= threshold,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = args.repository_root.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract)
    phase_b_root = resolve_under(
        repository_root, contract["study_source"]["phase_b_root"]
    )
    output_root = resolve_under(repository_root, contract["execution"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    frozen_contract = output_root / "contract.json"
    if frozen_contract.exists():
        if json.loads(frozen_contract.read_text(encoding="utf-8")) != contract:
            raise StabilityContractError("Output root contains a different contract.")
    else:
        atomic_json(frozen_contract, contract)
    completion_path = output_root / "generation_stability_completion.json"
    if completion_path.exists():
        return json.loads(completion_path.read_text(encoding="utf-8"))

    evaluator = contract["evaluator"]
    records = []
    for candidate in contract["candidates"]:
        for replicate in candidate["replicates"]:
            checkpoint = resolve_under(phase_b_root, replicate["checkpoint"])
            config = resolve_under(phase_b_root, replicate["resolved_config"])
            if (
                sha256_file(checkpoint) != replicate["checkpoint_sha256"]
                or sha256_file(config) != replicate["resolved_config_sha256"]
            ):
                raise StabilityContractError("Checkpoint or config hash differs.")
            for generation_seed in contract["generation_seeds"]:
                if generation_seed == 123:
                    result_path = resolve_under(
                        phase_b_root, replicate["generation_seed_123_evaluation"]
                    )
                    if sha256_file(result_path) != (
                        replicate["generation_seed_123_evaluation_sha256"]
                    ):
                        raise StabilityContractError("Reused seed-123 evaluation differs.")
                    source = "reused_phase_b"
                else:
                    result_dir = (
                        output_root
                        / "evaluations"
                        / candidate["label"]
                        / f"training_seed_{replicate['training_seed']}"
                        / f"generation_seed_{generation_seed}"
                    )
                    result_path = result_dir / "graphcl_f1pr.json"
                    if not result_path.exists():
                        if result_dir.exists():
                            raise StabilityContractError(
                                "Incomplete evaluation root exists; refusing overwrite."
                            )
                        command = [
                            str(args.python),
                            str(resolve_under(repository_root, evaluator["script"])),
                            "--run-dir",
                            str(phase_b_root),
                            "--config",
                            str(config),
                            "--checkpoint",
                            str(checkpoint),
                            "--cache-path",
                            str(resolve_under(repository_root, evaluator["cache_path"])),
                            "--reference",
                            str(resolve_under(repository_root, evaluator["reference"])),
                            "--encoder-bundle-manifest",
                            str(
                                resolve_under(
                                    repository_root,
                                    evaluator["encoder_bundle_manifest"],
                                )
                            ),
                            "--encoder-bundle-manifest-sha256",
                            evaluator["encoder_bundle_manifest_sha256"],
                            "--campaign-root",
                            str(resolve_under(repository_root, evaluator["campaign_root"])),
                            "--dependency-root",
                            str(resolve_under(repository_root, evaluator["dependency_root"])),
                            "--graphcl-runtime-sha256",
                            evaluator["graphcl_runtime_sha256"],
                            "--upstream-repo",
                            evaluator["upstream_repo"],
                            "--python",
                            str(args.python),
                            "--generation-seed",
                            str(generation_seed),
                            "--max-graphs",
                            str(evaluator["max_graphs"]),
                            "--generation-batch-size",
                            str(evaluator["generation_batch_size"]),
                            "--nearest-k",
                            str(evaluator["nearest_k"]),
                            "--adjacency-threshold",
                            str(evaluator["adjacency_threshold"]),
                            "--device",
                            args.device,
                            "--output-dir",
                            str(result_dir),
                        ]
                        completed = subprocess.run(
                            command,
                            cwd=str(repository_root),
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=float(evaluator["timeout_seconds"]),
                            check=False,
                        )
                        if completed.returncode:
                            raise StabilityContractError(
                                "GraphCL evaluation failed; incomplete root is preserved."
                            )
                    source = "new_evaluation"
                value = validate_evaluation(
                    result_path,
                    expected_seed=generation_seed,
                    expected_checkpoint_sha256=replicate["checkpoint_sha256"],
                )
                if generation_seed == 123 and value != replicate[
                    "generation_seed_123_value"
                ]:
                    raise StabilityContractError("Reused seed-123 objective differs.")
                records.append(
                    {
                        "candidate_label": candidate["label"],
                        "training_seed": replicate["training_seed"],
                        "generation_seed": generation_seed,
                        "value": value,
                        "source": source,
                        "evaluation_sha256": sha256_file(result_path),
                    }
                )
    stability = aggregate_records(contract, records)
    completion = {
        "schema_version": "lobster-graphcl-f1pr-gate5-generation-stability-v1",
        "study_contract_sha256": contract["study_source"]["phase_b_contract_sha256"],
        "stability_contract_sha256": canonical_sha256(contract),
        "selection_split": "validation",
        "test_access": False,
        "skip_final_evaluation": True,
        "records": records,
        "stability": stability,
        "gate5_decision_before_stability": "qualification_failed",
        "gate5_decision_after_stability": "qualification_failed",
        "adaptive_bo_authorized": False,
        "finished_at_unix": time.time(),
    }
    atomic_json(completion_path, completion)
    atomic_json(
        output_root / "COMPLETE.json",
        {
            "schema_version": "lobster-graphcl-f1pr-generation-stability-complete-v1",
            "stability_contract_sha256": completion["stability_contract_sha256"],
            "completion_sha256": sha256_file(completion_path),
            "test_access": False,
        },
    )
    return completion


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "completed": True,
                "dominance_rule_passed": result["stability"]["dominance_rule_passed"],
                "test_access": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
