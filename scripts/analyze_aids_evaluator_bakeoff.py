#!/usr/bin/env python3
"""Validate and analyze the matched AIDS Random-GIN/GraphCL bake-off."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from graphvae_attr_bo_distributed import (  # noqa: E402
    DistributedContractError,
    atomic_write_json,
)


CACHE_SHA256 = "6edcc3309fb1c3d366b0f87065aa1b2e2c7d23cbff92bc729053f44e874909bb"
VALIDATION_SPLIT_FINGERPRINT = (
    "ea6e38e034feb2c523263172d27f07af3ae1aaa99ea0ba875b739780706d6e66"
)
FEATURE_SCHEMA = "tu-quantile8-max40|export=decoded_node_edge"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DistributedContractError(f"Required bake-off JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DistributedContractError(f"Bake-off JSON must be an object: {path}")
    return value


def _finite_values(values: Sequence[Any], *, count: int, name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != count or not all(math.isfinite(value) for value in result):
        raise DistributedContractError(f"{name} must contain {count} finite values.")
    return result


def _assert_close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise DistributedContractError(f"{name} differs from its component values.")


def _manifest(path: Path, *, role: str, generation_seed: int, checkpoint_sha: str) -> dict:
    payload = _load_json(path)
    metadata = payload.get("metadata") or {}
    expected = {
        "dataset": "AIDS",
        "feature_mode": "decoded_node_edge",
        "feature_schema": FEATURE_SCHEMA,
        "split": "validation",
        "test_access": False,
        "generation_seed": generation_seed,
        "source_cache_sha256": CACHE_SHA256,
        "split_fingerprint": VALIDATION_SPLIT_FINGERPRINT,
        "checkpoint_sha256": checkpoint_sha,
        "collection_role": role,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise DistributedContractError(
                f"{role} collection metadata differs for {name}."
            )
    summary = payload.get("summary") or {}
    if (
        summary.get("graph_count") != 184
        or summary.get("node_feature_dim") != 56
        or summary.get("edge_feature_dim") != 3
    ):
        raise DistributedContractError(f"{role} collection dimensions/count differ.")
    digest = payload.get("collection_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise DistributedContractError(f"{role} collection digest is invalid.")
    return {"sha256": digest, "summary": summary, "metadata": metadata}


def _validate_random_gin(
    job_dir: Path,
    *,
    generation_seed: int,
    checkpoint_sha: str,
    evaluator_seeds: Sequence[int],
) -> dict[str, Any]:
    root = job_dir / "random_gin"
    payload = _load_json(root / "attributed_random_gin.json")
    exact = {
        "schema_version": "attributed-random-gin-v1",
        "split": "validation",
        "test_access": False,
        "skip_final_evaluation": True,
        "generation_seed": generation_seed,
        "evaluator_seeds": list(evaluator_seeds),
        "primary_mode": "decoded_node_edge",
    }
    for name, expected in exact.items():
        if payload.get(name) != expected:
            raise DistributedContractError(f"Random-GIN result differs for {name}.")
    counts = payload.get("graph_counts") or {}
    if any(counts.get(name) != 184 for name in (
        "accepted_per_collection",
        "generated_accepted",
        "reference_accepted",
        "validation_cache_count",
    )):
        raise DistributedContractError("Random-GIN graph counts differ from 184.")
    integrity = payload.get("integrity") or {}
    if integrity.get("cache_sha256") != CACHE_SHA256:
        raise DistributedContractError("Random-GIN cache hash differs.")
    if integrity.get("split_fingerprint") != VALIDATION_SPLIT_FINGERPRINT:
        raise DistributedContractError("Random-GIN validation fingerprint differs.")
    mode = (payload.get("evaluation") or {}).get("modes", {}).get(
        "decoded_node_edge", {}
    )
    values = _finite_values(
        (mode.get("per_repeat") or {}).get("f1_pr", []),
        count=len(evaluator_seeds),
        name="Random-GIN F1-PR repeats",
    )
    _assert_close(
        (mode.get("summary") or {}).get("f1_pr", {}).get("mean", float("nan")),
        statistics.fmean(values),
        name="Random-GIN F1-PR mean",
    )
    generated = _manifest(
        root / "generated_attributed_graphs.pt.json",
        role="generated",
        generation_seed=generation_seed,
        checkpoint_sha=checkpoint_sha,
    )
    reference = _manifest(
        root / "reference_attributed_graphs.pt.json",
        role="reference",
        generation_seed=generation_seed,
        checkpoint_sha=checkpoint_sha,
    )
    exports = payload.get("pyg_exports") or {}
    if (exports.get("generated") or {}).get("collection_sha256") != generated["sha256"]:
        raise DistributedContractError("Random-GIN generated export digest differs.")
    if (exports.get("reference") or {}).get("collection_sha256") != reference["sha256"]:
        raise DistributedContractError("Random-GIN reference export digest differs.")
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "generated_sha256": generated["sha256"],
        "reference_sha256": reference["sha256"],
    }


def _validate_graphcl(
    job_dir: Path,
    *,
    generation_seed: int,
    encoder_seeds: Sequence[int],
    generated_sha256: str,
    reference_sha256: str,
) -> dict[str, Any]:
    payload = _load_json(job_dir / "graphcl" / "evaluation.json")
    exact = {
        "checkpoint_count": len(encoder_seeds),
        "encoder": "graphcl",
        "engine": "contrastive-pyg-upstream",
        "feature_mode": "decoded_node_edge",
    }
    for name, expected in exact.items():
        if payload.get(name) != expected:
            raise DistributedContractError(f"GraphCL result differs for {name}.")
    rows = payload.get("per_checkpoint") or []
    if [row.get("checkpoint_seed") for row in rows] != list(encoder_seeds):
        raise DistributedContractError("GraphCL encoder seed order differs.")
    values = []
    for row in rows:
        if row.get("generated_sha256") != generated_sha256:
            raise DistributedContractError("GraphCL generated collection digest differs.")
        if row.get("reference_sha256") != reference_sha256:
            raise DistributedContractError("GraphCL reference collection digest differs.")
        for metadata_name in (
            "generated_metadata",
            "reference_metadata",
            "training_metadata",
        ):
            metadata = row.get(metadata_name) or {}
            if metadata.get("dataset") != "AIDS" or metadata.get("test_access") is not False:
                raise DistributedContractError(
                    f"GraphCL {metadata_name} is not AIDS/test-free."
                )
            if metadata.get("feature_schema") != FEATURE_SCHEMA:
                raise DistributedContractError(f"GraphCL {metadata_name} schema differs.")
        if row["generated_metadata"].get("generation_seed") != generation_seed:
            raise DistributedContractError("GraphCL generation seed differs.")
        if row["reference_metadata"].get("split") != "validation":
            raise DistributedContractError("GraphCL reference is not validation.")
        if row["training_metadata"].get("split") != "train":
            raise DistributedContractError("GraphCL encoder was not train-only.")
        values.append(float((row.get("metrics") or {}).get("f1_pr", float("nan"))))
    values = _finite_values(values, count=len(encoder_seeds), name="GraphCL F1-PR")
    _assert_close(
        (payload.get("summary") or {}).get("f1_pr", {}).get("mean", float("nan")),
        statistics.fmean(values),
        name="GraphCL F1-PR mean",
    )
    return {"values": values, "mean": statistics.fmean(values)}


def _method_summary(jobs: Mapping[tuple[str, int, int], dict], method: str) -> dict:
    paired_rows = []
    for training_seed in (0, 1, 2):
        for generation_seed in (123, 124, 125):
            selected = jobs[("selected", training_seed, generation_seed)][method]
            uniform = jobs[("uniform", training_seed, generation_seed)][method]
            differences = [a - b for a, b in zip(selected["values"], uniform["values"])]
            paired_rows.append(
                {
                    "training_seed": training_seed,
                    "generation_seed": generation_seed,
                    "mean_difference": statistics.fmean(differences),
                    "population_sd": statistics.pstdev(differences),
                }
            )
    overall_difference = statistics.fmean(
        [row["mean_difference"] for row in paired_rows]
    )
    overall_sign = 1 if overall_difference > 0 else -1 if overall_difference < 0 else 0
    sign_stability = sum(
        (1 if row["mean_difference"] > 0 else -1 if row["mean_difference"] < 0 else 0)
        == overall_sign
        for row in paired_rows
    )
    generation_rows = []
    for candidate in ("selected", "uniform"):
        for training_seed in (0, 1, 2):
            means = [
                jobs[(candidate, training_seed, generation_seed)][method]["mean"]
                for generation_seed in (123, 124, 125)
            ]
            generation_rows.append(
                {
                    "candidate": candidate,
                    "training_seed": training_seed,
                    "means": means,
                    "range": max(means) - min(means),
                }
            )
    return {
        "overall_mean_selected_minus_uniform": overall_difference,
        "mean_paired_difference_population_sd": statistics.fmean(
            [row["population_sd"] for row in paired_rows]
        ),
        "sign_stability_count_of_9": sign_stability,
        "mean_generation_seed_range": statistics.fmean(
            [row["range"] for row in generation_rows]
        ),
        "max_generation_seed_range": max(row["range"] for row in generation_rows),
        "paired_differences": paired_rows,
        "generation_seed_ranges": generation_rows,
    }


def analyze(contract_path: Path, campaign_root: Path) -> dict[str, Any]:
    contract = _load_json(contract_path)
    random_seeds = contract["sampling"]["random_gin"]["fixed_evaluator_seeds"]
    encoder_seeds = contract["sampling"]["graphcl"]["fixed_train_only_encoder_seeds"]
    jobs = {}
    for candidate, candidate_contract in contract["candidates"].items():
        for checkpoint in candidate_contract["checkpoints"]:
            training_seed = int(checkpoint["training_seed"])
            for generation_seed in contract["sampling"]["generation_seeds"]:
                job_id = f"{candidate}_train{training_seed}_gen{generation_seed}"
                job_dir = campaign_root / "jobs" / job_id
                random_gin = _validate_random_gin(
                    job_dir,
                    generation_seed=int(generation_seed),
                    checkpoint_sha=checkpoint["sha256"],
                    evaluator_seeds=random_seeds,
                )
                graphcl = _validate_graphcl(
                    job_dir,
                    generation_seed=int(generation_seed),
                    encoder_seeds=encoder_seeds,
                    generated_sha256=random_gin["generated_sha256"],
                    reference_sha256=random_gin["reference_sha256"],
                )
                jobs[(candidate, training_seed, int(generation_seed))] = {
                    "job_id": job_id,
                    "random_gin": random_gin,
                    "graphcl": graphcl,
                }
    if len(jobs) != 18:
        raise DistributedContractError("Bake-off did not contain exactly 18 jobs.")
    uniform_seed0 = next(
        row
        for row in contract["candidates"]["uniform"]["checkpoints"]
        if int(row["training_seed"]) == 0
    )
    replay_dir = campaign_root / "replay" / "uniform_train0_gen123"
    replay_random = _validate_random_gin(
        replay_dir,
        generation_seed=123,
        checkpoint_sha=uniform_seed0["sha256"],
        evaluator_seeds=random_seeds,
    )
    replay_graphcl = _validate_graphcl(
        replay_dir,
        generation_seed=123,
        encoder_seeds=encoder_seeds,
        generated_sha256=replay_random["generated_sha256"],
        reference_sha256=replay_random["reference_sha256"],
    )
    original = jobs[("uniform", 0, 123)]
    replay_exact = {
        "random_gin": replay_random == original["random_gin"],
        "graphcl": replay_graphcl == original["graphcl"],
    }
    random_summary = _method_summary(jobs, "random_gin")
    graphcl_summary = _method_summary(jobs, "graphcl")
    conditions = {
        "both_evaluators_replay_exactly": all(replay_exact.values()),
        "graphcl_paired_dispersion_at_least_20_percent_lower": (
            graphcl_summary["mean_paired_difference_population_sd"]
            <= 0.8 * random_summary["mean_paired_difference_population_sd"]
        ),
        "graphcl_mean_generation_range_no_greater": (
            graphcl_summary["mean_generation_seed_range"]
            <= random_summary["mean_generation_seed_range"]
        ),
        "graphcl_sign_stability_no_worse": (
            graphcl_summary["sign_stability_count_of_9"]
            >= random_summary["sign_stability_count_of_9"]
        ),
        "integrity_and_validation_only_guards": True,
    }
    selected = "graphcl" if all(conditions.values()) else "random_gin"
    return {
        "schema_version": "aids-evaluator-bakeoff-analysis-v1",
        "job_count": 18,
        "test_access": False,
        "held_out_access": False,
        "methods": {"random_gin": random_summary, "graphcl": graphcl_summary},
        "replay_exact": replay_exact,
        "decision_conditions": conditions,
        "selected_primary_evaluator": selected,
        "weight_improvement_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.contract.resolve(), args.campaign_root.resolve())
    atomic_write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
