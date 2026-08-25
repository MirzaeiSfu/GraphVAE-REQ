#!/usr/bin/env python3
"""Evaluate one GraphVAE checkpoint with the frozen LOBSTER GraphCL bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_EVALUATION_SRC = REPO_ROOT / "graph_evaluation" / "src"
for source in (REPO_ROOT, REPO_ROOT / "scripts", GRAPH_EVALUATION_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from evaluate_attributed_graph_realism_checkpoints import (  # noqa: E402
    _checkpoint_state_dict,
    _seed_generation,
    evaluator_input_integrity,
    generate_attributed_graphs,
    resolve_device,
    validate_feature_heads,
)
from freeze_lobster_graphcl_f1pr_bundle import (  # noqa: E402
    EXPECTED_FEATURE_SCHEMA,
    EXPECTED_SEEDS,
    EXPECTED_UPSTREAM_REVISION,
    graphcl_runtime_fingerprint,
)
from ggm_eval.adapters import attributed_arrays_to_pyg  # noqa: E402
from ggm_eval.contract import collection_digest, validate_collection  # noqa: E402
from ggm_eval.io import (  # noqa: E402
    load_pyg_collection_with_metadata,
    save_pyg_collection,
)
from ggm_eval.reporting import summarize_values  # noqa: E402
from ggm_eval.runner import evaluate_contrastive_checkpoints  # noqa: E402
from ggm_eval.upstreams import validate_contrastive_upstream  # noqa: E402
from graphvae_attr_bo_distributed import (  # noqa: E402
    DistributedContractError,
    atomic_write_json,
    canonical_json_bytes,
)
from graphvae_attr_bo_fingerprints import sha256_file  # noqa: E402
from resample_grid_checkpoints import (  # noqa: E402
    build_model,
    dataset_cache_path,
    load_cached_dataset,
    load_config,
)


CACHE_RELATIVE_PATH = Path(
    "cache_datasets/LOBSTER_split-paper_70_10_20_train0p7_val0p1_"
    "test0p2_seed123_loaderseed-0_bfs-legacy_first_component_"
    "features-lobster-optimal_v2.pkl"
)
CACHE_BYTE_LENGTH = 59295793
CACHE_SHA256 = "928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660"
VALIDATION_COLLECTION_SHA256 = (
    "0a5ad40ab717440f1739f0b203df3df253a6318089202aa467dd4fc6ee5c1832"
)
VALIDATION_SPLIT_FINGERPRINT = (
    "2c1853642157f09f68bc4950e0c5e0074dc207f7565f6bedca2dc2126dd33d62"
)
EXPECTED_GRAPH_COUNT = 10
EXPECTED_NODE_DIMENSION = 14
EXPECTED_EDGE_DIMENSION = 11
EXPECTED_NEAREST_K = 5
OUTPUT_FILENAME = "graphcl_f1pr.json"


def _validate_cache(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    expected = (REPO_ROOT / CACHE_RELATIVE_PATH).resolve()
    if resolved != expected or path.is_symlink() or not resolved.is_file():
        raise DistributedContractError("GraphCL-F1PR requires the exact cache path.")
    state = {
        "byte_length": resolved.stat().st_size,
        "mode": format(stat.S_IMODE(resolved.stat().st_mode), "04o"),
        "sha256": sha256_file(resolved),
    }
    if state != {
        "byte_length": CACHE_BYTE_LENGTH,
        "mode": "0444",
        "sha256": CACHE_SHA256,
    }:
        raise DistributedContractError("LOBSTER cache size, mode, or hash differs.")
    return state


def _inside(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise DistributedContractError("Encoder artifact escapes its campaign root.") from exc
    return resolved


def load_encoder_bundle(
    manifest_path: Path,
    *,
    campaign_root: Path,
    expected_manifest_sha256: str,
    dependency_root: Path,
    expected_runtime_sha256: str,
) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = manifest_path.expanduser().resolve()
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or stat.S_IMODE(manifest_path.stat().st_mode) != 0o444
    ):
        raise DistributedContractError("Encoder bundle manifest is not immutable.")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise DistributedContractError("Encoder bundle manifest file hash differs.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "lobster-graphcl-f1pr-encoder-bundle-v1":
        raise DistributedContractError("Unsupported encoder bundle schema.")
    material = dict(manifest)
    declared_bundle_sha = material.pop("bundle_sha256", None)
    actual_bundle_sha = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    if declared_bundle_sha != actual_bundle_sha:
        raise DistributedContractError("Encoder bundle semantic hash differs.")
    exact = {
        "encoder": "graphcl",
        "feature_mode": "decoded_node_edge",
        "feature_schema": EXPECTED_FEATURE_SCHEMA,
        "training_split": "train",
        "test_access": False,
        "seeds": list(EXPECTED_SEEDS),
        "checkpoint_count": 5,
    }
    for name, expected in exact.items():
        if manifest.get(name) != expected:
            raise DistributedContractError(f"Encoder bundle differs for {name}.")
    if manifest.get("upstream") != {
        "revision": EXPECTED_UPSTREAM_REVISION,
        "worktree_dirty": False,
    }:
        raise DistributedContractError("Encoder bundle upstream contract differs.")
    runtime = graphcl_runtime_fingerprint(dependency_root)
    if runtime.get("sha256") != expected_runtime_sha256:
        raise DistributedContractError("GraphCL runtime fingerprint differs.")
    if runtime != manifest.get("runtime"):
        raise DistributedContractError("GraphCL runtime differs from the frozen bundle.")

    encoders = manifest.get("encoders") or []
    if [entry.get("seed") for entry in encoders] != list(EXPECTED_SEEDS):
        raise DistributedContractError("Encoder bundle seed order differs.")
    checkpoints = []
    for entry in encoders:
        record = entry.get("checkpoint") or {}
        raw_relative = Path(str(record.get("path", "")))
        if raw_relative.is_absolute() or ".." in raw_relative.parts:
            raise DistributedContractError("Encoder checkpoint path is unsafe.")
        checkpoint = _inside(campaign_root / raw_relative, campaign_root)
        if (
            not checkpoint.is_file()
            or checkpoint.is_symlink()
            or checkpoint.stat().st_size != record.get("byte_length")
            or format(stat.S_IMODE(checkpoint.stat().st_mode), "04o") != "0444"
            or sha256_file(checkpoint) != record.get("sha256")
        ):
            raise DistributedContractError("Frozen encoder checkpoint differs.")
        checkpoints.append(checkpoint)
    if len(set(checkpoints)) != 5:
        raise DistributedContractError("Encoder checkpoints must be distinct.")
    return manifest, checkpoints


def validate_validation_reference(path: Path) -> tuple[list, dict[str, Any]]:
    graphs, metadata = load_pyg_collection_with_metadata(path)
    summary = validate_collection(graphs, name="LOBSTER validation reference")
    exact_metadata = {
        "dataset": "LOBSTER",
        "split": "validation",
        "feature_mode": "decoded_node_edge",
        "feature_schema": EXPECTED_FEATURE_SCHEMA,
        "node_feature_dimension": EXPECTED_NODE_DIMENSION,
        "edge_feature_dimension": EXPECTED_EDGE_DIMENSION,
        "source_cache_sha256": CACHE_SHA256,
        "split_fingerprint": VALIDATION_SPLIT_FINGERPRINT,
        "test_access": False,
    }
    for name, expected in exact_metadata.items():
        if metadata.get(name) != expected:
            raise DistributedContractError(
                f"Validation reference metadata differs for {name}."
            )
    if summary.to_dict() != {
        "graph_count": 10,
        "node_feature_dim": 14,
        "edge_feature_dim": 11,
        "total_nodes": 464,
        "directed_edge_count": 908,
    }:
        raise DistributedContractError("Validation reference dimensions/counts differ.")
    if collection_digest(graphs) != VALIDATION_COLLECTION_SHA256:
        raise DistributedContractError("Validation reference collection hash differs.")
    return graphs, metadata


def _atomic_save_collection(path: Path, graphs: Sequence, metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".graphcl-f1pr-", dir=str(path.parent)))
    temporary = temporary_root / path.name
    try:
        manifest = save_pyg_collection(temporary, graphs, metadata=metadata)
        os.replace(str(temporary), str(path))
        os.replace(
            str(temporary.with_suffix(temporary.suffix + ".json")),
            str(path.with_suffix(path.suffix + ".json")),
        )
        return manifest
    finally:
        for child in temporary_root.iterdir():
            child.unlink()
        temporary_root.rmdir()


def _summary_matches_per_checkpoint(payload: Mapping[str, Any]) -> None:
    per_checkpoint = payload.get("per_checkpoint") or []
    if [entry.get("seed") for entry in per_checkpoint] != list(EXPECTED_SEEDS):
        raise DistributedContractError("GraphCL result checkpoint seeds differ.")
    if len({entry.get("checkpoint_sha256") for entry in per_checkpoint}) != 5:
        raise DistributedContractError("GraphCL result repeats a checkpoint.")
    summary = payload.get("summary") or {}
    metric_names = set(per_checkpoint[0].get("metrics") or {})
    if not {"f1_pr", "precision", "recall"}.issubset(metric_names):
        raise DistributedContractError("GraphCL result lacks F1-PR diagnostics.")
    for entry in per_checkpoint:
        if set(entry.get("metrics") or {}) != metric_names:
            raise DistributedContractError("GraphCL per-checkpoint metrics differ.")
    for metric in metric_names:
        values = [float(entry["metrics"][metric]) for entry in per_checkpoint]
        if not all(math.isfinite(value) for value in values):
            raise DistributedContractError(f"GraphCL metric {metric} is nonfinite.")
        upper = 1.00001 + 1e-12 if metric == "f1_pr" else 1.0
        if metric in {"f1_pr", "precision", "recall"} and not all(
            0.0 <= value <= upper for value in values
        ):
            raise DistributedContractError(f"GraphCL metric {metric} is outside [0,1].")
        expected = summarize_values(values)
        actual = summary.get(metric) or {}
        if set(actual) != set(expected) or any(
            not math.isclose(float(actual[name]), expected[name], rel_tol=0.0, abs_tol=1e-12)
            for name in expected
        ):
            raise DistributedContractError(
                f"GraphCL summary for {metric} differs from per-checkpoint values."
            )


def parse_graphcl_f1pr_payload(
    payload: Mapping[str, Any],
    *,
    expected_bundle_sha256: str,
    expected_runtime_sha256: str,
    expected_generation_seed: int,
) -> float:
    exact = {
        "schema_version": "lobster-graphcl-f1pr-evaluation-v1",
        "engine": "contrastive-pyg-upstream",
        "encoder": "graphcl",
        "feature_mode": "decoded_node_edge",
        "checkpoint_count": 5,
        "split": "validation",
        "test_access": False,
        "skip_final_evaluation": True,
        "generation_seed": expected_generation_seed,
        "nearest_k": EXPECTED_NEAREST_K,
        "objective_json_path": "summary.f1_pr.mean",
        "encoder_bundle_sha256": expected_bundle_sha256,
        "graphcl_runtime_sha256": expected_runtime_sha256,
    }
    for name, expected in exact.items():
        if payload.get(name) != expected:
            raise DistributedContractError(f"GraphCL evaluation differs for {name}.")
    if payload.get("graph_counts") != {
        "generated_accepted": 10,
        "reference_accepted": 10,
        "validation_cache_count": 10,
        "generation_attempts": payload.get("graph_counts", {}).get(
            "generation_attempts"
        ),
    }:
        raise DistributedContractError("GraphCL evaluation graph counts differ.")
    attempts = payload["graph_counts"]["generation_attempts"]
    if not isinstance(attempts, int) or attempts < 10:
        raise DistributedContractError("GraphCL generation attempt count is invalid.")
    feature_source = payload.get("feature_source") or {}
    if feature_source != {
        "generated": "GraphVAE node_feature_decoder and edge_feature_decoder",
        "reference": "frozen LOBSTER validation node and edge one-hot attributes",
        "same_latent_decoding": True,
        "hand_made_topology_features": False,
    }:
        raise DistributedContractError("GraphCL evaluation feature provenance differs.")
    integrity = payload.get("integrity") or {}
    expected_integrity = {
        "cache_sha256": CACHE_SHA256,
        "validation_split_fingerprint": VALIDATION_SPLIT_FINGERPRINT,
        "validation_collection_sha256": VALIDATION_COLLECTION_SHA256,
        "upstream_revision": EXPECTED_UPSTREAM_REVISION,
    }
    for name, expected in expected_integrity.items():
        if integrity.get(name) != expected:
            raise DistributedContractError(f"GraphCL integrity differs for {name}.")
    dimensions = payload.get("feature_dimensions") or {}
    if dimensions != {"node": 14, "edge": 11}:
        raise DistributedContractError("GraphCL feature dimensions differ.")
    _summary_matches_per_checkpoint(payload)
    try:
        value = float(payload["summary"]["f1_pr"]["mean"])
        compatibility = float(
            payload["evaluation"]["modes"]["decoded_node_edge"]["summary"]
            ["f1_pr"]["mean"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DistributedContractError("GraphCL objective path is missing.") from exc
    if (
        not math.isfinite(value)
        or not 0.0 <= value <= 1.00001 + 1e-12
        or value != compatibility
    ):
        raise DistributedContractError("GraphCL objective is invalid or inconsistent.")
    return value


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_graphs != EXPECTED_GRAPH_COUNT:
        raise ValueError("LOBSTER GraphCL-F1PR requires exactly 10 validation graphs.")
    if args.nearest_k != EXPECTED_NEAREST_K:
        raise ValueError("LOBSTER GraphCL-F1PR requires nearest-k 5.")
    if args.generation_batch_size < 1:
        raise ValueError("Generation batch size must be positive.")
    if not 0.0 <= args.adjacency_threshold <= 1.0:
        raise ValueError("Adjacency threshold must be in [0,1].")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse GraphCL evaluation root: {output_dir}")
    output_dir.mkdir(parents=True)

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = _inside(args.checkpoint, run_dir)
    if not run_dir.is_dir() or not checkpoint_path.is_file():
        raise DistributedContractError("GraphVAE run/checkpoint is missing.")
    before = _validate_cache(args.cache_path)
    config = load_config(args.config.expanduser().resolve())
    if (
        config.get("dataset") != "LOBSTER"
        or config.get("split_mode") != "paper_70_10_20"
        or config.get("split_seed") != 123
        or config.get("use_feature") is not True
        or config.get("skip_final_evaluation") is not True
        or float(config.get("alpha_node_feat", 0.0)) <= 0.0
        or float(config.get("alpha_edge_feat", 0.0)) <= 0.0
    ):
        raise DistributedContractError("GraphVAE config violates the LOBSTER campaign.")
    declared_cache_dir = Path(str(config.get("dataset_cache_dir", "cache_datasets")))
    if declared_cache_dir.is_absolute() and (
        declared_cache_dir.resolve() != args.cache_path.expanduser().resolve().parent
    ):
        raise DistributedContractError("GraphVAE config declares a different cache root.")
    config["dataset_cache_dir"] = str(args.cache_path.expanduser().resolve().parent)
    configured_cache = dataset_cache_path(config).expanduser().resolve()
    if configured_cache != args.cache_path.expanduser().resolve():
        raise DistributedContractError("GraphVAE config selects a different cache.")
    cache = load_cached_dataset(config)
    cache_integrity = evaluator_input_integrity(cache, config, "validation")
    if (
        cache_integrity.get("cache_sha256") != CACHE_SHA256
        or cache_integrity.get("split_fingerprint") != VALIDATION_SPLIT_FINGERPRINT
        or cache_integrity.get("split_graph_count") != EXPECTED_GRAPH_COUNT
    ):
        raise DistributedContractError("Cache-backed validation integrity differs.")
    reference_graphs, reference_metadata = validate_validation_reference(
        args.reference
    )
    upstream = validate_contrastive_upstream(args.upstream_repo)
    if upstream["revision"] != EXPECTED_UPSTREAM_REVISION:
        raise DistributedContractError("GraphCL upstream revision differs.")
    bundle, checkpoints = load_encoder_bundle(
        args.encoder_bundle_manifest,
        campaign_root=args.campaign_root,
        expected_manifest_sha256=args.encoder_bundle_manifest_sha256,
        dependency_root=args.dependency_root,
        expected_runtime_sha256=args.graphcl_runtime_sha256,
    )

    device = resolve_device(args.device)
    state_dict = _checkpoint_state_dict(checkpoint_path, device)
    has_node_head, has_edge_head = validate_feature_heads(state_dict)
    if not has_node_head or not has_edge_head:
        raise DistributedContractError("GraphCL-F1PR requires both feature decoders.")
    model = build_model(config, cache, device)
    if model.node_feature_decoder is None or model.edge_feature_decoder is None:
        raise DistributedContractError("Reconstructed GraphVAE lacks a feature decoder.")
    model.load_state_dict(state_dict)
    model.eval()

    _seed_generation(args.generation_seed)
    generated_attributed, attempted = generate_attributed_graphs(
        model,
        EXPECTED_GRAPH_COUNT,
        device,
        cache,
        args.adjacency_threshold,
        args.generation_batch_size,
        EXPECTED_EDGE_DIMENSION,
    )
    generated_graphs = [
        attributed_arrays_to_pyg(
            graph.edges,
            graph.node_attributes,
            graph.edge_attributes,
            graph.source_node_ids,
            name=f"generated validation graph {index}",
        )
        for index, graph in enumerate(generated_attributed)
    ]
    generated_summary = validate_collection(generated_graphs, name="generated graphs")
    if (
        generated_summary.graph_count != 10
        or generated_summary.node_feature_dim != 14
        or generated_summary.edge_feature_dim != 11
    ):
        raise DistributedContractError("Generated GraphCL collection dimensions differ.")
    generated_path = output_dir / "generated_validation_graphs.pt"
    generated_manifest = _atomic_save_collection(
        generated_path,
        generated_graphs,
        {
            "dataset": "LOBSTER",
            "split": "generated_for_validation",
            "feature_mode": "decoded_node_edge",
            "feature_schema": EXPECTED_FEATURE_SCHEMA,
            "node_feature_dimension": 14,
            "edge_feature_dimension": 11,
            "source_cache_sha256": CACHE_SHA256,
            "generation_seed": args.generation_seed,
            "same_latent_decoding": True,
            "feature_source": "GraphVAE node_feature_decoder and edge_feature_decoder",
            "test_access": False,
        },
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw_evaluation = evaluate_contrastive_checkpoints(
        generated=generated_path,
        reference=args.reference,
        checkpoints=checkpoints,
        upstream_repository=args.upstream_repo,
        output_dir=output_dir / "encoder_evaluation",
        python_executable=args.python,
        device=args.device,
        nearest_k=args.nearest_k,
        max_graphs=args.max_graphs,
    )
    checkpoint_records = {
        entry["seed"]: entry["checkpoint"] for entry in bundle["encoders"]
    }
    per_checkpoint = []
    for result in raw_evaluation["per_checkpoint"]:
        seed = int(result["checkpoint_seed"])
        per_checkpoint.append(
            {
                "seed": seed,
                "checkpoint_sha256": checkpoint_records[seed]["sha256"],
                "metrics": result["metrics"],
                "activation_seconds": result["activation_seconds"],
                "activation_dimension": result["activation_dim"],
                "generated_collection_sha256": result["generated_sha256"],
                "validation_collection_sha256": result["reference_sha256"],
            }
        )
    summary = raw_evaluation["summary"]
    payload = {
        "schema_version": "lobster-graphcl-f1pr-evaluation-v1",
        "engine": raw_evaluation["engine"],
        "encoder": raw_evaluation["encoder"],
        "feature_mode": raw_evaluation["feature_mode"],
        "checkpoint_count": raw_evaluation["checkpoint_count"],
        "split": "validation",
        "test_access": False,
        "skip_final_evaluation": True,
        "generation_seed": args.generation_seed,
        "nearest_k": args.nearest_k,
        "objective_json_path": "summary.f1_pr.mean",
        "compatibility_objective_json_path": (
            "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
        ),
        "encoder_bundle_sha256": bundle["bundle_sha256"],
        "graphcl_runtime_sha256": args.graphcl_runtime_sha256,
        "graph_counts": {
            "generated_accepted": 10,
            "reference_accepted": len(reference_graphs),
            "validation_cache_count": cache_integrity["split_graph_count"],
            "generation_attempts": attempted,
        },
        "feature_dimensions": {"node": 14, "edge": 11},
        "feature_source": {
            "generated": "GraphVAE node_feature_decoder and edge_feature_decoder",
            "reference": "frozen LOBSTER validation node and edge one-hot attributes",
            "same_latent_decoding": True,
            "hand_made_topology_features": False,
        },
        "summary": summary,
        "per_checkpoint": per_checkpoint,
        "evaluation": {
            "modes": {"decoded_node_edge": {"summary": summary}}
        },
        "integrity": {
            "cache_sha256": CACHE_SHA256,
            "validation_split_fingerprint": VALIDATION_SPLIT_FINGERPRINT,
            "validation_collection_sha256": collection_digest(reference_graphs),
            "generated_collection_sha256": generated_manifest["collection_sha256"],
            "generated_file_sha256": sha256_file(generated_path),
            "graphvae_checkpoint_sha256": sha256_file(checkpoint_path),
            "config_sha256": sha256_file(args.config),
            "encoder_bundle_manifest_file_sha256": sha256_file(
                args.encoder_bundle_manifest
            ),
            "upstream_revision": upstream["revision"],
            "node_schema_fingerprint": cache_integrity["node_schema_fingerprint"],
            "edge_schema_fingerprint": cache_integrity["edge_schema_fingerprint"],
        },
        "reference_metadata": reference_metadata,
    }
    parse_graphcl_f1pr_payload(
        payload,
        expected_bundle_sha256=bundle["bundle_sha256"],
        expected_runtime_sha256=args.graphcl_runtime_sha256,
        expected_generation_seed=args.generation_seed,
    )
    after = _validate_cache(args.cache_path)
    if after != before:
        raise DistributedContractError("LOBSTER cache changed during evaluation.")
    atomic_write_json(output_dir / OUTPUT_FILENAME, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--encoder-bundle-manifest", type=Path, required=True)
    parser.add_argument("--encoder-bundle-manifest-sha256", required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--graphcl-runtime-sha256", required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--generation-seed", type=int, required=True)
    parser.add_argument("--max-graphs", type=int, default=10)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--adjacency-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    print(
        json.dumps(
            {
                "objective": payload["summary"]["f1_pr"]["mean"],
                "output": str((args.output_dir / OUTPUT_FILENAME).resolve()),
                "test_access": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
