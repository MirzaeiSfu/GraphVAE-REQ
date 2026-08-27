#!/usr/bin/env python3
"""Freeze or verify the ten-encoder train-only AIDS GraphCL bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_EVALUATION_SRC = REPO_ROOT / "graph_evaluation" / "src"
for source in (REPO_ROOT, REPO_ROOT / "scripts", GRAPH_EVALUATION_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from freeze_lobster_graphcl_f1pr_bundle import (  # noqa: E402
    graphcl_runtime_fingerprint,
)
from ggm_eval.upstreams import validate_contrastive_upstream  # noqa: E402
from graphvae_attr_bo_distributed import (  # noqa: E402
    DistributedContractError,
    atomic_write_json,
    canonical_json_bytes,
)
from graphvae_attr_bo_fingerprints import sha256_file  # noqa: E402


EXPECTED_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
EXPECTED_UPSTREAM_REVISION = "fb6bc26237eb21d7617fd41b22b4bb26ab29bf95"
EXPECTED_COLLECTION_SHA256 = (
    "3d979eb9fec89967832df54547fff6a9dfa2d6f9da8a5d357200dc5963f41650"
)
EXPECTED_SPLIT_FINGERPRINT = (
    "cbc4af1fdf217f72ea51e6b369f15aa611a72e0d23cb79cf4db24dd9b5b6e3b4"
)
EXPECTED_FEATURE_SCHEMA = "tu-quantile8-max40|export=decoded_node_edge"
EXPECTED_MODEL = {
    "input_dim": 56,
    "edge_dim": 3,
    "num_layers": 3,
    "hidden_dim": 32,
    "init": "orthogonal",
    "limit_lipschitz": True,
    "lipschitz_factor": 1.0,
}
EXPECTED_TRAINING_GRAPHS = {
    "graph_count": 1294,
    "total_nodes": 15685,
    "directed_edge_count": 32106,
    "node_feature_dim": 56,
    "edge_feature_dim": 3,
}


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DistributedContractError(
            f"Bundle artifact must be below the campaign root: {path}"
        ) from exc


def _file_record(path: Path, campaign_root: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DistributedContractError(f"Bundle artifact is missing or a symlink: {path}")
    return {
        "path": _relative(path, campaign_root),
        "byte_length": path.stat().st_size,
        "mode": format(stat.S_IMODE(path.stat().st_mode), "04o"),
        "sha256": sha256_file(path),
    }


def _find_seed_directory(training_roots: Iterable[Path], seed: int) -> Path:
    matches = [root.resolve() / f"seed_{seed}" for root in training_roots]
    matches = [path for path in matches if path.is_dir()]
    if len(matches) != 1:
        raise DistributedContractError(
            f"Expected exactly one completed directory for encoder seed {seed}."
        )
    return matches[0]


def _expected_metadata() -> dict[str, Any]:
    return {
        "dataset": "AIDS",
        "feature_mode": "decoded_node_edge",
        "feature_schema": EXPECTED_FEATURE_SCHEMA,
        "node_feature_dimension": 56,
        "edge_feature_dimension": 3,
        "split": "train",
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "test_access": False,
    }


def _assert_checkpoint(seed_dir: Path, seed: int) -> tuple[Path, Path, dict[str, Any]]:
    training_path = seed_dir / "training.json"
    checkpoint_path = seed_dir / "checkpoint.pt"
    if not training_path.is_file() or not checkpoint_path.is_file():
        raise DistributedContractError(f"Encoder seed {seed} is incomplete.")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    exact = {
        "checkpoint_format": "ggm-eval-upstream-gconv",
        "checkpoint_version": 1,
        "encoder": "graphcl",
        "feature_mode": "decoded_node_edge",
        "seed": seed,
        "model": EXPECTED_MODEL,
        "training": {"epochs": 100, "trained": True},
        "training_collection_sha256": EXPECTED_COLLECTION_SHA256,
        "training_graphs": EXPECTED_TRAINING_GRAPHS,
    }
    for name, expected in exact.items():
        if training.get(name) != expected:
            raise DistributedContractError(
                f"Encoder seed {seed} training manifest differs for {name}."
            )
    for name, expected in _expected_metadata().items():
        if (training.get("training_metadata") or {}).get(name) != expected:
            raise DistributedContractError(
                f"Encoder seed {seed} training metadata differs for {name}."
            )
    if training.get("upstream", {}).get("revision") != EXPECTED_UPSTREAM_REVISION:
        raise DistributedContractError(f"Encoder seed {seed} upstream revision differs.")
    if not math.isfinite(float(training.get("training_loss", float("nan")))):
        raise DistributedContractError(f"Encoder seed {seed} training loss is nonfinite.")
    if float(training.get("elapsed_seconds", 0.0)) <= 0.0:
        raise DistributedContractError(f"Encoder seed {seed} elapsed time is invalid.")
    if Path(str(training.get("checkpoint", ""))).resolve() != checkpoint_path.resolve():
        raise DistributedContractError(f"Encoder seed {seed} checkpoint path differs.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    for name, expected in (
        ("format", "ggm-eval-upstream-gconv"),
        ("version", 1),
        ("encoder", "graphcl"),
        ("feature_mode", "decoded_node_edge"),
        ("seed", seed),
        ("model", EXPECTED_MODEL),
        ("training", {"epochs": 100, "trained": True}),
        ("upstream_revision", EXPECTED_UPSTREAM_REVISION),
    ):
        if checkpoint.get(name) != expected:
            raise DistributedContractError(
                f"Encoder seed {seed} checkpoint differs for {name}."
            )
    for name, expected in _expected_metadata().items():
        if (checkpoint.get("training_metadata") or {}).get(name) != expected:
            raise DistributedContractError(
                f"Encoder seed {seed} checkpoint metadata differs for {name}."
            )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise DistributedContractError(f"Encoder seed {seed} state dict is empty.")
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()):
            raise DistributedContractError(f"Encoder seed {seed} has invalid tensor {name}.")
    return checkpoint_path, training_path, training


def _make_tree_read_only(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_symlink():
            raise DistributedContractError("Frozen bundle trees may not contain symlinks.")
        child.chmod(0o555 if child.is_dir() else 0o444)
    path.chmod(0o555)


def build_bundle_manifest(
    *,
    campaign_root: Path,
    training_roots: Sequence[Path],
    upstream_repo: Path,
    dependency_root: Path,
) -> dict[str, Any]:
    upstream = validate_contrastive_upstream(upstream_repo)
    if upstream["revision"] != EXPECTED_UPSTREAM_REVISION:
        raise DistributedContractError("GraphCL upstream revision differs.")
    runtime = graphcl_runtime_fingerprint(dependency_root)
    encoders = []
    recorded_versions = None
    for seed in EXPECTED_SEEDS:
        seed_dir = _find_seed_directory(training_roots, seed)
        checkpoint_path, training_path, training = _assert_checkpoint(seed_dir, seed)
        if recorded_versions is None:
            recorded_versions = training.get("versions")
        elif training.get("versions") != recorded_versions:
            raise DistributedContractError("Encoder runtime versions differ by seed.")
        encoders.append(
            {
                "seed": seed,
                "checkpoint": _file_record(checkpoint_path, campaign_root),
                "training_manifest": _file_record(training_path, campaign_root),
                "elapsed_seconds": training["elapsed_seconds"],
                "training_loss": training["training_loss"],
            }
        )
    manifest = {
        "schema_version": "aids-evaluator-bakeoff-graphcl-bundle-v1",
        "encoder": "graphcl",
        "feature_mode": "decoded_node_edge",
        "feature_schema": EXPECTED_FEATURE_SCHEMA,
        "training_split": "train",
        "test_access": False,
        "training_collection_sha256": EXPECTED_COLLECTION_SHA256,
        "training_split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "seeds": list(EXPECTED_SEEDS),
        "checkpoint_count": len(encoders),
        "upstream": {
            "revision": upstream["revision"],
            "worktree_dirty": upstream["worktree_dirty"],
        },
        "runtime": runtime,
        "training_versions": recorded_versions,
        "encoders": encoders,
    }
    manifest["bundle_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, action="append", required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    publication = parser.add_mutually_exclusive_group(required=True)
    publication.add_argument("--output", type=Path)
    publication.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--make-read-only", action="store_true")
    args = parser.parse_args(argv)
    training_roots = [path.expanduser().resolve() for path in args.training_root]
    if args.make_read_only:
        if args.output is None:
            raise ValueError("--make-read-only is valid only while publishing.")
        for root in training_roots:
            _make_tree_read_only(root)
    actual = build_bundle_manifest(
        campaign_root=args.campaign_root.expanduser().resolve(),
        training_roots=training_roots,
        upstream_repo=args.upstream_repo.expanduser().resolve(),
        dependency_root=args.dependency_root.expanduser().resolve(),
    )
    if args.verify_manifest is not None:
        expected = json.loads(args.verify_manifest.read_text(encoding="utf-8"))
        if actual != expected:
            raise DistributedContractError("Frozen AIDS GraphCL bundle differs.")
        print(json.dumps({"verified": True, **actual}, indent=2, sort_keys=True))
        return 0
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite bundle manifest: {args.output}")
    atomic_write_json(args.output, actual)
    os.chmod(args.output, 0o444)
    print(json.dumps(actual, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
