#!/usr/bin/env python3
"""Validate an existing GraphVAE cache and write its distributed BO manifest.

This command never creates or regenerates a dataset cache.  Create the canonical
cache once with the normal data pipeline, then use this command before staging.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from graphvae_attr_bo_distributed import (  # noqa: E402
    DistributedContractError,
    atomic_write_json,
)
from graphvae_attr_bo_fingerprints import (  # noqa: E402
    FINGERPRINT_SCHEMA_VERSION,
    feature_schema_fingerprint,
    feature_schema_payload,
    graph_fingerprint,
    sha256_file,
    split_fingerprint,
)
from resample_grid_checkpoints import (  # noqa: E402
    dataset_cache_path,
    validate_dataset_cache_metadata,
    build_dataset_cache_metadata,
)
from tune_graphvae_attribute_weights import flatten_config, load_yaml_mapping  # noqa: E402


def _split_arrays(cache: dict[str, Any], split: str):
    if split == "train":
        adjacency = cache["list_adj"]
        nodes = cache.get("list_noh_train")
        edges = cache.get("list_eoh_train")
    elif split == "validation":
        adjacency = cache["val_adj"]
        nodes = cache.get("list_noh_val")
        edges = cache.get("list_eoh_val")
    elif split == "test":
        adjacency = cache["test_list_adj"]
        nodes = cache.get("list_noh_test")
        edges = cache.get("list_eoh_test")
    else:  # pragma: no cover
        raise ValueError(split)
    if nodes is None:
        raise ValueError(f"Cache has no {split} node attributes.")
    if edges is None:
        raise ValueError(f"Cache has no {split} edge attributes.")
    if not (len(adjacency) == len(nodes) == len(edges)):
        raise ValueError(f"Cache {split} adjacency/node/edge lengths differ.")
    return list(adjacency), list(nodes), list(edges)


def _split_manifest(cache: dict[str, Any], split: str) -> dict[str, Any]:
    adjacencies, nodes, edges = _split_arrays(cache, split)
    graph_hashes = [
        graph_fingerprint(
            adjacency,
            nodes[index],
            edges[index],
            relation_axes={
                "node": cache.get("node_onehot_info"),
                "edge": cache.get("edge_onehot_info"),
            },
        )
        for index, adjacency in enumerate(adjacencies)
    ]
    return {
        "graph_count": len(graph_hashes),
        "fingerprint": split_fingerprint(graph_hashes),
        "graph_fingerprints": graph_hashes,
    }


def build_cache_manifest(
    cache_path: Path,
    cache: dict[str, Any],
    *,
    max_graphs: int,
) -> dict[str, Any]:
    resolved = cache_path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("Distributed cache must be staged beneath the repository.") from exc
    splits = {
        name: _split_manifest(cache, name) for name in ("train", "validation", "test")
    }
    validation_count = splits["validation"]["graph_count"]
    expected = validation_count if max_graphs == 0 else min(max_graphs, validation_count)
    sample_nodes = _split_arrays(cache, "validation")[1][0]
    sample_edges = _split_arrays(cache, "validation")[2][0]
    node_dimension = len(cache.get("node_onehot_info") or {}) or int(np.asarray(sample_nodes).shape[-1])
    edge_dimension = len(cache.get("edge_onehot_info") or {}) or int(np.asarray(sample_edges).shape[-3])
    node_schema = feature_schema_payload(
        cache.get("node_onehot_info"),
        total_dimension=node_dimension,
        dtype=str(np.asarray(sample_nodes).dtype),
    )
    edge_schema = feature_schema_payload(
        cache.get("edge_onehot_info"),
        total_dimension=edge_dimension,
        dtype=str(np.asarray(sample_edges).dtype),
    )
    return {
        "schema_version": "graphvae-attr-f1pr-cache-manifest-v1",
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "relative_path": relative,
        "filename": resolved.name,
        "byte_length": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "cache_metadata": cache.get("cache_metadata"),
        "split_mode": cache.get("split_mode"),
        "splits": splits,
        "split_fingerprint": splits["validation"]["fingerprint"],
        "expected_validation_graphs": expected,
        "node_feature_dimension": node_dimension,
        "edge_feature_dimension": edge_dimension,
        "node_schema": node_schema,
        "edge_schema": edge_schema,
        "node_schema_fingerprint": feature_schema_fingerprint(node_schema),
        "edge_schema_fingerprint": feature_schema_fingerprint(edge_schema),
    }


def verify_cache_manifest(
    cache_path: Path,
    cache: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Recompute and compare every staged cache and schema identity."""

    if expected.get("schema_version") != "graphvae-attr-f1pr-cache-manifest-v1":
        raise DistributedContractError("Unsupported cache manifest schema version.")
    expected_metadata = expected.get("cache_metadata")
    if not isinstance(expected_metadata, dict):
        raise DistributedContractError("Cache manifest has no validated metadata.")
    validate_dataset_cache_metadata(cache, expected_metadata, cache_path)
    expected_count = int(expected.get("expected_validation_graphs", 0))
    if expected_count < 3:
        raise DistributedContractError(
            "Cache manifest must require at least three validation graphs."
        )
    actual = build_cache_manifest(cache_path, cache, max_graphs=expected_count)
    for field in (
        "schema_version",
        "fingerprint_schema_version",
        "relative_path",
        "filename",
        "byte_length",
        "sha256",
        "cache_metadata",
        "split_mode",
        "splits",
        "split_fingerprint",
        "expected_validation_graphs",
        "node_feature_dimension",
        "edge_feature_dimension",
        "node_schema",
        "edge_schema",
        "node_schema_fingerprint",
        "edge_schema_fingerprint",
    ):
        if actual.get(field) != expected.get(field):
            raise DistributedContractError(f"Cache manifest mismatch for {field}.")
    if actual != expected:
        raise DistributedContractError("Cache manifest contains unexpected fields.")
    return actual


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    publication = parser.add_mutually_exclusive_group(required=True)
    publication.add_argument("--output", type=Path)
    publication.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--make-read-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_graphs in {1, 2} or args.max_graphs < 0:
        raise ValueError("--max-graphs must be 0 or at least 3.")
    expected_manifest = (
        None
        if args.verify_manifest is None
        else json.loads(args.verify_manifest.read_text(encoding="utf-8"))
    )
    config = (
        None
        if args.base_config is None
        else flatten_config(load_yaml_mapping(args.base_config))
    )
    cache_path = (
        args.cache_path.expanduser().resolve()
        if args.cache_path is not None
        else (
            dataset_cache_path(config).expanduser().resolve()
            if config is not None
            else None
        )
    )
    if cache_path is None:
        raise ValueError("--cache-path is required without --base-config.")
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Required existing dataset cache not found: {cache_path}. "
            "This command will not regenerate it."
        )
    before = (cache_path.stat().st_size, cache_path.stat().st_mtime_ns, sha256_file(cache_path))
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    if config is not None:
        validate_dataset_cache_metadata(
            cache, build_dataset_cache_metadata(config), cache_path
        )
    if expected_manifest is None:
        if config is None:
            raise ValueError("--base-config is required when creating a manifest.")
        manifest = build_cache_manifest(cache_path, cache, max_graphs=args.max_graphs)
    else:
        manifest = verify_cache_manifest(cache_path, cache, expected_manifest)
    after = (cache_path.stat().st_size, cache_path.stat().st_mtime_ns, sha256_file(cache_path))
    if before != after:
        raise RuntimeError("Dataset cache changed while its manifest was being built.")
    if args.output is not None:
        atomic_write_json(args.output, manifest)
    if args.make_read_only:
        cache_path.chmod(cache_path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    manifest_path = args.output if args.output is not None else args.verify_manifest
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "sha256": manifest["sha256"],
                "verified": args.verify_manifest is not None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
