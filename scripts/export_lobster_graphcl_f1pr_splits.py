#!/usr/bin/env python3
"""Export the frozen LOBSTER training/validation splits for GraphCL-F1PR."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_EVALUATION_SRC = REPO_ROOT / "graph_evaluation" / "src"
for source in (REPO_ROOT, REPO_ROOT / "scripts", GRAPH_EVALUATION_SRC):
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

from ggm_eval import save_pyg_collection  # noqa: E402
from graphvae_attr_bo_distributed import atomic_write_json  # noqa: E402
from graphvae_attr_bo_fingerprints import (  # noqa: E402
    graph_fingerprint,
    sha256_file,
    split_fingerprint,
)
from export_real_pyg_splits import graphs_to_pyg  # noqa: E402


CACHE_RELATIVE_PATH = Path(
    "cache_datasets/LOBSTER_split-paper_70_10_20_train0p7_val0p1_"
    "test0p2_seed123_loaderseed-0_bfs-legacy_first_component_"
    "features-lobster-optimal_v2.pkl"
)
CACHE_SHA256 = "928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660"
CACHE_BYTE_LENGTH = 59295793
FEATURE_SCHEMA = "lobster-optimal_v2|export=decoded_node_edge"
NODE_FEATURE_DIMENSION = 14
EDGE_FEATURE_DIMENSION = 11
EXPECTED_COUNTS = {"train": 70, "validation": 10}
OUTPUT_FILENAMES = {
    "train": "real_train_graphs.pt",
    "validation": "real_validation_graphs.pt",
}


class LobsterGraphCLExportError(RuntimeError):
    """Raised when the frozen cache/export contract is not satisfied."""


def _cache_state(path: Path) -> dict[str, Any]:
    return {
        "byte_length": path.stat().st_size,
        "mode": format(stat.S_IMODE(path.stat().st_mode), "04o"),
        "sha256": sha256_file(path),
    }


def validate_cache_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    expected = (REPO_ROOT / CACHE_RELATIVE_PATH).resolve()
    if path.is_symlink() or resolved != expected:
        raise LobsterGraphCLExportError(
            "GraphCL-F1PR export requires the exact non-symlink frozen cache path."
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Frozen LOBSTER cache is missing: {resolved}")
    state = _cache_state(resolved)
    if state != {
        "byte_length": CACHE_BYTE_LENGTH,
        "mode": "0444",
        "sha256": CACHE_SHA256,
    }:
        raise LobsterGraphCLExportError(
            "Frozen LOBSTER cache size, mode, or SHA-256 differs from the contract."
        )
    return state


def _split_values(cache: Mapping[str, Any], split: str) -> tuple[Sequence, Sequence, Sequence]:
    if split == "train":
        raw_graphs = cache.get("list_graphs")
        adjacencies = (
            None
            if raw_graphs is None
            else [
                graph[0] if isinstance(graph, (tuple, list)) else graph
                for graph in raw_graphs
            ]
        )
        values = (
            adjacencies,
            cache.get("list_noh_train"),
            cache.get("list_eoh_train"),
        )
    elif split == "validation":
        values = (
            cache.get("val_adj"),
            cache.get("list_noh_val"),
            cache.get("list_eoh_val"),
        )
    else:
        raise LobsterGraphCLExportError(
            "Only training and validation exports are allowed in this campaign."
        )
    if any(value is None for value in values):
        raise LobsterGraphCLExportError(
            f"Frozen cache is missing adjacency/node/edge values for {split}."
        )
    lengths = {len(value) for value in values}
    if lengths != {EXPECTED_COUNTS[split]}:
        raise LobsterGraphCLExportError(
            f"Frozen {split} split does not contain exactly {EXPECTED_COUNTS[split]} aligned graphs."
        )
    return values


def _raw_graph_hashes(
    adjacencies: Sequence,
    node_values: Sequence,
    edge_values: Sequence,
    *,
    node_feature_info: Mapping,
    edge_feature_info: Mapping,
) -> list[str]:
    return [
        graph_fingerprint(
            adjacency,
            nodes,
            edges,
            relation_axes={"node": node_feature_info, "edge": edge_feature_info},
        )
        for adjacency, nodes, edges in zip(adjacencies, node_values, edge_values)
    ]


def assert_disjoint_splits(train_hashes: Sequence[str], validation_hashes: Sequence[str]) -> None:
    overlap = set(train_hashes).intersection(validation_hashes)
    if overlap:
        raise LobsterGraphCLExportError(
            "Frozen training and validation graph fingerprints overlap."
        )


def _atomic_save_collection(path: Path, graphs: Sequence, metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".graphcl-export-", dir=str(path.parent)))
    temporary_path = temporary_root / path.name
    try:
        manifest = save_pyg_collection(temporary_path, graphs, metadata=metadata)
        temporary_manifest = temporary_path.with_suffix(temporary_path.suffix + ".json")
        destination_manifest = path.with_suffix(path.suffix + ".json")
        os.replace(str(temporary_path), str(path))
        os.replace(str(temporary_manifest), str(destination_manifest))
        return manifest
    finally:
        for child in temporary_root.iterdir():
            child.unlink()
        temporary_root.rmdir()


def export_frozen_splits(
    cache_path: Path,
    output_dir: Path,
    *,
    include_test: bool = False,
) -> dict:
    if include_test:
        raise LobsterGraphCLExportError(
            "Held-out/test export is forbidden by the GraphCL-F1PR campaign."
        )
    before = validate_cache_file(cache_path)
    with cache_path.open("rb") as stream:
        cache = pickle.load(stream)
    if not isinstance(cache, Mapping):
        raise LobsterGraphCLExportError("Frozen cache payload must be a mapping.")
    metadata = cache.get("cache_metadata") or {}
    expected_metadata = {
        "dataset": "LOBSTER",
        "feature_schema": "lobster-optimal_v2",
        "split_mode": "paper_70_10_20",
        "bfs_strategy": "legacy_first_component",
        "train_fraction": 0.7,
        "val_fraction": 0.1,
        "split_seed": 123,
        "dataset_loader_seed": 0,
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise LobsterGraphCLExportError(
                f"Frozen cache metadata mismatch for {name}."
            )
    node_info = cache.get("node_onehot_info") or {}
    edge_info = cache.get("edge_onehot_info") or {}
    if len(node_info) != NODE_FEATURE_DIMENSION or len(edge_info) != EDGE_FEATURE_DIMENSION:
        raise LobsterGraphCLExportError(
            "Frozen cache node/edge schema dimensions differ from 14/11."
        )

    split_payloads = {}
    split_hashes = {}
    for split in ("train", "validation"):
        adjacencies, node_values, edge_values = _split_values(cache, split)
        hashes = _raw_graph_hashes(
            adjacencies,
            node_values,
            edge_values,
            node_feature_info=node_info,
            edge_feature_info=edge_info,
        )
        split_hashes[split] = hashes
        pyg_graphs, rejected = graphs_to_pyg(
            adjacencies,
            node_values,
            edge_values,
            node_feature_info=node_info,
            edge_feature_info=edge_info,
            adjacency_threshold=0.5,
            split_name=split,
        )
        if rejected or len(pyg_graphs) != EXPECTED_COUNTS[split]:
            raise LobsterGraphCLExportError(
                f"{split} export rejected a graph or changed its exact count."
            )
        split_payloads[split] = pyg_graphs

    assert_disjoint_splits(split_hashes["train"], split_hashes["validation"])
    output_dir = output_dir.expanduser().resolve()
    artifacts = {}
    common_metadata = {
        "dataset": "LOBSTER",
        "feature_mode": "decoded_node_edge",
        "feature_schema": FEATURE_SCHEMA,
        "producer": "scripts/export_lobster_graphcl_f1pr_splits.py",
        "source_cache_relative_path": str(CACHE_RELATIVE_PATH),
        "source_cache_sha256": CACHE_SHA256,
        "split_mode": "paper_70_10_20",
        "split_seed": 123,
        "bfs_strategy": "legacy_first_component",
        "node_feature_dimension": NODE_FEATURE_DIMENSION,
        "edge_feature_dimension": EDGE_FEATURE_DIMENSION,
        "test_access": False,
    }
    for split in ("train", "validation"):
        destination = output_dir / OUTPUT_FILENAMES[split]
        manifest = _atomic_save_collection(
            destination,
            split_payloads[split],
            {
                **common_metadata,
                "split": split,
                "split_fingerprint": split_fingerprint(split_hashes[split]),
            },
        )
        summary = manifest["summary"]
        if (
            summary["graph_count"] != EXPECTED_COUNTS[split]
            or summary["node_feature_dim"] != NODE_FEATURE_DIMENSION
            or summary["edge_feature_dim"] != EDGE_FEATURE_DIMENSION
        ):
            raise LobsterGraphCLExportError(
                f"Published {split} PyG collection differs from the exact contract."
            )
        artifacts[split] = {
            "path": str(destination),
            "manifest": str(destination.with_suffix(destination.suffix + ".json")),
            "collection_sha256": manifest["collection_sha256"],
            "split_fingerprint": split_fingerprint(split_hashes[split]),
            "summary": summary,
        }

    after = validate_cache_file(cache_path)
    if after != before:
        raise LobsterGraphCLExportError("Frozen cache changed during export.")
    result = {
        "schema_version": "lobster-graphcl-f1pr-split-export-v1",
        "cache": before,
        "feature_schema": FEATURE_SCHEMA,
        "test_access": False,
        "exported_splits": ["train", "validation"],
        "split_overlap_count": 0,
        "artifacts": artifacts,
    }
    atomic_write_json(output_dir / "export_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=REPO_ROOT / CACHE_RELATIVE_PATH,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Forbidden campaign guard; any use fails before the cache is loaded.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_frozen_splits(
        args.cache_path,
        args.output_dir,
        include_test=args.include_test,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
