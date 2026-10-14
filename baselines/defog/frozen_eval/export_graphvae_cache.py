#!/usr/bin/env python3
"""Export exact GraphVAE cache identities to the frozen DeFoG package.

GraphVAE caches are trusted local pickle artifacts. This command verifies the
whole-file SHA-256 before unpickling and writes restricted tensor-only PyG
collections for cross-repository use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "scripts", ROOT / "graph_evaluation" / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from eval.attributed_gin import graph_from_dense_attributes  # noqa: E402
from ggm_eval import save_pyg_collection  # noqa: E402
from ggm_eval.adapters import attributed_arrays_to_pyg  # noqa: E402
from scripts.graphvae_attr_bo_fingerprints import (  # noqa: E402
    graph_fingerprint,
    split_fingerprint,
)


class FrozenExportError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read the campaign manifest.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FrozenExportError(f"Manifest must be a mapping: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def split_values(cache: Mapping[str, Any], split: str) -> tuple[list, list, None]:
    keys = {
        "train": ("list_adj", "list_noh_train"),
        "validation": ("val_adj", "list_noh_val"),
        "reference": ("test_list_adj", "list_noh_test"),
    }
    adjacency_key, node_key = keys[split]
    adjacencies = list(cache.get(adjacency_key) or [])
    node_values = list(cache.get(node_key) or [])
    if len(adjacencies) != len(node_values):
        raise FrozenExportError(
            f"{split} adjacency/node lengths differ: "
            f"{len(adjacencies)} versus {len(node_values)}"
        )
    return adjacencies, node_values, None


def topology_node_values(adjacencies: Sequence) -> list[np.ndarray]:
    return [
        np.ones((int(adjacency.shape[0]), 1), dtype=np.float32)
        for adjacency in adjacencies
    ]


def export_split(
    *,
    dataset: str,
    split: str,
    adjacencies: Sequence,
    node_values: Sequence,
    node_info: Mapping | None,
    feature_mode: str,
    feature_schema: str,
    cache_sha256: str,
    output_path: Path,
) -> dict:
    graph_hashes = [
        graph_fingerprint(
            adjacency,
            nodes,
            None,
            relation_axes={"node": node_info, "edge": None},
        )
        for adjacency, nodes in zip(adjacencies, node_values)
    ]
    accepted = []
    accepted_indices = []
    rejected = []
    removed_nodes = []
    for index, (adjacency, nodes) in enumerate(zip(adjacencies, node_values)):
        attributed = graph_from_dense_attributes(
            adjacency,
            nodes,
            None,
            node_feature_info=node_info,
            edge_feature_info=None,
            values_are_logits=False,
            adjacency_threshold=0.5,
        )
        if attributed is None:
            rejected.append({"source_index": index, "reason": "empty_after_normalization"})
            continue
        graph = attributed_arrays_to_pyg(
            attributed.edges,
            attributed.node_attributes,
            None,
            attributed.source_node_ids,
            name=f"{dataset} {split}[{index}]",
        )
        accepted.append(graph)
        accepted_indices.append(index)
        removed_nodes.append(int(adjacency.shape[0]) - int(graph.num_nodes))

    if not accepted:
        raise FrozenExportError(f"{dataset} {split} has no accepted graphs")
    metadata = {
        "dataset": dataset,
        "split": split,
        "split_mode": "paper_70_10_20",
        "split_seed": 123,
        "feature_mode": feature_mode,
        "feature_schema": feature_schema,
        "source_cache_sha256": cache_sha256,
        "source_split_fingerprint": split_fingerprint(graph_hashes),
        "source_graph_count": len(adjacencies),
        "accepted_source_indices": accepted_indices,
        "rejected": rejected,
        "postprocessing": {
            "adjacency_threshold": 0.5,
            "symmetry": "logical_or",
            "self_loops": "removed",
            "isolated_nodes": "removed",
            "connected_components": "deterministic_largest_component",
            "empty_graphs": "rejected",
        },
    }
    artifact = save_pyg_collection(output_path, accepted, metadata=metadata)
    return {
        "path": str(output_path),
        "source_graph_count": len(adjacencies),
        "accepted_graph_count": len(accepted),
        "rejected_graph_count": len(rejected),
        "accepted_source_indices": accepted_indices,
        "rejected": rejected,
        "removed_node_count": int(sum(removed_nodes)),
        "split_fingerprint": split_fingerprint(graph_hashes),
        "collection_sha256": artifact["collection_sha256"],
        "summary": artifact["summary"],
    }


def export_dataset(manifest_path: Path, dataset: str, output_root: Path) -> dict:
    manifest = load_yaml(manifest_path)
    try:
        spec = manifest["datasets"][dataset]
    except KeyError as exc:
        raise FrozenExportError(f"Unknown frozen dataset: {dataset}") from exc
    source = resolve_source(str(spec["source_cache"]))
    if not source.is_file():
        raise FileNotFoundError(f"Source cache is missing: {source}")
    actual_sha256 = sha256_file(source)
    if actual_sha256 != spec["source_cache_sha256"]:
        raise FrozenExportError(
            f"Source cache hash mismatch for {dataset}: {actual_sha256}"
        )

    # The hash check above is the trust boundary for this local pickle.
    with source.open("rb") as stream:
        cache = pickle.load(stream)
    if not isinstance(cache, Mapping):
        raise FrozenExportError("GraphVAE cache payload must be a mapping")
    metadata = cache.get("cache_metadata") or {}
    expected_metadata = {
        "dataset": dataset,
        "split_mode": "paper_70_10_20",
        "split_seed": 123,
        "train_fraction": 0.7,
        "val_fraction": 0.1,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise FrozenExportError(
                f"{dataset} cache metadata {key}={metadata.get(key)!r}; "
                f"expected {expected!r}"
            )

    dataset_dir = output_root.expanduser().resolve() / dataset.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "train": "real_train_graphs.pt",
        "validation": "real_validation_graphs.pt",
        "reference": "real_test_graphs.pt",
    }
    results = {}
    for split in ("train", "validation", "reference"):
        adjacencies, nodes, _ = split_values(cache, split)
        expected_raw = int(spec["raw_counts"][split])
        if len(adjacencies) != expected_raw:
            raise FrozenExportError(
                f"{dataset} {split} has {len(adjacencies)} source graphs; "
                f"expected {expected_raw}"
            )
        if spec["feature_mode"] == "topology_control":
            nodes = topology_node_values(adjacencies)
            node_info = {0: {"feature_name": "constant", "value": 1}}
        else:
            node_info = cache.get("node_onehot_info")
        result = export_split(
            dataset=dataset,
            split=split,
            adjacencies=adjacencies,
            node_values=nodes,
            node_info=node_info,
            feature_mode=spec["feature_mode"],
            feature_schema=spec["feature_schema"],
            cache_sha256=actual_sha256,
            output_path=dataset_dir / filenames[split],
        )
        expected_accepted = int(spec["accepted_counts"][split])
        if result["accepted_graph_count"] != expected_accepted:
            raise FrozenExportError(
                f"{dataset} {split} accepted {result['accepted_graph_count']} "
                f"graphs; expected {expected_accepted}"
            )
        dimensions = result["summary"]
        if dimensions["node_feature_dim"] != int(spec["node_feature_dim"]):
            raise FrozenExportError(f"{dataset} node feature dimension changed")
        if dimensions["edge_feature_dim"] != int(spec["edge_feature_dim"]):
            raise FrozenExportError(f"{dataset} edge feature dimension changed")
        results[split] = result

    summary = {
        "schema_version": "defog-graphvae-frozen-export-v1",
        "dataset": dataset,
        "source_cache": str(source),
        "source_cache_sha256": actual_sha256,
        "artifacts": results,
    }
    (dataset_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = export_dataset(
        args.manifest.expanduser().resolve(),
        args.dataset.upper(),
        args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
