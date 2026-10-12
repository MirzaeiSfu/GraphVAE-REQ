"""Versioned serialization for collections that satisfy the PyG contract.

The runtime API uses PyG ``Data`` objects.  On disk we store only tensors and
primitive metadata rather than pickled ``Data`` instances.  This remains a
PyG interchange format—the payload maps one-to-one to ``Data``—while allowing
modern PyTorch to use its restricted ``weights_only`` loader.

Raw ``torch.save(list_of_data)`` files are accepted only with ``trusted=True``.
They execute Python pickle semantics and are tied more closely to PyG versions.
"""

from __future__ import annotations

import inspect
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, List

import torch
from torch_geometric.data import Data

from .contract import (
    collection_digest,
    prepare_collection,
    validate_collection,
)


FORMAT_NAME = "ggm-eval-pyg-tensors"
FORMAT_VERSION = 1


def _tensor_record(graph: Data) -> dict:
    return {
        "num_nodes": int(graph.num_nodes),
        "x": graph.x.detach().cpu().to(torch.float32).contiguous(),
        "edge_index": graph.edge_index.detach()
        .cpu()
        .to(torch.int64)
        .contiguous(),
        "edge_attr": (
            None
            if getattr(graph, "edge_attr", None) is None
            else graph.edge_attr.detach().cpu().to(torch.float32).contiguous()
        ),
        "source_node_ids": (
            graph.source_node_ids.detach().cpu().to(torch.int64).contiguous()
            if getattr(graph, "source_node_ids", None) is not None
            else torch.arange(int(graph.num_nodes), dtype=torch.int64)
        ),
    }


def _from_record(record: dict, *, name: str) -> Data:
    required = {"num_nodes", "x", "edge_index", "edge_attr"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}.")
    graph = Data(
        x=record["x"],
        edge_index=record["edge_index"],
        edge_attr=record["edge_attr"],
        num_nodes=int(record["num_nodes"]),
    )
    if record.get("source_node_ids") is not None:
        graph.source_node_ids = record["source_node_ids"]
    return graph


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _validated_metadata(metadata: dict | None) -> dict:
    """Return detached JSON primitives or reject unsafe/custom objects."""

    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("Collection metadata must be a dictionary.")
    candidate = dict(metadata or {})
    try:
        encoded = json.dumps(candidate, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Collection metadata must contain only JSON-serializable "
            "primitive values."
        ) from exc
    return json.loads(encoded)


def save_pyg_collection(
    path,
    graphs: Iterable[Data],
    *,
    metadata: dict | None = None,
    normalize: bool = True,
) -> dict:
    """Save a collection and adjacent JSON manifest.

    The default normalization enforces the common largest-component policy
    before the collection crosses a repository boundary.
    """

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_metadata = _validated_metadata(metadata)
    materialized = list(graphs)
    prepared = (
        prepare_collection(
            materialized,
            mode="decoded_node_edge",
            name="saved graphs",
        )
        if normalize
        else materialized
    )
    summary = validate_collection(prepared, name="saved graphs")
    digest = collection_digest(prepared)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "collection_sha256": digest,
        "graphs": [_tensor_record(graph) for graph in prepared],
        "metadata": safe_metadata,
    }
    try:
        pyg_version = importlib_metadata.version("torch-geometric")
    except importlib_metadata.PackageNotFoundError:
        pyg_version = None
    torch.save(payload, destination)
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "collection_sha256": digest,
        "summary": summary.to_dict(),
        "metadata": safe_metadata,
        "torch_version": torch.__version__,
        "torch_geometric_version": pyg_version,
    }
    _manifest_path(destination).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _torch_load(path: Path, *, trusted: bool):
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            if not trusted:
                raise ValueError(
                    f"{path} is not a restricted tensor-only collection. "
                    "Pass trusted=True only for a file produced by a trusted "
                    "source."
                ) from exc
    elif not trusted:
        raise RuntimeError(
            "This PyTorch version cannot restrict pickle loading. Upgrade "
            "PyTorch or explicitly mark the input as trusted."
        )
    return torch.load(path, map_location="cpu")


def load_pyg_collection_with_metadata(
    path,
    *,
    trusted: bool = False,
    normalize: bool = True,
) -> tuple[List[Data], dict]:
    """Load graphs plus artifact metadata from safe or trusted input."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PyG graph collection not found: {source}")
    payload = _torch_load(source, trusted=trusted)
    expected_digest = None
    artifact_metadata = {}

    if isinstance(payload, dict) and payload.get("format") == FORMAT_NAME:
        if int(payload.get("version", -1)) != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported {FORMAT_NAME} version: {payload.get('version')}."
            )
        expected_digest = payload.get("collection_sha256")
        artifact_metadata = _validated_metadata(payload.get("metadata"))
        graphs = [
            _from_record(record, name=f"graphs[{index}]")
            for index, record in enumerate(payload.get("graphs", []))
        ]
    elif trusted and isinstance(payload, Data):
        graphs = [payload]
    elif trusted and isinstance(payload, (list, tuple)):
        graphs = list(payload)
    else:
        raise ValueError(
            f"{source} is not a {FORMAT_NAME} payload. Raw PyG objects require "
            "trusted=True."
        )

    if not graphs:
        raise ValueError(f"Graph collection is empty: {source}")
    if normalize:
        graphs = prepare_collection(
            graphs,
            mode="decoded_node_edge",
            name=str(source),
        )
    else:
        validate_collection(graphs, name=str(source))
    if expected_digest is not None:
        actual_digest = collection_digest(graphs)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Collection digest mismatch for {source}: expected "
                f"{expected_digest}, got {actual_digest}."
            )
    return graphs, artifact_metadata


def load_pyg_collection(
    path,
    *,
    trusted: bool = False,
    normalize: bool = True,
) -> List[Data]:
    """Load only graphs from the safe tensor payload or trusted raw PyG."""

    graphs, _ = load_pyg_collection_with_metadata(
        path,
        trusted=trusted,
        normalize=normalize,
    )
    return graphs
