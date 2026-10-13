"""Direct attributed-array/PyG/DGL adapters with explicit feature alignment.

No conversion passes through NetworkX.  DGL edge IDs are read in order and
PyG edge rows are written in exactly the same order.  Historical DGL inputs
may contain self-loops, a single direction, or repeated reverse directions;
the importer merges them into the strict bidirectional PyG contract and
rejects conflicting attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from .contract import normalize_pyg_graph, validate_pyg_graph
from .io import load_pyg_collection, save_pyg_collection


def attributed_arrays_to_pyg(
    edges,
    node_attributes,
    edge_attributes=None,
    source_node_ids=None,
    *,
    name: str = "attributed graph",
) -> Data:
    """Build a strict PyG graph from aligned undirected attribute arrays.

    ``edges`` must contain each undirected edge exactly once as a canonical
    pair ``u < v``.  The adapter adds the reverse direction and duplicates the
    corresponding edge-attribute row.  This small boundary lets dense graph
    decoders export the PyG interchange contract without constructing a DGL
    graph or losing the decoder's original node IDs.
    """

    edges_np = np.asarray(edges, dtype=np.int64)
    node_np = np.asarray(node_attributes, dtype=np.float32)
    if edges_np.ndim != 2 or edges_np.shape[1] != 2:
        raise ValueError(
            f"{name}.edges must have shape (E, 2), got {edges_np.shape}."
        )
    if node_np.ndim != 2 or node_np.shape[1] < 1:
        raise ValueError(
            f"{name}.node_attributes must have shape (N, D>=1), "
            f"got {node_np.shape}."
        )
    if not len(edges_np):
        raise ValueError(f"{name} must contain at least one undirected edge.")
    if np.any(edges_np[:, 0] >= edges_np[:, 1]):
        raise ValueError(
            f"{name}.edges must contain canonical undirected pairs with u < v."
        )
    if int(edges_np.min()) < 0 or int(edges_np.max()) >= len(node_np):
        raise ValueError(f"{name}.edges contains an invalid node index.")
    if len({tuple(row) for row in edges_np.tolist()}) != len(edges_np):
        raise ValueError(f"{name}.edges contains duplicate undirected pairs.")

    undirected_edge_attr = None
    if edge_attributes is not None:
        edge_np = np.asarray(edge_attributes, dtype=np.float32)
        if edge_np.ndim != 2 or edge_np.shape[0] != len(edges_np):
            raise ValueError(
                f"{name}.edge_attributes must have shape "
                f"({len(edges_np)}, D), got {edge_np.shape}."
            )
        if edge_np.shape[1] > 0:
            undirected_edge_attr = edge_np

    reverse_edges = edges_np[:, ::-1]
    edge_index = torch.as_tensor(
        np.concatenate((edges_np, reverse_edges), axis=0).T.copy(),
        dtype=torch.int64,
    )
    edge_attr = (
        None
        if undirected_edge_attr is None
        else torch.as_tensor(
            np.concatenate(
                (undirected_edge_attr, undirected_edge_attr), axis=0
            ),
            dtype=torch.float32,
        )
    )
    graph = Data(
        x=torch.as_tensor(node_np, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(node_np),
    )
    if source_node_ids is not None:
        source_ids_np = np.asarray(source_node_ids, dtype=np.int64)
        if source_ids_np.shape != (len(node_np),):
            raise ValueError(
                f"{name}.source_node_ids must have shape ({len(node_np)},), "
                f"got {source_ids_np.shape}."
            )
        graph.source_node_ids = torch.as_tensor(
            source_ids_np.copy(), dtype=torch.int64
        )
    return normalize_pyg_graph(graph, name=name)


def _require_dgl():
    try:
        import dgl
    except ImportError as exc:
        raise RuntimeError(
            "DGL is required only for DGL conversion and legacy evaluation. "
            "Install the DGL build matching your PyTorch/CUDA environment."
        ) from exc
    return dgl


def _require_dgl_float_matrix(value, *, name: str, rows: int):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 2 or int(value.shape[0]) != rows:
        raise ValueError(
            f"{name} must have shape ({rows}, D), got {tuple(value.shape)}."
        )
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values.")


def dgl_to_pyg(graph, *, name: str = "DGL graph") -> Data:
    """Convert one homogeneous DGL graph into strict bidirectional PyG."""

    dgl = _require_dgl()
    if not isinstance(graph, dgl.DGLGraph):
        raise TypeError(f"{name} must be a DGLGraph.")
    if not graph.is_homogeneous:
        raise TypeError(f"{name} must be homogeneous.")
    if int(getattr(graph, "batch_size", 1)) != 1:
        raise ValueError(f"{name} must be unbatched.")

    node_count = int(graph.num_nodes())
    if "attr" not in graph.ndata:
        raise ValueError(f"{name} is missing ndata['attr'].")
    x = graph.ndata["attr"]
    _require_dgl_float_matrix(x, name=f"{name}.ndata['attr']", rows=node_count)

    try:
        sources, targets = graph.edges(order="eid")
    except TypeError:  # DGL 0.6 compatibility
        sources, targets = graph.edges()
    sources_np = sources.detach().cpu().numpy().astype(np.int64, copy=False)
    targets_np = targets.detach().cpu().numpy().astype(np.int64, copy=False)
    edge_count = len(sources_np)

    raw_attr = graph.edata.get("attr")
    edge_attr_np = None
    if raw_attr is not None:
        _require_dgl_float_matrix(
            raw_attr, name=f"{name}.edata['attr']", rows=edge_count
        )
        edge_attr_np = raw_attr.detach().cpu().numpy().astype(
            np.float32, copy=False
        )

    rows_by_pair: dict[Tuple[int, int], List[np.ndarray | None]] = {}
    for edge_id, (raw_source, raw_target) in enumerate(
        zip(sources_np, targets_np)
    ):
        source = int(raw_source)
        target = int(raw_target)
        if source == target:
            continue
        pair = (min(source, target), max(source, target))
        row = None if edge_attr_np is None else edge_attr_np[edge_id]
        rows_by_pair.setdefault(pair, []).append(row)
    if not rows_by_pair:
        raise ValueError(f"{name} has no non-self-loop edges.")

    canonical_rows = {}
    for pair, rows in rows_by_pair.items():
        if edge_attr_np is None:
            canonical_rows[pair] = None
            continue
        first = rows[0]
        if any(
            not np.allclose(first, row, rtol=1e-5, atol=1e-7)
            for row in rows[1:]
        ):
            raise ValueError(
                f"{name} has conflicting attributes for undirected edge {pair}."
            )
        canonical_rows[pair] = np.mean(np.stack(rows), axis=0)

    pairs = sorted(canonical_rows)
    directed_edges = pairs + [(target, source) for source, target in pairs]
    edge_index = torch.tensor(directed_edges, dtype=torch.int64).T.contiguous()
    if edge_attr_np is None:
        edge_attr = None
    else:
        undirected_attr = np.stack([canonical_rows[pair] for pair in pairs])
        edge_attr = torch.as_tensor(
            np.concatenate((undirected_attr, undirected_attr), axis=0),
            dtype=torch.float32,
        )

    result = Data(
        x=x.detach().cpu().to(torch.float32).contiguous(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=node_count,
    )
    return normalize_pyg_graph(result, name=name)


def pyg_to_dgl(graph: Data, *, name: str = "PyG graph"):
    """Convert one strict PyG graph to DGL without changing edge order."""

    dgl = _require_dgl()
    validate_pyg_graph(graph, name=name)
    normalized = normalize_pyg_graph(graph, name=name)
    result = dgl.graph(
        (
            normalized.edge_index[0].detach().cpu(),
            normalized.edge_index[1].detach().cpu(),
        ),
        num_nodes=int(normalized.num_nodes),
    )
    result.ndata["attr"] = normalized.x.detach().cpu().to(torch.float32)
    if getattr(normalized, "edge_attr", None) is not None:
        result.edata["attr"] = normalized.edge_attr.detach().cpu().to(
            torch.float32
        )
    return result


def load_dgl_collection(path) -> list:
    """Load an individual-graph DGL binary collection."""

    dgl = _require_dgl()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DGL graph collection not found: {source}")
    graphs, _ = dgl.load_graphs(str(source))
    if not graphs:
        raise ValueError(f"DGL graph collection is empty: {source}")
    return list(graphs)


def convert_dgl_file_to_pyg(
    input_path,
    output_path,
    *,
    metadata: dict | None = None,
) -> dict:
    """Convert a DGL binary collection to the safe PyG tensor payload."""

    dgl_graphs = load_dgl_collection(input_path)
    pyg_graphs = [
        dgl_to_pyg(graph, name=f"DGL graphs[{index}]")
        for index, graph in enumerate(dgl_graphs)
    ]
    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        {
            "source_format": "dgl.save_graphs",
            "source_path": str(Path(input_path).expanduser().resolve()),
        }
    )
    return save_pyg_collection(
        output_path,
        pyg_graphs,
        metadata=artifact_metadata,
    )


def convert_pyg_file_to_dgl(
    input_path,
    output_path,
    *,
    trusted: bool = False,
) -> dict:
    """Convert a PyG collection into a DGL binary legacy artifact."""

    dgl = _require_dgl()
    pyg_graphs = load_pyg_collection(input_path, trusted=trusted)
    dgl_graphs = [
        pyg_to_dgl(graph, name=f"PyG graphs[{index}]")
        for index, graph in enumerate(pyg_graphs)
    ]
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(destination), dgl_graphs)
    return {
        "output": str(destination),
        "graph_count": len(dgl_graphs),
        "source": str(Path(input_path).expanduser().resolve()),
    }
