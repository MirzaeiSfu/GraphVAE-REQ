"""Strict PyTorch Geometric interchange contract.

The public boundary is a collection of individual, homogeneous
``torch_geometric.data.Data`` objects.  The evaluator intentionally accepts a
small subset of PyG:

* ``x`` is a finite floating-point matrix of shape ``[N, D_node]``;
* ``edge_index`` is an ``int64`` matrix of shape ``[2, E]``;
* ``edge_attr`` is absent or a finite float matrix ``[E, D_edge]``;
* undirected edges appear in both directions with matching attributes;
* self-loops and duplicate directed edges are forbidden.

These restrictions make the conversion to DGL lossless and auditable.  They
also prevent backend-specific transforms from silently changing feature/edge
alignment.  Normalization keeps the deterministic largest connected component
and preserves the producer's original node IDs in optional
``source_node_ids``.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch, Data


FEATURE_MODES = (
    "topology_control",
    "decoded_node",
    "decoded_edge",
    "decoded_node_edge",
)


@dataclass(frozen=True)
class CollectionSummary:
    """Shape and size metadata shared by one compatible graph collection."""

    graph_count: int
    node_feature_dim: int
    edge_feature_dim: int
    total_nodes: int
    directed_edge_count: int

    def to_dict(self) -> dict:
        """Return JSON-serializable summary metadata."""

        return asdict(self)


def _require_float_matrix(value, *, name: str, rows: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 2 or int(value.shape[0]) != rows:
        raise ValueError(
            f"{name} must have shape ({rows}, D), got {tuple(value.shape)}."
        )
    if not torch.is_floating_point(value):
        raise TypeError(
            f"{name} must be floating point. One-hot encode categorical IDs "
            "before evaluation."
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return value


def _edge_rows(graph: Data) -> Tuple[np.ndarray, np.ndarray | None]:
    edges = graph.edge_index.detach().cpu().numpy().T.astype(
        np.int64, copy=False
    )
    edge_attr = getattr(graph, "edge_attr", None)
    if edge_attr is None:
        return edges, None
    return (
        edges,
        edge_attr.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _validated_source_node_ids(graph: Data) -> torch.Tensor:
    """Return valid provenance IDs, creating local IDs when absent."""

    node_count = int(graph.num_nodes)
    source_node_ids = getattr(graph, "source_node_ids", None)
    if source_node_ids is None:
        return torch.arange(node_count, dtype=torch.int64)
    if not isinstance(source_node_ids, torch.Tensor):
        raise TypeError(f"{type(graph).__name__}.source_node_ids must be a tensor.")
    if source_node_ids.dtype != torch.int64:
        raise TypeError("source_node_ids must use torch.int64.")
    if source_node_ids.ndim != 1 or len(source_node_ids) != node_count:
        raise ValueError(
            "source_node_ids must have shape "
            f"({node_count},), got {tuple(source_node_ids.shape)}."
        )
    if int(torch.unique(source_node_ids).numel()) != node_count:
        raise ValueError("source_node_ids must be unique within a graph.")
    return source_node_ids.detach().cpu()


def validate_pyg_graph(
    graph: Data,
    *,
    name: str = "graph",
    require_bidirectional: bool = True,
) -> Tuple[int, int]:
    """Validate one graph and return ``(node_dim, edge_dim)``.

    ``Batch`` objects are rejected because collection boundaries must remain
    visible to the evaluator.  Call ``Batch.to_data_list()`` in the producer.
    """

    if isinstance(graph, Batch):
        raise TypeError(
            f"{name} is a PyG Batch. Pass individual Data objects instead."
        )
    if not isinstance(graph, Data):
        raise TypeError(
            f"{name} must be torch_geometric.data.Data, got "
            f"{type(graph).__module__}.{type(graph).__name__}."
        )

    if graph.num_nodes is None:
        raise ValueError(f"{name}.num_nodes must be defined.")
    node_count = int(graph.num_nodes)
    if node_count < 1:
        raise ValueError(f"{name} must contain at least one node.")
    _validated_source_node_ids(graph)

    x = getattr(graph, "x", None)
    if x is None:
        raise ValueError(
            f"{name}.x is required; topology-created fallback features are "
            "not part of the interchange contract."
        )
    _require_float_matrix(x, name=f"{name}.x", rows=node_count)
    if int(x.shape[1]) < 1:
        raise ValueError(f"{name}.x needs at least one feature channel.")

    edge_index = getattr(graph, "edge_index", None)
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"{name}.edge_index must be a torch.Tensor.")
    if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
        raise ValueError(
            f"{name}.edge_index must have shape (2, E), got "
            f"{tuple(edge_index.shape)}."
        )
    if edge_index.dtype != torch.int64:
        raise TypeError(
            f"{name}.edge_index must use torch.int64, got {edge_index.dtype}."
        )
    edge_count = int(edge_index.shape[1])
    if edge_count < 1:
        raise ValueError(f"{name} must contain at least one directed edge.")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count:
        raise ValueError(f"{name}.edge_index contains an invalid node index.")

    edge_attr = getattr(graph, "edge_attr", None)
    edge_dim = 0
    if edge_attr is not None:
        _require_float_matrix(
            edge_attr, name=f"{name}.edge_attr", rows=edge_count
        )
        edge_dim = int(edge_attr.shape[1])
        if edge_dim < 1:
            raise ValueError(
                f"{name}.edge_attr should be absent rather than have zero columns."
            )

    edges, edge_rows = _edge_rows(graph)
    directed: dict[Tuple[int, int], int] = {}
    for edge_id, (raw_source, raw_target) in enumerate(edges):
        source = int(raw_source)
        target = int(raw_target)
        if source == target:
            raise ValueError(
                f"{name} contains self-loop ({source}, {target}). Evaluator "
                "backends add loops according to their own published protocol."
            )
        pair = (source, target)
        if pair in directed:
            raise ValueError(f"{name} contains duplicate directed edge {pair}.")
        directed[pair] = edge_id

    if require_bidirectional:
        for pair, edge_id in directed.items():
            reverse = (pair[1], pair[0])
            reverse_id = directed.get(reverse)
            if reverse_id is None:
                raise ValueError(
                    f"{name} is missing reverse edge {reverse} for {pair}."
                )
            if edge_rows is not None and not np.allclose(
                edge_rows[edge_id],
                edge_rows[reverse_id],
                rtol=1e-5,
                atol=1e-7,
            ):
                raise ValueError(
                    f"{name} has conflicting edge_attr values on {pair} and "
                    f"{reverse}."
                )

    return int(x.shape[1]), edge_dim


def _largest_component_node_ids(graph: Data) -> List[int]:
    """Return the deterministic largest component using a small union-find."""

    node_count = int(graph.num_nodes)
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    edges = graph.edge_index.detach().cpu().numpy()
    for source, target in zip(edges[0], edges[1]):
        union(int(source), int(target))

    components: dict[int, List[int]] = {}
    for node in range(node_count):
        components.setdefault(find(node), []).append(node)
    nontrivial = [nodes for nodes in components.values() if len(nodes) > 1]
    if not nontrivial:
        raise ValueError("Graph has no connected component containing an edge.")
    return max(nontrivial, key=lambda nodes: (len(nodes), -min(nodes)))


def normalize_pyg_graph(graph: Data, *, name: str = "graph") -> Data:
    """Validate and retain the deterministic largest connected component."""

    validate_pyg_graph(graph, name=name)
    retained = sorted(_largest_component_node_ids(graph))
    input_source_node_ids = _validated_source_node_ids(graph)
    if len(retained) == int(graph.num_nodes):
        result = Data(
            x=graph.x.detach().cpu().to(torch.float32).contiguous(),
            edge_index=graph.edge_index.detach().cpu().to(torch.int64).contiguous(),
            edge_attr=(
                None
                if getattr(graph, "edge_attr", None) is None
                else graph.edge_attr.detach().cpu().to(torch.float32).contiguous()
            ),
            num_nodes=int(graph.num_nodes),
        )
        result.source_node_ids = input_source_node_ids.clone()
        return result

    retained_set = set(retained)
    local_id = {source_id: index for index, source_id in enumerate(retained)}
    raw_edges = graph.edge_index.detach().cpu().numpy().T
    edge_ids = [
        edge_id
        for edge_id, (source, target) in enumerate(raw_edges)
        if int(source) in retained_set and int(target) in retained_set
    ]
    local_edges = torch.tensor(
        [
            [local_id[int(raw_edges[edge_id, 0])], local_id[int(raw_edges[edge_id, 1])]]
            for edge_id in edge_ids
        ],
        dtype=torch.int64,
    ).T.contiguous()

    result = Data(
        x=graph.x.detach().cpu()[retained].to(torch.float32).contiguous(),
        edge_index=local_edges,
        edge_attr=(
            None
            if getattr(graph, "edge_attr", None) is None
            else graph.edge_attr.detach().cpu()[edge_ids]
            .to(torch.float32)
            .contiguous()
        ),
        num_nodes=len(retained),
    )
    result.source_node_ids = input_source_node_ids[retained].clone()
    validate_pyg_graph(result, name=f"normalized {name}")
    return result


def apply_feature_mode(graph: Data, mode: str) -> Data:
    """Create the input view used to train and evaluate one encoder.

    Unlike the historical Random-GIN ablation, trained encoders use the
    natural dimensionality of each view.  A topology-only model therefore gets
    one constant node channel and no edge attributes.  The same mode must be
    used for encoder training and evaluation.
    """

    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode {mode!r}: {FEATURE_MODES}.")
    edge_attr = getattr(graph, "edge_attr", None)
    if mode in {"decoded_edge", "decoded_node_edge"} and edge_attr is None:
        if mode == "decoded_edge":
            raise ValueError("decoded_edge mode requires edge_attr.")

    x = (
        graph.x.detach().cpu().to(torch.float32).contiguous()
        if mode in {"decoded_node", "decoded_node_edge"}
        else torch.ones((int(graph.num_nodes), 1), dtype=torch.float32)
    )
    selected_edges = (
        edge_attr.detach().cpu().to(torch.float32).contiguous()
        if mode in {"decoded_edge", "decoded_node_edge"} and edge_attr is not None
        else None
    )
    result = Data(
        x=x,
        edge_index=graph.edge_index.detach().cpu().to(torch.int64).contiguous(),
        edge_attr=selected_edges,
        num_nodes=int(graph.num_nodes),
    )
    if hasattr(graph, "source_node_ids"):
        result.source_node_ids = graph.source_node_ids.detach().cpu().clone()
    validate_pyg_graph(result, name=f"{mode} graph")
    return result


def _materialize(graphs: Iterable[Data], *, name: str) -> List[Data]:
    if isinstance(graphs, (Data, Batch, str, bytes)):
        raise TypeError(f"{name} must be a collection of individual Data objects.")
    try:
        result = list(graphs)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a collection of individual Data objects."
        ) from exc
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def validate_collection(
    graphs: Iterable[Data],
    *,
    name: str = "graphs",
    minimum_graphs: int = 1,
) -> CollectionSummary:
    """Validate dimensions and return collection-level metadata."""

    materialized = _materialize(graphs, name=name)
    if len(materialized) < minimum_graphs:
        raise ValueError(
            f"{name} needs at least {minimum_graphs} graphs, got "
            f"{len(materialized)}."
        )
    dimensions = [
        validate_pyg_graph(graph, name=f"{name}[{index}]")
        for index, graph in enumerate(materialized)
    ]
    node_dims = {item[0] for item in dimensions}
    edge_dims = {item[1] for item in dimensions}
    if len(node_dims) != 1 or len(edge_dims) != 1:
        raise ValueError(
            f"{name} has inconsistent feature dimensions: "
            f"node={sorted(node_dims)}, edge={sorted(edge_dims)}."
        )
    return CollectionSummary(
        graph_count=len(materialized),
        node_feature_dim=next(iter(node_dims)),
        edge_feature_dim=next(iter(edge_dims)),
        total_nodes=sum(int(graph.num_nodes) for graph in materialized),
        directed_edge_count=sum(
            int(graph.edge_index.shape[1]) for graph in materialized
        ),
    )


def prepare_collection(
    graphs: Iterable[Data],
    *,
    mode: str = "decoded_node_edge",
    name: str = "graphs",
    minimum_graphs: int = 1,
) -> List[Data]:
    """Normalize a collection and apply the encoder's feature mode."""

    materialized = _materialize(graphs, name=name)
    prepared = [
        apply_feature_mode(
            normalize_pyg_graph(graph, name=f"{name}[{index}]"),
            mode,
        )
        for index, graph in enumerate(materialized)
    ]
    validate_collection(
        prepared, name=f"prepared {name}", minimum_graphs=minimum_graphs
    )
    return prepared


def _hash_tensor(hasher, name: str, tensor: torch.Tensor | None):
    hasher.update(name.encode("utf-8"))
    if tensor is None:
        hasher.update(b"<none>")
        return
    array = tensor.detach().cpu().contiguous().numpy()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(str(tuple(array.shape)).encode("ascii"))
    hasher.update(array.tobytes(order="C"))


def collection_digest(
    graphs: Sequence[Data],
    *,
    mode: str = "decoded_node_edge",
) -> str:
    """Hash normalized topology and feature tensors in collection order."""

    prepared = prepare_collection(graphs, mode=mode, name="digest graphs")
    hasher = hashlib.sha256()
    hasher.update(f"ggm-eval-pyg-v1:{mode}".encode("ascii"))
    for index, graph in enumerate(prepared):
        hasher.update(str(index).encode("ascii"))
        _hash_tensor(hasher, "x", graph.x)
        _hash_tensor(hasher, "edge_index", graph.edge_index)
        _hash_tensor(hasher, "edge_attr", getattr(graph, "edge_attr", None))
    return hasher.hexdigest()
